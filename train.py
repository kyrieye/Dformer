
import argparse
import datetime
import os
import pprint
import random
import time
from importlib import import_module
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel
from val_mm import evaluate, evaluate_msf

from builder import EncoderDecoder as segmodel
from dataloader import get_train_loader, get_val_loader
from RGBXDataset import RGBXDataset
from engine import Engine
from logger import get_logger
from init_func import configure_optimizers, group_weight
from lr_policy import WarmUpPolyLR
from pyt_utils import all_reduce_tensor

import warnings
warnings.filterwarnings("ignore")
# from eval import evaluate_mid


parser = argparse.ArgumentParser()
parser.add_argument("--config", help="train config file path")
parser.add_argument("--gpus", default=2, type=int, help="used gpu number")
# parser.add_argument('-d', '--devices', default='0,1', type=str)
parser.add_argument("-v", "--verbose", default=False, action="store_true")
parser.add_argument("--epochs", default=0)
parser.add_argument("--show_image", "-s", default=False, action="store_true")
parser.add_argument("--save_path", default=None)
parser.add_argument("--checkpoint_dir")
parser.add_argument("--continue_fpath")
parser.add_argument("--sliding", default=False, action='store_true')
parser.add_argument("--compile", default=False, action='store_true')
parser.add_argument("--compile_mode", default="default")
parser.add_argument("--syncbn", default=True, action='store_true')
parser.add_argument("--mst", default=True, action='store_true')
parser.add_argument("--amp", default=True, action='store_true')
parser.add_argument("--val_amp", default=True, action='store_true')
parser.add_argument("--pad_SUNRGBD", default=False, action='store_true')
parser.add_argument("--use_seed", default=True, action='store_true')
parser.add_argument("--local-rank", default=0)
# parser.add_argument('--save_path', '-p', default=None)

# os.environ['MASTER_PORT'] = '169710'
torch.set_float32_matmul_precision("high")
import torch._dynamo

torch._dynamo.config.suppress_errors = True
# torch._dynamo.config.automatic_dynamic_shapes = False


def is_eval(epoch, config):
    return epoch > int(config.checkpoint_start_epoch) or epoch == 1 or epoch % 5 == 0


class gpu_timer:
    def __init__(self, beta=0.6) -> None:
        self.start_time = None
        self.stop_time = None
        self.mean_time = None
        self.beta = beta
        self.first_call = True

    def start(self):
        torch.cuda.synchronize()
        self.start_time = time.perf_counter()

    def stop(self):
        if self.start_time is None:
            print("Use start() before stop(). ")
        torch.cuda.synchronize()
        self.stop_time = time.perf_counter()
        elapsed = self.stop_time - self.start_time
        self.start_time = None
        if self.first_call:
            self.mean_time = elapsed
            self.first_call = False
        else:
            self.mean_time = self.beta * self.mean_time + (1 - self.beta) * elapsed


def set_seed(seed):
    # seed init.
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # torch seed init.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True  # train speed is slower after enabling this opts.

    # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    # avoiding nondeterministic algorithms (see https://pytorch.org/docs/stable/notes/randomness.html)
    torch.use_deterministic_algorithms(True, warn_only=True)


with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()
    config = getattr(import_module(args.config), "C")
    logger = get_logger(config.log_dir, config.log_file, rank=engine.local_rank)
    config.pad = args.pad_SUNRGBD
    if args.use_seed:
        set_seed(config.seed)
        logger.info(f"set seed {config.seed}")
    else:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        logger.info("use random seed")

    train_loader, train_sampler = get_train_loader(engine, RGBXDataset, config)

    val_dl_factor = 1.5
    val_loader, val_sampler = get_val_loader(
        engine,
        RGBXDataset,
        config,
        val_batch_size=int(config.batch_size * val_dl_factor) if config.dataset_name != "SUNRGBD" else int(args.gpus),
    )
    logger.info(f"val dataset len:{len(val_loader) * int(args.gpus)}")

    logger.info("args parsed:")
    for k in args.__dict__:
        logger.info(k + ": " + str(args.__dict__[k]))

    criterion = nn.CrossEntropyLoss(reduction="none", ignore_index=config.background)

    if args.syncbn:
        BatchNorm2d = nn.SyncBatchNorm
        logger.info("using syncbn")
    else:
        BatchNorm2d = nn.BatchNorm2d
        logger.info("using regular bn")

    model = segmodel(
        cfg=config,
        criterion=criterion,
        norm_layer=BatchNorm2d,
        syncbn=args.syncbn,
    )
    '''
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    '''
    
    base_lr = config.lr

    params_list = []
    params_list = group_weight(params_list, model, BatchNorm2d, base_lr)

    if config.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            params_list,
            lr=base_lr,
            betas=(0.9, 0.999),
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "SGDM":
        optimizer = torch.optim.SGD(
            params_list,
            lr=base_lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    else:
        raise NotImplementedError

    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = WarmUpPolyLR(
        base_lr,
        config.lr_power,
        total_iteration,
        config.niters_per_epoch * config.warm_up_epoch,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    engine.register_state(dataloader=train_loader, model=model, optimizer=optimizer)
    if engine.continue_state_object:
        engine.restore_checkpoint()

    optimizer.zero_grad()

    logger.info("begin trainning:")
    data_setting = {
        "rgb_root": config.rgb_root_folder,
        "rgb_format": config.rgb_format,
        "gt_root": config.gt_root_folder,
        "gt_format": config.gt_format,
        "transform_gt": config.gt_transform,
        "x_root": config.x_root_folder,
        "x_format": config.x_format,
        "x_single_channel": config.x_is_single_channel,
        "class_names": config.class_names,
        "train_source": config.train_source,
        "eval_source": config.eval_source,
    }

    all_dev = [0]

    p, best_p, p_epoch = 0.0, 0.0, -1
    r, best_r, r_epoch = 0.0, 0.0, -1
    f, best_f, f_epoch = 0.0, 0.0, -1


    for epoch in range(engine.state.epoch, config.nepochs + 1):
        model.train()
        dataloader = iter(train_loader)
        sum_loss = 0
        i = 0
        for idx in range(config.niters_per_epoch):
            engine.update_iteration(epoch, idx)
            minibatch = next(dataloader)
            imgs = minibatch["data"]
            gts = minibatch["label"]

            imgs = imgs.cuda(non_blocking=True)
            gts = gts.cuda(non_blocking=True)

            loss = model(imgs,label=gts)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_idx = (epoch - 1) * config.niters_per_epoch + idx
            lr = lr_policy.get_lr(current_idx)

            for i in range(len(optimizer.param_groups)):
                optimizer.param_groups[i]["lr"] = lr

            sum_loss += loss
            print_str = (
                f"Epoch {epoch}/{config.nepochs} "
                + f"Iter {idx + 1}/{config.niters_per_epoch}: "
                + f"lr={lr:.4e} loss={loss:.4f} total_loss={(sum_loss / (idx + 1)):.4f}"
            )

            if ((idx + 1) % 5 == 0 or idx == 0):
                print(print_str)

            del loss
        logger.info(print_str)
        if is_eval(epoch, config):
            torch.cuda.empty_cache()
            with torch.no_grad():
                f1s=[]
                ps = []
                rs = []
                for idx, minibatch in enumerate(val_loader):
                    model.eval()
                    device = torch.device("cuda")
                
                    imgs = minibatch["data"]
                    gts = minibatch["label"]
                    
                    imgs = imgs.cuda(non_blocking=True)

                    index = model.evaluate(imgs)

                    f1s.append(f1_score( gts, index, average='macro' ))
                    ps.append(precision_score(gts, index, average='macro'))
                    rs.append(recall_score(gts, index, average='macro'))

                f1s = sum(f1s)/len(f1s)
                ps = sum(ps)/len(ps)
                rs = sum(rs)/len(rs)
                print(f1s, ps, rs)

                if ps > best_p:
                    best_p = ps
                    p_epoch = epoch
                if rs > best_r:
                    best_r = rs
                    r_epoch = epoch
                if f1s > best_f:
                    best_f = f1s
                    f_epoch = epoch
            logger.info("Epoch {0} validation result: P {1:.4f}, R {2:.4f}, F {3:.4f}".format(epoch,ps,rs,f1s))
            logger.info("best P {0:.4f} Epoch {1}, best R {2:.4f} Epoch {3}, best F1 {4:.4f} Epoch {5}".format(best_p,p_epoch,best_r,r_epoch,best_f,f_epoch))
