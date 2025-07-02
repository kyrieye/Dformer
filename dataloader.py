import cv2
import torch
import numpy as np
from torch.utils import data
import random

# from config import config
# from train import config
from transforms import (
    generate_random_crop_pos,
    random_crop_pad_to_shape,
    normalize,
)


def random_mirror(rgb):
    if random.random() >= 0.5:
        rgb = cv2.flip(rgb, 1)

    return rgb


def random_scale(rgb, scales):
    scale = random.choice(scales)
    sh = int(rgb.shape[0] * scale)
    sw = int(rgb.shape[1] * scale)
    rgb = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_LINEAR)

    return rgb, scale


class TrainPre(object):
    def __init__(self, norm_mean, norm_std, sign=False, config=None):
        self.config = config
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.sign = sign

    def __call__(self, rgb):
        rgb = random_mirror(rgb)
        if self.config.train_scale_array is not None:
            rgb, scale = random_scale(rgb, self.config.train_scale_array)

        rgb = normalize(rgb, self.norm_mean, self.norm_std)

        # return rgb.transpose(2, 0, 1), gt, modal_x.transpose(2, 0, 1)

        crop_size = (self.config.image_height, self.config.image_width)
        crop_pos = generate_random_crop_pos(rgb.shape[:2], crop_size)

        p_rgb, _ = random_crop_pad_to_shape(rgb, crop_pos, crop_size, 0)

        p_rgb = p_rgb.transpose(2, 0, 1)
        # p_rgb = p_rgb
        # p_modal_x = p_modal_x

        return p_rgb


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(0)


class ValPre(object):
    def __init__(self, norm_mean, norm_std, sign=False, config=None):
        self.config = config
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.sign = sign

    def __call__(self, rgb):
        # rgb = cv2.resize(
        #     rgb,
        #     (self.config.image_width, self.config.image_height),
        #     interpolation=cv2.INTER_LINEAR,
        # )
        # gt = cv2.resize(
        #     gt,
        #     (self.config.image_width, self.config.image_height),
        #     interpolation=cv2.INTER_NEAREST,
        # )
        # modal_x = cv2.resize(
        #     modal_x,
        #     (self.config.image_width, self.config.image_height),
        #     interpolation=cv2.INTER_LINEAR,
        # )

        rgb = normalize(rgb, self.norm_mean, self.norm_std)
        return rgb.transpose(2, 0, 1)
        # return rgb, gt, modal_x


def get_train_loader(engine, dataset, config):
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
        "class_names": config.class_names,
        "dataset_name": config.dataset_name,
        "backbone": config.backbone,
    }
    train_preprocess = TrainPre(config.norm_mean, config.norm_std, config.x_is_single_channel, config)

    train_dataset = dataset(
        data_setting,
        "train",
        train_preprocess,
        config.batch_size * config.niters_per_epoch,
    )

    train_sampler = None
    is_shuffle = True
    batch_size = config.batch_size

    if engine.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        batch_size = config.batch_size // engine.world_size
        is_shuffle = False

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=config.num_workers,
        drop_last=True,
        shuffle=is_shuffle,
        pin_memory=True,
        sampler=train_sampler,
        # worker_init_fn=seed_worker,
        # generator=g,
    )

    return train_loader, train_sampler


def get_val_loader(engine, dataset, config, val_batch_size=1):
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
        "class_names": config.class_names,
        "dataset_name": config.dataset_name,
        "backbone": config.backbone,
    }
    val_preprocess = ValPre(config.norm_mean, config.norm_std, config.x_is_single_channel, config)

    val_dataset = dataset(data_setting, "val", val_preprocess)

    val_sampler = None
    is_shuffle = False
    batch_size = val_batch_size

    if engine.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        batch_size = val_batch_size // engine.world_size
        is_shuffle = False

    val_loader = data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=config.num_workers,
        drop_last=False,
        shuffle=is_shuffle,
        pin_memory=True,
        sampler=val_sampler,
        # worker_init_fn=seed_worker,
        # generator=g,
    )

    return val_loader, val_sampler
