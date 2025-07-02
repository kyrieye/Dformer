import torch
import torch.nn.functional as F

a = torch.randn((2,3,2))
b = torch.randn((2,3,2))
vec1 = a.repeat(1,1,3).reshape(2,3*3,2)
vec2 = b.repeat(1,3,1).reshape(2,3*3,2)

print(a,b)

print(vec1,vec2)

cos_sim = F.cosine_similarity(vec1, vec2, dim=-1)
print(cos_sim.shape) 