import torch
import torch.nn as nn

a = torch.zeros(2, 512, 512, 3).cuda().permute(0, 3, 1, 2).contiguous()
conv1 = nn.Conv2d(3, 3, 1, 1, 0).cuda()
y = conv1(a)
y.sum().backward()
print(y.shape)