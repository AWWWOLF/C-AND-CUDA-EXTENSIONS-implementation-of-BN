from BNLink import BN
from BNLink_cuda import BNCUDA
import torch
import time
from tqdm import tqdm
import torch.nn.functional as F

assert torch.cuda.is_available()
cuda_device = torch.device("cuda")  # device object representing GPU

torch.manual_seed(0)
N, D, H, W = 5, 3, 10, 10
X = torch.randn(N, D, H, W, device=cuda_device)
dX = torch.randn(N, D, H, W, device=cuda_device)
gamma = torch.randn(D, device=cuda_device)
beta = torch.randn(D, device=cuda_device)

bn = BN(D).to(cuda_device)
bn1 = BNCUDA(D).to(cuda_device)
bn2 = torch.nn.BatchNorm2d(D).to(cuda_device)
forward = 0
backward = 0

for _ in tqdm(range(10000)):
    start = time.time()
    output = bn(X)
    forward += time.time() - start
    start = time.time()
    loss = (output - dX).pow(2).sum()    
    loss = loss.requires_grad_(True)    
    loss.backward()
    backward += time.time() - start
print('c++ result: Forward: {:.3f} us | Backward {:.3f} us'.format(forward * 1e6/1e4, backward * 1e6/1e4))

forward = 0
backward = 0
for _ in tqdm(range(10000)):
    start = time.time()
    output = bn1(X)
    forward += time.time() - start
    start = time.time()
    loss = (output - dX).pow(2).sum()    
    loss = loss.requires_grad_(True)    
    loss.backward()
    backward += time.time() - start
print('c++/CUDA result: Forward: {:.3f} us | Backward {:.3f} us'.format(forward * 1e6/1e4, backward * 1e6/1e4))

forward = 0
backward = 0
for _ in tqdm(range(10000)):
    start = time.time()
    output = bn2(X)
    forward += time.time() - start
    start = time.time()
    loss = (output - dX).pow(2).sum()
    loss = loss.requires_grad_(True)
    loss.backward()
    backward += time.time() - start
print('Pytorch result: Forward: {:.3f} us | Backward {:.3f} us'.format(forward * 1e6/1e4, backward * 1e6/1e4))