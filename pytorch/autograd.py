'''
https://www.jianshu.com/p/a105858567df
'''


import torch
from torch.autograd import Variable

x = torch.tensor([[1.,2.,3.],[4.,5.,6.]],requires_grad=True)
y = x+1
z = 2*y*y
J = torch.mean(z)
J.backward()
print(x.grad)