'''
Descripttion:
version:
Author: YinFeiyu
Date: 2022-11-02 17:26:54
LastEditors: YinFeiyu
LastEditTime: 2022-11-16 16:13:47
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import utils
import scipy
import scipy.interpolate
import time

class Mole(nn.Module):
    def __init__(self,relation_nums,source_nums):
        super(Mole, self).__init__()
        self.molecule_linear=nn.Sequential(
                nn.Linear(source_nums,source_nums),
                nn.Linear(source_nums,3),
                nn.Tanh(),
                nn.Linear(3,1))
        # 只需要一层网络
        # 输入分别是元素的量
        self.T_a=torch.nn.Parameter(torch.tensor(0.01),requires_grad=True)
        self.T_b=torch.nn.Parameter(torch.tensor(-6.0),requires_grad=True)


    def forward(self,x,T):
        x=x.to(torch.float32)
        y_1=self.molecule_linear(x)
        y_1=torch.tanh(y_1)*4 # 控制这个值的值域在0到4之间
        T=T.T
        y_2=torch.tanh(self.T_a*T+self.T_b)*2

        return 24+y_1+y_2



