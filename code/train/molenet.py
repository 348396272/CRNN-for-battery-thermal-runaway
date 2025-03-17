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
                nn.LayerNorm(source_nums+3),
                nn.Linear(source_nums+3,18),
                nn.Tanh(),
                nn.Linear(18,7),
                nn.Tanh(),
                nn.Linear(7,5),
                nn.Tanh(),
                nn.Linear(5,1),
                nn.Tanh(),

        )


    def forward(self,x,T,P):
        x=torch.cat((x,T.T/100,1/P.T,T.T/100*1/P.T),dim=1)
        x=x.to(torch.float32)
        y_1=self.molecule_linear(x)*18

        return 28+y_1



