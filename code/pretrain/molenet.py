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
                # nn.Linear(5,1),
                # nn.Tanh()
        )
        # self.molecule_linear=nn.Sequential(
        #         nn.LayerNorm(source_nums+2),
        #         nn.Linear(source_nums+2,18),
        #         nn.Tanh(),
        #         # nn.Linear(source_nums*4,10),
        #         # nn.Tanh(),
        #         nn.Linear(18,5),
        #         nn.Tanh(),
        #         nn.Linear(5,1),
        #         nn.Tanh(),
        #         # nn.Linear(5,1),
        #         # nn.Tanh()
        # )

        

        # self.lay1 = nn.Linear(source_nums+2,source_nums)
        # 只需要一层网络
        # 输入分别是元素的量
        # self.T_a=torch.nn.Parameter(torch.tensor(0.01),requires_grad=True)
        # self.T_b=torch.nn.Parameter(torch.tensor(-6.0),requires_grad=True)

    def start(self):
        self.T_a=torch.nn.Parameter(torch.tensor(-2.5),requires_grad=True)
        self.T_b=torch.nn.Parameter(torch.tensor(700.0),requires_grad=True)


    def forward(self,x,T,P):
        x=torch.cat((x,T.T/100,1/P.T,T.T/100*1/P.T),dim=1)
        x=x.to(torch.float32)
        # T1 = torch.relu(T-690)
        T1=T-self.T_b
        y_1=self.molecule_linear(x)*18 + self.T_a*torch.sigmoid(T1.T)
        # y_1=torch.clamp(y_1,min=-10,max=10)
        # y_1=self.lay1(x)
        # y_1=torch.tanh(y_1)*10 # 控制这个值的值域在0到4之间
        # y_1=self.molecule_linear(x)
        # y_1=torch.tanh(y_1)*4 # 控制这个值的值域在0到4之间
        # T=T.T
        # y_2=torch.tanh(self.T_a*T+self.T_b)*2

        return 28+y_1



