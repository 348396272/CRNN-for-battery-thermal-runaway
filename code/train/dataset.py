'''
Descripttion:
version:
Author: YinFeiyu
Date: 2022-11-04 17:29:55
LastEditors: YinFeiyu
LastEditTime: 2022-11-23 19:34:32
'''
import numpy as np
import torch
from torch.utils.data import Dataset

class Dataset_wlfc(Dataset):
    def __init__(self,expdata,ini_state,ns,init_m):
        # expdata 是读取到的数据
        # ini_state 是初始状态
        self.expdata=expdata
        self.ini_state=ini_state
        self.ns=ns # 反应物的数量

    def __len__(self):
        #return 1
        return len(self.expdata)

    def __getitem__(self,idx):
        # 依据 torchdiffeq 的 官方示例
        # 这里需要返回 ini_state(初始状态) t(序列) Ylabel(实际标签) Y0(元素初始量)
        # 注意 这里的 ylabel是一个相对值
        # 同样 如果 你要设定反应物的初始值 都需要在全局额外设置一个值乘上这个玩意

        ini_state=self.ini_state[idx]
        t=self.expdata[idx][:,0] # 整个时间序列
        Ylabel=self.expdata[idx][:,-1] # 整个标签序列
        X0=torch.tensor(np.zeros(self.ns,dtype=np.float64)) # 元素初始量
        X0[0]=6  # 除了第一个元素设定为5g
        Temp=self.expdata[idx][:,1]
        Pressure=self.expdata[idx][:,2]
        return ini_state,t,Ylabel,X0,Temp,Pressure