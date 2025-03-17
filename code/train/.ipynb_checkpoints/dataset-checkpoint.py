'''
Descripttion:
version:
Author: YinFeiyu
Date: 2022-11-04 17:29:55
LastEditors: YinFeiyu
LastEditTime: 2022-11-17 17:09:17
'''
import numpy as np
import torch
from torch.utils.data import Dataset

class Dataset_wlfc(Dataset):
    def __init__(self,expdata,ini_state,ini_X0,ini_mole,ns,init_m):
        # expdata 是读取到的数据
        # ini_state 是初始状态
        self.expdata=expdata
        self.ini_state=ini_state
        self.ns=ns # 反应物的数量
        self.ini_X0=ini_X0
        self.ini_mole=ini_mole

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

        # # 方法1：取front的state vector[-1]去掉gas的质量，作为back的初始state vector
        # X0[:-1]=self.ini_X0[idx,:-1]
        # Temp=self.expdata[idx][:,1]

        # return ini_state,t,Ylabel,X0,Temp

        # # 方法2：取front的average mol value乘以真实值gas的mol量作为gas的质量，求出质量差值再按比例分配
        # X0[-1]=Ylabel[0]*self.ini_mole[idx]
        # other_m=5.0-Ylabel[0]*self.ini_mole[idx]  #减去初始气体后的固体量
        # # 然后按照比例分配
        # X0_proportion=self.ini_X0[idx,:-1]/self.ini_X0[idx,:-1].sum()
        # X0[:-1]=other_m*X0_proportion

        # Temp=self.expdata[idx][:,1]

        # return ini_state,t,Ylabel,X0,Temp

        # trick：质量差值按比例分配, 气体质量对齐back的初始状态
        other_m=5.0-Ylabel[0]*self.ini_mole[idx]  #减去初始气体后的固体量
        X0_proportion=self.ini_X0[idx,:-1]/self.ini_X0[idx,:-1].sum()
        X0[:-1]=other_m*X0_proportion # 按照比例分配
        if idx!=len(self.ini_mole)-1:
            X0[-1] = Ylabel[0]*self.ini_mole[idx]
        else:
            X0[-1] = Ylabel[0]*21.19005594

        Temp=self.expdata[idx][:,1]

        return ini_state,t,Ylabel,X0,Temp