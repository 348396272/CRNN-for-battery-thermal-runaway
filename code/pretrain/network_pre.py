'''
Descripttion:
version:
Author: liushuaiqi
Date: 2022-11-02 16:12:29
LastEditors: liushuaiqi
LastEditTime: 2025-1-11 18:03:14
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

class CRNN(nn.Module):
    def __init__(self,relation_nums,source_nums,p_cutoff,lb):
        super(CRNN, self).__init__()
        self.relation_nums=relation_nums
        self.source_nums=source_nums
        self.p_cutoff=p_cutoff # 实际上是一个负数
        self.lb=lb
        # 我的评价是 后面设置那么多约束条件 可能也没什么用

        # self.w_out=torch.nn.Parameter(torch.randn((source_nums,relation_nums),),requires_grad=True)
        # self.w_b=torch.nn.Parameter(torch.randn(relation_nums),requires_grad=True)
        # self.w_in_Ea=torch.nn.Parameter(torch.randn(relation_nums),requires_grad=True)
        # self.w_in_b=torch.nn.Parameter(torch.randn(relation_nums),requires_grad=True)
        # self.w_in_ocen=torch.nn.Parameter(torch.randn(relation_nums),requires_grad=True)


        #self.data_p=torch.nn.Parameter(torch.randn((source_nums+4)*relation_nums,dtype=torch.float64),requires_grad=True) # nr * (ns+4)
        #self.data_p = np.loadtxt(open("file1.csv","rb"))
        #self.data_p=torch.tensor(self.data_p)
        self.data_p=torch.randn((source_nums+4)*relation_nums+1,dtype=torch.float64)*0.01
        self.data_p[-1]=0.15
        self.data_p[0:relation_nums]+=0.8
        self.data_p[relation_nums*(source_nums+1):relation_nums*(source_nums+2)]+=0.8
        self.data_p[relation_nums*(source_nums+2):relation_nums*(source_nums+3)]+=0.1
        self.data_p[relation_nums*(source_nums+3):relation_nums*(source_nums+4)]+=0.1

        self.data_p=torch.nn.Parameter(self.data_p,requires_grad=True)
        self.hyp_ocen=torch.nn.Parameter(torch.rand(1),requires_grad=True)
        self.hyp_T=torch.nn.Parameter(torch.tensor(4.0),requires_grad=True)
        self.R=-1.0 / 8.314e-3
        self.droplayer=torch.nn.Dropout(0.5)

        self.xiu = True
        self.xiu2 = False
        # self.molecule_linear=nn.Sequential(
        #         nn.Linear(source_nums,3),
        #         nn.ReLU(),
        #         nn.Linear(3,1)
        #     )
       # self.test_linear=torch.nn.Linear(3,4)


    # 注 我只保证我重写的函数功能与 原版一致 至于这部分之前说不需要理解和修改
    # 我对这部分是否有效以及是否会起到正确的作用 持保留意见。
    # 初始化参数 给一个先验值？
    # 实际上后面 梯度更新的时候并没有做同样的限制
    # 处于后续对接的考虑 我依然重写了这部分代码
    # 实在是 典急孝崩麻
    def init_w_b(self,w_b): # 检验正确
        w_b=w_b*self.slope*10
        w_b=torch.clamp(w_b,min=0,max=50)
        return w_b
    def init_w_in_Ea(self,w_in_Ea): # 检验正确
        w_in_Ea=torch.abs(w_in_Ea)*(self.slope*100.0)
        w_in_Ea=torch.clamp(w_in_Ea,min=40,max=300)
        return w_in_Ea
    def init_w_in_b(self,w_in_b): #检验正确
        w_in_b=torch.abs(w_in_b)
        return w_in_b
    def init_w_in_ocen(self,w_in_ocen):#检验正确
        w_in_ocen=torch.abs(w_in_ocen)
        w_in_ocen=torch.clamp(w_in_ocen,0.0,1.5)
        if self.p_cutoff > 0.0: # 由于实际上这个参数是负的 实际上根本不会生效
            w_in_ocen[torch.where(torch.abs(w_in_ocen) < self.p_cutoff)] = 0.0

        return w_in_ocen
    
    def init_w_out(self,w_out):
        # 这个是最复杂的 而且我完全看不懂他要干什么
        #w_out=torch.clamp(w_out,-3.,0)
        w_out0=torch.clamp(w_out[0,:],min=-3.0,max=0.0).unsqueeze(0) #lib
        # w_out0=-torch.clamp(torch.abs(w_out[0,:]),min=0.0,max=3.0).unsqueeze(0) #lib



        w_out_last=torch.clamp(torch.abs(w_out[-1,:]),min=0.0,max=3.0).unsqueeze(0) #gas

        w_out_update=torch.cat([w_out0,w_out[1:-1,:],w_out_last],dim=0)




        if self.p_cutoff > 0.0: # 同 w_in_ocen
            w_out_update[torch.where(torch.abs(w_out_update) < self.p_cutoff)] = 0.0
        w_out_sourcelastsecond = -torch.sum(w_out_update[0:self.source_nums-2, :], dim=0) - w_out_update[self.source_nums-1, :]
        w_out_update2=torch.cat([w_out_update[0:self.source_nums-2,:],w_out_sourcelastsecond.unsqueeze(0),w_out_update[self.source_nums-1,:].unsqueeze(0)],dim=0)
        # w_out[self.source_nums-2, :] = \
        # -torch.sum(w_out[0:self.source_nums-2, :], dim=0) - w_out[self.source_nums-1, :]

        if self.p_cutoff > 0.0: # 同 w_in_ocen
            w_out_update2[torch.where(torch.abs(w_out_update2) < 0.0006)] = 0.0

        #让前两个方程除了lib全为生成物
        w_out1=torch.clamp(torch.abs(w_out[1:,0:2]),min=0,max=3.0)
        w_out1_lib = -torch.sum(w_out1,dim=0).unsqueeze(0)
        w_out1 = torch.cat((w_out1_lib,w_out1),dim=0)
        w_out_update3 = torch.cat((w_out1,w_out_update2[:,2:]),dim=1)

        return w_out_update3

    # def init_w_out(self,w_out):
    #     # 这个是最复杂的 而且我完全看不懂他要干什么
    #     #w_out=torch.clamp(w_out,-3.,0)
    #     w_out0=torch.clamp(torch.abs(w_out[0,:]),min=-3.0,max=0.0).unsqueeze(0) #lib
    #     w_out_last=torch.clamp(torch.abs(w_out[-1,:]),min=0.0,max=3.0).unsqueeze(0) #gas
    #     # w_out_lastsecond = torch.clamp(torch.abs(w_out[-2,:]),min=0.0,max=3.0).unsqueeze(0) #someting

    #     w_out_update=torch.cat([w_out0,w_out[1:-1,:],w_out_last],dim=0)

    #     if self.p_cutoff > 0.0: # 同 w_in_ocen
    #         w_out_update[torch.where(torch.abs(w_out_update) < self.p_cutoff)] = 0.0
    #     # w_out_sourcelastsecond = -torch.sum(w_out_update[0:self.source_nums-2, :], dim=0) - w_out_update[self.source_nums-1, :]
    #     # w_out_update2=torch.cat([w_out_update[0:self.source_nums-2,:],w_out_sourcelastsecond.unsqueeze(0),w_out_update[self.source_nums-1,:].unsqueeze(0)],dim=0)
    #     # w_out[self.source_nums-2, :] = \
    #     # -torch.sum(w_out[0:self.source_nums-2, :], dim=0) - w_out[self.source_nums-1, :]
    #     return w_out_update


    def init_state(self,ini_state,t,temp):
        self.T0=ini_state[0]
        self.beta=ini_state[1]
        self.ocen=ini_state[2]
        self.slope=self.data_p[-1]*10 #:)
        relation_nums=self.relation_nums
        source_nums=self.source_nums
        self.w_b=self.data_p[0:relation_nums]
        self.w_in_Ea=self.data_p[relation_nums*(source_nums+1):relation_nums*(source_nums+2)]
        self.w_in_b=self.data_p[relation_nums*(source_nums+2):relation_nums*(source_nums+3)]
        self.w_in_ocen=self.data_p[relation_nums*(source_nums+3):relation_nums*(source_nums+4)]
        self.w_out=self.data_p[relation_nums:relation_nums*(source_nums+1)].reshape(relation_nums,source_nums).swapaxes(0,1)
        #temp=torch.clamp(-self.w_out,0.0,4.0)
        #self.w_in=torch.concat([temp,self.w_in_Ea.unsqueeze(0),self.w_in_b.unsqueeze(0),self.w_in_ocen.unsqueeze(0)],dim=0) # 验证完成
        #self.w_in=torch.concat([-self.w_out,self.w_in_Ea.unsqueeze(0),self.w_in_b.unsqueeze(0),self.w_in_ocen.unsqueeze(0)],dim=0) # 验证完成
        #self.w_in=torch.nn.Parameter(self.w_in,requires_grad=True) # 转化为可训练参数
        # self.w_b=torch.nn.Parameter(self.w_b,requires_grad=True)
        # self.w_out=torch.nn.Parameter(self.w_out,requires_grad=True)
        # self.w_in_b=torch.nn.Parameter(self.w_in_b,requires_grad=True)
        # self.w_in_ocen=torch.nn.Parameter(self.w_in_ocen,requires_grad=True)
        # self.w_in_Ea=torch.nn.Parameter(self.w_in_Ea,requires_grad=True)
        self.w_b=self.init_w_b(self.w_b)
        self.w_in_Ea=self.init_w_in_Ea(self.w_in_Ea)
        self.w_in_b=self.init_w_in_b(self.w_in_b)
        self.w_in_ocen=self.init_w_in_ocen(self.w_in_ocen)
        self.w_out=self.init_w_out(self.w_out)

        self.inter_temp=scipy.interpolate.interp1d(t.cpu().numpy(), temp.cpu().numpy(), kind='slinear',fill_value="extrapolate")
        self.min_t=t.min()
        self.max_t=t.max()
        # time=[i for i in range(int(self.min_t),int(self.max_t)+1)]
        # temp=self.inter_temp(time)
        # import matplotlib.pyplot as plt
        # plt.plot(time, temp.squeeze(), 'o')
        # plt.show()


    def forward(self,t,x,type="default"):

        for param in self.parameters():
            if torch.isnan(param).any():
                print("NaN detected in model parameters")
                raise ValueError("NaN detected in model parameters")


        if torch.isnan(x).any():
            print("NaN detected in input data")
            raise ValueError("NaN detected in input data")
            
        
        if isinstance(x,torch.Tensor):
            pass
        else:
            x=torch.from_numpy(x)
            t=torch.tensor(t)

        xorgin=x
        x=x.clamp(self.lb,10.0)
        # if t>self.max_t:
        #     t=self.max_t+100
        # if t<self.min_t:
        #     t=self.min_t
        logx=torch.log(x)

        # threshold = -45
        # logx=torch.where(logx < threshold, torch.full_like(logx,-1000), logx)

        self.ocen_update=torch.log(self.ocen*torch.abs(self.hyp_ocen))
        T=self.inter_temp(t.cpu().detach().numpy())
        T=torch.tensor(np.array(T))
        if T.dim() == 0:
            T = T.unsqueeze(0)
        if T.dim() == 2:
            T = T.squeeze(1)
        input_x=torch.cat((logx.squeeze(), (self.R / T), self.hyp_T*torch.log(T), self.ocen_update),dim=0).unsqueeze(1)


        # self.w_out1=torch.where(self.w_out.abs() < 0.0005, torch.zeros_like(self.w_out), self.w_out)



        if self.xiu:
            # w_in=torch.concat([torch.clamp(-self.w_out,0.0,10.0),self.w_in_Ea.unsqueeze(0),self.w_in_b.unsqueeze(0),self.w_in_ocen.unsqueeze(0)],dim=0) # 验证完成
            # w_in_x = torch.mm(w_in.T, input_x)
            # w_in_x2=torch.exp(w_in_x+self.w_b.unsqueeze(1))
            # logic = (xorgin.t() <= self.lb) & (self.w_out < 0)
            # logic=~torch.all(~logic,dim=0).unsqueeze(1)
            # w_in_x3 = torch.where(logic,w_in_x2-w_in_x2,w_in_x2)
            # du3=torch.mm(self.w_out,w_in_x3)

            w_in=torch.concat([torch.clamp(-self.w_out,0.0,10.0),self.w_in_Ea.unsqueeze(0),self.w_in_b.unsqueeze(0),self.w_in_ocen.unsqueeze(0)],dim=0) # 验证完成
            w_in_x = torch.mm(w_in.T, input_x)
            w_in_x2=torch.exp(w_in_x+self.w_b.unsqueeze(1))
      
            du1=torch.mm(self.w_out,w_in_x2)
            logic = (xorgin.t() <= self.lb) & (du1 < 0)
            temp=torch.where(~logic, torch.zeros_like(du1), du1)
            temp=torch.sum(torch.abs(temp))
            du2=torch.where(logic, torch.zeros_like(du1), du1)
            if temp<=du2[-1]:
                du2[-1]-=temp
            else:
                temp=temp-du2[-1]                
                du2[-1]-=du2[-1]
                # du2[1]-=temp

            du3=du2

        else:

            w_in=torch.concat([torch.clamp(-self.w_out,0.0,10.0),self.w_in_Ea.unsqueeze(0),self.w_in_b.unsqueeze(0),self.w_in_ocen.unsqueeze(0)],dim=0) # 验证完成
            w_in_x = torch.mm(w_in.T, input_x)
            w_in_x2=torch.exp(w_in_x+self.w_b.unsqueeze(1))
            
            threshold = 1e-6
            w_in_x3 = torch.where(abs(w_in_x2) < threshold, torch.zeros_like(w_in_x2), w_in_x2)


            du3=torch.mm(self.w_out,w_in_x3)



        # du = torch.clamp(du,-2.5,2.5)

        if torch.isnan(w_in_x).any():
            print("NaN detected in w_in_x")
            raise ValueError("NaN detected in w_in_x")
        if torch.isnan(w_in_x2).any():
            print("NaN detected in w_in_x2")
            raise ValueError("NaN detected in w_in_x2")



        if self.xiu2:
            if xorgin[0][0].item()<self.lb:
                du3=torch.zeros_like(du3)

        return du3.squeeze().unsqueeze(0)
        # else:
        #     x=x.clamp(self.lb,10.0)
        #     logx=torch.log(x)
        #     Temp=t.squeeze().unsqueeze(-1)
        #     ocen_update=torch.log(self.ocen*torch.abs(self.hyp_ocen))
        #     ocen_update=ocen_update.expand(Temp.shape[0],1)
        #     R_T=(self.R/Temp)
        #     T_=torch.log(Temp)
        #     input_x=torch.cat([logx,R_T,T_,ocen_update],dim=-1)
        #     w_in=torch.concat([torch.clamp(-self.w_out,0.0,4.0),self.w_in_Ea.unsqueeze(0),self.w_in_b.unsqueeze(0),self.w_in_ocen.unsqueeze(0)],dim=0)
        #     w_in_x = torch.mm(w_in.T, input_x.T)
        #     w_in_x=torch.exp(w_in_x+self.w_b.unsqueeze(1).expand(-1,w_in_x.shape[-1]))
        #     du=torch.mm(self.w_out,w_in_x)
        #     mole=self.molecule_linear(du.T.to(torch.float32))
        #     return torch.tanh(mole)*4




