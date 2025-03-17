'''
Descripttion:
version:
Author: YinFeiyu
Date: 2022-11-02 16:16:19
LastEditors: YinFeiyu
LastEditTime: 2022-11-15 20:49:51
'''
import numpy as np
import torch
import os
import pandas as pd
# 我们 julia 真的是 太快辣！ = =
class Readdata:

    def __init__(self,sample_size,llb,front_or_back):
        self.sample_size=sample_size # 数据集大小
        self.llb=llb # 不知道是什么
        self.data=[]
        self.ini_state=np.zeros((sample_size,3)) # 设置初始状态
        for (i, value) in enumerate(range(sample_size)):
            if front_or_back=="FRONT":
                filename = os.path.join(f"exp_data_2/front_exp_{i+1}.txt") # python 从0 开始呢
            elif front_or_back=="BACK":
                filename=os.path.join(f"exp_data_2/back_exp_{i+1}.txt")
            elif front_or_back=='DOUBLE':
                filename=os.path.join(f"exp_data_2/front_and_back_exp_{i+1}.txt")
            else:
                print("ERROR: front_or_back should be FRONT or BACK")
                break
            data_read = self.load_file(filename,i)
            self.data.append(data_read)
            self.ini_state[i,0]=data_read[0,1] # 读入 初始的温度
        self.ini_state_fun()
        print(1)

    def load_file(self,filename,num):
        data=pd.read_csv(filename,header=None,delimiter='\t')

        # 很显然 time的设置没必要设置如此长
        # 而且后续的t 我们直接设置插值函数
        # 所以这里我的建议是直接缩放
        data.drop_duplicates(subset=[0],keep='first',inplace=True)
        data=data.values
        #data[:,2]/=max(data[:,2])  ##/ 按照需求 设置质量，则不需要归一化 归一化也行吧

        # if num==3:
        #     data=data[0:60,:]
        # elif num==4:
        #     data=data[0:58,:]
        # elif num==5:
        #     data=data[0:60,:]
        # elif num==6:
        #     data=data[0:71,:]
        return data

    def ini_state_fun(self):
        beta=pd.read_csv("exp_data_2/beta.txt",header=None).values
        ocen=pd.read_csv("exp_data_2/ocen.txt",header=None).values
        # l_exp_info[:, 2] = readdlm("new_data/beta.txt")[l_exp];
        # l_exp_info[:, 3] = log.(readdlm("new_data/ocen.txt")[l_exp] .+ llb);
        self.ini_state[:,1]=beta.reshape(-1)
        self.ini_state[:,2]=(ocen+self.llb).reshape(-1)
        # self.ini_state[:,1]=np.concatenate([beta,beta]).reshape(-1)
        # self.ini_state[:,2]=np.log(np.concatenate([ocen,ocen])+self.llb).reshape(-1)




