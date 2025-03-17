'''
Descripttion:
version:
Author: YinFeiyu
Date: 2022-11-03 16:40:35
LastEditors: YinFeiyu
LastEditTime: 2022-12-11 19:20:16
'''
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from TorchDiffEqPack.odesolver import odesolve
from torchdiffeq import odeint
import matplotlib.pyplot as plt
def getsampletemp(t, T0, beta): # 验证完成
    if beta < 100:
        T = T0 + beta / 60 * t  # K/min to K/s  # 实际上 根本不会>=100 初始值beta在 10-20之间
    else: # 无用代码
        tc = torch.tensor(np.array([999.0, 1059.0]) * 60.0)
        Tc = torch.tensor([beta, 370.0, 500.0]) + 273.0
        HR = 40.0 / 60.0
        if t <= tc[0]:
            T = Tc[0]
        elif t <= tc[1]:
            T=Tc[1] if Tc[0] + HR * (t - tc[0])<Tc[1] else Tc[0] + HR * (t - tc[0])
            #T = torch.min([Tc[0] + HR * (t - tc[0]), Tc[1]])
        else:
            T=Tc[2] if Tc[1] + HR * (t - tc[1])<Tc[2] else Tc[1] + HR * (t - tc[1])
            #T = torch.min(([Tc[1] + HR * (t - tc[1]), Tc[2]]))
    return T

def mass_laws(mass,gass,all_m):

    mass_m=torch.relu(mass)
    gas_m=torch.relu(gass)

    mass_m=mass[:,:-1].sum(-1)
    gass_m=gass

    now_m=mass_m+gass_m
    loss_masslaws=torch.abs((all_m-now_m).sum()) # 质量守恒

    # # 物质质量不能为负数
    # mass_reverse=-mass
    # gass_reverse=-gass
    # mass_reverse=F.relu(mass_reverse) # 显然 如果一开始 都是正数 然后取反 再过relu将全部被设置为0
    # gass_reverse=F.relu(gass_reverse)
    # loss_zero=torch.norm(mass_reverse)+torch.norm(gass_reverse)
    return loss_masslaws


def plot_show(dataloader,network,net_mole,epoch,test_or_train,result_dir):
    network=network
    for i,(ini_state,t,Y_label,X0,Temp) in enumerate(dataloader):
        ini_state=ini_state.squeeze()
        t=t.squeeze()
        Y_label=Y_label.squeeze()
        X0=X0.squeeze()
        network.init_state(ini_state,t,Temp)
        options = {}
        options.update({'method': 'ode23s'})
        options.update({'h': None})
        options.update({'t0': t.min()})
        options.update({'t1': t.max()})
        options.update({'rtol': 1e-3})
        options.update({'atol': 1e-3})
        options.update({'t_eval':t})
        options.update({'interpolation_method':'linear'})
        y_predict = odesolve(network, X0.unsqueeze(0), options = options).squeeze()
        
        # y_predict=torch.clamp(y_predict,0.,5.)
        mole=net_mole(y_predict,Temp)
        y_predict=torch.relu(y_predict)
        mass=y_predict[:,:-1]
        gas=y_predict[:,-1]

        gas_v=gas/(mole.squeeze())

        # gas_v=torch.clamp(gas_v,0,Y_label[-1])
        mass=y_predict[:,:-1]
        gas=y_predict[:,-1]
        # mass=torch.relu(mass)
        # gas=torch.relu(gas)
        fig, ax = plt.subplots()
        ax.ticklabel_format(useOffset=False)
        ax.plot(t, Y_label, marker='o',label='True',lw=2) # 作y1 = x 图，并标记此线名为linear
        ax.plot(t, gas_v, marker='o',label='Predict',lw=2) # 作y2 = x^2 图，并标记此线名为quadratic
        plt.savefig(f"{result_dir}/{epoch}_PRE{i}_{test_or_train}.png")
        ax.cla()
        ax.ticklabel_format(useOffset=False)
        ax.plot(t,torch.relu(y_predict).numpy(),lw=2)
        plt.savefig(f"{result_dir}/{epoch}_item{i}_{test_or_train}.png")
        ax.cla()
        ax.ticklabel_format(useOffset=False)
        ax.plot(t,mole,lw=2)
        plt.savefig(f"{result_dir}/{epoch}_mole{i}_{test_or_train}.png")
        ax.cla()
        plt.close()

        df = pd.DataFrame({
            'times': t.numpy(),
            'Y_label': Y_label.numpy(),
            'gas_v': gas_v.numpy(),
            'mole': mole.squeeze().numpy()
        })
        df.to_csv(f"{result_dir}/{epoch}_GasPred{i}_{test_or_train}.csv",index=False)

