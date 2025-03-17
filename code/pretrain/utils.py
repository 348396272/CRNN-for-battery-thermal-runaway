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
import sys
from pathlib import Path
# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent # 根据实际层级调整
sys.path.append(str(project_root))

from modifiedTorchDiffEqPack import odesolve
from torchdiffeq import odeint
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d  
import torchode as to



def variable_step_tensor(start, end, length ,alpha = 3):
    """
    生成一个变步长的一维张量，确保最后一个值是end。
    
    参数:
    start -- 序列的起始值
    end -- 序列的结束值
    length -- 生成的张量长度
    alpha -- 控制步长变化的参数,越大则点越在后方密集
    返回:
    一个一维张量，其步长逐渐减小，且最后一个值是end。
    """
    # 确保起始值小于结束值
    if start >= end:
        raise ValueError("起始值必须小于结束值")
    
    # 计算总步长
    total_step = end - start
    
    # 初始化张量
    t1=torch.linspace(0,1,length)
    t2=torch.tanh(alpha*t1)
    t2=t2/t2.max()
    out = start+t2*total_step
    
    return out




def my_solver(ode_func, t_eval ,y0 ,method='Tsit5', rtol=1e-5, atol=1e-5, **kwargs):
    counters = 0
    while True:  
        term = to.ODETerm(ode_func)
        if method == 'dopri5':
            step_method = to.Dopri5(term=term)
        elif method == 'euler':
            step_method = to.Euler(term=term)
        elif method == 'Heun':
            step_method = to.Heun(term=term)
        elif method == 'Tsit5':
            step_method = to.Tsit5(term=term)
        step_size_controller = to.IntegralController(atol=atol, rtol=rtol, term=term)
        solver = to.AutoDiffAdjoint(step_method, step_size_controller)
        jit_solver = torch.compile(solver)
        sol = jit_solver.solve(to.InitialValueProblem(y0=y0,t_start=t_eval.min().unsqueeze(0),t_end=t_eval.max().unsqueeze(0), t_eval=t_eval.unsqueeze(0)))
        if torch.any(torch.isnan(sol.ys.squeeze(0))):
            counters += 1
            atol *= 0.1
            rtol *= 0.1
        else:
            break
    return sol.ys.squeeze(0)

def solveode(front_net_crnn, X0, options):

    predict=odesolve(front_net_crnn, X0, options = options).squeeze()

    return predict


def plot_grad_flow(named_parameters,name):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.'''
    ave_grads = []
    max_grads = []
    layers = []
    ave_params = []
    max_params = []
    for n, p in named_parameters:
        if p.requires_grad and "bias" not in n:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
            ave_params.append(p.detach().abs().mean())
            max_params.append(p.detach().abs().max())
    
    plt.figure(figsize=(10, 10))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    # plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([plt.Line2D([0], [0], color="c", lw=4),
                plt.Line2D([0], [0], color="b", lw=4),
                plt.Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.savefig(name+"_grad_flow.png")
    plt.close()


    plt.figure(figsize=(10, 10))
    plt.bar(np.arange(len(max_params)), max_params, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_params)), ave_params, alpha=0.1, lw=1, color="b")
    plt.xticks(range(0,len(ave_params), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_params))
    # plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average weights")
    plt.title("weights")
    plt.grid(True)
    plt.legend([plt.Line2D([0], [0], color="c", lw=4),
                plt.Line2D([0], [0], color="b", lw=4)
                ], ['max_params', 'mean_params'])
    plt.savefig(name+"_para.png")
    plt.close()


    flag = None
    for n, p in named_parameters:
        if torch.isnan(p.grad).any():
            print(n, 'nan grad')
            print(p.grad)
            flag = 'nan'

    return flag


# 假设你有一个模型实例model
# 在训练循环中，执行反向传播后调用plot_grad_flow
# for epoch in range(num_epochs):
#     for batch in dataloader:
#         # 前向传播、计算损失、反向传播
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, targets)
#         loss.backward()
#         optimizer.step()
#     # 在每个epoch结束后绘制梯度流动图
#     plot_grad_flow(model.named_parameters())

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


def plot_show(front_dataloader,back_dataloader,front_net_crnn,back_net_crnn,net_mole,epoch,test_or_train,back_result_dir,front_result_dir,convert_time,train_list,val_list,result_data):
    if test_or_train == 'test':
        data_list  = val_list
    if test_or_train == 'train':
        data_list  = train_list

    for i in data_list:
        data = result_data[i]
        t_total = data['t_total']
        Y_label = data['Y_label']
        gas_v = data['gas_v']   
        front_end = data['front_end']   
        y_predict = data['y_predict']
        mole = data['mole']




        
    

        fig, ax = plt.subplots()
        ax.ticklabel_format(useOffset=False)

        ax.plot(t_total[0:front_end+1], Y_label[0:front_end+1], marker='o',label='True',lw=2) 
        ax.plot(t_total[0:front_end+1], gas_v[0:front_end+1], marker='o',label='Predict',lw=2) 
        plt.savefig(f"{front_result_dir}/{epoch}_PRE{i}_{test_or_train}.png")
        ax.cla()
        ax.ticklabel_format(useOffset=False)
        ax.plot(t_total[front_end:], Y_label[front_end:], marker='o',label='True',lw=2) 
        ax.plot(t_total[front_end:], gas_v[front_end:], marker='o',label='Predict',lw=2) 
        # plt.xlim(t_total[front_end], t_total[front_end+100])
        
        plt.savefig(f"{back_result_dir}/{epoch}_PRE{i}_{test_or_train}.png")
        ax.cla()
        # ax.ticklabel_format(useOffset=False)
        # ax.plot(t,torch.relu(y_predict).numpy(),lw=2)
        for ii in range(len(y_predict.T)): 
            ax.ticklabel_format(useOffset=False)
            ax.plot(t_total[0:front_end+1],torch.relu(y_predict[:,ii][0:front_end+1]).numpy(),lw=2,label=str(ii))
        plt.legend()
        plt.savefig(f"{front_result_dir}/{epoch}_item{i}_{test_or_train}.png")
        ax.cla()
        for ii in range(len(y_predict.T)): 
            ax.ticklabel_format(useOffset=False)
            ax.plot(t_total[front_end:],torch.relu(y_predict[:,ii][front_end:]).numpy(),lw=2,label=str(ii))
        plt.legend()
        # plt.xlim(t_total[front_end], t_total[front_end+100])
        plt.savefig(f"{back_result_dir}/{epoch}_item{i}_{test_or_train}.png")


        ax.cla()
        ax.ticklabel_format(useOffset=False)
        ax.plot(t_total[0:front_end+1],mole[0:front_end+1],lw=2)
        plt.savefig(f"{front_result_dir}/{epoch}_mole{i}_{test_or_train}.png")
   
        ax.cla()
        ax.ticklabel_format(useOffset=False)
        ax.plot(t_total[front_end:],mole[front_end:],lw=2)
        # plt.xlim(t_total[front_end], t_total[front_end+100])
        plt.savefig(f"{back_result_dir}/{epoch}_mole{i}_{test_or_train}.png")
   
        plt.close()

        df = pd.DataFrame({
            'times': t_total.numpy(),
            'Y_label': Y_label.numpy(),
            'gas_v': gas_v.numpy(),
            'mole': mole.squeeze().numpy()
        })
        df.to_csv(f"{front_result_dir}/{epoch}_GasPred{i}_{test_or_train}.csv",index=False)

