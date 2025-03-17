'''
Descripttion:
version:
Author: YinFeiyu
Date: 2022-11-02 16:12:29
LastEditors: YinFeiyu
LastEditTime: 2022-12-11 18:03:14
'''
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import Settings
from dataset import Dataset_wlfc
from torch.utils.data import DataLoader
import Readdata
from network_pre import CRNN
import torch.optim as optim
import torch.nn.functional as F
from TorchDiffEqPack.odesolver import odesolve
from torchdiffeq import odeint
from molenet import Mole
from earlystopping import EarlyStopping
import utils
from scipy.interpolate import interp1d  
import csv
from matplotlib import pyplot as plt
from utils import solveode
# torch.set_default_dtype(torch.float64)



if __name__ == '__main__':
    seed=42
    pretrain = False
    using_pretrain_model = False
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    

    front_lossgain=10
    back_lossgain=1
    front_number = 250
    back_number = 60
    front_alpha = 2.5
    back_alpha = 1
    acc_interp = 10
    acc_interp2 = 5
    gas_gain = 700
    mass_gain = 1
    mass_gain2 = 10
    tail_number=10
    head_number=20
    tail_time = 5
    time_delta = [0,0,0,0]
    time_head = [10,10,15,5]
    scheduler_time = 500
    setting=Settings.Settings()
    setting.parse() # 配置参数

    # 然后要加载数据咯
    # 首先要读入对吧
    front_readdata=Readdata.Readdata(setting.sample_size,setting.llb,"FRONT")
    back_readdata=Readdata.Readdata(setting.sample_size,setting.llb,"BACK")
    # 因为数据已经分段
    # 所以你必须这样处理

    # 一开始 觉得可能没有必要写一个 dataset
    # 想想还是算了 写吧 模块化就要模块化到底

#     train_list=[1,2,3,4,5]
#     # train_list=[0,2,3,4,6,7,9,10,12,13] # 注意下标从0开始哦
#     val_list=[0]
    
    train_list=[0,1,1,3,3]
    val_list=[2]

    # comment="_train0313_test2_adddata_old_frontlossgain10_500decay"
    comment='_01133_wholeloss_vairablestep2_molegai_withloss3_lib4.5_withtail5_test15'
    convert_time={0:49085.0,1:51400.0,2:61070.08,3:89409.0}
    for i in time_delta:
        convert_time[i]+=time_delta[i]

    #back过程数据
    back_result_dir = "./backsave/back{}{}_{}".format(setting.ns,setting.nr,setting.nr_back)+comment
    if not os.path.exists(back_result_dir):
        os.mkdir(back_result_dir)
    back_model_dir = f'./backmodelsave/model{setting.ns}{setting.nr}_{setting.nr_back}'+comment
    if not os.path.exists(back_model_dir):
        os.mkdir(back_model_dir)          

    #front过程数据
    front_result_dir = "./frontsave/front{}{}_{}".format(setting.ns,setting.nr,setting.nr_back)+comment
    if not os.path.exists(front_result_dir):
        os.mkdir(front_result_dir)
    front_model_dir = f'./frontmodelsave/model{setting.ns}{setting.nr}_{setting.nr_back}'+comment
    if not os.path.exists(front_model_dir):
        os.mkdir(front_model_dir)    

    mole_model_dir = f'./molemodelsave/model{setting.ns}{setting.nr}_{setting.nr_back}'+comment
    if not os.path.exists(mole_model_dir):
        os.mkdir(mole_model_dir)          
    
    model_best_dir = f'./modelbest/model{setting.ns}{setting.nr}_{setting.nr_back}'+comment
    if not os.path.exists(model_best_dir):
        os.mkdir(model_best_dir)    



    if pretrain:
        back_result_dir=back_result_dir+'/pretrain'
        if not os.path.exists(back_result_dir):
            os.mkdir(back_result_dir)      

        back_model_dir=back_model_dir+'/pretrain'
        if not os.path.exists(back_model_dir):
            os.mkdir(back_model_dir)

        front_result_dir=front_result_dir+'/pretrain'
        if not os.path.exists(front_result_dir):
            os.mkdir(front_result_dir)

        front_model_dir=front_model_dir+'/pretrain'
        if not os.path.exists(front_model_dir):
            os.mkdir(front_model_dir)
        mole_model_dir=mole_model_dir+'/pretrain'
        if not os.path.exists(mole_model_dir):
            os.mkdir(mole_model_dir)

        model_best_pretrain_dir=model_best_dir+'/pretrain'
        if not os.path.exists(model_best_pretrain_dir):
            os.mkdir(model_best_pretrain_dir)

        

    
    front_train_data=[]
    front_val_data=[]
    front_train_ini_state=[]
    front_val_ini_state=[]
    for n in train_list:
        front_train_data.append(front_readdata.data[n])
        front_train_ini_state.append(front_readdata.ini_state[n])
    for n in val_list:
        front_val_data.append(front_readdata.data[n])
        front_val_ini_state.append(front_readdata.ini_state[n])


    back_train_data=[]
    back_val_data=[]
    back_train_ini_state=[]
    back_val_ini_state=[]
    for n in train_list:
        back_train_data.append(back_readdata.data[n])
        back_train_ini_state.append(back_readdata.ini_state[n])
    for n in val_list:
        back_val_data.append(back_readdata.data[n])
        back_val_ini_state.append(back_readdata.ini_state[n])



    #dataset
    front_train_dataset = Dataset_wlfc(front_train_data, front_train_ini_state,setting.ns,setting.init_m)
    front_test_dataset = Dataset_wlfc(front_val_data,front_val_ini_state,setting.ns,setting.init_m)

    back_train_dataset = Dataset_wlfc(back_train_data, back_train_ini_state,setting.ns,setting.init_m)
    back_test_dataset = Dataset_wlfc(back_val_data,back_val_ini_state,setting.ns,setting.init_m)

    #dataloader
    front_dataloader = DataLoader(front_train_dataset, batch_size=1, shuffle=False)
    front_dataloader_test=DataLoader(front_test_dataset,batch_size=1,shuffle=False)

    back_dataloader = DataLoader(back_train_dataset, batch_size=1, shuffle=False)
    back_dataloader_test=DataLoader(back_test_dataset,batch_size=1,shuffle=False)

    bar = tqdm(total=setting.n_epoch) # 2500
    bar.set_description('Training')

    #net
    front_net_crnn=CRNN(setting.nr,setting.ns,setting.p_cutoff,setting.lb)
    back_net_crnn=CRNN(setting.nr_back,setting.ns,setting.p_cutoff,setting.lb)
    net_mole=Mole(setting.nr,setting.ns)

    # net_crnn.load_state_dict(torch.load('model/back_net_crnn1799.pth',weights_only=True))
    # net_mole.load_state_dict(torch.load('model/back_net_mole1799.pth',weights_only=True))

    optimizer = optim.Adam(
        [
            {'params': front_net_crnn.parameters(),'lr':setting.lr_max,'momentum':0.9,'weight_decay':setting.w_decay,'eps':1e-4},
            {'params': back_net_crnn.parameters(),'lr':setting.lr_max,'momentum':0.9,'weight_decay':setting.w_decay,'eps':1e-4}
        ],
    )

    optimizer2 = optim.Adam(
        [
            {'params': front_net_crnn.parameters(),'lr':setting.lr_max,'momentum':0.9,'weight_decay':setting.w_decay,'eps':1e-4},
            {'params': back_net_crnn.parameters(),'lr':setting.lr_max,'momentum':0.9,'weight_decay':setting.w_decay,'eps':1e-4}
        ],
    )

    optimizer_mole = optim.Adam(
        [
            {'params': net_mole.parameters(),'lr':10*setting.lr_max,'momentum':0.9,'weight_decay':setting.w_decay,'eps':1e-4}
        ],
    )
    early_stopping = EarlyStopping(patience=500, verbose=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, scheduler_time, gamma=0.2)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, scheduler_time, gamma=0.2)
    scheduler_mole = torch.optim.lr_scheduler.StepLR(optimizer_mole, scheduler_time, gamma=0.2)
    loss_func = torch.nn.MSELoss()

    loss_train, loss_test = [], []
    loss_gas, loss_mass, loss_penalty = [], [], []
    train_gas, train_mass, penaltys = [], [], []
    loss_mass2,train_mass2=[],[]



    loss_train_back, loss_test_back = [], []
    loss_gas_back, loss_mass_back, loss_penalty_back = [], [], []
    train_gas_back, train_mass_back, penaltys_back = [], [], []


    loss_train_front, loss_test_front = [], []
    loss_gas_front, loss_mass_front, loss_penalty_front = [], [], []
    train_gas_front, train_mass_front, penaltys_front = [], [], []

    loss_train_plot=[]

    loss_best_total = 1e10

    if pretrain:
        if not using_pretrain_model:
            front_net_crnn.load_state_dict(torch.load(model_best_dir+f'/front_net_crnn_ns{setting.ns}_nr{setting.nr}_best.pth',weights_only=True))
            back_net_crnn.load_state_dict(torch.load(model_best_dir+f'/back_net_crnn_ns{setting.ns}_nr{setting.nr}_best.pth',weights_only=True))
            net_mole.load_state_dict(torch.load(model_best_dir+f'/net_mole_ns{setting.ns}_nr{setting.nr}_best.pth',weights_only=True))
        else:
            front_net_crnn.load_state_dict(torch.load(model_best_pretrain_dir+f'/front_net_crnn_ns{setting.ns}_nr{setting.nr}_best.pth',weights_only=True))
            back_net_crnn.load_state_dict(torch.load(model_best_pretrain_dir+f'/back_net_crnn_ns{setting.ns}_nr{setting.nr}_best.pth',weights_only=True))
            net_mole.load_state_dict(torch.load(model_best_pretrain_dir+f'/net_mole_ns{setting.ns}_nr{setting.nr}_best.pth',weights_only=True))

        optimizer = optim.Adam(
            [
                {'params': front_net_crnn.parameters(),'lr':1e-6,'momentum':0.9},
                {'params': back_net_crnn.parameters(),'lr':1e-6,'momentum':0.9},
                {'params': net_mole.parameters(),'lr':1e-6,'momentum':0.9}
            ],
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.2)

    for epoch in range(setting.n_epoch):

        result_data={}


        loss_tensor=torch.tensor(0,dtype=torch.float32)

        loss_list=[]
        loss1_list=[]
        loss2_list=[]
        loss3_list=[]
        lossp_list=[]
        yps = []
        loss_total = 0

        front_loss_list=[]
        front_loss1_list=[]
        front_loss2_list=[]
        front_lossp_list=[]
        front_yps = []

        back_loss_list=[]
        back_loss1_list=[]
        back_loss2_list=[]
        back_lossp_list=[]
        back_yps = []

        
        for i,(ini_state,t,Y_label,X0,Temp,Pressure) in enumerate(front_dataloader):


            for ii,(ini_state_back,t_back,Y_label_back,X0_back,Temp_back,Pressure_back) in enumerate(back_dataloader):
                if ii == i:
                    break




            #设置模型为训练模式
            front_net_crnn.train()
            back_net_crnn.train()
            net_mole.train()



            case = train_list[i]
            if case == 3:
                front_alpha = 1.2

            time_end = convert_time[train_list[i]]

            #拼接
            t=torch.cat((t,t_back[0][1:].unsqueeze(0)),dim=1)
            Y_label=torch.cat((Y_label,Y_label_back[0][1:].unsqueeze(0)),dim=1)
            Temp=torch.cat((Temp,Temp_back[0][1:].unsqueeze(0)),dim=1)
            Pressure=torch.cat((Pressure,Pressure_back[0][1:].unsqueeze(0)),dim=1)

            #先算front
            #插值
            f = interp1d(t[0], Y_label[0])  
            f_T =interp1d(t[0], Temp[0])  
            f_P = interp1d(t[0], Pressure[0])


            # t_front = torch.linspace(t[0].min(), time_end, front_number).unsqueeze(0)

            #变步长
            t_front = utils.variable_step_tensor(t[0].min(), time_end, front_number ,front_alpha).unsqueeze(0)

            #尾端加密

            # t_front1 = torch.linspace(t[0].min(), (time_end*0.95-t[0].min()), int(0.7*front_number))
            # t_front2 = torch.linspace((time_end*0.95-t[0].min()), time_end, front_number-int(0.7*front_number)+1)
            # t_front = torch.cat((t_front1,t_front2[1:]),dim=0).unsqueeze(0)







            Y_label = torch.from_numpy(f(t_front[0])).unsqueeze(0)
            Temp = torch.from_numpy(f_T(t_front[0])).unsqueeze(0)
            Pressure = torch.from_numpy(f_P(t_front[0])).unsqueeze(0)

            ini_state=ini_state.squeeze()
            t_front=t_front.squeeze()
            Y_label=Y_label.squeeze()
            X0=X0.squeeze()
            front_net_crnn.init_state(ini_state,t_front,Temp)

            #y_predict=odeint(net_crnn,X0,t,method='adaptive_heun',atol=1e-6)
            options = {}
            options.update({'method': 'Dopri5'})
            # options.update({'h': 1e-3})
            options.update({'t0': t_front.min()})
            options.update({'t1': t_front.max()})
            options.update({'rtol': 1e-4})
            options.update({'atol': 1e-4})
            options.update({'step_dif_ratio': 1e-4})
            options.update({'neval_max': 8000000})
            options.update({'t_eval':t_front})
            options.update({'safety': 0.4})
            options.update({'interpolation_method':'cubic'})
            options.update({'h_max':acc_interp*(t_front.max()-t_front.min())/front_number})
            options.update({'h_min':1e-4})
            front_y_predict = solveode(front_net_crnn, X0.unsqueeze(0), options = options).squeeze()
            front_y_predict=torch.relu(front_y_predict)
            


            lib = front_y_predict[-1][0]
            if lib >4:
                loss4=(lib-4)*20
            else:
                loss4=0





            #back过程
            #插值
            # t_back= utils.variable_step_tensor(time_end, t.max(), back_number ,back_alpha).unsqueeze(0)
            head = torch.linspace(time_end, time_end+time_head[case], head_number).unsqueeze(0)
            t_back= torch.linspace(time_end+time_head[case]+0.1, t.max(), back_number).unsqueeze(0)

            t_back = torch.cat((head,t_back),dim=1)

            Y_label_back = torch.from_numpy(f(t_back[0])).unsqueeze(0)
            Temp_back = torch.from_numpy(f_T(t_back[0])).unsqueeze(0)
            Pressure_back = torch.from_numpy(f_P(t_back[0])).unsqueeze(0)



            tail = torch.linspace(t.max()+0.1,t.max()+tail_time,tail_number).unsqueeze(0)
            Y_label_tail = torch.tensor([Y_label_back[0][-1].item()]*tail_number).unsqueeze(0)
            Temp_tail = torch.tensor([Temp_back[0][-1].item()]*tail_number).unsqueeze(0)
            Pressure_tail = torch.tensor([Pressure_back[0][-1].item()]*tail_number).unsqueeze(0)

            t_back=torch.cat((t_back,tail),dim=1)
            Y_label_back=torch.cat((Y_label_back,Y_label_tail),dim=1)
            Temp_back=torch.cat((Temp_back,Temp_tail),dim=1)
            Pressure_back=torch.cat((Pressure_back,Pressure_tail),dim=1)


            ini_state=ini_state.squeeze()
            t_back=t_back.squeeze()
            Y_label_back=Y_label_back.squeeze()
            X0=X0.squeeze()
            back_net_crnn.init_state(ini_state,t_back,Temp_back)

            options.update({'t0': t_back.min()})
            options.update({'t1': t_back.max()})
            options.update({'t_eval':t_back})
            options.update({'h_max':acc_interp2*(t_back.max()-t_back.min())/back_number})
            options.update({'h_min':1e-4})
            back_y_predict = solveode(back_net_crnn, front_y_predict[-1].unsqueeze(0), options = options).squeeze()


            y_predict = torch.cat((front_y_predict,back_y_predict[1:]),dim=0)
            Temp = torch.cat((Temp,Temp_back[0][1:].unsqueeze(0)),dim=1)
            Pressure = torch.cat((Pressure,Pressure_back[0][1:].unsqueeze(0)),dim=1)
            Y_label = torch.cat((Y_label,Y_label_back[1:]),dim=0)


            #计算梯度##############################

            Y_label_np=Y_label.numpy()
            Y_label_gradient = np.abs(np.gradient(Y_label_np))*1e5
            Y_label_weight=torch.from_numpy(Y_label_gradient+1)
            Y_label_weight=torch.sqrt(Y_label_weight)
            Y_label_weight=torch.clamp(Y_label_weight,1,30)
            ##############################################################


            y_predict=torch.relu(y_predict)

            #y_predict=odeint(net_crnn,X0.unsqueeze(0),t,atol=1e-6,rtol=1e-3).squeeze()
            # pre_temp=torch.cat([y_predict,Temp.squeeze().unsqueeze(1)],dim=-1)
            mole=net_mole(y_predict,Temp,Pressure) # 这里

            #将结束后的mole拉平
            molelast=mole[front_number+head_number+back_number-2]
            molelast=torch.tensor([molelast.item()]*(tail_number)).unsqueeze(1)
            mole=torch.cat((mole[:front_number+head_number+back_number-1], molelast),dim=0)

            mass=y_predict[:,:-1]
            gas=y_predict[:,-1]


            # mass_m=mass[:,:-1].sum(-1)
            mass_m=mass.sum(-1)
            gas_v=gas/(mole.squeeze())



            loss1_all=torch.abs(gas_v-Y_label)
            loss2_all=torch.abs(mass_m-(X0[0]-Y_label*mole.squeeze()))
            loss3_all=torch.abs(torch.sum(y_predict,dim=1)-X0[0])





            loss1=loss1_all.mean()*gas_gain
            #if ini_state[2] < 1000.:
            loss2=loss2_all.mean()*mass_gain
            loss3=loss3_all.mean()*mass_gain2
            
            front_end_idx = len(t_front)-1

            
            loss5 = (gas_v[front_end_idx]-Y_label[front_end_idx]).abs()*100


            loss1_front_all=loss1_all[0:front_end_idx+1]
            loss2_front_all=loss2_all[0:front_end_idx+1]
            loss3_front_all=loss3_all[0:front_end_idx+1]
            loss1_back_all=loss1_all[front_end_idx+1:]
            loss2_back_all=loss2_all[front_end_idx+1:]
            loss3_back_all=loss3_all[front_end_idx+1:]



            loss1_front=loss1_front_all.mean()
            loss2_front=loss2_front_all.mean()
            loss3_front=loss3_front_all.mean()
            loss1_back=loss1_back_all.mean()
            loss2_back=loss2_back_all.mean()
            loss3_back=loss3_back_all.mean()


            loss1_fb=(loss1_front*front_lossgain+loss1_back*back_lossgain)*gas_gain
            loss2_fb=(loss2_front*front_lossgain+loss2_back*back_lossgain)*mass_gain
            loss3_fb=(loss3_front*front_lossgain+loss3_back*back_lossgain)*mass_gain2

            loss_all=torch.cat((loss1_front_all*front_lossgain,loss1_back_all*back_lossgain),dim=0)*gas_gain+torch.cat((loss2_front_all*front_lossgain,loss2_back_all*back_lossgain),dim=0)*mass_gain

         
            # y_penalty = torch.clamp(y_predict.sum(-1) - X0[0], min=0) 
            # penalty = (y_penalty ** 2).mean() * 20

            sum_penalty = torch.clamp(y_predict.sum(-1) - X0[0], min=0).mean()
            time_weights = torch.linspace(1, y_predict.shape[0], y_predict.shape[0]).to(y_predict.device)
            weighted_penalty = torch.sum(time_weights * sum_penalty)
            smooth_penalty = torch.sum((y_predict[1:] - y_predict[:-1]) ** 2)
            penalty = 5*weighted_penalty + 0.05*smooth_penalty
            

            
            loss=(loss1+loss2)+loss3
            # loss_fb=loss1_fb+loss2_fb+loss3_fb
            loss_tensor+=loss


            loss_list.append(loss.item())
            loss1_list.append(loss1.item()/gas_gain)
            loss2_list.append(loss2.item()/mass_gain)
            loss3_list.append(loss3.item()/mass_gain2)
            lossp_list.append(penalty.item())



            loss_front = (loss1_front+loss2_front)
            front_loss_list.append(loss_front.item())
            front_loss1_list.append(loss1_front.item())
            front_loss2_list.append(loss2_front.item())
            front_lossp_list.append(penalty.item())

            loss_back = (loss1_back+loss2_back)
            back_loss_list.append(loss_back.item())
            back_loss1_list.append(loss1_back.item())
            back_loss2_list.append(loss2_back.item())
            back_lossp_list.append(penalty.item())

            loss_total+=loss.item()

            loss_total+=(loss1+loss2)

################################采用权重形式################################

            # loss=torch.mul(loss_all,Y_label_weight).mean()

###########################################################################
            # optimizer.zero_grad()
            # optimizer_mole.zero_grad()
            # loss.backward()
            # torch.nn.utils.clip_grad_norm_(front_net_crnn.parameters(), 5e-3, 2)
            # torch.nn.utils.clip_grad_norm_(back_net_crnn.parameters(), 1e-2, 2)
            # torch.nn.utils.clip_grad_norm_(net_mole.parameters(), 1e-2, 2)

            # utils.plot_grad_flow(front_net_crnn.named_parameters(),'front_net_crnn')
            # utils.plot_grad_flow(back_net_crnn.named_parameters(),'back_net_crnn')
            # utils.plot_grad_flow(net_mole.named_parameters(),'net_mole')

            # optimizer.step()

            # optimizer_mole.step()
            #print("\n",loss.item(),loss1.item(),loss2.item())
            
            # yps.append(torch.clamp(y_predict,0.,5.).detach().numpy())
            yps.append(y_predict.detach().numpy())




            #记录结果##################################
            result_data[case]={}

            result_data[case]['Y_label']=Y_label
            result_data[case]['y_predict']=y_predict
            result_data[case]['front_end']=front_end_idx
            result_data[case]['t_total']=torch.cat((t_front,t_back[1:]),0)
            result_data[case]['gas_v']=gas_v
            result_data[case]['mole']=mole
            ##########################################

        optimizer2.zero_grad()
        optimizer_mole.zero_grad()

        loss_tensor=loss_tensor/len(train_list)

        loss_tensor.backward()
        try:
            torch.nn.utils.clip_grad_norm_(front_net_crnn.parameters(), 1e-2, 2 ,error_if_nonfinite=True)
            torch.nn.utils.clip_grad_norm_(back_net_crnn.parameters(), 1e-2, 2 ,error_if_nonfinite=True)
            torch.nn.utils.clip_grad_norm_(net_mole.parameters(), 1e-1, 2 ,error_if_nonfinite=True)
        except:
            for p in front_net_crnn.parameters():
                if torch.isnan(p.grad).any():
                    print("nan")
                if torch.isinf(p.grad).any():
                    print("inf")
                print(loss_tensor)

            print("have nan 3")
        havewrong=utils.plot_grad_flow(front_net_crnn.named_parameters(),'front_net_crnn')
        utils.plot_grad_flow(back_net_crnn.named_parameters(),'back_net_crnn')
        utils.plot_grad_flow(net_mole.named_parameters(),'net_mole')

        if havewrong:
            print(havewrong)
            print(loss_tensor)
            raise ValueError("have nan 1")

      
        if epoch < 300:
            optimizer.step()
            scheduler.step()
        if epoch >= 300:
            optimizer_mole.step()
            optimizer2.step()
            scheduler_mole.step()
            scheduler2.step()  



        for  p in front_net_crnn.parameters():
            if torch.isnan(p.grad).any():
                raise ValueError("have nan 2")





        loss_train_plot.append((sum(loss1_list)*gas_gain+sum(loss2_list)*mass_gain)/len(loss_list))
        # loss_train_plot.append(sum(loss_list)/len(loss_list))


        train_gas.append((sum(loss1_list))/len(loss_list))
        train_mass.append((sum(loss2_list))/len(loss_list))
        train_mass2.append((sum(loss3_list))/len(loss_list))
        penaltys.append((sum(lossp_list))/len(loss_list))
        loss_train.append((sum(loss1_list)+sum(loss2_list))/len(loss_list))
        
        train_gas_front.append((sum(front_loss1_list))/len(front_loss_list))
        train_mass_front.append((sum(front_loss2_list))/len(front_loss_list))
        penaltys_front.append((sum(front_lossp_list))/len(front_loss_list))
        loss_train_front.append((sum(front_loss1_list)+sum(front_loss2_list))/len(front_loss_list))


        train_gas_back.append((sum(back_loss1_list))/len(back_loss_list))
        train_mass_back.append((sum(back_loss2_list))/len(back_loss_list))
        penaltys_back.append((sum(back_lossp_list))/len(back_loss_list))
        loss_train_back.append((sum(back_loss1_list)+sum(back_loss2_list))/len(back_loss_list))

        
        loss1_list=[]
        loss2_list=[]
        loss_list=[]
        lossp_list=[]


        front_loss_list=[]
        front_loss1_list=[]
        front_loss2_list=[]
        front_lossp_list=[]
        front_yps = []

        back_loss_list=[]
        back_loss1_list=[]
        back_loss2_list=[]
        back_lossp_list=[]
        back_yps = []
        with torch.no_grad():
            for i,(ini_state,t,Y_label,X0,Temp,Pressure) in enumerate(front_dataloader_test):

                #再算back
                for ii,(ini_state_back,t_back,Y_label_back,X0_back,Temp_back,Pressure_back) in enumerate(back_dataloader_test):
                    if ii == i:
                        break


                #设置模型为评估模式
                front_net_crnn.eval()
                back_net_crnn.eval()
                net_mole.eval()





                case = val_list[i]


                time_end = convert_time[val_list[i]]

                #拼接
                t=torch.cat((t,t_back[0][1:].unsqueeze(0)),dim=1)
                Y_label=torch.cat((Y_label,Y_label_back[0][1:].unsqueeze(0)),dim=1)
                Temp=torch.cat((Temp,Temp_back[0][1:].unsqueeze(0)),dim=1)
                Pressure=torch.cat((Pressure,Pressure_back[0][1:].unsqueeze(0)),dim=1)

                #先算front
                #插值
                f = interp1d(t[0], Y_label[0])  
                f_T =interp1d(t[0], Temp[0])  
                f_P = interp1d(t[0], Pressure[0])


                # t_front = torch.linspace(t[0].min(), time_end, front_number).unsqueeze(0)

                #变步长
                t_front = utils.variable_step_tensor(t[0].min(), time_end, front_number ,front_alpha).unsqueeze(0)




                Y_label = torch.from_numpy(f(t_front[0])).unsqueeze(0)
                Temp = torch.from_numpy(f_T(t_front[0])).unsqueeze(0)
                Pressure = torch.from_numpy(f_P(t_front[0])).unsqueeze(0)

                ini_state=ini_state.squeeze()
                t_front=t_front.squeeze()
                Y_label=Y_label.squeeze()
                X0=X0.squeeze()
                front_net_crnn.init_state(ini_state,t_front,Temp)

                #y_predict=odeint(net_crnn,X0,t,method='adaptive_heun',atol=1e-6)
                options = {}
                options.update({'method': 'Dopri5'})
                # options.update({'h': 1e-3})
                options.update({'t0': t_front.min()})
                options.update({'t1': t_front.max()})
                options.update({'rtol': 1e-4})
                options.update({'atol': 1e-4})
                options.update({'step_dif_ratio': 1e-4})
                options.update({'neval_max': 8000000})
                options.update({'t_eval':t_front})
                options.update({'safety': 0.4})
                options.update({'interpolation_method':'cubic'})
                options.update({'h_max':acc_interp*(t_front.max()-t_front.min())/front_number})
                options.update({'h_min':1e-4})
                #options.update({'regenerate_graph':True})
                front_y_predict = solveode(front_net_crnn, X0.unsqueeze(0), options = options).squeeze()
                front_y_predict=torch.relu(front_y_predict)






                #back过程
                #插值
                # t_back= utils.variable_step_tensor(time_end, t.max(), back_number, back_alpha).unsqueeze(0)
                head = torch.linspace(time_end, time_end+time_head[case], head_number).unsqueeze(0)
                t_back= torch.linspace(time_end+time_head[case]+0.1, t.max(), back_number).unsqueeze(0)

                t_back = torch.cat((head,t_back),dim=1)

                Y_label_back = torch.from_numpy(f(t_back[0])).unsqueeze(0)
                Temp_back = torch.from_numpy(f_T(t_back[0])).unsqueeze(0)
                Pressure_back = torch.from_numpy(f_P(t_back[0])).unsqueeze(0)



                tail = torch.linspace(t.max()+0.1,t.max()+tail_time,tail_number).unsqueeze(0)
                Y_label_tail = torch.tensor([Y_label_back[0][-1].item()]*tail_number).unsqueeze(0)
                Temp_tail = torch.tensor([Temp_back[0][-1].item()]*tail_number).unsqueeze(0)
                Pressure_tail = torch.tensor([Pressure_back[0][-1].item()]*tail_number).unsqueeze(0)

                t_back=torch.cat((t_back,tail),dim=1)
                Y_label_back=torch.cat((Y_label_back,Y_label_tail),dim=1)
                Temp_back=torch.cat((Temp_back,Temp_tail),dim=1)
                Pressure_back=torch.cat((Pressure_back,Pressure_tail),dim=1)

                ini_state=ini_state.squeeze()
                t_back=t_back.squeeze()
                Y_label_back=Y_label_back.squeeze()
                X0=X0.squeeze()
                back_net_crnn.init_state(ini_state,t_back,Temp_back)

                options.update({'t0': t_back.min()})
                options.update({'t1': t_back.max()})
                options.update({'t_eval':t_back})
                options.update({'h_max':acc_interp2*(t_back.max()-t_back.min())/back_number})
                options.update({'h_min':1e-4})
                back_y_predict = solveode(back_net_crnn, front_y_predict[-1].unsqueeze(0), options = options).squeeze()


                y_predict = torch.cat((front_y_predict,back_y_predict[1:]),dim=0)
                Temp = torch.cat((Temp,Temp_back[0][1:].unsqueeze(0)),dim=1)
                Pressure = torch.cat((Pressure,Pressure_back[0][1:].unsqueeze(0)),dim=1)
                Y_label = torch.cat((Y_label,Y_label_back[1:]),dim=0)

                #y_predict=odeint(net_crnn,X0.unsqueeze(0),t,atol=1e-6,rtol=1e-3).squeeze()
                # pre_temp=torch.cat([y_predict,Temp.squeeze().unsqueeze(1)],dim=-1)
                
                y_predict=torch.relu(y_predict)                
                mole=net_mole(y_predict,Temp,Pressure) # 这里

                #将结束后的mole拉平
                molelast=mole[front_number+head_number+back_number-2]
                molelast=torch.tensor([molelast.item()]*(tail_number)).unsqueeze(1)
                mole=torch.cat((mole[:front_number+head_number+back_number-1], molelast),dim=0)


                mass=y_predict[:,:-1]
                gas=y_predict[:,-1]


                # mass_m=mass[:,:-1].sum(-1)
                mass_m=mass.sum(-1)
                gas_v=gas/(mole.squeeze())

                loss1=torch.abs(gas_v-Y_label).mean()*gas_gain
                #if ini_state[2] < 1000.:
                loss2=torch.abs(mass_m-(X0[0]-Y_label*mole.squeeze())).mean()*mass_gain
                
                loss3=torch.abs(torch.sum(y_predict,dim=1)-X0[0]).mean()*mass_gain


                front_end_idx = len(t_front)-1
                loss1_front=torch.abs(gas_v[0:front_end_idx+1]-Y_label[0:front_end_idx+1]).mean()
                loss2_front=torch.abs(mass_m[0:front_end_idx+1]-(X0[0]-Y_label[0:front_end_idx+1]*mole.squeeze()[0:front_end_idx+1])).mean()
                loss1_back=torch.abs(gas_v[front_end_idx+1:]-Y_label[front_end_idx+1:]).mean()
                loss2_back=torch.abs(mass_m[front_end_idx+1:]-(X0[0]-Y_label[front_end_idx+1:]*mole.squeeze()[front_end_idx+1:])).mean()
                
                # loss1=(loss1_front*front_lossgain+loss1_back*back_lossgain)*gas_gain
                # loss2=(loss2_front*front_lossgain+loss2_back*back_lossgain)*mass_gain

                sum_penalty = torch.clamp(y_predict.sum(-1) - X0[0], min=0).mean()
                time_weights = torch.linspace(1, y_predict.shape[0], y_predict.shape[0]).to(y_predict.device)
                weighted_penalty = torch.sum(time_weights * sum_penalty)
                smooth_penalty = torch.sum((y_predict[1:] - y_predict[:-1]) ** 2)
                penalty = 1*weighted_penalty + 0.05*smooth_penalty

                loss_temp = loss1.item()+loss2.item()
                loss_test.append(loss_temp)

                loss_total+=(loss1+loss2)




                loss1_list.append(loss1.item()/gas_gain)
                loss2_list.append(loss2.item()/mass_gain)
                lossp_list.append(penalty.item())




                # yps.append(torch.clamp(y_predict,0.,5.).detach().numpy())
                yps.append(y_predict.detach().numpy())
                # loss_test.append(loss1.item()+loss2.item())
                loss_gas.append(loss1.item()/gas_gain)
                loss_mass.append(loss2.item()/mass_gain)
                loss_mass2.append(loss3.item()/mass_gain2)
                loss_penalty.append(penalty.item())


                loss_front = (loss1_front+loss2_front)
                front_loss_list.append(loss_front.item())
                front_loss1_list.append(loss1_front.item())
                front_loss2_list.append(loss2_front.item())
                front_lossp_list.append(penalty.item())

                loss_back = (loss1_back+loss2_back)
                back_loss_list.append(loss_back.item())
                back_loss1_list.append(loss1_back.item())
                back_loss2_list.append(loss2_back.item())
                back_lossp_list.append(penalty.item())    



                #记录结果
                result_data[case]={}

                result_data[case]['Y_label']=Y_label
                result_data[case]['y_predict']=y_predict
                result_data[case]['front_end']=front_end_idx
                result_data[case]['t_total']=torch.cat((t_front,t_back[1:]),0)
                result_data[case]['gas_v']=gas_v
                result_data[case]['mole']=mole


            loss_gas_front.append((sum(front_loss1_list))/len(front_loss_list))
            loss_mass_front.append((sum(front_loss2_list))/len(front_loss_list))
            loss_penalty_front.append((sum(front_lossp_list))/len(front_loss_list))
            loss_test_front.append((sum(front_loss1_list)+sum(front_loss2_list))/len(front_loss_list))

            loss_gas_back.append((sum(back_loss1_list))/len(back_loss_list))
            loss_mass_back.append((sum(back_loss2_list))/len(back_loss_list))
            loss_penalty_back.append((sum(back_lossp_list))/len(back_loss_list))
            loss_test_back.append((sum(back_loss1_list)+sum(back_loss2_list))/len(back_loss_list))





            if (epoch+1)%100==0:
                
                utils.plot_show(front_dataloader,back_dataloader,front_net_crnn,back_net_crnn,net_mole,epoch,'train',back_result_dir,front_result_dir,convert_time,train_list,val_list,result_data)
                utils.plot_show(front_dataloader_test,back_dataloader_test,front_net_crnn,back_net_crnn,net_mole,epoch,'test',back_result_dir,front_result_dir,convert_time,train_list,val_list,result_data)
                torch.save(front_net_crnn.state_dict(),front_model_dir+f"/front_net_crnn{epoch}.pth")
                torch.save(back_net_crnn.state_dict(),back_model_dir+f"/back_net_crnn{epoch}.pth")
                torch.save(net_mole.state_dict(),mole_model_dir+f"/net_mole{epoch}.pth")
                
                w_in=torch.concat([front_net_crnn.w_in_Ea.unsqueeze(0),front_net_crnn.w_in_b.unsqueeze(0)],dim=0)
                w_out=front_net_crnn.w_out
                w_b=front_net_crnn.w_b
                w_in_ocen=front_net_crnn.w_in_ocen
                print_target=np.array(torch.concat([w_in.T,w_b.unsqueeze(1),w_in_ocen.unsqueeze(1),w_out.T],dim=-1))
                df = pd.DataFrame(print_target)
                df.to_csv(front_result_dir+f'/result{epoch}.csv',index=False)

                w_in=torch.concat([back_net_crnn.w_in_Ea.unsqueeze(0),back_net_crnn.w_in_b.unsqueeze(0)],dim=0)
                w_out=back_net_crnn.w_out
                w_b=back_net_crnn.w_b
                w_in_ocen=back_net_crnn.w_in_ocen
                print_target=np.array(torch.concat([w_in.T,w_b.unsqueeze(1),w_in_ocen.unsqueeze(1),w_out.T],dim=-1))
                df = pd.DataFrame(print_target)
                df.to_csv(back_result_dir+f'/result{epoch}.csv',index=False)

            early_stopping(loss_temp, front_net_crnn,back_net_crnn, net_mole, front_dataloader,back_dataloader, front_dataloader_test,back_dataloader_test, epoch, front_result_dir,back_result_dir, setting.ns, setting.nr, train_list, val_list, yps,front_model_dir,back_model_dir,mole_model_dir,model_best_dir,convert_time,result_data)

            for idx, value in enumerate(train_list):
                df = pd.DataFrame(yps[idx])
                df.to_csv(front_result_dir+'/ypredict_train{}.csv'.format(value),index=False) 
        
            for idx, value in enumerate(val_list):
                df = pd.DataFrame(yps[len(train_list)+idx])
                df.to_csv(front_result_dir+'/ypredict_test{}.csv'.format(value),index=False)
        

        
            loss_dict = {'train_loss':loss_train,
                         'gas_loss':train_gas,
                         'mass_loss':train_mass,
                         'mass_loss2':train_mass2,
                         'ybe5':penaltys}
            df = pd.DataFrame(loss_dict)
            df.to_csv(front_result_dir+'/train_loss.csv',index=False)
         
            loss_dict_test = {'test_loss':loss_test,
                              'gas_loss':loss_gas,
                              'mass_loss':loss_mass,
                              'mass_loss2':loss_mass2,
                              'ybe5':penaltys}
            df = pd.DataFrame(loss_dict_test)
            df.to_csv(front_result_dir+'/test_loss.csv',index=False)

            loss_dict_front = {'train_loss':loss_train_front,
                         'gas_loss':train_gas_front,
                         'mass_loss':train_mass_front,
                         'ybe5':penaltys_front}
            df = pd.DataFrame(loss_dict_front)
            df.to_csv(front_result_dir+'/train_loss_front.csv',index=False)

            loss_dict_back = {'train_loss':loss_train_back,
                         'gas_loss':train_gas_back,
                         'mass_loss':train_mass_back,
                         'ybe5':penaltys_back}
            df = pd.DataFrame(loss_dict_back)
            df.to_csv(front_result_dir+'/train_loss_back.csv',index=False)

            loss_dict_test_front = {'test_loss':loss_test_front,
                              'gas_loss':loss_gas_front,
                              'mass_loss':loss_mass_front,
                              'ybe5':loss_penalty_front}
            df = pd.DataFrame(loss_dict_test_front)
            df.to_csv(front_result_dir+'/test_loss_front.csv',index=False)

            loss_dict_test_back = {'test_loss':loss_test_back,
                              'gas_loss':loss_gas_back,
                              'mass_loss':loss_mass_back,
                              'ybe5':loss_penalty_back}
            df = pd.DataFrame(loss_dict_test_back)
            df.to_csv(front_result_dir+'/test_loss_back.csv',index=False)

            loss_total =  loss_total/(len(train_list)+len(val_list))

            if loss_total<loss_best_total:
                loss_best_total = loss_total
                utils.plot_show(front_dataloader,back_dataloader,front_net_crnn,back_net_crnn,net_mole,epoch,'train',back_result_dir,front_result_dir,convert_time,train_list,val_list,result_data)
                utils.plot_show(front_dataloader_test,back_dataloader_test,front_net_crnn,back_net_crnn,net_mole,epoch,'test',back_result_dir,front_result_dir,convert_time,train_list,val_list,result_data)
                torch.save(front_net_crnn.state_dict(),model_best_dir+f"/front_net_crnn_ns{setting.ns}_nr{setting.nr}_best.pth")
                torch.save(back_net_crnn.state_dict(),model_best_dir+f"/back_net_crnn_ns{setting.ns}_nr{setting.nr}_best.pth")
                torch.save(net_mole.state_dict(),model_best_dir+f"/net_mole_ns{setting.ns}_nr{setting.nr}_best.pth")


        bar.update(1)
        plt.figure()
        plt.plot(loss_test,label = 'test_loss')
        plt.plot(loss_train_plot,label = 'train_loss')
        # plt.ylim(ymax=20,ymin=0)
        plt.legend()
        if not pretrain:
            plt.savefig(front_result_dir+'/0.png')
            plt.savefig('0.png')
        else:
            plt.savefig(front_result_dir+'/0_pre.png')
            plt.savefig('0_pre.png')
        plt.close()

        plt.figure()
        plt.plot(loss_test_front,label = 'test_loss')
        plt.plot(loss_train_front,label = 'train_loss')
        # plt.ylim(ymax=20)
        plt.legend()
        if not pretrain:
            plt.savefig(front_result_dir+'/0_front.png')
            plt.savefig('0_front.png')
        else:
            plt.savefig(front_result_dir+'/0_front_pre.png')
            plt.savefig('0_front_pre.png')
        plt.close()

        plt.figure()
        plt.plot(loss_test_back,label = 'test_loss')
        plt.plot(loss_train_back,label = 'train_loss')
        # plt.ylim(ymax=20)
        plt.legend()
        if not pretrain:
            plt.savefig(front_result_dir+'/0_back.png')
            plt.savefig('0_back.png')
        else:
            plt.savefig(front_result_dir+'/0_back_pre.png')
            plt.savefig('0_back_pre.png')
        plt.close()