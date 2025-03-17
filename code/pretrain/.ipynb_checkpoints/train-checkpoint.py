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
import csv
if __name__ == '__main__':
    seed=42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    setting=Settings.Settings()
    setting.parse() # 配置参数

    # 然后要加载数据咯
    # 首先要读入对吧
    readdata=Readdata.Readdata(setting.sample_size,setting.llb,"BACK")
    # 因为数据已经分段
    # 所以你必须这样处理

    # 一开始 觉得可能没有必要写一个 dataset
    # 想想还是算了 写吧 模块化就要模块化到底

#     train_list=[1,2,3,4,5]
#     # train_list=[0,2,3,4,6,7,9,10,12,13] # 注意下标从0开始哦
#     val_list=[0]
    
    train_list=[0,1,3]
    val_list=[2]

    result_dir = "./Back{}{}".format(setting.ns,setting.nr)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    model_dir = f'model{setting.ns}{setting.nr}'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)          
        
    train_data=[]
    val_data=[]
    train_ini_state=[]
    val_ini_state=[]
    for n in train_list:
        train_data.append(readdata.data[n])
        train_ini_state.append(readdata.ini_state[n])
    for n in val_list:
        val_data.append(readdata.data[n])
        val_ini_state.append(readdata.ini_state[n])

    # epoch 1703
    ini_X0=np.load('../front/pre_list/result_pre_optimial.npy') #last raw of each ypredict.csv 
    ini_X0=torch.tensor(ini_X0)
    mole_X0=np.load('../front/pre_list/mole_pre_optimial.npy')
    mole_X0=torch.tensor(mole_X0)

    ini_X0_train=ini_X0[train_list]
    ini_X0_test=ini_X0[val_list]

    mole_X0_train=mole_X0[train_list]
    mole_X0_test=mole_X0[val_list]

    train_dataset = Dataset_wlfc(train_data,train_ini_state,ini_X0_train,mole_X0_train,setting.ns,setting.init_m)
    test_dataset = Dataset_wlfc(val_data,val_ini_state,ini_X0_test,mole_X0_test,setting.ns,setting.init_m)

    # train_dataset = Dataset_wlfc(train_data,train_ini_state,setting.ns,setting.init_m)
    # test_dataset = Dataset_wlfc(val_data,val_ini_state,setting.ns,setting.init_m)

    dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    #dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    dataloader_test=DataLoader(test_dataset,batch_size=1,shuffle=False)
    bar = tqdm(total=setting.n_epoch) # 2500
    bar.set_description('Training')
    net_crnn=CRNN(setting.nr,setting.ns,setting.p_cutoff,setting.lb)
    net_mole=Mole(setting.nr,setting.ns)
    # net_crnn.load_state_dict(torch.load('model/back_net_crnn1799.pth',weights_only=True))
    # net_mole.load_state_dict(torch.load('model/back_net_mole1799.pth',weights_only=True))

    optimizer = optim.Adam(
        [
            {'params': net_crnn.parameters(),'lr':setting.lr_max},
            {'params': net_mole.parameters(),'lr':setting.lr_max},
        ],
    )

    early_stopping = EarlyStopping(patience=1000, verbose=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 500, gamma=0.2)
    loss_func = torch.nn.MSELoss()

    loss_train, loss_test = [], []
    loss_gas, loss_mass, loss_penalty = [], [], []
    train_gas, train_mass, penaltys = [], [], []
    for epoch in range(setting.n_epoch):
        loss_list=[]
        loss1_list=[]
        loss2_list=[]
        lossp_list=[]
        yps = []
        for ini_state,t,Y_label,X0,Temp in dataloader:
            ini_state=ini_state.squeeze()
            t=t.squeeze()
            Y_label=Y_label.squeeze()
            X0=X0.squeeze()
            net_crnn.init_state(ini_state,t,Temp)

            #y_predict=odeint(net_crnn,X0,t,method='adaptive_heun',atol=1e-6)
            options = {}
            options.update({'method': 'ode23s'})
            options.update({'h': None})
            options.update({'t0': t.min()})
            options.update({'t1': t.max()})
            options.update({'rtol': 1e-3})
            options.update({'atol': 1e-3})
            options.update({'t_eval':t})
            options.update({'interpolation_method':'linear'})
            #options.update({'regenerate_graph':True})
            y_predict = odesolve(net_crnn, X0.unsqueeze(0), options = options).squeeze()
            #y_predict=odeint(net_crnn,X0.unsqueeze(0),t,atol=1e-6,rtol=1e-3).squeeze()
            # pre_temp=torch.cat([y_predict,Temp.squeeze().unsqueeze(1)],dim=-1)
            mole=net_mole(y_predict,Temp) # 这里

            y_predict=torch.relu(y_predict)
            mass=y_predict[:,:-1]
            gas=y_predict[:,-1]

            # mass_m=mass[:,:-1].sum(-1)
            mass_m=mass.sum(-1)
            gas_v=gas/(mole.squeeze())

            loss1=torch.abs(gas_v-Y_label).mean()*100
            #if ini_state[2] < 1000.:
            loss2=torch.abs(mass_m-(X0[0]-Y_label*mole.squeeze())).mean()*5
            
            # y_penalty = torch.clamp(y_predict.sum(-1) - X0[0], min=0) 
            # penalty = (y_penalty ** 2).mean() * 20

            sum_penalty = torch.clamp(y_predict.sum(-1) - X0[0], min=0).mean()
            time_weights = torch.linspace(1, y_predict.shape[0], y_predict.shape[0]).to(y_predict.device)
            weighted_penalty = torch.sum(time_weights * sum_penalty)
            smooth_penalty = torch.sum((y_predict[1:] - y_predict[:-1]) ** 2)
            penalty = 5*weighted_penalty + 0.05*smooth_penalty
            
            loss=(loss1+loss2+penalty)
            loss_list.append(loss.item())
            loss1_list.append(loss1.item())
            loss2_list.append(loss2.item())
            lossp_list.append(penalty.item())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net_crnn.parameters(), 1e-1, 2)

            optimizer.step()
            #print("\n",loss.item(),loss1.item(),loss2.item())
            
            # yps.append(torch.clamp(y_predict,0.,5.).detach().numpy())
            yps.append(y_predict.detach().numpy())
        train_gas.append((sum(loss1_list)/100)/len(loss_list))
        train_mass.append((sum(loss2_list)/5)/len(loss_list))
        penaltys.append((sum(lossp_list))/len(loss_list))
        loss_train.append((sum(loss1_list)/100+sum(loss2_list)/5)/len(loss_list))
        
        scheduler.step()
        with torch.no_grad():
            for i,(ini_state,t,Y_label,X0,Temp) in enumerate(dataloader_test):
                ini_state=ini_state.squeeze()
                t=t.squeeze()
                Y_label=Y_label.squeeze()
                X0=X0.squeeze()
                net_crnn.init_state(ini_state,t,Temp)
                options = {}
                options.update({'method': 'ode23s'})
                options.update({'h': None})
                options.update({'t0': t.min()})
                options.update({'t1': t.max()})
                options.update({'rtol': 1e-4})
                options.update({'atol': 1e-6})
                options.update({'t_eval':t})
                options.update({'interpolation_method':'cubic'})
                y_predict = odesolve(net_crnn, X0.unsqueeze(0), options = options).squeeze()
                # y_predict=odeint(network,X0.unsqueeze(0),t,atol=1e-6,rtol=1e-3).squeeze()
#                 y_predict=torch.clamp(y_predict,0.,5.)
                mole=net_mole(y_predict,Temp) # 这里

                y_predict=torch.relu(y_predict)
                mass=y_predict[:,:-1]
                gas=y_predict[:,-1]

                # mass_m=mass[:,:-1].sum(-1)
                mass_m=mass.sum(-1)
                gas_v=gas/(mole.squeeze())

                loss1=torch.abs(gas_v-Y_label).mean()
                loss2=torch.abs(mass_m-(X0[0]-Y_label*mole.squeeze())).mean()
                
                sum_penalty = torch.clamp(y_predict.sum(-1) - X0[0], min=0).mean()
                time_weights = torch.linspace(1, y_predict.shape[0], y_predict.shape[0]).to(y_predict.device)
                weighted_penalty = torch.sum(time_weights * sum_penalty)
                smooth_penalty = torch.sum((y_predict[1:] - y_predict[:-1]) ** 2)
                penalty = 1*weighted_penalty + 0.05*smooth_penalty

                loss_temp = loss1.item()+loss2.item()
                loss_test.append(loss_temp)
                
                # yps.append(torch.clamp(y_predict,0.,5.).detach().numpy())
                yps.append(y_predict.detach().numpy())
                # loss_test.append(loss1.item()+loss2.item())
                loss_gas.append(loss1.item())
                loss_mass.append(loss2.item())
                loss_penalty.append(penalty.item())
                
            if (epoch+1)%500==0:
                utils.plot_show(dataloader,net_crnn,net_mole,epoch,'train',result_dir)
                utils.plot_show(dataloader_test,net_crnn,net_mole,epoch,'test',result_dir)
                torch.save(net_crnn.state_dict(),f"model{setting.ns}{setting.nr}/back_net_crnn{epoch}.pth")
                torch.save(net_mole.state_dict(),f"model{setting.ns}{setting.nr}/back_net_mole{epoch}.pth")
                
                w_in=torch.concat([net_crnn.w_in_Ea.unsqueeze(0),net_crnn.w_in_b.unsqueeze(0)],dim=0)
                w_out=net_crnn.w_out
                w_b=net_crnn.w_b
                w_in_ocen=net_crnn.w_in_ocen
                print_target=np.array(torch.concat([w_in.T,w_b.unsqueeze(1),w_in_ocen.unsqueeze(1),w_out.T],dim=-1))
                df = pd.DataFrame(print_target)
                df.to_csv(result_dir+f'/result{epoch}.csv',index=False)

            early_stopping(loss_temp, net_crnn, net_mole, dataloader, dataloader_test, epoch, result_dir, setting.ns, setting.nr, train_list, val_list, yps)

            for idx, value in enumerate(train_list):
                df = pd.DataFrame(yps[idx])
                df.to_csv(result_dir+'/ypredict_train{}.csv'.format(value),index=False) 
        
            for idx, value in enumerate(val_list):
                df = pd.DataFrame(yps[len(train_list)+idx])
                df.to_csv(result_dir+'/ypredict_test{}.csv'.format(value),index=False)
        
            loss_dict = {'train_loss':loss_train,
                         'gas_loss':train_gas,
                         'mass_loss':train_mass,
                         'ybe5':penaltys}
            df = pd.DataFrame(loss_dict)
            df.to_csv(result_dir+'/train_loss.csv',index=False)
         
            loss_dict_test = {'test_loss':loss_test,
                              'gas_loss':loss_gas,
                              'mass_loss':loss_mass,
                              'ybe5':penaltys}
            df = pd.DataFrame(loss_dict_test)
            df.to_csv(result_dir+'/test_loss.csv',index=False)
            
        bar.update(1)



