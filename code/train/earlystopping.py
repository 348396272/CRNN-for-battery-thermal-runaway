import torch
import utils
import numpy as np
import pandas as pd

class EarlyStopping:
    def __init__(self, patience=200, verbose=False, delta=0, path='checkpoint.pth', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss, front_net_crnn,back_net_crnn, net_mole, front_dataloader,back_dataloader, front_dataloader_test,back_dataloader_test, epoch, front_result_dir,back_result_dir, ns, nr, train_list, val_list, yps,front_model_dir,back_model_dir,mole_model_dir,model_best_dir,convert_time,result_data):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, front_net_crnn,back_net_crnn, net_mole, front_dataloader,back_dataloader, front_dataloader_test,back_dataloader_test, epoch, front_result_dir,back_result_dir, ns, nr, train_list, val_list, yps,front_model_dir,back_model_dir,mole_model_dir,model_best_dir,convert_time,result_data)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, front_net_crnn,back_net_crnn, net_mole, front_dataloader,back_dataloader, front_dataloader_test,back_dataloader_test, epoch, front_result_dir,back_result_dir, ns, nr, train_list, val_list, yps,front_model_dir,back_model_dir,mole_model_dir,model_best_dir,convert_time,result_data)
            self.counter = 0

    def save_checkpoint(self, val_loss, front_net_crnn,back_net_crnn, net_mole, front_dataloader,back_dataloader, front_dataloader_test,back_dataloader_test, epoch, front_result_dir,back_result_dir, ns, nr, train_list, val_list, yps,front_model_dir,back_model_dir,mole_model_dir,model_best_dir,convert_time,result_data):
        '''Saves model when validation loss decrease.'''
        for idx, value in enumerate(train_list):
            df = pd.DataFrame(yps[idx])
            df.to_csv(front_result_dir+'/epoch{}_ypredict_train{}.csv'.format(epoch,value),index=False) 

        for idx, value in enumerate(val_list):
            df = pd.DataFrame(yps[len(train_list)+idx])
            df.to_csv(front_result_dir+'/epoch{}_ypredict_test{}.csv'.format(epoch,value),index=False) 
        
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(front_net_crnn.state_dict(), front_model_dir+f"/net_crnn_ns{ns}_nr{nr}_{epoch}.pth")
        torch.save(back_net_crnn.state_dict(), back_model_dir+f"/net_crnn_ns{ns}_nr{nr}_{epoch}.pth")
        torch.save(net_mole.state_dict(), mole_model_dir+f"/net_mole_ns{ns}_nr{nr}_{epoch}.pth")
        
        torch.save(front_net_crnn.state_dict(), model_best_dir+f"/front_net_crnn_ns{ns}_nr{nr}_best.pth")
        torch.save(back_net_crnn.state_dict(), model_best_dir+f"/back_net_crnn_ns{ns}_nr{nr}_best.pth")
        torch.save(net_mole.state_dict(), model_best_dir+f"/net_mole_ns{ns}_nr{nr}_best.pth")

        utils.plot_show(front_dataloader,back_dataloader,front_net_crnn,back_net_crnn,net_mole,epoch,'train',back_result_dir,front_result_dir,convert_time,train_list,val_list,result_data)
        utils.plot_show(front_dataloader_test,back_dataloader_test,front_net_crnn,back_net_crnn,net_mole,epoch,'test',back_result_dir,front_result_dir,convert_time,train_list,val_list,result_data)

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

        self.val_loss_min = val_loss