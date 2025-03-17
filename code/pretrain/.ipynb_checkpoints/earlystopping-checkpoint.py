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

    def __call__(self, val_loss, net_crnn, net_mole, dataloader, dataloader_test, epoch, result_dir, ns, nr, train_list, val_list, yps):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, net_crnn, net_mole, dataloader, dataloader_test, epoch, result_dir, ns, nr, train_list, val_list, yps)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, net_crnn, net_mole, dataloader, dataloader_test, epoch, result_dir, ns, nr, train_list, val_list, yps)
            self.counter = 0

    def save_checkpoint(self, val_loss, net_crnn, net_mole, dataloader, dataloader_test, epoch, result_dir, ns, nr, train_list, val_list, yps):
        '''Saves model when validation loss decrease.'''
        for idx, value in enumerate(train_list):
            df = pd.DataFrame(yps[idx])
            df.to_csv(result_dir+'/epoch{}_ypredict_train{}.csv'.format(epoch,value),index=False) 

        for idx, value in enumerate(val_list):
            df = pd.DataFrame(yps[len(train_list)+idx])
            df.to_csv(result_dir+'/epoch{}_ypredict_test{}.csv'.format(epoch,value),index=False) 
        
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(net_crnn.state_dict(), f"model{ns}{nr}/net_crnn_ns{ns}_nr{nr}_{epoch}.pth")
        torch.save(net_mole.state_dict(), f"model{ns}{nr}/net_mole_ns{ns}_nr{nr}_{epoch}.pth")
        
        utils.plot_show(dataloader,net_crnn,net_mole,epoch,'train',result_dir)
        utils.plot_show(dataloader_test,net_crnn,net_mole,epoch,'test',result_dir)

        w_in=torch.concat([net_crnn.w_in_Ea.unsqueeze(0),net_crnn.w_in_b.unsqueeze(0)],dim=0)
        w_out=net_crnn.w_out
        w_b=net_crnn.w_b
        w_in_ocen=net_crnn.w_in_ocen
        print_target=np.array(torch.concat([w_in.T,w_b.unsqueeze(1),w_in_ocen.unsqueeze(1),w_out.T],dim=-1))
        df = pd.DataFrame(print_target)
        df.to_csv(result_dir+f'/ES_result_ns{ns}_nr{nr}_{epoch}.csv',index=False)
        
        self.val_loss_min = val_loss