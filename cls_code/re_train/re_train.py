import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys 
sys.path += ['../helper']

import numpy as np
import re_dataset
from helper.model_helper import build_model
from helper.opt_helper import build_opt
from helper.test_evaluate import evalute
device = torch.device('cuda')
            
def load_dataset( main_opt,train_labidx, test_idx ,pse_idx, pse_lab):
    trainlabidx_pseunlabidx_dataset,train_labidx_val_dataset,test_dataset,reshuffle_train_labidx,data_information= \
        re_dataset.create_dataset( main_opt,train_labidx, test_idx ,pse_idx, pse_lab,224)
    trainlabidx_pseunlabidx_loader = DataLoader(trainlabidx_pseunlabidx_dataset, batch_size=main_opt.batch_size, shuffle=True,
                              num_workers=2,pin_memory=True)    
    train_labidx_val_loader = DataLoader(train_labidx_val_dataset, batch_size=main_opt.batch_size, num_workers=2,pin_memory=True)    
    test_loader = DataLoader(test_dataset, batch_size=main_opt.batch_size, num_workers=2,pin_memory=True)   
    return trainlabidx_pseunlabidx_loader,train_labidx_val_loader,test_loader,reshuffle_train_labidx,data_information

def load_txt(fname):
    file_for_idx= open(fname)
    idx_str=file_for_idx.read()
    if idx_str!='':
        idx=idx_str.split(',')
        idx_list=[int(i) for i in idx]
    else:
        idx_list='No_num'
    file_for_idx.close()
    return np.array(idx_list)

def save_data_information(file_name,file):
    f_open = open(file_name, 'a')
    f_open.write("train_labidx train 1:%d    train_labidx train 0:%d    train_labidx train 1 ratio:%.6f\n" %(file['reshuffle_train_labidx_train_1'], file['reshuffle_train_labidx_train_0'],file['reshuffle_train_labidx_train_1ratio']))
    f_open.write("train_labidx val 1:%d    train_labidx val 0:%d    train_labidx val 1 ratio:%.6f\n" %(file['reshuffle_train_labidx_val_1'], file['reshuffle_train_labidx_val_0'],file['reshuffle_train_labidx_val_1ratio']))
    f_open.write("pse_train_unlabidx_1:%d    pse_train_unlabidx_0:%d    pse_train_unlabidx_1ratio:%.6f\n" %(file['pse_train_unlabidx_1'], file['pse_train_unlabidx_0'],file['pse_train_unlabidx_1ratio']))
    f_open.close()

def save_txt_file(file_name,file):
    f_open = open(file_name, 'w')
    file_str=','.join(str(i) for i in file.tolist())
    f_open.write(file_str)
    f_open.close()

def main(second_train_path,pse_path,retrain_path,main_opt):
    mdol=main_opt.model
    lss=main_opt.loss
    sel_mdol=main_opt.select_model
    epochs = main_opt.epochs
    data_name=main_opt.data_name
    fid=main_opt.fid
    root_path=second_train_path.rsplit('/',1)[0]

    f_log_file=retrain_path+'/'+str(mdol)+'_'+str(lss)+'_'+str(sel_mdol)+'_'+str(data_name)+'_'+str(fid)+'_log.txt'
    f_log=open(f_log_file,'w')
    f_train_labidx=second_train_path+'/train_labidx.txt'

    f_test_idx = root_path+'/test_idx.txt'
    f_pseidx= pse_path + '/pse_idx.txt'
    f_pselab= pse_path + '/pse_lab.txt'

    f_reshuffle_train_labidx=retrain_path+'/train_labidx.txt'
    f_data_information=retrain_path+'/data_information.txt'

    # load dataset
    pse_idx, pse_lab=load_txt(f_pseidx),load_txt(f_pselab)
    train_labidx, test_idx=load_txt(f_train_labidx),load_txt(f_test_idx)
    trainlabidx_pseunlabidx_loader,train_labidx_val_loader,test_loader,reshuffle_train_labidx,data_information=load_dataset( main_opt,train_labidx, test_idx ,pse_idx, pse_lab)
    # build model and opt
    save_txt_file(f_reshuffle_train_labidx,reshuffle_train_labidx)
    save_data_information(f_data_information,data_information)
    model = build_model(mdol)
    model.to(device)

    optimizer,scheduler = build_opt(model)
    # get loss
    if lss=='CE':    
        criteon = nn.CrossEntropyLoss()
    else:
        print('Invalid loss')
        sys.exit()
    #(preform, epoch)    
    best_perform = {         
        'Pre':(0,0),
        'Acc':(0,0),
        'Rec':(0,0),
        'F1': (0,0),
        'Sum':(0,0),
        'Mcc':(0,0),
        'Ap': (0,0)
    }
    assert(sel_mdol in best_perform.keys())

    for epoch in range(epochs):
        total=len(trainlabidx_pseunlabidx_loader)
        for step, (x,y) in enumerate(trainlabidx_pseunlabidx_loader):
            x, y = x.type(torch.FloatTensor).to(device), y.to(device)
            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            f_log.write("Epoch %d: [%d / %d]    loss:%.6f\n" %(epoch, step,total-1,loss))
            print("Epoch %d: [%d / %d]    loss:%.6f" %(epoch, step,total-1,loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if (epoch+1)%1==0:
            val_res = evalute(model, train_labidx_val_loader,device)
            f_log.write('--------------Epoch %d Evaluation--------------\n'%epoch)
            print('--------------Epoch %d Evaluation--------------'%epoch)
            f_log.write(str(val_res)[1:-1]+'\n')
            print(str(val_res)[1:-1])
            #get best info
            if best_perform[sel_mdol][0] <= val_res[sel_mdol]:
                best_perform[sel_mdol] = (val_res[sel_mdol],epoch)
                torch.save(model.state_dict(), retrain_path+'/best.pth')
                if main_opt.online_test =='yes':
                    test_res = evalute(model, test_loader,device)
                    f_log.write('--------------Best Test--------------\n')
                    print('--------------Best Test--------------')
                    f_log.write(str(test_res)[1:-1]+'\n')
                    print(str(test_res)[1:-1])
        scheduler.step()
    
    f_log.write('best val %s:%f      best epoch:%d\n'%(sel_mdol, best_perform[sel_mdol][0], best_perform[sel_mdol][1]))
    print('best val %s:%f      best epoch:%d'%(sel_mdol, best_perform[sel_mdol][0], best_perform[sel_mdol][1]))

    if main_opt.final_test =='yes':
        model.load_state_dict(torch.load(retrain_path+'/best.pth'))
        f_log.write('loaded from best val checkpoint!\n')
        print('loaded from best val checkpoint!')
    
        test_res = evalute(model, test_loader,device)
        f_log.write(str(test_res)[1:-1])
        print(str(test_res)[1:-1])
    f_log.close()

