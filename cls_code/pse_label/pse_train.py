import torch
from torch.utils.data import DataLoader
from sys import path
path.append('../helper')
import numpy as np
import pse_dataset
import pse_test 
from helper.model_helper import build_model
device = torch.device('cuda')
            
def load_train_unlabidx_data(main_opt,train_unlabidx):
    train_unlabidx_dataset= pse_dataset.create_dataset(main_opt, 224,train_unlabidx)
    train_unlabidx_loader = DataLoader(train_unlabidx_dataset, batch_size=main_opt.batch_size, num_workers=2,pin_memory=True)   #19
    return train_unlabidx_loader

def get_train10ratio(fname):
    file_for_ratio=open(fname)
    file_str=file_for_ratio.read()
    train10ratio=float(file_str.split('\n')[0].split(':')[-1])
    return train10ratio

def get_idx(fname):
    file_for_idx= open(fname)
    idx_str=file_for_idx.read()
    idx=idx_str.split(',')
    idx_list=[int(i) for i in idx]
    file_for_idx.close()
    return np.array(idx_list)

def save_txt_file(file_name,file):
    f_open = open(file_name, 'w')
    file_str=','.join(str(i) for i in file.tolist())
    f_open.write(file_str)
    f_open.close()
    
def main(second_train_path,pse_path,main_opt):
    mdol=main_opt.model
    root_path=second_train_path.rsplit('/',1)[0]

    f_train_unlabidx = root_path+'/train_unlabidx.txt'
    f_data_information=second_train_path+'/data_information.txt'
    
    ckpt_fname = second_train_path+'/best.pth'
    f_pseidx= pse_path + '/pse_idx.txt'
    f_pselab= pse_path + '/pse_lab.txt'
   
    train_unlabidx=get_idx(f_train_unlabidx)
    train10ratio=get_train10ratio(f_data_information)
    
    # load dataset
    train_unlabidx_loader=load_train_unlabidx_data(main_opt,train_unlabidx)
    # build model and opt
    model = build_model(mdol)
    model.to(device)

    model.load_state_dict(torch.load(ckpt_fname))
    print('loaded checkpoint from {}'.format(ckpt_fname))
    pse_testlabs,pse_testidxs=pse_test.pseduo(model, train_unlabidx_loader,train_unlabidx,main_opt.threshold,train10ratio,device)

    save_txt_file(f_pselab,pse_testlabs)
    save_txt_file(f_pseidx,pse_testidxs)
    print("save pse_test idxs and labs!")

