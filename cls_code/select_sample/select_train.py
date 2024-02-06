import torch
from torch.utils.data import DataLoader
import sys 
sys.path += ['./helper']
import numpy as np

import select_dataset
import select_idx 
from model_helper import build_model
device = torch.device('cuda')
            
def load_train_unlabidx_data(main_opt,first_train_unlabidx):
    train_unlabidx_dataset= select_dataset.create_dataset(main_opt, 224,first_train_unlabidx)
    train_unlabidx_loader = DataLoader(train_unlabidx_dataset, batch_size=main_opt.batch_size, num_workers=2,pin_memory=True)   #19
    return train_unlabidx_loader

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

def main(first_train_path,select_sample_path,main_opt):
    mdol=main_opt.model
    root_path=first_train_path.rsplit('/',1)[0]

    f_first_train_unlabidx = first_train_path+'/first_train_unlabidx.txt'
    ckpt_fname = first_train_path+'/best.pth'
    f_select_train_idx= select_sample_path + '/select_train_idx.txt'
    f_train_unlabidx= root_path + '/train_unlabidx.txt'

    first_train_unlabidx=get_idx(f_first_train_unlabidx)#9360
    total_num=main_opt.train_labidx_num
    
    # load dataset
    train_unlabidx_loader=load_train_unlabidx_data(main_opt,first_train_unlabidx)
    # build model and opt
    model = build_model(mdol)
    model.to(device)
 
    model.load_state_dict(torch.load(ckpt_fname))
    print('loaded checkpoint from {}'.format(ckpt_fname))
    select_train_idx=select_idx.select(main_opt,model,total_num,train_unlabidx_loader,first_train_unlabidx,device)
    train_unlabidx=np.array(list(set(first_train_unlabidx)-set(select_train_idx)))
    
    save_txt_file(f_train_unlabidx,train_unlabidx)
    save_txt_file(f_select_train_idx,select_train_idx)
    print("save select idxs and train_unlabidxs!")

