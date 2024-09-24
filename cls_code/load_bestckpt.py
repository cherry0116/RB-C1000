import torch
import time
import os
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd
import argparse
import sys 

sys.path += ['./load_ckpt','./helper']
import load_Dataset
from helper.model_helper import build_model
from helper.load_test_evaluate import evalute
from helper.get_label_score import get_label_score
device = torch.device('cuda')
            
def load_dataset(load_opt,data_name,ori_fid,fid,train_unlabidx,nouse_idx,test_idx):
    train_unlabidx_dataset,nouse_dataset,test_dataset,train_unlabdix_test_idx_nouse_dataset,new_train_unlabidx,\
        new_nouse_idx,new_test_idx,new_fuse_idx=load_Dataset.create_dataset(load_opt,data_name,ori_fid,fid,train_unlabidx,nouse_idx,test_idx,224)
    train_unlabidx_loader = DataLoader(train_unlabidx_dataset, batch_size=load_opt.batch_size, num_workers=2,pin_memory=True) 
    test_loader = DataLoader(test_dataset, batch_size=load_opt.batch_size, num_workers=2,pin_memory=True)   
    nouse_loader = DataLoader(nouse_dataset, batch_size=load_opt.batch_size, num_workers=2,pin_memory=True)  
    train_unlabdix_test_idx_loader = DataLoader(train_unlabdix_test_idx_nouse_dataset, batch_size=load_opt.batch_size, num_workers=2,pin_memory=True) 
    return train_unlabidx_loader,test_loader,nouse_loader,train_unlabdix_test_idx_loader,new_train_unlabidx,new_nouse_idx,new_test_idx,new_fuse_idx

def parse_opt():
    """Parses the input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default='./save_log_when_training/fuse_level1_1000_hard/fuse_2024.09.17_002340_rseed9_level1_1000_0.1_6,4_first_select_ratio0.9_hardyes/retrain_2_time')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--train_unlabidx', type=str, default='no')
    parser.add_argument('--test_idx', type=str, default='yes')
    parser.add_argument('--nouse', type=str, default='no')
    parser.add_argument('--train_unlabdix_test_idx_nouse', type=str, default='no')
    parser.add_argument('--get_result', type=str, default='yes')
    parser.add_argument('--test_dataset_path', type=str, default='../../DATASET/new_cls_dataset')
    
    parser.add_argument('--fid', type=str, default='zg')
    #parser.add_argument('--fid', type=str, default='zr')
    #parser.add_argument('--fid', type=str, default='fuse')
    
    args = parser.parse_args()
    return args

def get_string(vari):
    str_list=[str(i) for i in vari]
    return np.array(str_list)

def get_idx(fname):
    file_for_idx= open(fname)
    idx_str=file_for_idx.read()
    idx=idx_str.split(',')
    idx_list=[int(i) for i in idx]
    file_for_idx.close()
    return np.array(idx_list)

def main():
    start_time=time.time()

    load_opt = parse_opt()
    load_path=load_opt.resume
    root_path=load_path.rsplit('/',1)[0]
    time_now=time.strftime("%Y.%m.%d_%H%M%S", time.localtime())
    data_name='new_data'
    save_fname=data_name+'_'+str(time_now)+'_random_number'+root_path.split('rseed')[1]

    for file_name in os.listdir(load_path):
        if file_name.endswith('_log.txt'):
            mdol=file_name.split('_',1)[0]
            save_file_name=file_name
            ori_fid=file_name.split('_')[-2]

    save_path='./save_load_from_ckpt/'+str(save_fname)
    os.makedirs(save_path,exist_ok=True)
    f_log_file=save_path+'/'+str(save_file_name)
    f_log=open(f_log_file,'w')
    f_train_unlabidx = root_path+'/train_unlabidx.txt'
    f_test_idx = root_path+'/test_idx.txt'

    train_unlabidx=get_idx(f_train_unlabidx)
    test_idx=get_idx(f_test_idx)

    if data_name=='new_data':
        f_nouse_idx = root_path+'/nouse_idx.txt'
        nouse_idx=get_idx(f_nouse_idx)
    else:
        print('Invalid dataset!')
        exit()
    
    f_log.write('Arguments:\n')
    for k in load_opt.__dict__.keys():
      f_log.write('%s : %s \n'%(k, str(load_opt.__dict__[k])))
    f_log.write('ori_fid:%s  now_fid:%s\n'%(ori_fid,load_opt.fid))

    train_unlabidx_loader,test_loader,nouse_loader,train_unlabdix_test_idx_nouse_loader,new_train_unlabidx,new_nouse_idx,new_test_idx,new_fuse_idx=\
        load_dataset(load_opt,data_name,ori_fid,load_opt.fid,train_unlabidx,nouse_idx,test_idx)

    # build model and opt
    model = build_model(mdol)
    model.to(device)

    if load_opt.resume is not None:
        model.load_state_dict(torch.load(load_opt.resume+'/best.pth'))
        f_log.write('loaded checkpoint from {}\n'.format(load_opt.resume+'/best.pth'))
        print('loaded checkpoint from {}'.format(load_opt.resume+'/best.pth'))
    else:
        print('Invalid resume!')
        exit()

    load_time=time.time()
    
    if load_opt.get_result=='yes':
        if load_opt.train_unlabidx=='yes':
            train_unlabidx_res,train_unlabidx_con_matrix = evalute(model, train_unlabidx_loader,device)
            f_log.write('unlab:%s'%(str(train_unlabidx_res)[1:-1]+'\n'))
            print('unlab:',str(train_unlabidx_res)[1:-1])
            f_log.write('unlab:%s'%(str(train_unlabidx_con_matrix)[1:-1]+'\n'))            
            print('unlab:',str(train_unlabidx_con_matrix)[1:-1])
        if load_opt.test_idx=='yes':
            test_res,test_con_matrix = evalute(model, test_loader,device)
            f_log.write('test:%s'%(str(test_res)[1:-1]+'\n'))
            print('test:',str(test_res)[1:-1])
            f_log.write('test:%s'%(str(test_con_matrix)[1:-1]+'\n'))
            print('test:',str(test_con_matrix)[1:-1])
        if load_opt.nouse=='yes':
            nouse_res,nouse_con_matrix = evalute(model, nouse_loader,device)
            f_log.write('nouse:%s'%(str(nouse_res)[1:-1]+'\n'))
            print('nouse:',str(nouse_res)[1:-1])
            f_log.write('nouse:%s'%(str(nouse_con_matrix)[1:-1]+'\n'))
            print('nouse:',str(nouse_con_matrix)[1:-1])
        if load_opt.train_unlabdix_test_idx_nouse=='yes':
            train_unlabdix_test_nouse_res,train_unlabdix_test_nouse_con_matrix = evalute(model, train_unlabdix_test_idx_nouse_loader,device)
            f_log.write('unlab and nouse and test:%s'%(str(train_unlabdix_test_nouse_res)[1:-1]+'\n'))
            print('unlab and nouse and test:',str(train_unlabdix_test_nouse_res)[1:-1])
            f_log.write('unlab and nouse and test:%s'%(str(train_unlabdix_test_nouse_con_matrix)[1:-1]+'\n'))
            print('unlab and nouse and test:',str(train_unlabdix_test_nouse_con_matrix)[1:-1])
    else:
        if load_opt.train_unlabidx=='yes':
            scores,preds,labels = get_label_score(model, train_unlabidx_loader,device)     
            train_unlab_data_content=np.concatenate((np.expand_dims(np.array(get_string(new_train_unlabidx)),axis=1),np.expand_dims(np.array(get_string(scores)),axis=1),\
                np.expand_dims(np.array(get_string(preds)),axis=1),np.expand_dims(np.array(get_string(labels)),axis=1)),axis=1)
            train_unlab_head_dic = ["data_idx","score","pred", "gt_label"] 
            train_unlab_frame = pd.DataFrame(train_unlab_data_content)
            train_unlab_frame.to_csv(os.path.join(save_path,"train_unlab_results.csv"), index=False,header=train_unlab_head_dic) 
        if load_opt.test_idx=='yes':
            scores,preds,labels = get_label_score(model, test_loader,device)
            test_data_content=np.concatenate((np.expand_dims(np.array(get_string(new_test_idx)),axis=1),np.expand_dims(np.array(get_string(scores)),axis=1),\
                np.expand_dims(np.array(get_string(preds)),axis=1),np.expand_dims(np.array(get_string(labels)),axis=1)),axis=1)
            test_head_dic = ["data_idx","score","pred", "gt_label"] 
            test_frame = pd.DataFrame(test_data_content)
            test_frame.to_csv(os.path.join(save_path,"test_results.csv"), index=False,header=test_head_dic) 
        if load_opt.nouse=='yes':
            scores,preds,labels = get_label_score(model, nouse_loader,device)
            nouse_data_content=np.concatenate((np.expand_dims(np.array(get_string(new_nouse_idx)),axis=1),np.expand_dims(np.array(get_string(scores)),axis=1),\
                np.expand_dims(np.array(get_string(preds)),axis=1),np.expand_dims(np.array(get_string(labels)),axis=1)),axis=1)
            nouse_head_dic = ["data_idx","score","pred", "gt_label"] 
            nouse_frame = pd.DataFrame(nouse_data_content)
            nouse_frame.to_csv(os.path.join(save_path,"nouse_results.csv"), index=False,header=nouse_head_dic) 
        if load_opt.train_unlabdix_test_idx_nouse=='yes':
            scores,preds,labels = get_label_score(model, train_unlabdix_test_idx_nouse_loader,device)
            unlab_nouse_test_data_content=np.concatenate((np.expand_dims(np.array(get_string(new_fuse_idx)),axis=1),np.expand_dims(np.array(get_string(scores)),axis=1),\
                np.expand_dims(np.array(get_string(preds)),axis=1),np.expand_dims(np.array(get_string(labels)),axis=1)),axis=1)
            unlab_nouse_test_head_dic = ["data_idx","score","pred", "gt_label"] 
            unlab_nouse_test_frame = pd.DataFrame(unlab_nouse_test_data_content)
            unlab_nouse_test_frame.to_csv(os.path.join(save_path,"unlab_nouse_test_results.csv"), index=False,header=unlab_nouse_test_head_dic) 

    end_time=time.time()
    test_time=end_time-load_time
    total_time=end_time-start_time
    f_log.write('total_time:%s   total_len:%s  each_time:%s'%(str(total_time),str(len(new_test_idx)),str(total_time/len(new_test_idx))))
    print('total_time:%s   total_len:%s  each_time:%s'%(str(total_time),str(len(new_test_idx)),str(total_time/len(new_test_idx))))
    f_log.write('test_time:%s   test_len:%s  each_time:%s'%(str(test_time),str(len(new_test_idx)),str(test_time/len(new_test_idx))))
    print('test_time:%s   test_len:%s  each_time:%s'%(str(test_time),str(len(new_test_idx)),str(test_time/len(new_test_idx))))
    f_log.close()

if __name__ == '__main__':
    main()