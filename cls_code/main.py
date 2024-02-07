import random
import numpy as np
import torch
import time
import argparse
import os
import sys 
import warnings
warnings.filterwarnings("ignore")

sys.path += ['./first_train', './select_sample','./second_train','./pse_label','./re_train']
import first_train
import select_train
import second_train
import pse_train
import re_train


def parse_opt():
    """Parses the input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_labidx_num', type=int,default=1000)  #train_labidx_num
    parser.add_argument('--test_ratio', type=float,default=0.1)    #test_idx_split_ratio
    parser.add_argument('--trainval_ratio', type=str,default='6,4')    #train_labidx_train,train_labidx_val
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--loss', type=str, default='CE')    
    parser.add_argument('--select_model', type=str, default='Sum')
    parser.add_argument('--hard_select', type=str, default='yes')
    parser.add_argument('--online_test', type=str, default='yes')
    parser.add_argument('--final_test', type=str, default='yes')
    parser.add_argument('--repeat_times', type=int, default=5)
    parser.add_argument('--first_select_ratio', type=float, default=0.9)
    parser.add_argument('--bias_select_ratio', type=float, default=0.001)
    parser.add_argument('--threshold', type=float, default=0.95)
    parser.add_argument('--unlab_level', type=float, default=1)
    parser.add_argument('--dataset_path', type=str, default='../../../Datasets/astronomy/transient_field/new_collection')  
    parser.add_argument('--data_name', type=str, default='new_data')

    #parser.add_argument('--fid', type=str, default='zg')
    #parser.add_argument('--fid', type=str, default='zr')
    parser.add_argument('--fid', type=str, default='fuse')

    args = parser.parse_args()
    return args

def main(random_number):
    main_opt = parse_opt()
    time_now=time.strftime("%Y.%m.%d_%H%M%S", time.localtime())
    save_root_path='./save_log_when_training/'+str(main_opt.data_name)+'_'+str(time_now)+'_random_number'+str(random_number)+'_level'+str(main_opt.unlab_level)+'_'+str(main_opt.train_labidx_num)+'_'+str(main_opt.test_ratio)+'_'+str(main_opt.trainval_ratio)
    first_train_path=save_root_path+'/first_train'
    os.makedirs(first_train_path,exist_ok=True)
    first_train.main(first_train_path,main_opt)
    select_sample_path=save_root_path+'/select_sample'
    os.makedirs(select_sample_path,exist_ok=True)
    select_train.main(first_train_path,select_sample_path,main_opt)
    second_train_path=save_root_path+'/second_train'
    os.makedirs(second_train_path,exist_ok=True)
    second_train.main(first_train_path,select_sample_path,second_train_path,main_opt)
    for repeat_time in range(main_opt.repeat_times):
        print('This is the '+str(repeat_time)+' repeat time!')
        pse_path=save_root_path+'/pse_'+str(repeat_time)+'_generate'
        os.makedirs(pse_path,exist_ok=True)
        pse_train.main(second_train_path,pse_path,main_opt)
        retrain_path=save_root_path+'/retrain_'+str(repeat_time)+'_time'
        os.makedirs(retrain_path,exist_ok=True)
        re_train.main(second_train_path,pse_path,retrain_path,main_opt)
        second_train_path=retrain_path

if __name__ == '__main__':
    random_number=0
    np.random.seed(random_number)
    torch.manual_seed(random_number)
    random.seed(random_number)
    torch.cuda.manual_seed(random_number)
    torch.cuda.manual_seed_all(random_number)
    torch.backends.cudnn.deterministic=True
    main(random_number)