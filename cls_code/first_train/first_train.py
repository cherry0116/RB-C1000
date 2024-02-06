import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys 
sys.path += ['../helper']
import first_Dataset

from helper.model_helper import build_model
from helper.opt_helper import build_opt
from helper.test_evaluate import evalute
device = torch.device('cuda')
            
def load_dataset(main_opt):
    first_train_labidx_train_dataset,first_train_labidx_val_dataset,test_dataset,train_labidx,train_unlabidx,nouse_idx,test_idx,data_information\
        = first_Dataset.create_dataset(main_opt, 224)
    first_train_labidx_train_loader = DataLoader(first_train_labidx_train_dataset, batch_size=main_opt.batch_size, shuffle=True,
                              num_workers=2,pin_memory=True)     #147
    first_train_labidx_val_loader = DataLoader(first_train_labidx_val_dataset, batch_size=main_opt.batch_size, num_workers=2,pin_memory=True)    #17
    test_loader = DataLoader(test_dataset, batch_size=main_opt.batch_size, num_workers=2,pin_memory=True)   #19
    return first_train_labidx_train_loader,first_train_labidx_val_loader,test_loader,train_labidx,train_unlabidx,nouse_idx,test_idx,data_information

def save_txt(file_name,file):
    file_open = open(file_name, 'w')
    file_str=','.join(str(i) for i in file.tolist())
    file_open.write(file_str)
    file_open.close()

def save_data_information(file_name,file):
    f_open = open(file_name, 'w')
    f_open.write("total_train_num:%d\n" %(file['total_train_num']))
    f_open.write("train_labidx train 1:%d    train_labidx train 0:%d    train_labidx train 1 ratio:%.6f\n" %(file['train_labidx_train_1'], file['train_labidx_train_0'],file['train_labidx_train_1ratio']))
    f_open.write("train_labidx val 1:%d    train_labidx val 0:%d    train_labidx val 1 ratio:%.6f\n" %(file['train_labidx_val_1'], file['train_labidx_val_0'],file['train_labidx_val_1ratio']))
    f_open.close()

def main(first_train_path,main_opt):
    mdol=main_opt.model
    lss=main_opt.loss
    sel_mdol=main_opt.select_model
    epochs = main_opt.epochs
    fid=main_opt.fid
    data_name=main_opt.data_name
    root_path=first_train_path.rsplit('/',1)[0]

    f_log_file=first_train_path+'/'+str(mdol)+'_'+str(lss)+'_'+str(sel_mdol)+'_'+str(data_name)+'_'+str(fid)+'_log.txt'
    f_log=open(f_log_file,'w')
    f_first_train_labidx = first_train_path+'/first_train_labidx.txt'
    f_first_train_unlabidx = first_train_path+'/first_train_unlabidx.txt'
    f_test_idx = root_path+'/test_idx.txt'
    f_data_information=first_train_path+'/data_information.txt'

    f_log.write('Arguments:\n')
    for k in main_opt.__dict__.keys():
      f_log.write('%s : %s \n'%(k, str(main_opt.__dict__[k])))

   # load dataset
    first_train_labidx_train_loader,first_train_labidx_val_loader,test_loader,first_train_labidx,\
        first_train_unlabidx,nouse_idx,test_idx,data_information=load_dataset(main_opt)
    save_txt(f_first_train_labidx,first_train_labidx)
    save_txt(f_first_train_unlabidx,first_train_unlabidx)
    f_nouse_idx = root_path+'/nouse_idx.txt'
    save_txt(f_nouse_idx,nouse_idx)
    save_txt(f_test_idx,test_idx)
    save_data_information(f_data_information,data_information)

    # build model and opt
    model = build_model(mdol)
    model.to(device)

    optimizer,scheduler = build_opt(model)
    # get loss
    if lss=='CE':    
        criteon = nn.CrossEntropyLoss()
    else:
        print('Invalid loss!')
        exit()
    #(preform, epoch)    
    best_perform = {         
        'Pre':(0,0),
        'Acc':(0,0),
        'Rec':(0,0),
        'F1': (0,0),
        'Mcc':(0,0),
        'Sum':(0,0),
        'Ap': (0,0)
    }
    assert(sel_mdol in best_perform.keys())
    
    for epoch in range(epochs):
        total=len(first_train_labidx_train_loader)
        for step, (x,y) in enumerate(first_train_labidx_train_loader):
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
            val_res = evalute(model, first_train_labidx_val_loader,device)
            f_log.write('--------------Epoch %d Evaluation--------------\n'%epoch)
            print('--------------Epoch %d Evaluation--------------'%epoch)
            f_log.write(str(val_res)[1:-1]+'\n')
            print(str(val_res)[1:-1])
            #get best info
            if best_perform[sel_mdol][0] <= val_res[sel_mdol]:
                best_perform[sel_mdol] = (val_res[sel_mdol],epoch)
                torch.save(model.state_dict(), first_train_path+'/best.pth')
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
        model.load_state_dict(torch.load(first_train_path+'/best.pth'))
        f_log.write('loaded from best val checkpoint!\n')
        print('loaded from best val checkpoint!')
    
        test_res = evalute(model, test_loader,device)
        f_log.write(str(test_res)[1:-1])
        print(str(test_res)[1:-1])
    f_log.close()

