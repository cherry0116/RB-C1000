import  torch
import  os
import  random
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from astropy.table import Table

def load_imglabel_new_data(root):
    fname_true=str(root.rsplit('/',1)[1])+'_true_data.fits'
    fname_fake=str(root.rsplit('/',1)[1])+'_false_data.fits'
    tab_true= Table.read(os.path.join(root, fname_true))
    tab_fake= Table.read(os.path.join(root, fname_fake))
    imgs_true=np.array(tab_true['star_stamp']) 
    imgs_fake=np.array(tab_fake['star_stamp'])
    imgs=np.concatenate((imgs_true, imgs_fake), axis=0)  
    labels=np.array([1]*len(imgs_true)+[0]*len(imgs_fake))
    imgs=np.nan_to_num(imgs).transpose(0,3,1,2) 
    return imgs, labels

def select_train_idx_process(bias_select_ratio,select_train_idx,select_train_idxs_labs,train_labidx_train10ratio):
    lab0_index=[i for i,x in enumerate(select_train_idxs_labs) if x==0]
    lab1_index=[i for i,x in enumerate(select_train_idxs_labs) if x==1]

    idxs1=select_train_idx[lab1_index]
    labels1=select_train_idxs_labs[lab1_index]
    idxs0=select_train_idx[lab0_index]
    labels0=select_train_idxs_labs[lab0_index]

    sample1=int(len(lab0_index)*(train_labidx_train10ratio+bias_select_ratio)/(1-train_labidx_train10ratio-bias_select_ratio))
    sample0=int(len(lab1_index)/(train_labidx_train10ratio-bias_select_ratio)-len(lab1_index))

    if len(lab1_index)>=sample1:
        idxs1_we_need=idxs1[:sample1]
        labels1_we_need=labels1[:sample1]
        idxs10_we_need=idxs1_we_need.tolist()+idxs0.tolist()
        labels10_we_need=labels1_we_need.tolist()+labels0.tolist()
    elif len(lab0_index)>=sample0:
        idxs0_we_need=idxs0[:sample0]
        labels0_we_need=labels0[:sample0]
        idxs10_we_need=idxs1.tolist()+idxs0_we_need.tolist()
        labels10_we_need=labels1.tolist()+labels0_we_need.tolist()
    else:
        idxs10_we_need=idxs1.tolist()+idxs0.tolist()
        labels10_we_need=labels1.tolist()+labels0.tolist()

    shuffle_idx = np.arange(len(idxs10_we_need))
    np.random.shuffle(shuffle_idx)
    idxs10=np.array(idxs10_we_need)[shuffle_idx]
    labels10=np.array(labels10_we_need)[shuffle_idx]

    return labels10,idxs10

def create_dataset(main_opt,first_train_labidx,select_train_idx,test_idx,img_size):
    root=main_opt.dataset_path
    data_name=main_opt.data_name
    trainval_ratio=main_opt.trainval_ratio
    fid=main_opt.fid
    train_val_ratio=np.array([float(i) for i in trainval_ratio.split(',')])/10
    bias_select_ratio=main_opt.bias_select_ratio

    assert train_val_ratio.sum()==1.0 
    train_val_ratio[1]+=train_val_ratio[0]

    if data_name=='new_data':
        root=os.path.join(root,fid)
        imgs_np,labels_np= load_imglabel_new_data(root)
    else:
        print('Invalid dataset!')
        exit()

    first_train_labidx_labs,select_train_idxs_labs=labels_np[first_train_labidx],labels_np[select_train_idx] 
    first_train_labidx_1=first_train_labidx_labs.tolist().count(1)    
    select_select_train_idxs_labs,select_select_train_idxs=select_train_idx_process(bias_select_ratio,select_train_idx,select_train_idxs_labs,first_train_labidx_1/len(first_train_labidx_labs))
    shuffle_idx = np.arange(first_train_labidx.shape[0]+select_select_train_idxs.shape[0])
    np.random.shuffle(shuffle_idx)
    train_labidx=np.array(first_train_labidx.tolist()+select_select_train_idxs.tolist())
    train_labidx=train_labidx[shuffle_idx]

    train_val_num=(np.array(train_val_ratio)*train_labidx.shape[0]).astype(int)
    train_labidx_train,train_labidx_val=train_labidx[:train_val_num[0]],train_labidx[train_val_num[0]:]
          
    #get different split 
    train_labidx_train_imgs,train_labidx_val_imgs,test_imgs =   imgs_np[train_labidx_train],\
                                                                imgs_np[train_labidx_val],\
                                                                imgs_np[test_idx]
    train_labidx_train_labs,train_labidx_val_labs,test_labs =   labels_np[train_labidx_train],\
                                                                labels_np[train_labidx_val],\
                                                                labels_np[test_idx]                                
     
    
    train_labidx_train_1=train_labidx_train_labs.tolist().count(1)  
    train_labidx_val_1=train_labidx_val_labs.tolist().count(1)     
    select_select_train_idxs_1=select_select_train_idxs_labs.tolist().count(1) 
    if []==select_select_train_idxs_labs.tolist():
        select_select_train_idxs_labs=[0] 

    data_information={
        'first_train_labidx_1':first_train_labidx_1,
        'first_train_labidx_0':len(first_train_labidx_labs)-first_train_labidx_1,
        'first_train_labidx_1ratio':first_train_labidx_1/len(first_train_labidx_labs),
        'select_train_idxs_1':select_select_train_idxs_1,
        'select_train_idxs_0':len(select_select_train_idxs_labs)-select_select_train_idxs_1,
        'select_train_idxs_1ratio':select_select_train_idxs_1/len(select_select_train_idxs_labs),
        'train_labidx_train_1':train_labidx_train_1,
        'train_labidx_train_0':len(train_labidx_train_labs)-train_labidx_train_1,
        'train_labidx_train_1ratio':train_labidx_train_1/len(train_labidx_train_labs),
        'train_labidx_val_1':train_labidx_val_1,
        'train_labidx_val_0':len(train_labidx_val_labs)-train_labidx_val_1,
        'train_labidx_val_1ratio':train_labidx_val_1/len(train_labidx_val_labs),
    } 
    
    #get dataset
    train_labidx_train_dataset = ztf_dataset(train_labidx_train_imgs,train_labidx_train_labs,img_size,'train')   
    train_labidx_val_dataset = ztf_dataset(train_labidx_val_imgs,train_labidx_val_labs,img_size,'val')  
    test_dataset = ztf_dataset(test_imgs,test_labs,img_size,'test')      
    return train_labidx_train_dataset,train_labidx_val_dataset,test_dataset,train_labidx,data_information

class ztf_dataset(Dataset):

    def __init__(self,images,labels,size,split):
        super(ztf_dataset, self).__init__()
        self.size = size
        self.images,self.labels=images,labels
        self.split = split
        
    def Vertical(self,x_nparry):
        return np.flip(x_nparry,1)
    
    def Horizontal(self,x_nparry):
        return np.flip(x_nparry,2)
    
    def Resize(self,x_nparry):
        x_tensor=torch.tensor(x_nparry).unsqueeze(0)
        x_resize = F.interpolate(x_tensor, [self.size,self.size], mode='bilinear', align_corners=True).squeeze(0).numpy()
        return x_resize
        
    def transnp(self,x_nparry):
        x_resize=self.Resize(x_nparry)   #3*63*63->3*224*224
        if random.random()>=0.5 and self.split=='train':
            x_resize=self.Vertical(x_resize)
        if random.random()>=0.5 and self.split=='train':
            x_resize=self.Horizontal(x_resize)   
        return x_resize

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, label = self.images[idx], self.labels[idx]
        img= self.transnp(img)
        image = img.copy()
        image = torch.from_numpy(image)
        return image, label
