import  torch
import  os
import  random
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from astropy.table import Table

def load_imglabel_new_data(root):
    fname_true='new_real_'+str(root.rsplit('/',1)[1])+'.fits'
    fname_fake='new_bogus_'+str(root.rsplit('/',1)[1])+'.fits'
    tab_true= Table.read(os.path.join(root, fname_true))
    tab_fake= Table.read(os.path.join(root, fname_fake))
    imgs_true=np.array(tab_true['star_stamp']) 
    imgs_fake=np.array(tab_fake['star_stamp'])
    imgs=np.concatenate((imgs_true, imgs_fake), axis=0)  
    labels=np.array([1]*len(imgs_true)+[0]*len(imgs_fake))
    imgs=np.nan_to_num(imgs).transpose(0,3,1,2)   
    return imgs, labels

def create_dataset(main_opt,img_size):
    root,data_name,train_labidx_num,test_ratio,trainval_ratio,first_select_ratio,fid=\
        main_opt.dataset_path, main_opt.data_name,main_opt.train_labidx_num,\
        main_opt.test_ratio,main_opt.trainval_ratio,main_opt.first_select_ratio,main_opt.fid

    train_val_ratio=np.array([float(i) for i in trainval_ratio.split(',')])/10
    assert train_val_ratio.sum()==1.0 
    train_val_ratio[1]+=train_val_ratio[0]

    if data_name=='new_data':
        root=os.path.join(root,fid)
        imgs_np,labels_np= load_imglabel_new_data(root)
    else:
        print('Invalid dataset!')
        exit()

    shuffle_idx = np.arange(imgs_np.shape[0])
    np.random.shuffle(shuffle_idx)

    first_train_labidx_num=int(train_labidx_num*first_select_ratio)
    test_num=int(test_ratio*imgs_np.shape[0])
    first_train_unlabidx_num=imgs_np.shape[0]-first_train_labidx_num-test_num
    train_labidx,train_unlabidx,nouse_idx,test_idx=\
        shuffle_idx[:first_train_labidx_num],shuffle_idx[first_train_labidx_num:int(first_train_labidx_num+10000*main_opt.unlab_level)],\
            shuffle_idx[int(first_train_labidx_num+10000*main_opt.unlab_level):first_train_labidx_num+first_train_unlabidx_num],shuffle_idx[first_train_labidx_num+first_train_unlabidx_num:]
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
    data_information={
        'total_train_num':train_labidx_num,
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
    return train_labidx_train_dataset,train_labidx_val_dataset,test_dataset,train_labidx,train_unlabidx,nouse_idx,test_idx,data_information

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
