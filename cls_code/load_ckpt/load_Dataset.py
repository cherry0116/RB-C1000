import  torch
import  os
import  random
import numpy as np
import torch.nn.functional as F
from    torch.utils.data import Dataset
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

def create_dataset(load_opt,data_name,ori_fid,fid,train_unlabidx,nouse_idx,test_idx,img_size):
    root=load_opt.test_dataset_path
    if data_name=='new_data':
        root=os.path.join(root,fid)
        imgs_np,labels_np= load_imglabel_new_data(root)
    else:
        print('Invalid dataset!')
        exit()

    if ori_fid!=fid and data_name=='new_data':
        shuffle_idx = np.arange(imgs_np.shape[0])
        np.random.shuffle(shuffle_idx)
        train_unlabidx=shuffle_idx[:len(train_unlabidx)]
        test_idx=shuffle_idx[len(train_unlabidx):len(train_unlabidx)+len(test_idx)]
        nouse_idx=shuffle_idx[len(train_unlabidx)+len(test_idx):len(train_unlabidx)+len(test_idx)+len(nouse_idx)]
         
    train_unlabidx_imgs,train_unlabidx_labs=imgs_np[train_unlabidx],labels_np[train_unlabidx]                  
    test_imgs,test_labs=imgs_np[test_idx],labels_np[test_idx]                                
    nouse_imgs,nouse_labs=imgs_np[nouse_idx],labels_np[nouse_idx]  
    fuse_idx=train_unlabidx.tolist()+nouse_idx.tolist()+test_idx.tolist()
    train_unlabdix_test_idx_nouse_imgs,train_unlabdix_test_idx_nouse_labs=imgs_np[fuse_idx],labels_np[fuse_idx] 

    #get dataset   
    train_unlabidx_dataset= ztf_dataset(train_unlabidx_imgs,train_unlabidx_labs,img_size,'test')   
    test_dataset = ztf_dataset(test_imgs,test_labs,img_size,'test') 
    nouse_dataset=ztf_dataset(nouse_imgs,nouse_labs,img_size,'test')
    train_unlabdix_test_idx_nouse_dataset = ztf_dataset(train_unlabdix_test_idx_nouse_imgs,train_unlabdix_test_idx_nouse_labs,img_size,'test')    
    return train_unlabidx_dataset,nouse_dataset,test_dataset,train_unlabdix_test_idx_nouse_dataset,train_unlabidx,nouse_idx,test_idx,fuse_idx

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
