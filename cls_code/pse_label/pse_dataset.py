import  torch
import  os
import  random
import numpy as np
import torch.nn.functional as F
from  torch.utils.data import Dataset
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

def create_dataset(main_opt,img_size,train_unlabidx):
    root=main_opt.dataset_path
    data_name=main_opt.data_name
    fid=main_opt.fid
  
    if data_name=='new_data':
        root=os.path.join(root,fid)
        imgs_np,_= load_imglabel_new_data(root)
    else:
        print('Invalid dataset!')
        exit()
         
    train_unlabidx_imgs_np= imgs_np[train_unlabidx]
    #get dataset  
    train_unlabidx_dataset = ztf_dataset(train_unlabidx_imgs_np,img_size,'pse')      
    return train_unlabidx_dataset

class ztf_dataset(Dataset):

    def __init__(self,images,size,split):
        super(ztf_dataset, self).__init__()
        self.size = size
        self.images=images
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
        img = self.images[idx]
        img= self.transnp(img)
        image = img.copy()
        image = torch.from_numpy(image)
        return image

