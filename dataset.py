


import torch
from torch.utils.data import Dataset,DataLoader,SubsetRandomSampler,random_split
import os




class MyDataset(Dataset):
    def __init__(self,cry_root_dir,label_root_dir):
        self.cry_root_dir=cry_root_dir
        self.cry_path_name=os.listdir(self.cry_root_dir)
        #self.cry_path_name.remove('.ipynb_checkpoints')
        self.cry_path_name= sorted(self.cry_path_name, key=lambda x: int(x.split('.')[0]))
        for i in range(0,len(self.cry_path_name)):
            self.cry_path_name[i]=self.cry_root_dir+'/'+self.cry_path_name[i]
        
        self.label_root_dir=label_root_dir
        self.label_path_name=os.listdir(self.label_root_dir)
        #self.label_path_name.remove('.ipynb_checkpoints')
        self.label_path_name= sorted(self.label_path_name, key=lambda x: int(x.split('.')[0]))
        for i in range(0,len(self.label_path_name)):
            self.label_path_name[i]=self.label_root_dir+'/'+self.label_path_name[i]
    def __getitem__(self,idx):
        cry_path=self.cry_path_name[idx]
        cry=np.load(cry_path)
        cry=torch.tensor(cry,dtype=torch.float32).cuda(0)
        cry=cry.permute(3, 0, 1, 2)
        label_path=self.label_path_name[idx]
        label=np.load(label_path)
        label=torch.tensor(label,dtype=torch.float32).cuda(0)
        return cry,label
    
    def __len__(self):
        return len(self.cry_path_name)



def train_loader(args):
    torch.manual_seed(args.seed)
    mydataset=MyDataset(args.path_npy2,args.path_label)
    train_size=(int(args.lengths*len(mydataset)))
    test_size=(len(mydataset)-train_size)
    train_dataset, test_dataset = random_split(mydataset, [train_size, test_size])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,num_workers=num_workers)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=True, num_workers=num_workers) 
    
    return train_dataloader


def test_loader(args):
    torch.manual_seed(args.seed)
    mydataset=MyDataset(args.path_npy2,args.path_label)
    train_size=(int(args.lengths*len(mydataset)))
    test_size=(len(mydataset)-train_size)
    train_dataset, test_dataset = random_split(mydataset, [train_size, test_size])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,num_workers=num_workers)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=True, num_workers=num_workers) 
    
    return test_dataloader






