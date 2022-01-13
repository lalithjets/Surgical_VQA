import os
import glob
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as transforms

import pandas as pd
import torch
import numpy as np

from torchvision import transforms
from torch.utils.data import DataLoader



class PathVQA(Dataset):
    def __init__(self, folder, img_folder, QA, transform=None):
        
        self.transform = transform
        
        self.img_path = []

        QA_file = os.path.join(folder, QA)
        df = pd.read_pickle(QA_file)
       
        self.img_path = []
        self.q = []
        self.ans = []
        for item in df:
            path = os.path.join(folder, img_folder, item['image']) + '.jpg'
            self.img_path.append(path)
            self.q.append(item['question'])
            self.ans.append(item['answer'])

        c=0
        for file in os.listdir(os.path.join(folder, img_folder)): c +=1

        print('Total files: %d | Total question: %.d' %(c, len(self.q)))
              
    def __len__(self):
        return len(self.q)

    def __getitem__(self, idx):
        
        # img
        Img = []
        for path in self.img_path:
            img = Image.open(path)
            if self.transform: img = self.transform(img)
            Img.append(img)
            
        # question and answer
        question = self.q
        label = self.ans

        return Img, question, label


os.environ["CUDA_VISIBLE_DEVICES"]="1"

def seed_everything(seed=27):
    '''
    Set random seed for reproducible experiments
    Inputs: seed number 
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
     
    # Set random seed
    seed_everything()  
    
    # Device Count
    num_gpu = torch.cuda.device_count()
    
    # hyperparameters
    bs = 32
        
        # train and test dataloader
    folder = 'VQA/Data/PathVQA'
    train_img_folder = 'images/train'
    val_img_folder = 'images/val'
    test_img_folder = 'images/test'
    train_QA = 'qas/train/train_qa.pkl'
    val_QA = 'qas/val/val_qa.pkl'
    test_QA = 'qas/test/test_qa.pkl'

    transform = transforms.Compose([
                transforms.Resize((300,256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ])

    # train_dataset
    train_dataset = PathVQA(folder, train_img_folder, train_QA, transform=transform)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size= bs, shuffle=True)

    # Val_dataset
    val_dataset = PathVQA(folder, val_img_folder, val_QA, transform=transform)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size= bs, shuffle=False)

    # Test_dataset
    test_dataset = PathVQA(folder, test_img_folder, test_QA, transform=transform)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size= bs, shuffle=False)
