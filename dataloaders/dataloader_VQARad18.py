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



class VQA_Rad18(Dataset):
    def __init__(self, folder, img_folder, QA, transform=None):
        
        self.transform = transform
        
        self.img_path = []

        QA_file = os.path.join(folder, QA)
        df = pd.read_excel(QA_file)
        image_files = np.array(df['IMAGEID'])
        for file in image_files:
            self.img_path.append(os.path.join(folder, img_folder, file.strip('/')[-1]))
        self.q = np.array(df['QUESTION'])
        self.ans = np.array(df['ANSWER'])

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
    train_split = 0.8
        
    # train and test dataloader
    folder = 'VQA/Data/VQA-Med/ImageClef-2018-VQA-Med-RAD'
    img_folder = 'VQA_RAD Image Folder'
    QA = 'VQA_RAD Dataset Public.xlsx'

    transform = transforms.Compose([
                transforms.Resize((300,256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ])

    dataset = VQA_Rad18(folder, img_folder, QA, transform=transform)

    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # train_dataset
    train_dataloader = DataLoader(dataset=train_dataset, batch_size= bs, shuffle=True)

    # Val_dataset
    val_dataloader = DataLoader(dataset=val_dataset, batch_size= bs, shuffle=False)
