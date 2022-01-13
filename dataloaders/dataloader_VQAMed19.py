import os
import glob
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os
import torch

from torchvision import transforms
from torch.utils.data import DataLoader


class VQA_Med19(Dataset):
    def __init__(self, folder, img_folder, QA, ID, transform=None):
        
        self.transform = transform
        
        self.img_path = []
        self.q = []
        self.ans = []

        QA_file = os.path.join(folder, QA)
        file_data = open(QA_file, "r")
        lines = [line.strip("\n") for line in file_data if line != "\n"]
        file_data.close()
        for line in lines: 
            data = line.strip('|')
            i_path = os.path.join(folder, img_folder, data[0]) + '.jpg'
            
            self.img_path.append(i_path)
            self.q.append(data[1])
            self.ans.append(data[2])

        text = open(os.path.join(folder, ID), "r")
        imgs = [line.strip("\n") for line in text if line != "\n"]

        print('Total files: %d | Total question: %.d' %(len(imgs), len(lines)))
              
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
    train_folder = 'VQA/Data/VQA-Med/ImageClef-2019-VQA-Med-Training'
    val_folder = 'VQA/Data/VQA-Med/ImageClef-2019-VQA-Med-Validation'
    train_img_folder = 'Train_images'
    val_img_folder = 'Val_images'
    train_QA = 'All_QA_Pairs_train.txt'
    val_QA = 'All_QA_Pairs_val.txt'
    train_ID = 'train_ImageIDs.txt'
    val_ID = 'val_ImageIDs.txt'

    transform = transforms.Compose([
                transforms.Resize((300,256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ])

    # train_dataset
    train_dataset = VQA_Med19(train_folder, train_img_folder, train_QA, train_ID, transform=transform)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size= bs, shuffle=True)

    # Val_dataset
    val_dataset = VQA_Med19(val_folder, val_img_folder, val_QA, val_ID, transform=transform)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size= bs, shuffle=False)
