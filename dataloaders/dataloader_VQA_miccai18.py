import os
import glob
from PIL import Image

from torch.utils.data import Dataset


class SurgicalSentenceVQADataset(Dataset):
    def __init__(self, seq, folder_head, folder_tail, transform=None):
        
        self.transform = transform
        
        # files, question and answers
        filenames = []
        for curr_seq in seq: filenames = filenames + glob.glob(folder_head + str(curr_seq) + folder_tail)
        self.vqas = []
        for file in filenames:
            file_data = open(file, "r")
            lines = [line.strip("\n") for line in file_data if line != "\n"]
            file_data.close()
            for line in lines: self.vqas.append([file, line])
        print('Total files: %d | Total question: %.d' %(len(filenames), len(self.vqas)))
        
    def __len__(self):
        return len(self.vqas)

    def __getitem__(self, idx):
        
        # img
        loc = self.vqas[idx][0].split('/')
        img_loc = os.path.join(loc[0],loc[1],loc[2], 'left_frames',loc[-1].split('_')[0]+'.png')
        img = Image.open(img_loc)
        if self.transform: img = self.transform(img)
            
        # question and answer
        question = self.vqas[idx][1].split('|')[0]
        label = self.vqas[idx][1].split('|')[1]

        return img, question, label