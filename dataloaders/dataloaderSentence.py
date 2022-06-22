'''
Description     : Dataloader for Sentence task.
Paper           : Surgical-VQA: Visual Question Answering in Surgical Scenes Using Transformers
Author          : Lalithkumar Seenivasan, Mobarakol Islam, Adithya Krishna, Hongliang Ren
Lab             : MMLAB, National University of Singapore
'''

import os
import glob
import h5py

import torch
from torch.utils.data import Dataset


'''
MedVQA Sentence Dataloader
'''
class MedVQASentence(Dataset):
    def __init__(self, datafolder, imgfolder, filename, patch_size = 4):
        
        self.data_folder_loc = datafolder
        self.img_folder_loc = imgfolder
        self.file_name = filename
        self.patch_size = patch_size

        self.vqas = []
        file_data = open((self.data_folder_loc+self.file_name), "r")
        lines = [line.strip("\n") for line in file_data if line != "\n"]
        file_data.close()
        for line in lines: self.vqas.append([line])
        
        print('Total question: %.d' %len(lines))
              
    def __len__(self):
        return len(self.vqas)

    def __getitem__(self, idx):
        
        # img
        visual_feature_loc = self.data_folder_loc+ 'vqa/img_features/'+(str(self.patch_size)+'x'+str(self.patch_size))+ '/'+ self.vqas[idx][0].split('|')[0]+'.hdf5'
        frame_data = h5py.File(visual_feature_loc, 'r')    
        visual_features = torch.from_numpy(frame_data['visual_features'][:])
        # question and answer
        question = self.vqas[idx][0].split('|')[1]
        answer = self.vqas[idx][0].split('|')[2]

        return self.vqas[idx][0].split('|')[0], visual_features, question, answer


'''
EndoVis18 Sentence Dataloader
'''
class EndoVis18VQASentence(Dataset):
    def __init__(self, seq, folder_head, folder_tail, patch_size):
        
        self.patch_size = patch_size
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
        visual_feature_loc = os.path.join(loc[0],loc[1],loc[2], 'vqa/img_features',(str(self.patch_size)+'x'+str(self.patch_size)) ,loc[-1].split('_')[0]+'.hdf5')
        frame_data = h5py.File(visual_feature_loc, 'r')    
        visual_features = torch.from_numpy(frame_data['visual_features'][:])
        # question and answer
        question = self.vqas[idx][1].split('|')[0]
        label = self.vqas[idx][1].split('|')[1]

        return loc[-1].split('_')[0], visual_features, question, label


'''
EndoVis18 video Sentence Dataloader
'''
class EndoVis18VidVQASentence(Dataset):
    def __init__(self, seq, folder_head, folder_tail, temporal_size, patch_size):
        
        self.temporal_size = temporal_size
        self.patch_size = patch_size
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
        visual_feature_loc = os.path.join(loc[0],loc[1],loc[2], ('vqa/vid_features'+str(self.temporal_size)),(str(self.patch_size)+'x'+str(self.patch_size)) ,loc[-1].split('_')[0]+'.hdf5')
        frame_data = h5py.File(visual_feature_loc, 'r')    
        visual_features = torch.from_numpy(frame_data['visual_features'][:])
        # question and answer
        question = self.vqas[idx][1].split('|')[0]
        label = self.vqas[idx][1].split('|')[1]

        return loc[-1].split('_')[0], visual_features, question, label


'''
Cholec80 Sentence Dataloader
'''
class Cholec80VQASentence(Dataset):
    def __init__(self, seq, folder_head, folder_tail, patch_size):

        self.patch_size = patch_size
        # files, question and answers
        filenames = []
        for curr_seq in seq: filenames = filenames + glob.glob(folder_head + str(curr_seq) + folder_tail)
        new_filenames = []
        for filename in filenames:
            frame_num = int(filename.split('/')[-1].split('.')[0].split('_')[0])
            if frame_num % 100 == 0: new_filenames.append(filename)
    		
        self.vqas = []
        for file in new_filenames:
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
        visual_feature_loc = os.path.join(loc[0],loc[1], 'cropped_image',loc[3],'vqa/img_features',(str(self.patch_size)+'x'+str(self.patch_size)) ,loc[-1].split('_')[0]+'.hdf5')
        frame_data = h5py.File(visual_feature_loc, 'r')    
        visual_features = torch.from_numpy(frame_data['visual_features'][:])
        # question and answer
        question = self.vqas[idx][1].split('|')[0]
        label = self.vqas[idx][1].split('|')[1]

        return loc[-1].split('_')[0], visual_features, question, label


'''
Cholec80 TCN Sentence Dataloader
'''
class Cholec80VidVQASentence(Dataset):
    def __init__(self, seq, folder_head, folder_tail, temporal_size, patch_size):
        
        self.temporal_size = temporal_size
        self.patch_size = patch_size
        # files, question and answers
        filenames = []
        for curr_seq in seq: filenames = filenames + glob.glob(folder_head + str(curr_seq) + folder_tail)
        new_filenames = []
        for filename in filenames:
            frame_num = int(filename.split('/')[-1].split('.')[0].split('_')[0])
            if frame_num % 100 == 0: new_filenames.append(filename)
    		
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
        visual_feature_loc = os.path.join(loc[0],loc[1], 'cropped_image',loc[3], ('vqa/vid2_features'+str(self.temporal_size)),(str(self.patch_size)+'x'+str(self.patch_size)) ,loc[-1].split('_')[0]+'.hdf5')
        frame_data = h5py.File(visual_feature_loc, 'r')    
        visual_features = torch.from_numpy(frame_data['visual_features'][:])
        # question and answer
        question = self.vqas[idx][1].split('|')[0]
        label = self.vqas[idx][1].split('|')[1]

        return loc[-1].split('_')[0], visual_features, question, label
