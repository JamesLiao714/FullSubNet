import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import config as cfg
import soundfile as sf
import glob
import pandas as pd
import random
from sklearn.model_selection import train_test_split


def create_dataloader(mode,type=0, snr=0):
    if mode == 'train':
        return DataLoader(
            dataset=Wave_Dataset(mode, type, snr),
            batch_size=cfg.batch,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
            sampler=None
        )
    elif mode == 'valid':
        return DataLoader(
            dataset=Wave_Dataset(mode, type, snr),
            batch_size=cfg.batch, shuffle=False, num_workers=0
        )
    elif mode == 'test':
        return DataLoader(
            dataset=Wave_Dataset(mode, type, snr),
            batch_size=cfg.batch, shuffle=False, num_workers=0
        )

# Dataloader for aicup competetion
model = 1
class Wave_Dataset(Dataset):
    def __init__(self, mode, type, snr):
        # Load Data
        # train_point.pkl has less training data (select segment > 32000 (2s)) 
        # best score: 1224.4674(30 epochs for 216_backup)
        # Train_point2.pkl has more training data (select segment > 16000 (1s)) 
        #df_all = pd.read_pickle("./dataset/train_point2.pkl")
        df_all = pd.read_pickle("./dataset/train_point2.pkl")
        df_test = pd.read_pickle("./dataset/dataset_test.pkl")
        if mode == 'train':
            # Load training data
            # df[data]: training file path
            # df[label]: label file path
            # df[split]: split point (for sounfile longer than 5 sec)
            self.mode = 'train'
            print('<Training dataset>')
            print('Load the data...')
            df = df_all
            self.df = df[['data', 'label', 'split']].reset_index(drop = True)
        if mode == 'valid':
            self.mode = 'valid'
            # Split 0.02 data for validation,
            # use random state to control randomness
            train, valid = train_test_split(df_all, test_size = 0.02, random_state=42)
            self.df = valid[['data', 'label', 'split']].reset_index(drop = True)
            #self.df.to_pickle("./valid.pkl")  
        elif mode == 'test':
            # Load testing data
            self.mode = 'test'          
            print('<Test dataset>')
            print('Load the data...')
            self.df = df_test[['file',  'data']].reset_index(drop = True)

            
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        max_l = 80000 # 5sec * 16000 (sample rate)
        d = self.df.iloc[idx]
        if self.mode == 'test':
            fn = d['file']
            test_path = d['data']
            test_path = 'dataset/'+ test_path
            inputs, samplerate = sf.read(test_path)
            ol = len(inputs) # original test soundfile length
            l1 = 0
            l2 = 0
            l3 = 0
            input1 = np.zeros(max_l) #(8000,)
            input2 = np.zeros(max_l)# (8000,)
            input3 = np.zeros(max_l) #(8000,)
            # inputs over 5 min
            if ol > max_l:
                l1 = max_l
                l2 = len(inputs[max_l:])
                input1 = inputs[:max_l]
                if l2 > max_l:
                    l2 = max_l 
                    l3 = len(inputs[2*max_l:])
                    input2 = inputs[max_l:2*max_l]
                    input3[:l3] = inputs[2*max_l:]
                else:
                    input2[:l2] = inputs[max_l:]
            # test file less than or equal 5 min
            elif ol <= max_l:
                l1 = ol
                input1[:ol] = inputs[:ol]

            input1 = torch.from_numpy(input1)
            input2 = torch.from_numpy(input2)
            input3 = torch.from_numpy(input3)
            # asser length of model input 
            try:
                assert len(input1) == 80000 and len(input2) == 80000 and len(input3) == 80000
            except:
                print(len(input1), len(input2))
            return fn, input1, input2, input3, ol, l1, l2, l3
        # for training&validation data
        elif self.mode == 'train' or self.mode == 'valid': 
            # read soundfiles (.flac)
            inputs_path = d['data']
            inputs_path = 'dataset/' + inputs_path
            # read label sounfiles
            targets_path = d['label']
            targets_path = 'dataset/'+ targets_path
            # retrieve split point for file segmentation
            beg = d['split']
            inputs, samplerate = sf.read(inputs_path)
            inputs = inputs[beg:]
            inputs = list(inputs)
            # noise file length
            noise_l = len(inputs)
            targets, samplerate = sf.read(targets_path)
            targets = targets[beg:]
            targets = list(targets)
            # clean file length
            clean_l = len(targets)
            # check length match, two size should be equal
            try:
                assert clean_l == noise_l
            except:
                print(d)
 
            if noise_l < max_l:
                #padd with 0
                pad = [0] * (max_l - noise_l)
                inputs.extend(pad)
                targets.extend(pad)
            elif noise_l > max_l:
                inputs = inputs[:max_l]
                targets = targets[:max_l]

            inputs = np.array(inputs)
            targets = np.array(targets)
            # assert model input length == max_len
            try:
                assert len(inputs) == max_l
            except:
                print(d)
            try:
                assert len(targets) == max_l
            except:
                print(d)
            # transform to torch from numpy
            inputs = torch.from_numpy(inputs)
            targets = torch.from_numpy(targets)
            return inputs, targets

       
