import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import config as cfg
import soundfile as sf
import glob
import pandas as pd
import random
import time




# # If you don't set the data type to object when saving the data... 
# np_load_old = np.load
# np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)




def create_dataloader(mode, type=0, snr=0):
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


class Wave_Dataset(Dataset):
    def __init__(self, mode, type, snr):
        # load data
        df_all = pd.read_pickle("./dataset/dataset.pkl")
        df_test = pd.read_pickle("./dataset/dataset_test.pkl")
        if mode == 'train':
            self.mode = 'train'
            print('<Training dataset>')
            print('Load the data...')
            df = df_all
            self.df = df[['data', 'label']].reset_index(drop = True)
        elif mode == 'valid':
            self.mode = 'valid'
            print('<Validation dataset>')
            print('Load the data...')
            df = df_all[df_all['type'] == 'valid']
            self.df = df[['data', 'label']].reset_index(drop = True)
            # # if you want to use a part of the dataset
            # self.input = self.input[:500]
        elif mode == 'test':
            self.mode = 'test'          
            print('<Test dataset>')
            print('Load the data...')
            self.df = df_test[['file', 'data']].reset_index(drop = True)
            
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        max_l = 80000 # 16000*5sec
        d = self.df.iloc[idx]
        if self.mode == 'test':
            fn = d['file']
            test_path = d['data']
            test_path = 'dataset/'+ test_path
            inputs, samplerate = sf.read(test_path)
            ol = len(inputs) #original len
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
            #inputs less than or equal 5 min
            elif ol <= max_l:
                l1 = ol
                input1[:ol] = inputs[:ol]
            
            input1 = torch.from_numpy(input1)
            input2 = torch.from_numpy(input2)
            input3 = torch.from_numpy(input3)
            try:
                assert len(input1) == 80000 and len(input2) == 80000 and len(input3) == 80000
            except:
                print(len(input1), len(input2))
            return fn, input1, input2, input3, ol, l1, l2, l3
        # for train, val
        else: 
            # random slicing
            random.seed(time.time())
            # read sounds (.flac)
            inputs_path = d['data']
            inputs_path = 'dataset/' + inputs_path
            inputs, samplerate = sf.read(inputs_path)
            # noise file length
            noise_l = len(inputs)
            # randomly choose beg index for file that > 5s 
            beg = random.randint(0, 1000000) % noise_l

            inputs = list(inputs)
            if noise_l < max_l:
                #padd with 0
                pad = [0] * (max_l - noise_l)
                inputs.extend(pad)
            elif noise_l > max_l:
                #random slice
                end = beg + max_l
                if end > noise_l:
                    left = end - noise_l
                    inputs = inputs[beg:] + inputs[:left] 
                else:
                    inputs = inputs[beg:end]
            try:
                assert len(inputs) == max_l
            except:
                print(d)

            inputs = np.array(inputs)

            targets_path = d['label']
            targets_path = 'dataset/'+ targets_path
            targets, samplerate = sf.read(targets_path)
            clean_l = len(targets)
            try:
                assert clean_l == noise_l
            except:
                print(d)

            targets = list(targets)
            if clean_l < max_l:
                pad = [0] * (max_l - clean_l)
                targets.extend(pad)
            elif len(targets) > max_l:
                end = beg + max_l
                if end > clean_l:
                    left = end - clean_l
                    targets = targets[beg:] + targets[:left] 
                else:
                    targets = targets[beg:end]
            try:
                assert len(targets) == max_l
            except:
                print(d)
            targets = np.array(targets)
            # transform to torch from numpy
            inputs = torch.from_numpy(inputs)
            targets = torch.from_numpy(targets)
            return inputs, targets
