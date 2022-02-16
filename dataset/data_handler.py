  
import numpy as np
import glob
import tqdm    
import pickle as pkl
import pandas as pd
from random import sample, shuffle
import soundfile as sf
import os

SEARCH = False
ANALYSIS = 0
GenTrainDF = False
GenTestDF = False
MATCH = False
GenTrainNp = 0

from scipy.io import wavfile
from pesq import pesq



df_test = pd.read_pickle("dataset_test.pkl")
df_test = df_test[['file', 'data']]
ttl_f = df_test['file'].to_numpy()
for idx in range(len(df_test)):
    df = df_test.iloc[idx]
    org = df['data'] 
    our = '../result/vocal_' + df['file'] + '.flac'
    #print(len(ref))
    try:
        deg, rate= sf.read(org)
        ref, rate= sf.read(our)
        #print(len(deg))
        #print(rate)
        
        if pesq(rate, ref, deg, 'wb') > 4:
            print(org, our)
        assert len(deg) == len(ref)

    except:
        print(org, our)
        
        
       


if ANALYSIS:
    f =  glob.glob('test/*.flac')
    max = 0    
    a = 0
    min = 999999
    for p in tqdm.tqdm(f):
        data, samplerate = sf.read(p)
        if len(data) > max:
            max = len(data)
        if len(data) < min:
            min = len(data)
        a += len(data)

    print('max:', max)
    print('average: ', a/len(f))
    print('min:', min)
#======================================
if SEARCH:
    p =  glob.glob('test/*00289*')   
        
    data, samplerate = sf.read(p[0])
    data2, samplerate = sf.read(p[1])
    print(len(data), len(data2))
#======================================
if GenTrainDF:
    train =  glob.glob('train/mixed*.flac')    
    label = glob.glob('train/vocal*.flac')

    print('train:' , len(train), 'label: ',  len(label))


    train_data = []
    test_data = []
    file_list = []

    for p in tqdm.tqdm(train):
        n = p.split('_')
        file_list.append(n[1])

    print('File list number:', len(file_list))

    data_dic = {}
    label_dic = {}

    df = pd.DataFrame({'file': file_list})

    l = len(file_list)
    t = ['train']*41000 + ['valid']*828

    PATH = './np_data'
    if not os.path.exists(PATH):
        os.mkdir(PATH)

    for p in tqdm.tqdm(train):
        n = p.split('_')
        name = n[1]
        data_dic[name] = p



    for p in tqdm.tqdm(label):
        #data, samplerate = sf.read(p)
        #padd with 0
        #data += [0] * (518400 - len(data))
        s = p
        p = p.replace('.', '_')
        n = p.split('_')
        name = n[1]
        label_dic[name] = s


    df['type'] = t
    df['label']= df['file'].apply(lambda t : label_dic[t])
    df['data'] = df['file'].apply(lambda t : data_dic[t])
    print(df.head())
    df.to_pickle("./dataset.pkl")

if GenTestDF:

    test =  glob.glob('test/*.flac')    
    

    print('test:' , len(test))
    test_data = []
    file_list = []

    for p in tqdm.tqdm(test):
        n = p.split('_')
        file_list.append(n[1])

    print('File list number:', len(file_list))

    data_dic = {}

    df = pd.DataFrame({'file': file_list})



    for p in tqdm.tqdm(test):
        n = p.split('_')
        name = n[1]
        data_dic[name] = p

    df['data'] = df['file'].apply(lambda t : data_dic[t])
    print(df.head())
    print(len(df))
    df.to_pickle("./dataset_test_lite.pkl")


