  
import numpy as np
import glob
import tqdm    
import pickle as pkl
import pandas as pd
from random import sample, shuffle
import soundfile as sf
import os
from pesq import pesq

noise_type =  ['blower.flac', 'dog', 'cleaner.flac', 'garbage', 'street', 'air', 'fan.flac', 'train.flac', 'drilling.flac', 'music.flac', 'market.flac', 'grinding.flac', 'gun', 'traffic.flac', 'jackhammer.flac', 'children', 'car', 'engine', 'silence.flac', 'rainy.flac', 'siren.flac']

def main():
    # df = pd.read_pickle("../valid.pkl") 
    # print(df[125:130])
    # p = 'train/mixed_25895_siren.flac'
    # data, samplerate = sf.read(p)
    # print(len(data))
    #analyData()
    #GenTrainDF()
    #splitForTrain()
    GenTestDF()


# transform data(.flac) to numpy
# not faster than just reading .flac file in dataloader
def dataToNumpy():
    df = pd.read_pickle("./dataset.pkl")
    print(df.head())
    #path for saved np
    train_path = 'train_np/'
    label_path = 'label_np/'
    # train numpy generate
    for p, name in tqdm.tqdm(zip(df['data'], df['file'])):
        data, samplerate = sf.read(p)
        np.save(train_path + name, data)
    # label numpy generate
    for p, name in tqdm.tqdm(zip(df['label'], df['file'])):
        data, samplerate = sf.read(p)
        np.save(label_path + name, data)
    print('FINISHED GENERATING')


# generate start idex for training and validation
def genIdx(p):
    data, samplerate = sf.read(p)
    split_idx = [0]
    left = data
    beg = 0
    while len(left) - 80000> 32000:
        beg += 80000
        split_idx.append(beg)
        left = data[beg:]
    
    return split_idx

def splitForTrain():
    df = pd.read_pickle("dataset_w_cat.pkl")
    print(df.head())
    df['split'] = df.data.apply(lambda t: genIdx(t)) 
    df2 = df.explode('split').reset_index(drop=True)
    print(len(df2))
    print(df2.head())
    df2.to_pickle("./train_point3.pkl")
#splitForTrain()
# for checking output format
def calScore():
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
            
# analyze sound file for training datas 
def analyData():
    f =  glob.glob('train/mixed*.flac')
    max = 0    
    a = 0
    min = 999999
    z = []
    cat = {}
    for p in tqdm.tqdm(f):
        #data, samplerate = sf.read(p)
        n = p.split('_')[2]
        if n in cat.keys():
            cat[n] += 1
        else:
            cat[n] = 1
        # if len(data) > 80000:
        #     z.append(len(data))
        
        # if len(data) > max:
        #     max = len(data)
        # if len(data) < min:
        #     min = len(data)
        # a += len(data)
    print(cat)
    print('Noise types:', len(cat.keys()))

    # print(len(z))
    # print(z)
    # print('max:', max)
    # print('average: ', a/len(f))
    # print('min:', min)

def cls(t):
    if t in noise_type[:10]:
        return 1
    else:
        return 2
def GenTrainDF():
    train =  glob.glob('train/mixed*.flac')    
    label = glob.glob('train/vocal*.flac')

    print('train:' , len(train), 'label: ',  len(label))

    file_list = []

    for p in tqdm.tqdm(train):
        n = p.split('_')
        file_list.append(n[1])

    print('File list number:', len(file_list))

    data_dic = {}
    label_dic = {}
    cat_dic = {}
    df = pd.DataFrame({'file': file_list})


    PATH = './np_data'
    if not os.path.exists(PATH):
        os.mkdir(PATH)

    for p in tqdm.tqdm(train):
        n = p.split('_')    
        name = n[1]
        type = n[2]
        data_dic[name] = p
        cat_dic[name] = type


    for p in tqdm.tqdm(label):
        s = p
        p = p.replace('.', '_')
        n = p.split('_')
        name = n[1]
        label_dic[name] = s
   
    print('noise type: ', noise_type)
    df['model'] = df['file'].apply(lambda t : cls(cat_dic[t]))
    df['label']= df['file'].apply(lambda t : label_dic[t])
    df['data'] = df['file'].apply(lambda t : data_dic[t])
    print(df.head())
    df1 = df[df['model'] == 1]
    df2 = df[df['model'] == 2]
    print(len(df1), len(df2))
    assert len(df1) + len(df2) == len(df), "ERROR shape"
    df.to_pickle("./dataset_w_cat.pkl")

def GenTestDF():
    test =  glob.glob('test/*.flac')    
    print('test:' , len(test))
    file_list = []

    for p in tqdm.tqdm(test):
        n = p.split('_')
        type = n[2]
        file_list.append(n[1])

    print('File list number:', len(file_list))

    data_dic = {}
    cat_dic = {}
    df = pd.DataFrame({'file': file_list})

    for p in tqdm.tqdm(test):
        n = p.split('_')
        name = n[1]
        cat_dic[name] = n[2]
        data_dic[name] = p
    print(set(cat_dic.values()))
    assert len(set(cat_dic.values())) == len(noise_type), "ERROR shape1"
    #assert set(cat_dic.values()) == noise_type, "ERROR shape1"
    df['model'] = df['file'].apply(lambda t : cls(cat_dic[t]))
    df['data'] = df['file'].apply(lambda t : data_dic[t])
    print(df.head())
    df1 = df[df['model'] == 1]
    df2 = df[df['model'] == 2]
    print(len(df1), len(df2))
    assert len(df1) + len(df2) == len(df), "ERROR shape"
    #df.to_pickle("./dataset_test_w_cat.pkl")


main()