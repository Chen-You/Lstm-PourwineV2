import re
import os
import glob
import pickle
from multiprocessing import Pool

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler as srs

from config import params

def logged(func):
    def wrapper(*args, **kwargs):
        print(f"{func.__name__} was called")
        return func(*args, **kwargs)
    return wrapper

@logged
def loadData(file_path, task, cols):
    """Returns time-step data

    Arguments:
        file_path (str): the csv files path
            e.g., xxx/*.csv
        task (str): select rows by task
        cols (list): select columns by cols

    Returns:
        data (list): numpy array
    """

    files = glob.glob(file_path)
    assert len(files)>0, 'Not found any file'

    data = []

    for f in files:
        tmp = pd.read_csv(f)
        tmp = tmp[tmp['task']==task][cols]
        data.append(tmp.values)

    return data

@logged
def findSeqMaxLength(data):
    """Returns max sequence length

    Arguments:
        data (list): numpy array

    Returns:
        seq (int): max sequence length
    """

    seqs = [i.shape[0] for i in data]
    return max(seqs)

# @logged
def validInterp(data, d=0.016):
    tmp = []
    for i in range(1, len(data)):
        diff = np.abs(data[i]-data[i-1])
        idx = np.argmax(diff)
        t, t1 = data[i-1][idx], data[i][idx]
        n = int(abs(t1-t)/d+1+1)
        x = np.arange(2)
        y = data[i-1:i+1]
        f = interpolate.interp1d(x=x, y=y, kind='linear', axis=0)
        xnew = np.linspace(0, 1, n)
        ynew = f(xnew)[:-1]
        tmp.append(ynew)

    new_data = np.concatenate(tmp, axis=0)
    
    check = np.max(np.abs(np.diff(new_data, axis=0)))
    assert check < 0.016, print(check)

    return new_data

@logged
def interpolation(data_arr):
    """Returns data with same time-step

    Arguments:
        data_arr (list): numpy array
        # max_len (int): max sequence length

    Returns:
        result (list): numpy array in a list
            all numpy array has same shape
    """
    with Pool() as pool:
        interp_data = pool.starmap(validInterp, zip(data_arr))

    max_len = findSeqMaxLength(interp_data)

    result = []
    for idx, d in enumerate(interp_data):
        x = np.arange(d.shape[0])
        f = interpolate.interp1d(x=x, y=d, kind='cubic', axis=0)
        xnew = np.linspace(0, d.shape[0]-1, max_len)
        ynew = f(xnew)
        result.append(ynew)

    # print(result[0].shape)
    # plt.scatter(np.arange(len(result[0][:, 9])), result[0][:, 9])
    # plt.show()

    return result

@logged
def standardize(task, data):
    """Returns data with standardization

    Arguments:
        data (list): numpy array

    Returns:
        data (list): data with standardization
    """

    arr = np.concatenate(data, axis=0)
    arr_min = np.min(arr, axis=0)
    arr_max = np.max(arr, axis=0)

    paths = [params['pickle_path'], task+'_stand.pkl']
    with open(os.path.join(*paths), 'wb') as f:
        norm = {'min': arr_min, 'max': arr_max}
        pickle.dump(norm, f)

    print(f"Save min, max as {task}_stand.pkl")

    new_data = []
    for idx, d in enumerate(data):
        tmp_d = (d-arr_min)/(arr_max-arr_min)
        new_data.append(tmp_d)

    # check = np.concatenate(new_data, axis=0)
    # print(check.min(axis=0), check.max(axis=0))

    return new_data, norm


@logged
def trainValidSplit(file_path, valid_ratio):
    """Returns training data and valid data

    Arguments:
        valid_ratio (float): ratio of train data and valid data

    Returns:
        train_index (np array): 
        valid_index (np array):
    """
    indexs = len(glob.glob(file_path))
    indexs = np.arange(indexs)
    np.random.shuffle(indexs)
    point = int(valid_ratio*len(indexs))
    train_index = indexs[point:]
    valid_index = indexs[:point]
    return train_index, valid_index


def dataPreprocess(task, file_path, valid_ratio):

    cols = None

    if task == 'move_to_drink':
        cols = [
            'hand_x', 'hand_y', 'hand_z',
            'coke_x', 'coke_y', 'coke_z',
            'rm1ps', 'rm2ps', 'rm3ps', 'rm4ps', 'rm5ps', 'LRMps', 'UDMps'
        ]
    elif task == 'grasp':
        cols = [
            'tb1ps', 'tb2ps',
            'if1ps', 'if2ps',
            'md0ps', 'rg0ps', 'lt0ps'
        ]
    elif task == 'move_up' or task == 'put_drink' or task == 'pour' or task == 'unpour':
        cols = [
            'hand_x', 'hand_y', 'hand_z',
            'rm1ps', 'rm2ps', 'rm3ps', 'rm4ps', 'rm5ps', 'LRMps', 'UDMps'
        ]

    elif task == 'move_to_cup':
        cols = [
            'hand_x', 'hand_y', 'hand_z',
            'cup_x', 'cup_y', 'cup_z',
            'rm1ps', 'rm2ps', 'rm3ps', 'rm4ps', 'rm5ps', 'LRMps', 'UDMps'
        ]
    elif task == 'release':
        cols = [
            'tb1ps', 'tb2ps',
            'if1ps', 'if2ps',
            'md0ps', 'rg0ps', 'lt0ps'
        ]

    assert cols != None

    data = loadData(file_path, task, cols)
    data = interpolation(data)
    data, norm = standardize(task, data)
    train_index, valid_index = trainValidSplit(file_path, valid_ratio)
    # print(train_index, valid_index)

    train_dataset = CustomDataset(data, train_index, norm)
    valid_dataset = CustomDataset(data, valid_index, norm)
    train_loader = DataLoader(
        train_dataset,
        batch_size=params['batch_size'],
        shuffle=params['shuffle'],
        num_workers=params['num_workers'],
        drop_last=params['drop_last']
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=params['batch_size'],
        shuffle=params['shuffle'],
        num_workers=params['num_workers'],
        drop_last=params['drop_last']
    )

    return train_loader, valid_loader, norm


class CustomDataset(Dataset):
    def __init__(self, data, index, norm):
        self.data = data
        self.index = index
        self.norm = norm

    def __getitem__(self, index):
        select = self.index[index]
        sample = self.data[select]
        sample = sample.astype(np.float32)
        return sample

    def __len__(self):
        return len(self.index)


if __name__ == '__main__':
    train_loader, valid_loader, norm = dataPreprocess(params['task'], params['file_path'], params['valid_ratio'])
    print(len(train_loader), len(valid_loader))

    tmp = next(iter(train_loader))
    print(tmp.size())

    tmp = next(iter(valid_loader))
    print(tmp.size())

   
