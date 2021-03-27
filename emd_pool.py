# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 22:53:32 2021

@author: Mert Ketenci
"""
import functools

from mlxtend.data import loadlocal_mnist
import numpy as np
import cv2
from cv2 import EMD
from tqdm import tqdm

from multiprocessing import Pool

DATADIR = './data/mnist'

def calc_emd(x_tr_, heldout_, coordinates_):
    x_tr_emd_ = []
    for j in range(heldout_.shape[0]):
        img1 = np.hstack((x_tr_.reshape((-1,784)).T/np.sum(x_tr_),np.array(coordinates_).T))
        img2 = np.hstack((heldout_[j].reshape((-1,784)).T/np.sum(heldout_[j]),np.array(coordinates_).T))
        x_tr_emd_.append(EMD(np.array(img1, np.float32), np.array(img2, np.float32), cv2.DIST_L1)[0])
    return np.asarray(x_tr_emd_)

def imap_unordered_bar(func, args, n_processes = 2):
    p = Pool(n_processes)
    res_list = []
    with tqdm(total = len(args)) as pbar:
        for i, res in tqdm(enumerate(p.imap(func, args))):
            pbar.update()
            res_list.append(res)
    pbar.close()
    p.close()
    p.join()
    return res_list

if __name__ == '__main__':

    x_tr, y_tr = loadlocal_mnist(
            images_path= DATADIR + '/train-images.idx3-ubyte',
            labels_path= DATADIR + '/train-labels.idx1-ubyte')

    x_te, y_te = loadlocal_mnist(
        images_path= DATADIR + '/t10k-images.idx3-ubyte',
        labels_path= DATADIR + '/t10k-labels.idx1-ubyte')

    x_tr = np.asarray(x_tr/255, dtype=np.float32)#.reshape(-1,28,28)
    x_te = np.asarray(x_te/255, dtype=np.float32)#.reshape(-1,28,28)

    choice = np.random.choice(x_tr.shape[0], size=x_tr.shape[1], replace=False)
    heldout = x_tr[choice,:]#/255

    y_heldout = y_tr[choice]

    coordinates = np.unravel_index(np.arange(784), (28,28))

    x_tr_emd = np.asarray(imap_unordered_bar(functools.partial(calc_emd, heldout_=heldout,
                                 coordinates_=coordinates), x_tr[:10], n_processes=4))

    x_te_emd = np.asarray(imap_unordered_bar(functools.partial(calc_emd, heldout_=heldout,
                                 coordinates_=coordinates), x_te[:10], n_processes=4))

    np.save(DATADIR + '/x_tr_emd.npy', x_tr_emd)
    np.save(DATADIR + '/y_tr.npy', y_tr)
    np.save(DATADIR + '/x_te_emd.npy', x_te_emd)
    np.save(DATADIR + '/y_te.npy', y_te)
