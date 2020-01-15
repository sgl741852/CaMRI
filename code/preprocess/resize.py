import os
import numpy as np
import gzip
import h5py
import nibabel as nib
import matplotlib.pyplot as plt
import cv2

def actual_size_index(img):
    l0,l1 = img.shape
    v0 = np.sum(np.abs(img), axis = 1)
    v1 = np.sum(np.abs(img), axis = 0)
    a0 = np.where(v0!=0)[0][0]
    a1 = np.where(v0!=0)[0][-1]
    b0 = np.where(v1!=0)[0][0]
    b1 = np.where(v1!=0)[0][-1]
    return a0,a1,b0,b1
def actual_size(img):
    a0,a1,b0,b1=actual_size_index(img)
    return a1-a0+1,b1-b0+1

path = 'Calcium/round1'
animals = ['animal55', 'animal56', 'animal73']
filename = ['2019-03-21_EPI1_REST_preprocessed_regress_preprocessed_regress_new.mat','2019-03-21_EPI1_REST_preprocessed_regress_preprocessed_regress_new.mat','2019-05-09_EPI1_REST_preprocessed_regress_preprocessed_regress_new.mat']

for j,animal in enumerate(animals):
    # load images
    file_ = os.path.join(path, animal, filename[j])
    f = h5py.File(file_,'r')
    size = np.array(f['sz']).astype(int).reshape([-1,])
    images = np.array(f['dA'])
    idx = [500*i+500 for i in range(10)]
    img = images[idx,:].reshape([10,size[1],size[0]])
    print('load successfully: ' + file_)
    
    # save original images
    imgpath = 'SampleImages/round1_'+animal+'_EPI1'
    if not os.path.isdir(imgpath):
        os.makedirs(imgpath)
    for i,ind in enumerate(idx):
        plt.imsave(imgpath+'/original_%d.jpeg'%ind, img[i], cmap = 'binary', vmin = np.min(img), vmax = np.max(img))
    
    # cut images
    a0,a1,b0,b1=actual_size_index(img[0])
    img_cut = img[:,a0:a1+1,b0:b1+1]
    
    # save 128x128 images
    for i,ind in enumerate(idx):
        img128 = cv2.resize(img_cut[i], (128,128), interpolation = cv2.INTER_LINEAR)
        plt.imsave(imgpath+'/128_%d.jpeg'%ind, img128, cmap = 'binary', vmin = np.min(img), vmax = np.max(img))
    
    # save 64x64 images
    for i,ind in enumerate(idx):
        img64 = cv2.resize(img_cut[i], (64,64), interpolation = cv2.INTER_LINEAR)
        plt.imsave(imgpath+'/64_%d.jpeg'%ind, img64, cmap = 'binary', vmin = np.min(img), vmax = np.max(img))
    

