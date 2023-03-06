import tensorflow as tf
import numpy as np
import scipy.io as scio
from scipy import signal
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import backend as K
from tensorflow.keras.models import model_from_json
from src.HSIdata import *

def get_data(datasetsname=None, cfg=0):
    if(datasetsname == 'mnist'):
        data, label = load_mnist()
    elif(datasetsname == 'har'):
        data, label = load_har()
    elif(datasetsname == 'fmnist'):
        data, label = load_fmnist()
    elif(datasetsname == 'indiapines'):
        data, label = load_IndianPines()
    elif(datasetsname == 'Botswana'):
        data, label = load_Botswana()
    elif(datasetsname == 'paviacentre'):
        data, label = load_paviacentre()
    elif(datasetsname == 'paviauni'):
        data, label = load_paviauni()
    elif(datasetsname == 'DFTC2013'):
        data, label = load_DFTC2013()
    elif(datasetsname == 'DFTC2018'):
        data, label = load_DFS2018()
    elif(datasetsname == 'saline'):
        data, label = load_saline()
    else:
        data, label = None
    data = np.array(data, dtype='float32')

    if(cfg['spatial'] > 1):
        maskss = cfg['spatial']**2
        in2 = np.array(np.ones(shape=(cfg['spatial'],cfg['spatial']))/maskss)
        mdata = np.empty_like(data)
        for i in range(np.shape(data)[-1]):
             mdata[:,:,i] = signal.convolve2d(data[:,:,i], in2, 'same')
        data = mdata

    if(len(np.shape(data)) > 2):
        traindatashape = np.shape(data)
        x_train = data.reshape([-1, traindatashape[-1]])
        y_train = label.flatten()
        x_train = x_train[np.nonzero(y_train)]
        y_train = y_train[np.nonzero(y_train)]
    else:
        x_train = data
        y_train = label
    index = np.random.choice(x_train.shape[0], x_train.shape[0], replace=False)
    x_train = x_train[index, :]
    y_train = y_train[index]


    return x_train, y_train, index, label

class DataLoader():
    def __init__(self, cfg=None, datasets = None):
        (self.train_data, self.train_label), (self.test_data, self.test_label) = datasets
        self.train_data = self.train_data.astype(np.float32)
        self.test_data = self.test_data.astype(np.float32)
        self.train_label = self.train_label.astype(np.int32)
        self.test_label = self.test_label.astype(np.int32)
        self.num_train_data, self.num_test_data = self.train_data.shape[0], self.test_data.shape[0]

    def get_batch(self, batch_size):
        # 从数据集中随机取出batch_size个元素并返回
        index = np.random.randint(0, self.num_train_data, batch_size)
        return self.train_data[index, :], self.train_label[index]