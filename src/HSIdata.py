import os, sys
import h5py
import cv2
import mat73
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.datasets import mnist
from keras.models import model_from_json
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import scipy.io as scio
from osgeo import gdal, gdal_array

# plt.switch_backend('agg')

PATH = '/home/user2/projects/DSCRLE/data/'

def load_HSIdata(params):
    '''
    Convenience function: reads from disk, downloads, or generates the data specified in params
    '''
    if params['dset'] == 'reuters':
        with h5py.File('../../data/reuters/reutersidf_total.h5', 'r') as f:
            x = np.asarray(f.get('data'), dtype='float32')
            y = np.asarray(f.get('labels'), dtype='float32')

            n_train = int(0.9 * len(x))
            x_train, x_test = x[:n_train], x[n_train:]
            y_train, y_test = y[:n_train], y[n_train:]
    elif params['dset'] == 'Botswana':
        x_train, y_train = load_Botswana()
    elif params['dset'] == 'AVIRIS':
        x_train, y_train = load_AVIRIS()
    elif params['dset'] == 'KennedySpaceCenter':
        x_train, y_train = load_KennedySpaceCenter()
    elif params['dset'] == 'DFS2018':
        x_train, y_train = load_DFS2018()
    else:
        raise ValueError('Dataset provided ({}) is invalid!'.format(params['dset']))

    # minmaxscaler = MinMaxScaler()
    # x_train = minmaxscaler.fit_transform(x_train)

    # x_train = StandardScaler().fit_transform(x_train)

    return x_train, y_train


def embed_data(x, dset):
    '''
    Convenience function: embeds x into the code space using the corresponding
    autoencoder (specified by dset).
    '''
    if not len(x):
        return np.zeros(shape=(0, 10))
    if dset == 'reuters':
        dset = 'reuters10k'

    json_path = '../pretrain_weights/ae_{}.json'.format(dset)
    weights_path = '../pretrain_weights/ae_{}_weights.h5'.format(dset)

    with open(json_path) as f:
        pt_ae = model_from_json(f.read())
    pt_ae.load_weights(weights_path)
    x = x.reshape(-1, np.prod(x.shape[1:]))
    get_embeddings = K.function([pt_ae.input], [pt_ae.layers[3].output])
    get_reconstruction = K.function([pt_ae.layers[4].input], [pt_ae.output])
    x_embedded = predict_with_K_fn(get_embeddings, x)[0]
    # x_recon = predict_with_K_fn(get_reconstruction, x_embedded)[0]
    # reconstruction_mse = np.mean(np.square(x - x_recon))
    # print("using pretrained embeddings; sanity check, total reconstruction error:", np.mean(reconstruction_mse))

    del pt_ae

    return x_embedded


def predict_with_K_fn(K_fn, x, bs=1000):
    '''
    Convenience function: evaluates x by K_fn(x), where K_fn is
    a Keras function, by batches of size 1000.
    '''
    if not isinstance(x, list):
        x = [x]
    num_outs = len(K_fn.outputs)
    y = [np.empty((len(x[0]), output_.get_shape()[1])) for output_ in K_fn.outputs]
    recon_means = []
    for i in range(int(x[0].shape[0] / bs + 1)):
        x_batch = []
        for x_ in x:
            x_batch.append(x_[i * bs:(i + 1) * bs])
        temp = K_fn(x_batch)
        for j in range(num_outs):
            y[j][i * bs:(i + 1) * bs] = temp[j]

    return y


def split_data(x, y, split, permute=None):
    '''
    Splits arrays x and y, of dimensionality n x d1 and n x d2, into
    k pairs of arrays (x1, y1), (x2, y2), ..., (xk, yk), where both
    arrays in the ith pair is of shape split[i-1]*n x (d1, d2)

    x, y:       two matrices of shape n x d1 and n x d2
    split:      a list of floats of length k (e.g. [a1, a2,..., ak])
                where a, b > 0, a, b < 1, and a + b == 1
    permute:    a list or array of length n that can be used to
                shuffle x and y identically before splitting it

    returns:    a tuple of tuples, where the outer tuple is of length k
                and each of the k inner tuples are of length 3, of
                the format (x_i, y_i, p_i) for the corresponding elements
                from x, y, and the permutation used to shuffle them
                (in the case permute == None, p_i would simply be
                range(split[0]+...+split[i-1], split[0]+...+split[i]),
                i.e. a list of consecutive numbers corresponding to the
                indices of x_i, y_i in x, y respectively)
    '''
    n = len(x)
    if permute is not None:
        if not isinstance(permute, np.ndarray):
            raise ValueError("Provided permute array should be an np.ndarray, not {}!".format(type(permute)))
        if len(permute.shape) != 1:
            raise ValueError("Provided permute array should be of dimension 1, not {}".format(len(permute.shape)))
        if len(permute) != len(x):
            raise ValueError(
                "Provided permute should be the same length as x! (len(permute) = {}, len(x) = {}".format(len(permute),
                                                                                                          len(x)))
    else:
        permute = np.arange(len(x))

    if np.sum(split) != 1:
        raise ValueError("Split elements must sum to 1!")

    ret_x_y_p = []
    prev_idx = 0
    for s in split:
        idx = prev_idx + np.round(s * n).astype(np.int)
        p_ = permute[prev_idx:idx]
        x_ = x[p_]
        y_ = y[p_]
        prev_idx = idx
        ret_x_y_p.append((x_, y_, p_))

    return tuple(ret_x_y_p)


def load_mnist(path=PATH + '/mnist/mnist.npz'):
    (train_data, train_label), (test_data, test_label) = tf.keras.datasets.mnist.load_data(path)
    train_data = np.expand_dims(train_data.astype(np.float32) / 255.0, axis=-1)
    train_data = train_data.reshape((train_data.shape[0], -1))
    return train_data, train_label


def load_har(path=PATH + '/har/'):
    data = scio.loadmat(path + 'HAR.mat')
    X = data['X']
    X = X.astype('float32')
    Y = data['Y'] - 1
    X = X[:10200]
    Y = Y[:10200]
    Y = np.squeeze(Y)
    return X, Y


def load_IndianPines(path=PATH + '/Indian Pines/'):
    x_train = scio.loadmat(path + 'Indian_pines.mat')['indian_pines']
    y_train = scio.loadmat(path + 'Indian_pines_gt.mat')['indian_pines_gt']
    return x_train, y_train

def load_Botswana(path=PATH + '/Botswana/'):
    x_train = scio.loadmat(path + 'Botswana.mat')['Botswana']
    # traindatashape = np.shape(traindata)
    # x_train = traindata.reshape([-1, traindatashape[-1]])  # an zhao 1 hang 1 hang de reshape
    y_train = scio.loadmat(path + 'Botswana_gt.mat')['Botswana_gt']
    # y_train = y_train.flatten()
    # x_train = x_train[np.nonzero(y_train)]
    # y_train = y_train[np.nonzero(y_train)]

    return x_train, y_train


def load_Chikusei(path=PATH + '/Chikusei/'):
    x_train = mat73.loadmat(path + 'HyperspecVNIR_Chikusei_20140729.mat')['chikusei']
    # traindatashape = np.shape(traindata)
    # x_train = traindata.reshape([-1, traindatashape[-1]])  # an zhao 1 hang 1 hang de reshape
    y_train = scio.loadmat(path + 'ggtt.mat')['aa']
    # y_train = y_train.flatten()
    # x_train = x_train[np.nonzero(y_train)]
    # y_train = y_train[np.nonzero(y_train)]

    return x_train, y_train


def load_DFS2018(path=PATH + '/2018IEEE_Contest/'):
    groundmapinfo = np.array([272056.000, 3290290.000])
    hsmapinfo = np.append(271460.000, 3290891.000)
    offsetxy = ((groundmapinfo - hsmapinfo)).astype(int)

    labelfile = gdal.Open(path + 'TrainingGT/2018_IEEE_GRSS_DFC_GT_TR')
    im_width = labelfile.RasterXSize
    im_height = labelfile.RasterYSize
    y_train = labelfile.ReadAsArray(0, 0, im_width, im_height)
    y_train = y_train[::2, ::2]
    im_width = int(im_width/2)
    im_height = int(im_height/2)
    # y_train_flatten = y_train.flatten()

    pix_arr = gdal_array.LoadFile(path + 'FullHSIDataset/20170218_UH_CASI_S4_NAD83.pix')
    pix_arr = pix_arr.transpose(1, 2, 0)
    # pix_arr_resize = cv2.resize(pix_arr, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    pix_arr_crop = pix_arr[-offsetxy[1]:-offsetxy[1] + im_height, offsetxy[0]:offsetxy[0] + im_width]
    # traindatashape = np.shape(pix_arr_crop)
    # x_train = pix_arr_crop.reshape([-1, traindatashape[-1]])  # an zhao 1 hang 1 hang de reshape
    #
    # img_arr = np.stack((pix_arr_crop[:, :, 16], pix_arr_crop[:, :, 32], pix_arr_crop[:, :, 48]), axis=-1)
    # img_arr = np.uint8(np.floor(img_arr / pix_arr_crop.max() * 255.0))
    #
    # label_arr = cv2.cvtColor(np.uint8(y_train / y_train.max() * 255.0), cv2.COLOR_GRAY2RGB)
    # label_arr = cv2.applyColorMap(label_arr, cv2.COLORMAP_JET)
    #
    # plt.subplot(2,1,1)
    # plt.imshow(img_arr)
    # plt.subplot(2,1,2)
    # plt.imshow(label_arr)
    # plt.show()
    # print(np.bincount(y_train))
    # mbins = len(np.unique(y_train))
    # plt.hist(y_train, bins=mbins)
    # plt.show()
    # x_train = x_train[np.nonzero(y_train_flatten)]
    # y_train_flatten = y_train_flatten[np.nonzero(y_train_flatten)]

    return pix_arr_crop, y_train


def load_AVIRIS(path='/home/jinli/PycharmProjects/test/data/AVIRIS/'):  # some problems
    labelfile = gdal.Open(path + 'NS-line_Project_and_Ground_Reference_Files/19920612_AVIRIS_IndianPine_NS-line_gr.tif')
    im_width = labelfile.RasterXSize
    im_height = labelfile.RasterYSize
    y_train = labelfile.ReadAsArray(0, 0, im_width, im_height)
    # y_train = y_train.flatten()

    pix_arr = gdal_array.LoadFile(path + 'aviris_hyperspectral_data/19920612_AVIRIS_IndianPine_NS-line.tif')
    x_train = pix_arr.transpose(1, 2, 0)
    # traindatashape = np.shape(pix_arr)
    # x_train = pix_arr.reshape([-1, traindatashape[-1]])  # an zhao 1 hang 1 hang de reshape

    # x_train = x_train[np.nonzero(y_train)]
    # y_train = y_train[np.nonzero(y_train)]

    return x_train, y_train


def load_KennedySpaceCenter(path='/home/jinli/PycharmProjects/test/data/Kennedy Space Center/'):
    x_train = scio.loadmat(path + 'KSC.mat')['KSC']
    # traindatashape = np.shape(traindata)
    # x_train = traindata.reshape([-1, traindatashape[-1]])  # an zhao 1 hang 1 hang de reshape
    y_train = scio.loadmat(path + 'KSC_gt.mat')['KSC_gt']
    # y_train = y_train.flatten()
    # x_train = x_train[np.nonzero(y_train)]
    # y_train = y_train[np.nonzero(y_train)]

    return x_train, y_train


def load_paviacentre(path=PATH + '/pavia centre/'):
    x_train = scio.loadmat(path + 'Pavia.mat')['pavia']
    # traindatashape = np.shape(traindata)
    # x_train = traindata.reshape([-1, traindatashape[-1]])  # an zhao 1 hang 1 hang de reshape
    y_train = scio.loadmat(path + 'Pavia_gt.mat')['pavia_gt']
    # y_train = y_train.flatten()
    # x_train = x_train[np.nonzero(y_train)]
    # y_train = y_train[np.nonzero(y_train)]

    return x_train, y_train


def load_paviauni(path=PATH + '/pavia uni/'):
    x_train = scio.loadmat(path + 'PaviaU.mat')['paviaU']
    # traindatashape = np.shape(traindata)
    # x_train = traindata.reshape([-1, traindatashape[-1]])  # an zhao 1 hang 1 hang de reshape
    y_train = scio.loadmat(path + 'PaviaU_gt.mat')['paviaU_gt']
    # y_train = y_train.flatten()
    # x_train = x_train[np.nonzero(y_train)]
    # y_train = y_train[np.nonzero(y_train)]

    return x_train, y_train


def load_saline(path=PATH + '/saline/'):
    x_train = scio.loadmat(path + 'salinas.mat')['Img']
    # traindatashape = np.shape(traindata)
    # x_train = traindata.reshape([-1, traindatashape[-1]])  # an zhao 1 hang 1 hang de reshape
    y_train = scio.loadmat(path + 'salinas_GT.mat')['GT']
    # y_train = y_train.flatten()
    # x_train = x_train[np.nonzero(y_train)]
    # y_train = y_train[np.nonzero(y_train)]

    return x_train, y_train


def load_xiongan(path='/home/jinli/PycharmProjects/test/data/xiongan/'):
    labelfile = gdal.Open(path + 'farm_roi.img')
    im_width = labelfile.RasterXSize
    im_height = labelfile.RasterYSize
    y_train = labelfile.ReadAsArray(0, 0, im_width, im_height)
    # y_train = y_train.flatten()

    pix_arr = gdal_array.LoadFile(path + 'XiongAn.img')
    x_train = pix_arr.transpose(1, 2, 0)
    # traindatashape = np.shape(pix_arr)
    # x_train = pix_arr.reshape([-1, traindatashape[-1]])  # an zhao 1 hang 1 hang de reshape

    # x_train = x_train[np.nonzero(y_train)]
    # y_train = y_train[np.nonzero(y_train)]

    return x_train, y_train


def load_teatree(path=PATH + '/teatree/'):
    labelfile = gdal.Open(path + 'PHI_GroundTruthFanglu/PHI_GroundTruthFanglu.img')
    im_width = labelfile.RasterXSize
    im_height = labelfile.RasterYSize
    y_train = labelfile.ReadAsArray(0, 0, im_width, im_height)
    # y_train = y_train.flatten()

    pix_arr = gdal_array.LoadFile(path + 'PHI_FangluTeaFarm/PHI_FangluTeaFarm.img')
    x_train = pix_arr.transpose(1, 2, 0)
    # traindatashape = np.shape(pix_arr)
    # x_train = pix_arr.reshape([-1, traindatashape[-1]])  # an zhao 1 hang 1 hang de reshape
    #
    # x_train = x_train[np.nonzero(y_train)]
    # y_train = y_train[np.nonzero(y_train)]

    return x_train, y_train


def load_DFTC2013(path=PATH + '/2013_DFTC/'):
    filenamelist = ['/home/user2/Datasets/data/2013_DFTC/gt/tr_water',
                    '/home/user2/Datasets/data/2013_DFTC/gt/tr_commercial',
                    '/home/user2/Datasets/data/2013_DFTC/gt/tr_grass_healthy',
                    '/home/user2/Datasets/data/2013_DFTC/gt/tr_grass_stressed',
                    '/home/user2/Datasets/data/2013_DFTC/gt/tr_grass_synthetic',
                    '/home/user2/Datasets/data/2013_DFTC/gt/tr_highway',
                    '/home/user2/Datasets/data/2013_DFTC/gt/tr_parking_lot1',
                    '/home/user2/Datasets/data/2013_DFTC/gt/tr_parking_lot2',
                    '/home/user2/Datasets/data/2013_DFTC/gt/tr_railway',
                    '/home/user2/Datasets/data/2013_DFTC/gt/tr_residential',
                    '/home/user2/Datasets/data/2013_DFTC/gt/tr_road',
                    '/home/user2/Datasets/data/2013_DFTC/gt/tr_running_track',
                    '/home/user2/Datasets/data/2013_DFTC/gt/tr_soil',
                    '/home/user2/Datasets/data/2013_DFTC/gt/tr_tennis_court',
                    '/home/user2/Datasets/data/2013_DFTC/gt/tr_tree',
                    '/home/user2/Datasets/data/2013_DFTC/gt/tr_water']

    pix_arr = gdal_array.LoadFile(path + '2013_IEEE_GRSS_DF_Contest_CASI.tif')
    x_train = pix_arr.transpose(1, 2, 0)
    # traindatashape = np.shape(pix_arr)
    # x_train = pix_arr.reshape([-1, traindatashape[-1]])  # an zhao 1 hang 1 hang de reshape

    gt = np.zeros([349, 1905])
    for i in range(1, 16):
        file = open(filenamelist[i])
        text = file.read()
        bb = np.frombuffer(bytes(text, 'utf8'), dtype=np.uint8) * i
        cc = bb.reshape((349, 1905))
        gt = gt + cc
        file.close()
    y_train = gt
    # y_train = gt.flatten()
    # x_train = x_train[np.nonzero(y_train)]
    # y_train = y_train[np.nonzero(y_train)]

    return x_train, y_train


def load_fmnist(path=PATH + '/fashionminst/'):
    import gzip
    train_labels_path = os.path.join(path, 'train-labels-idx1-ubyte.gz')
    train_images_path = os.path.join(path, 'train-images-idx3-ubyte.gz')
    test_labels_path = os.path.join(path, 't10k-labels-idx1-ubyte.gz')
    test_images_path = os.path.join(path, 't10k-images-idx3-ubyte.gz')

    with gzip.open(train_labels_path, 'rb') as lbpath:
        train_labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)
    with gzip.open(train_images_path, 'rb') as imgpath:
        train_images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(train_labels), 784)
    with gzip.open(test_labels_path, 'rb') as lbpath:
        test_labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)
    with gzip.open(test_images_path, 'rb') as imgpath:
        test_images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(test_labels), 784)

    return train_images, train_labels
# #
# params = {}
# params['dset'] = 'DFS2018'
# #
# load_HSIdata(params)
