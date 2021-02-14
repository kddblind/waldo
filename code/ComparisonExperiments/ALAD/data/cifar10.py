# CIFAR10 Downloader

import logging
import pickle
import math
import os
import errno
import tarfile
import shutil
import numpy as np
import urllib3
from sklearn.model_selection import train_test_split
from utils.adapt_data import adapt_labels_outlier_task
from collections import Counter
seed =11
np.random.seed(seed)
#th.manual_seed(seed)

logger = logging.getLogger(__name__)

def get_train(label=-1, centered=True, normalize=True, ratio_outliers=0.5):
    return _get_adapted_dataset("train", label, centered, normalize, ratio_outliers=0.5)

def get_test(label=-1, centered=True, normalize=True, ratio_outliers=0.5):
    return _get_adapted_dataset("test", label, centered, normalize, ratio_outliers=0.5)

def get_valid(label=-1, centered=True, normalize=True, ratio_outliers=0.5):
    return _get_adapted_dataset("valid", label, centered, normalize, ratio_outliers=0.5)

def get_shape_input():
    return (None, 32, 32, 3)

def get_shape_input_flatten():
    return (None, 32*32*3)

def get_shape_label():
    return (None,)

def num_classes():
    return 10

def get_anomalous_proportion():
    return 0.9

def _unpickle_file(filename):
    logger.debug("Loading pickle file: {}".format(filename))

    with open(filename, mode='rb') as file:
        data = pickle.load(file, encoding='bytes')

    # Reorder the data
    img = data[b'data']
    img = img.reshape([-1, 3, 32, 32])
    img = img.transpose([0, 2, 3, 1])
    # Load labels
    lbl = np.array(data[b'labels'])

    return img, lbl

def _get_dataset(split, centered=False, normalize=False):
    '''
    Gets the adapted dataset for the experiments
    Args :
            split (str): train or test
            normalize (bool): (Default=True) normalize data
            centered (bool): (Default=False) data centered to [-1, 1]
    Returns :
            (tuple): <training, testing> images and labels
    '''
    path = "data"
    dirname = "cifar-10-batches-py"
    data_url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

    if not os.path.exists(os.path.join(path, dirname)):
        # Extract or download data
        try:
            os.makedirs(path)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

        file_path = os.path.join(path, data_url.split('/')[-1])
        if not os.path.exists(file_path):
            # Download
            logger.warn("Downloading {}".format(data_url))
            with urllib3.PoolManager().request('GET', data_url, preload_content=False) as r, \
                 open(file_path, 'wb') as w:
                    shutil.copyfileobj(r, w)

        logger.warn("Unpacking {}".format(file_path))
        # Unpack data
        tarfile.open(name=file_path, mode="r:gz").extractall(path)

    # Import the data
    if split == 'test':
        filenames = ["test_batch"]
    # elif split == 'valid':
    #     filenames = ["data_batch_5"]
    else:
        filenames = ["data_batch_{}".format(i) for i in range(1, 6)]

    imgs = []
    lbls = []
    for f in filenames:
        img, lbl = _unpickle_file(os.path.join(path, dirname, f))
        imgs.append(img)
        lbls.append(lbl)

    # Now we flatten the arrays
    imgs = np.concatenate(imgs)
    lbls = np.concatenate(lbls)

    # Convert images to [0..1] range
    if normalize:
        imgs = imgs.astype(np.float32)/255.0
    if centered:
        imgs = imgs.astype(np.float32)*2. - 1.
    return imgs.astype(np.float32), lbls

def _get_adapted_dataset(split, label=None, centered=False, normalize=False, ratio_outliers=0.5):
    """
    Gets the adapted dataset for the experiments
    Args :
            split (str): train or test
            mode (str): inlier or outlier
            label (int): int in range 0 to 10, is the class/digit
                         which is considered inlier or outlier
            rho (float): proportion of anomalous classes INLIER
                         MODE ONLY
            centered (bool): (Default=False) data centered to [-1, 1]
    Returns :
            (tuple): <training, testing> images and labels
    """

    dataset = {}

    data={}
    data['x_train'],  data['y_train'] = _get_dataset('train',
                                                           centered=centered,
                                                           normalize=normalize)
    data['x_test'], data['y_test'] = _get_dataset('test', centered=centered,
                                                           normalize=normalize)
    print("data['y_test']=", np.unique(data['y_test']))

    full_x_data =data['x_train']
    full_y_data =data['y_train']
    plabel=label
    normal_x_data = full_x_data[full_y_data== plabel]
    normal_y_data = full_y_data[full_y_data== plabel]

    maxInlierRatio = 1

    test_x_data= data['x_test']
    test_y_data = data['y_test']
    testIN_x= test_x_data[test_y_data==plabel]
    testIN_y= test_y_data[test_y_data==plabel]
    testOUT_x = test_x_data[test_y_data!=plabel]
    testOUT_y = test_y_data[test_y_data!=plabel]

    unlabeledSize = int(testIN_x.shape[0]*(1.0/maxInlierRatio))
    outSize = int(unlabeledSize * ratio_outliers)
    inSize = int(unlabeledSize * (1-ratio_outliers))
    unlabeledSize = outSize + inSize
    unlabeledOUT_x = testOUT_x[:outSize]
    unlabeledOUT_y = testOUT_y[:outSize]
    unlabeledIN_x = testIN_x[:inSize]
    unlabeledIN_y = testIN_y[:inSize]

    print("insize=",inSize,"outSize=",outSize)
    testing_x_data =np.concatenate([unlabeledIN_x,unlabeledOUT_x],axis=0)
    testing_y_data =np.concatenate([unlabeledIN_y,unlabeledOUT_y],axis=0)

    #Shuffle the unlabeled data, otherwise in the training it will see first all the inliers, then the outliers

    shuffle_idx = np.arange(unlabeledSize)
    np.random.shuffle(shuffle_idx)
    testing_x_data = testing_x_data[shuffle_idx]
    testing_y_data = testing_y_data[shuffle_idx]

    training_x_data = np.concatenate([normal_x_data, testing_x_data], axis=0)
    training_y_data = np.concatenate([normal_y_data, testing_y_data], axis=0)


    key_img = 'x_' + split
    key_lbl = 'y_' + split

    if split == 'train' or split == 'valid':

        dataset[key_img] = training_x_data.astype(np.float32)
        dataset[key_lbl] = training_y_data.astype(np.float32)

    elif split == 'test':

        dataset[key_img] = testing_x_data.astype(np.float32)
        dataset[key_lbl] = testing_y_data.astype(np.float32)
        print(" test...dataset[key_lbl]=",set(dataset[key_lbl]), len(set(dataset[key_lbl])),set(dataset[key_lbl].astype(int)))
        print("keycount=", Counter(dataset[key_lbl]).keys()) # equals to list(set(words))
        print("valuecount=",Counter(dataset[key_lbl]).values())           # seed for randomness

    if centered:
        dataset[key_img] = dataset[key_img].astype(np.float32)
        dataset[key_img] = dataset[key_img] * 2. - 1.

    dataset[key_lbl] = adapt_labels_outlier_task(dataset[key_lbl], label)

    return (dataset[key_img], dataset[key_lbl])
