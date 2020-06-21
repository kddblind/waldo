import numpy as np
import waldo_mnist_suppl as model
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

args = dict()
args['dim_h'] = 40         # factor controlling size of hidden layers
args['n_channel'] = 1      # number of channels in the input data (MNIST = 1, greyscale)
args['n_z'] = 20           # number of dimensions in latent space.
args['sigma'] = 1.0        # variance in n_z
args['lr'] = 0.0001        # learning rate for Adam optimizer
args['epochs'] = 401       # how many epochs to run for
args['batch_size'] = 2048   # batch size for SGD
args['save'] = True       # save weights at each epoch of training if True
args['train'] = True       # train networks if True, else load networks from saved weights

args['scheduler'] = True   # use scheduler for learning rate or not
args['inClass'] = 0        # set which is the class of the inliers

args['subspace_reg']=0.0   # weight for subspace regularization, 0 if not applied
args['wae']=0.5           # WAE-GAN discriminator lambda, 0 if not WAE
args['lipschitz']=0        # Lipschitz penalty weight, 0 if not applied
args['advantage']=True     # Advantage penalty

args['outratio'] = 0.3     # outlier ration in unlabeled dataset
args['seed'] = 2           # seed for randomness

args['step1set'] = 'x'
args['step2set'] = 'x'
args['paramscloning'] = False


all_data = pd.DataFrame()

combinations = [ args.copy() for i in range(2)]

combinations[0]['subspace_reg']=0.1
combinations[0]['wae']=0
combinations[0]['advantage']=False
combinations[0]['model'] = 'CORA'

combinations[1]['subspace_reg']=0.1
combinations[1]['wae']=0
combinations[1]['advantage']=True
combinations[1]['model'] = 'CORA'

all_data = pd.DataFrame()

for outratio in [0.3]:
    for setting in combinations:
        for seed in range(13,21):
            setting['outratio'] = outratio
            setting['seed'] = seed
            print(setting)
            epochs_data = model.run(setting)
            all_data = all_data.append(epochs_data, ignore_index=True)

all_data.to_csv('convergence_experiment_CORA.csv')
