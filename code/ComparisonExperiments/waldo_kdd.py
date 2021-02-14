#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import torch as th
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
from matplotlib import pyplot as plt
from torchvision import datasets
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score, average_precision_score,auc,roc_curve, precision_recall_fscore_support, auc, precision_recall_curve, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import resplot
import pandas as pd


# ## Set parameters and define helper functions

# In[2]:


DEVICE = th.device("cuda:0" if th.cuda.is_available() else "cpu")

#USEFUL FUNCTIONS

# methods to freeze/free parameters for optimization
def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True

def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False

def min_max_normalization(tensor, min_value, max_value):
    min_tensor = tensor.min()
    tensor = (tensor - min_tensor)
    max_tensor = tensor.max()
    tensor = tensor / max_tensor
    tensor = tensor * (max_value - min_value) + min_value
    return tensor


# ## Neural Networks for Encoder, Decoders, Discriminator

# In[3]:


## create encoder model and decoder model kdd
class WAE_Encoder(nn.Module):
    def __init__(self, args):
        super(WAE_Encoder, self).__init__()

        self.n_channel = args['n_channel']
        self.dim_h = args['dim_h']
        self.n_z = args['n_z']

        self.fc = nn.Sequential(
                nn.Linear(121,64),
               nn.LeakyReLU(),
                nn.Linear(64,self.n_z),
               nn.ReLU())

    def forward(self, x):
        x = self.fc(x)
        return x



class WAE_Decoder(nn.Module):
    def __init__(self, args):
        super(WAE_Decoder, self).__init__()

        self.n_channel = args['n_channel']
        self.dim_h = args['dim_h']
        self.n_z = args['n_z']

        self.fc = nn.Sequential(
            nn.Linear(self.n_z,64),
            nn.LeakyReLU(),
            nn.Linear(64,121),
            nn.Tanh())

    def forward(self, x):
        xhat = self.fc(x)
        return xhat


# define the descriminator
class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()

        self.dim_h = args['dim_h']
        self.n_z = args['n_z']

        # main body of discriminator, returns [0,1]
        self.main =  nn.Sequential(
            nn.Linear(self.n_z, 32),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.main(x)
        return x

def _get_dataset(scale,args):
    """ Gets the basic dataset
    Returns :
            dataset (dict): containing the data
                dataset['x_train'] (np.array): training images shape
                (?, 120)
                dataset['y_train'] (np.array): training labels shape
                (?,)
                dataset['x_test'] (np.array): testing images shape
                (?, 120)
                dataset['y_test'] (np.array): testing labels shape
                (?,)
    """
    col_names = _col_names()
    df = pd.read_csv("./kddcup.data_10_percent_corrected", header=None, names=col_names)
    text_l = ['protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_host_login', 'is_guest_login']

    for name in text_l:
        _encode_text_dummy(df, name)

    labels = df['label'].copy()
    labels[labels != 'normal.'] = 0
    labels[labels == 'normal.'] = 1

    df['label'] = labels

    df = df.sample(frac=0.1, random_state=args['seed']) #<------THIS IS THE ONLY ROW ADDED to reduce dataset

    df_train = df.sample(frac=0.5, random_state=args['seed'])
    df_test = df.loc[~df.index.isin(df_train.index)]
    dataset = {}
    full_x_data, full_y_data= _to_xy(df_train, target='label')
    x_train = full_x_data[full_y_data!=1]
    y_train = full_y_data[full_y_data!=1]


    maxInlierRatio = 0.95
    maxOutlierRatio = 0.5

    test_x_data, test_y_data= _to_xy(df_test, target='label')
    testIN_x= test_x_data[test_y_data!=1]
    testIN_y= test_y_data[test_y_data!=1]
    testOUT_x = test_x_data[test_y_data==1]
    testOUT_y = test_y_data[test_y_data==1]

    unlabeledSize = min( int(testIN_x.shape[0]*(1.0/maxInlierRatio)), int(testOUT_x.shape[0]*(1.0/maxOutlierRatio)) )
    outSize = int(unlabeledSize * args['outratio'])
    inSize = int(unlabeledSize * (1-args['outratio']))
    unlabeledSize = outSize + inSize
    unlabeledOUT_x = testOUT_x[:outSize]
    unlabeledOUT_y = testOUT_y[:outSize]
    unlabeledIN_x = testIN_x[:inSize]

    unlabeledIN_y = testIN_y[:inSize]


    x_test =np.concatenate([unlabeledIN_x,unlabeledOUT_x])
    y_test =np.concatenate([unlabeledIN_y,unlabeledOUT_y])



    if scale:
        scaler = MinMaxScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

    shuffle_idx = np.arange(unlabeledSize)
    np.random.shuffle(shuffle_idx)
    x_test = x_test[shuffle_idx]
    y_test = y_test[shuffle_idx]

    dataset = {}
    dataset['x_train'] = x_train.astype(np.float32)[:unlabeledSize]
    dataset['y_train'] = y_train.astype(np.float32)[:unlabeledSize]

    dataset['x_test'] = x_test.astype(np.float32)
    dataset['y_test'] = y_test.astype(np.float32)


    return dataset



def _encode_text_dummy(df, name):
    """Encode text values to dummy variables(i.e. [1,0,0],[0,1,0],[0,0,1]
    for red,green,blue)
    """
    dummies = pd.get_dummies(df.loc[:,name])
    for x in dummies.columns:
        dummy_name = "{}-{}".format(name, x)
        df.loc[:, dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)

def _to_xy(df, target):
    """Converts a Pandas dataframe to the x,y inputs that TensorFlow needs"""
    result = []
    for x in df.columns:
        if x != target:
            result.append(x)
    dummies = df[target]
    return df.as_matrix(result).astype(np.float32), dummies.as_matrix().flatten().astype(int)

def _col_names():
    """Column names of the dataframe"""
    return ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]




def run(args):
    seed = args['seed']
    np.random.seed(seed)
    th.manual_seed(seed)

    zero = th.tensor(0, dtype=th.float).cuda().to(DEVICE)
    one = th.tensor(1, dtype=th.float).cuda().to(DEVICE)

    # ## Prepare dataset

    dt=_get_dataset(True,args)

    positive_loader = DataLoader(
        dataset=th.FloatTensor(dt['x_train']).cuda().to(DEVICE),
        batch_size=args['batch_size'],
        drop_last=False,
        shuffle=False
    )

    unlabeled_loader = DataLoader(
        dataset=th.FloatTensor(dt['x_test']).cuda().to(DEVICE),
        batch_size=args['batch_size'],
        drop_last=False,
        shuffle=False
    )

    # ## Initialize models and optimizers

    # In[5]:


    error = nn.MSELoss(reduction='none')
    criterion = nn.MSELoss()

    encoder = WAE_Encoder(args).cuda().to(DEVICE)
    decoderI = WAE_Decoder(args).cuda().to(DEVICE)
    decoderO = WAE_Decoder(args).cuda().to(DEVICE)
    discriminator = Discriminator(args).cuda().to(DEVICE)

    if args['paramscloning']:
        decoderI.load_state_dict(decoderO.state_dict())

    enc_optim = th.optim.Adam(encoder.parameters(), lr = args['lr'])
    decI_optim = th.optim.Adam(decoderI.parameters(), lr = args['lr'])
    decO_optim = th.optim.Adam(decoderO.parameters(), lr = args['lr'])
    dis_optim = th.optim.Adam(discriminator.parameters(), lr = args['lr'])

    if args['scheduler']:
        enc_scheduler = th.optim.lr_scheduler.StepLR(enc_optim, step_size=30, gamma=0.5)
        decI_scheduler = th.optim.lr_scheduler.StepLR(decI_optim, step_size=30, gamma=0.5)
        decO_scheduler = th.optim.lr_scheduler.StepLR(decO_optim, step_size=30, gamma=0.5)
        dis_scheduler = th.optim.lr_scheduler.StepLR(dis_optim, step_size=30, gamma=0.5)


    # ## Semi-supervised training

    # In[6]:


    values = []
    advantageI = th.tensor([0], dtype=th.float).cuda().to(DEVICE)
    for epoch in range(args['epochs']):
        y_pred = []
        scoreI = []
        for cnt, x in enumerate(zip(positive_loader, unlabeled_loader)):
            #split the dataset int positive p, unlabeled u, both combined x
            p = x[0]
            u = x[1]
            x = th.cat((x[0],x[1]),0)

            #zero the gradients
            encoder.zero_grad()
            decoderI.zero_grad()
            decoderO.zero_grad()
            discriminator.zero_grad()

            if args['wae'] > 0:
                ###################### TRAIN DISCRIMINATOR ######################

                # freeze auto encoder params
                frozen_params(decoderI)
                frozen_params(decoderO)
                frozen_params(encoder)
                # free discriminator params
                free_params(discriminator)

                #set discriminator dataset based on args setting
                if args['step1set'] == 'p':
                    dset1 = p
                elif args['step1set'] == 'x':
                    dset1 = x
                elif args['step1set'] == 'u':
                    dset1 = u

                if args['step2set'] == 'p':
                    dset2 = p
                elif args['step2set'] == 'x':
                    dset2 = x
                elif args['step2set'] == 'u':
                    dset2 = u

                # run discriminator against randn draws
                z = (th.randn(dset1.size()[0], args['n_z']) * args['sigma']).cuda().to(DEVICE)
                d_z = discriminator(z)

                # run discriminator against encoder z's
                z_hat = encoder(dset1)
                d_z_hat = discriminator(z_hat)

                d_z_loss = args['wae']*th.log(d_z).mean()
                d_z_hat_loss = args['wae']*th.log(1 - d_z_hat).mean()

                # formula for ascending the descriminator -- -one reverses the direction of the gradient.
                d_z_loss.backward(-one)
                d_z_hat_loss.backward(-one)

                dis_optim.step()


            ###################### TRAIN AUTOENCODER ######################

            # flip which networks are frozen, which are not
            free_params(decoderI)
            free_params(decoderO)
            free_params(encoder)
            frozen_params(discriminator)


            # encode positive and unlabeled
            z_hatP = encoder(p)
            z_hatU = encoder(u)

            # decode positive with inlier decoder
            x_hatP = decoderI(z_hatP)

            # decode unlabeled with both decoders
            x_hatUO = decoderO(z_hatU)
            x_hatUI = decoderI(z_hatU)

            #resplot.plot_rows(unlabeled,args,encoder,decoders = [decoderI,decoderO], seed = seed)

            lossI = error(x_hatP,p).mean(axis=(1))
            lossUI = error(x_hatUI,u).mean(axis=(1))
            lossUO = error(x_hatUO,u).mean(axis=(1))

            #compute advantage on best reconstructed sample to make training stable.
            if args['advantage']:
                advantageI = lossUO.min() - lossI.min()

            # compute y based on the decoder with lower error.
            y = (lossUO<(lossUI+advantageI)).float()

            # append this prediction to evaluate when the epoch ends

            y_pred.extend(list(y.cpu().numpy()))
            scoreI.extend(list(lossUI.detach().cpu().numpy()))


            # compute competitive AE loss
            loss = th.mean(y*lossUO + (1-y)*lossUI) + th.mean(lossI)

            if(args['subspace_reg']>0):
                z_hat2 = th.cat((z_hatP,z_hatU),0).detach()
                ws = th.exp(-th.pdist(x.reshape(x.shape[0],-1)/args['subspace_reg'],2))
                o_len = (int)(x.shape[0]/4)
                ws[-o_len:]=0
                diff = th.pow(th.pdist(z_hat2,1),2)
                loss += args['subspace_reg'] * th.mean(ws*diff)

            if(args['wae'])>0:
                z_hat = encoder(dset2)
                # discriminate latents
                d_z_hat = discriminator(z_hat)
                # calculate discriminator loss
                d_loss = args['wae'] * (th.log(d_z_hat)).mean()
                # compute gradient to fool discriminator
                d_loss.backward(-one)

            loss.backward(one)

            enc_optim.step()
            decI_optim.step()
            decO_optim.step()


        if epoch % 20 == 0:
            try:
                precision, recall, F1, _ = precision_recall_fscore_support(dt['y_test'],
                                                                   y_pred,
                                                                   average='binary')
            except:
                precision, recall, F1, _ = 0, 0 ,0 ,0

            try:
                AUC = roc_auc_score(dt['y_test'],scoreI)
            except:
                AUC = 0

            try:
                tn, fp, fn, tp = confusion_matrix(dt['y_test'], y_pred).ravel()
            except:
                tn, fp, fn, tp = 0 , 0, 0,0

            try:
                ps,rs,ts = precision_recall_curve(dt['y_test'], scoreI)
                AUPRC = auc(rs, ps)
            except:
                AUPRC = 0

            values.append((epoch,loss.item(),advantageI.item(),precision, recall,F1,AUC,AUPRC,tn, fp, fn, tp))





    # In[7]:


    return(values)
