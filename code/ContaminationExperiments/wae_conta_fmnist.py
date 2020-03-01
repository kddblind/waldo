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

import resplot


# ## Set parameters and define helper functions

# In[2]:


DEVICE = th.device("cuda:1" if th.cuda.is_available() else "cpu")

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


class WAE_Encoder(nn.Module):
    def __init__(self, args):
        super(WAE_Encoder, self).__init__()

        self.n_channel = args['n_channel']
        self.dim_h = args['dim_h']
        self.n_z = args['n_z']

        # convolutional filters, work excellent with image data
        self.conv = nn.Sequential(
            nn.Conv2d(self.n_channel, self.dim_h, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(self.dim_h, self.dim_h * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.ReLU(True),
            nn.Conv2d(self.dim_h * 2, self.dim_h * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 4),
            nn.ReLU(True),
            nn.Conv2d(self.dim_h * 4, self.dim_h * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 8),
            nn.ReLU(True),
        )

        # final layer is fully connected
        self.fc = nn.Linear(self.dim_h * (2 ** 3), self.n_z)
    def forward(self, x):
        x = self.conv(x)
        x = x.squeeze()
        x = self.fc(x)
        return x

class WAE_Decoder(nn.Module):
    def __init__(self, args):
        super(WAE_Decoder, self).__init__()

        self.n_channel = args['n_channel']
        self.dim_h = args['dim_h']
        self.n_z = args['n_z']

        # first layer is fully connected
        self.fc = nn.Sequential(
            nn.Linear(self.n_z, self.dim_h * 8 * 7 * 7),
            nn.ReLU()
        )

        # deconvolutional filters, essentially the inverse of convolutional filters
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(self.dim_h * 8, self.dim_h * 4, 4),
            nn.BatchNorm2d(self.dim_h * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 4, self.dim_h * 2, 4),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 2, 1, 4, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.dim_h * 8, 7, 7)
        x = self.deconv(x)
        return x



# define the descriminator
class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()

        self.dim_h = args['dim_h']
        self.n_z = args['n_z']

        # main body of discriminator, returns [0,1]
        self.main = nn.Sequential(
            nn.Linear(self.n_z, self.dim_h * 4),
            nn.ReLU(True),
            nn.Linear(self.dim_h * 4, self.dim_h * 4),
            nn.ReLU(True),
            nn.Linear(self.dim_h * 4, self.dim_h * 4),
            nn.ReLU(True),
            nn.Linear(self.dim_h * 4, self.dim_h * 4),
            nn.ReLU(True),
            nn.Linear(self.dim_h * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.main(x)
        return x



def run(args):
    seed = args['seed']
    np.random.seed(seed)
    th.manual_seed(seed)

    zero = th.tensor([0], dtype=th.float).cuda().to(DEVICE)
    one = th.tensor([1], dtype=th.float).cuda().to(DEVICE)

    # ## Prepare dataset

    # In[4]:


    #Download MNIST

    img_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = datasets.FashionMNIST(root = 'fmnist', train = True, download = True, transform = img_transform)
    testset = datasets.FashionMNIST(root = 'fmnist', train = False, download = True, transform = img_transform)


    #Get unlabeled dataset from testing
    testIN = min_max_normalization(testset.test_data[np.where(testset.test_labels == args['inClass'])].type('torch.FloatTensor'),0.0,1.0)
    testOUT = min_max_normalization(testset.test_data[np.where(testset.test_labels != args['inClass'])].type('torch.FloatTensor'),0.0,1.0)

    #We want to have the same size of data for all the runs, even when we change the outlier ratio.
    #We consider outlier ratio from 0.1 to 0.5. Thus, inlier ratio from 0.9 to 0.5
    #Given that usually inlier class is only one and outliers all the rest, with 0.1 outlier ratio and 0.9
    #inlier ratio we have the smallest unlabeled dataset, we will use this size for the unlabeled dataset.
    maxInlierRatio = 0.95
    unlabeledSize = int(testIN.shape[0]*(1.0/maxInlierRatio))

    outSize = int(unlabeledSize * args['outratio'])
    inSize = int(unlabeledSize * (1-args['outratio']))
    #Update unlabeledSize given that the above rounding may change the total
    unlabeledSize = outSize + inSize
    unlabeledOUT = testOUT[:outSize]
    unlabeledIN = testIN[:inSize]

    unlabeled = np.concatenate((unlabeledIN,unlabeledOUT))

    unlabeled_labels = np.concatenate((np.zeros(inSize),np.ones(outSize)))


    #reshape for convolutional and convert
    unlabeled = th.FloatTensor(unlabeled.reshape(-1,1,28,28)).cuda().to(DEVICE)

    #Shuffle the unlabeled data, otherwise in the training it will see first all the inliers, then the outliers
    shuffle_idx = np.arange(unlabeledSize)
    np.random.shuffle(shuffle_idx)
    unlabeled = unlabeled[shuffle_idx]
    unlabeled_labels = unlabeled_labels[shuffle_idx]

    #Get positive data from training set
    positive = min_max_normalization(trainset.train_data[np.where(trainset.train_labels == args['inClass'])].type('torch.FloatTensor'),0.0,1.0)
    contaminated = min_max_normalization(trainset.train_data[np.where(trainset.train_labels != args['inClass'])].type('torch.FloatTensor'),0.0,1.0)
    contacount = int(unlabeledSize*args['conta'])
    #Set positive subset to be equal in size to unlabeled subset
    positive = np.array(positive[:unlabeledSize])
    positive = np.concatenate((np.array(positive[:(unlabeledSize-contacount)]), np.array(contaminated[:contacount])))
    #positive = np.array(positive)
    #Reshape for convolutional net and convert
    positive = th.FloatTensor(positive.reshape(-1,1,28,28)).cuda().to(DEVICE)


    positive_loader = DataLoader(
        dataset=positive,
        batch_size=int(args['batch_size']*(positive.shape[0]/unlabeled.shape[0])),
        drop_last=False,
        shuffle=False
    )

    unlabeled_loader = DataLoader(
        dataset=unlabeled,
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
    discriminator = Discriminator(args).cuda().to(DEVICE)

    enc_optim = th.optim.Adam(encoder.parameters(), lr = args['lr'])
    decI_optim = th.optim.Adam(decoderI.parameters(), lr = args['lr'])
    dis_optim = th.optim.Adam(discriminator.parameters(), lr = args['lr'])

    if args['scheduler']:
        enc_scheduler = th.optim.lr_scheduler.StepLR(enc_optim, step_size=30, gamma=0.5)
        decI_scheduler = th.optim.lr_scheduler.StepLR(decI_optim, step_size=30, gamma=0.5)
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
            discriminator.zero_grad()

            if args['wae'] > 0:
                ###################### TRAIN DISCRIMINATOR ######################

                # freeze auto encoder params
                frozen_params(decoderI)
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
            free_params(encoder)
            frozen_params(discriminator)


            # encode positive and unlabeled
            z_hatP = encoder(p)
            z_hatU = encoder(u)

            # decode positive with inlier decoder
            x_hatP = decoderI(z_hatP)

            # decode unlabeled with both decoders
            x_hatUI = decoderI(z_hatU)

            #resplot.plot_rows(unlabeled,args,encoder,decoders = [decoderI,decoderO], seed = seed)

            lossI = error(x_hatP,p).mean(axis=(1,2,3))
            lossUI = error(x_hatUI,u).mean(axis=(1,2,3))

            scoreI.extend(list(lossUI.detach().cpu().numpy()))


            # compute competitive AE loss
            loss = th.mean(lossI)

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


        if epoch % 20 == 0:
            threshold = np.percentile(scoreI, 100 - (args['outratio']*100))
            scoreI = np.array(scoreI)
            y_pred = (scoreI>=threshold) * 1.0
            precision, recall, F1, _ = precision_recall_fscore_support(unlabeled_labels,
                                                                   y_pred,
                                                                   average='binary')
            AUC = roc_auc_score(unlabeled_labels,scoreI)
            tn, fp, fn, tp = confusion_matrix(unlabeled_labels, y_pred).ravel()

            ps,rs,ts = precision_recall_curve(unlabeled_labels, scoreI)
            AUPRC = auc(rs, ps)

            values.append((epoch,loss.item(),advantageI.item(),precision, recall,F1,AUC,AUPRC,tn, fp, fn, tp))





    # In[7]:


    return(values)
