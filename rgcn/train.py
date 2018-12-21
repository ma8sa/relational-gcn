from __future__ import print_function

from keras.layers import Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
#import keras
from keras.utils import np_utils

from rgcn.layers.graph import GraphConvolution
from rgcn.layers.input_adj import InputAdj
from rgcn.utils import *

import pickle as pkl

import os
import sys
import time
import argparse

import numpy as np
import scipy.sparse as sp

np.random.seed()

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default="aifb",
                help="Dataset string ('aifb', 'mutag', 'bgs', 'am')")
ap.add_argument("-e", "--epochs", type=int, default=50,
                help="Number training epochs")
ap.add_argument("-hd", "--hidden", type=int, default=16,
                help="Number hidden units")
ap.add_argument("-do", "--dropout", type=float, default=0.,
                help="Dropout rate")
ap.add_argument("-b", "--bases", type=int, default=-1,
                help="Number of bases used (-1: all)")
ap.add_argument("-lr", "--learnrate", type=float, default=0.01,
                help="Learning rate")
ap.add_argument("-l2", "--l2norm", type=float, default=0.,
                help="L2 normalization of input weights")

fp = ap.add_mutually_exclusive_group(required=False)
fp.add_argument('--validation', dest='validation', action='store_true')
fp.add_argument('--testing', dest='validation', action='store_false')
ap.set_defaults(validation=True)

args = vars(ap.parse_args())
print(args)

# Define parameters
DATASET = args['dataset']
NB_EPOCH = args['epochs']
VALIDATION = args['validation']
LR = args['learnrate']
L2 = args['l2norm']
HIDDEN = args['hidden']
BASES = args['bases']
DO = args['dropout']

dirname = os.path.dirname(os.path.realpath(sys.argv[0]))

with open(dirname + '/' + DATASET + '.pickle', 'rb') as f:
    data = pkl.load(f)

A = data['A']
y = data['y']
# check if this fucler is also predicting the edges( i mean the network ) although there doesnt seem any mention of that kind of oss , which reminds me that i have to look at the loss term again
# make K X NXN matrix( which will act as oour adj matrices ) the convert them to csr_sparse
# y is the labels of form NXC then converted in spares matrix
train_idx = data['train_idx']
test_idx = data['test_idx']
#set train to all  and test as zero
A = [ sp.csr_matrix(np.eye(10)) for _ in range(2) ]

pkl.dump(A,open("save.p", "wb"))
A = pkl.load(open("A0000.pkl","rb"))

B2 = [ sp.csr_matrix(np.eye(10)) for _ in range(2) ]
num_nodes = A[0].shape[0]

y = sp.csr_matrix(np.zeros((num_nodes,2))) 
y = pkl.load(open("labels0000.pkl","rb"))
#y = keras.utils.to_categorical(y, num_classes=2)
y = sp.csr_matrix(np_utils.to_categorical(y, 2))


train_idx = np.arange(num_nodes)
test_idx = train_idx 

# Get dataset splits
# somehow pass this
y_train, y_val, y_test, idx_train, idx_val, idx_test = get_splits(y, train_idx,
                                                                  test_idx,
                                                                  False)
train_mask = sample_mask(idx_train, y.shape[0])
num_nodes = A[0].shape[0]
support = len(A)

# Define empty dummy feature matrix (input is ignored as we set featureless=True)
# define New X and New A,
# can we bypass the splits
# In case features are available, define them here and set featureless=False.
# what is y

# TODO experiment with features
X = sp.csr_matrix(A[0].shape)
print(X.shape[1])
input()
#print(X.todense().shape)
print( y.todense() )
input()
#X = pkl.load(open("X0000.pkl","rb"))

# Normalize adjacency matrices individually
for i in range(len(A)):
    d = np.array(A[i].sum(1)).flatten()
    d_inv = 1. / d
    d_inv[np.isinf(d_inv)] = 0.
    D_inv = sp.diags(d_inv)
    A[i] = D_inv.dot(A[i]).tocsr()

A_in = [InputAdj(sparse=True) for _ in range(support)]
X_in = Input(shape=(X.shape[1],), sparse=True)

# Define model architecture
H = GraphConvolution(HIDDEN, support, num_bases=BASES, featureless=False,
                     activation='relu',
                     W_regularizer=l2(L2))([X_in] + A_in)
H = Dropout(DO)(H)
Y = GraphConvolution(y_train.shape[1], support, num_bases=BASES,
                     activation='softmax')([H] + A_in)

# Compile model

model = Model(input=[X_in] + A_in, output=Y)
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=LR))

preds = None
# defining what do we need
# X : most problaay feature vector , size and shape unknown
# A : list of adj matrix , evry matrix represt edge of one type
# y_train : mask for training i think
# train_mask : IDK
# batch size : 
# idx_train
# y_train
# idx_val
# y_val
# evalutae_preds
# how to give input in keras
# we need to change how the netwrork works as , we have lots graphs to train from , we dont need splits 
# We need to find where is the training hapeening
# maybe we should go through some keras videos

# Fit
for epoch in range(1, NB_EPOCH + 1):

    # Log wall-clock time
    t = time.time()

    # Single training iteration
    model.fit([X] + A, y_train, 
              batch_size=num_nodes, nb_epoch=1, shuffle=False, verbose=0)

    if epoch % 1 == 0:

        # Predict on full dataset
        preds = model.predict([X] + A, batch_size=num_nodes)

        # Train / validation scores
        train_val_loss, train_val_acc = evaluate_preds(preds, [y_train, y_val],
                                                       [idx_train, idx_val])

        print("Epoch: {:04d}".format(epoch),
              "train_loss= {:.4f}".format(train_val_loss[0]),
              "train_acc= {:.4f}".format(train_val_acc[0]),
              "val_loss= {:.4f}".format(train_val_loss[1]),
              "val_acc= {:.4f}".format(train_val_acc[1]),
              "time= {:.4f}".format(time.time() - t))

    else:
        print("Epoch: {:04d}".format(epoch),
              "time= {:.4f}".format(time.time() - t))

# Testing
test_loss, test_acc = evaluate_preds(preds, [y_test], [idx_test])
print("Test set results:",
      "loss= {:.4f}".format(test_loss[0]),
      "accuracy= {:.4f}".format(test_acc[0]))
