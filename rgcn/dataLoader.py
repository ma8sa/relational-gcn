import numpy as np
import keras
from keras.utils import np_utils
import pickle as pkl


class DataGenerator():
    'Generates data for Keras'
    def __init__(self, list_IDs=None, 
                 n_classes=2, shuffle=True):
        'Initialization'
        #self.dim = dim
        #self.batch_size = batch_size
        #self.labels = labels
        #self.list_IDs = list_IDs
        #self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        print("aassd")

    def __len__(self):
        'Denotes the number of batches per epoch'
        #return int(len(self.list_IDs) )
        return int(3)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        # Find list of IDs
        # Generate data
        print(" YOOOOOOO ")
        X, y = self.__data_generation(index)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(3)
        #self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, ID):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization

        # Generate data
        # Change to reading your data
        # LOAD A from pickle from IDA
        A = pkl.load(open("A" + str(ID).zfill(4) + ".pkl","rb"))
        print(A.todense().shape)
        print(" fuck yeah ")
        input()
        y = pkl.load(open("labels" + str(ID).zfill(4) + ".pkl","rb"))
        y = sp.csr_matrix(np_utils.to_categorical(y, 2))
        
        X = pkl.load(open("X" + str(ID).zfill(4) + ".pkl","rb")).todense()
        X = sp.csr_matrix(np_utils.to_categorical(X, 4))
        
        # LOAD X from ID
        # LOAD Y 
        for i in range(len(A)):
                    d = np.array(A[i].sum(1)).flatten()
                    d_inv = 1. / d
                    d_inv[np.isinf(d_inv)] = 0.
                    D_inv = sp.diags(d_inv)
                    A[i] = D_inv.dot(A[i]).tocsr()


        return [X]+A, y 
