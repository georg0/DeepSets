import pdb
import h5py
import numpy as np
from tqdm import tqdm, trange

# TODO: clean up and investigating if the model actually learns anything...

class DataIterator(object):
    def __init__(self, fname, batch_size, shuffle=False):

        self.fname = fname
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Load data from Matlab
        with h5py.File(fname) as f:
            #pdb.set_trace()
            self.n_s = np.squeeze(f['n'][()].astype('int32')) #is not shuffled
            self.L = len(self.n_s)
            self.sum_n = sum(self.n_s)
            self.y = np.reshape(f['q'][()].astype('float32'),[-1,1])

            self.lines = f['lines'][()]
            self.d = self.lines.shape[0]

            self.X = np.zeros([self.sum_n, 1+1, self.d], 'float32') # +1dimension for the sample counter
            self.sample_indices = np.array(range(self.L)) #is shuffled
            data_index = 0
            for i, n in enumerate(self.n_s):
                self.X[data_index:(data_index + n), 0, :] = np.transpose(np.array(
                    self.lines[:, data_index : (data_index + n)].astype('float32')))
                self.X[data_index:(data_index + n), 1, :] = i
                data_index = data_index + n
        
        assert self.L >= self.batch_size, \
            'Batch size larger than number of training examples'
            
    def __len__(self):
        return self.L//self.batch_size

    def get_iterator(self, loss=0.0):
        if self.shuffle:
            rng_state = np.random.get_state()
            #np.random.shuffle(self.X) 
            np.random.shuffle(self.sample_indices)
            data_index = 0
            for i, n in enumerate(self.n_s):
                self.X[data_index:(data_index + n), 1, :] = np.where(self.sample_indices == i)
                data_index = data_index + n
            self.X = self.X[np.argsort(self.X[:,1,0])]

            np.random.set_state(rng_state)
            np.random.shuffle(self.y)
        return tqdm(self.next_batch(),
                    desc='Train loss: {:.4f}'.format(loss),
                    total=len(self), mininterval=1.0, ncols=80)
                    
    def next_batch(self):
        start = 0
        start_i = 0
        end = self.batch_size
        end_i = sum(self.n_s[self.sample_indices[start:end]])
        while end <= self.L:
            yield self.X[start_i:end_i, :, :], self.y[start:end]
            start = end
            start_i = end_i
            end += self.batch_size
            end_i += sum(self.n_s[self.sample_indices[start:end]])
