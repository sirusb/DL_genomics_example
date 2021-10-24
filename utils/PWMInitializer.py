from utils.meme import parseMeme
import numpy as np
import random
import os
from copy import deepcopy
from tensorflow.keras.initializers import Initializer, truncated_normal
import tensorflow as tf
from scipy.stats import truncnorm




def _truncated_normal(mean,
                      stddev,
                      seed=None,
                      normalize=True,
                      alpha=0.01):
    ''' Add noise with truncnorm from numpy.
    Bounded (0.001,0.999)
    '''
    # within range ()
    # provide entry to chose which adding noise way to use
    if seed is not None:
        np.random.seed(seed)
    if stddev == 0:
        X = mean
    else:
        gen_X = truncnorm((alpha - mean) / stddev,
                          ((1 - alpha) - mean) / stddev,
                          loc=mean, scale=stddev)
        X = gen_X.rvs() + mean
        if normalize:
            # Normalize, column sum to 1
            col_sums = X.sum(1)
            X = X / col_sums[:, np.newaxis]
    return X

class PWMInitializer(Initializer):

    def __init__(self, pwm_file, max_motifs=1000, seed=None, std =0.02):

        # check that the file exist 
        if not os.path.exists(pwm_file):
            raise ValueError("Couldn't find file: %s" % pwm_file)            

        # read the list of pwms
        self.pwm_list = parseMeme(pwm_file, max_motifs=max_motifs)
        self.std = std

        if seed is None:
            seed = np.random.choice(100000)

        self.seed = seed
    
    def __call__(self, shape,dtype=None):

        
        filter_size = shape[0]
        
        nb_filters = shape[2]        
        
        newpwm = random.choices(self.pwm_list, k=nb_filters)        
        # reshape the pwms to the same size 
        for pwm in newpwm:
            pwm.reshape(filter_size)
        
        newpwm = [x.pwm for x in newpwm]

        pwms = np.stack(newpwm, axis=-1)

        pwm = _truncated_normal(mean=pwms,
                                    stddev=self.std,
                                    seed=self.seed)

        b = np.array([0.25,0.25,0.25,0.25])
        b = b.reshape([1, 4, 1])
        m =  np.log(pwm / b).astype(pwm.dtype)

        pwm = tf.convert_to_tensor(m, dtype=dtype)
        
        return pwm






        
