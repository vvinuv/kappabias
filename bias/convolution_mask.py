import numpy as np
import numpy.ma as ma
import pylab as pl
from numpy import random
from scipy.signal import convolve2d, fftconvolve
import scipy.ndimage as nd

def Gaussian(sigma, size=11, ndim=2):
    """size should always be odd number"""
    if size % 2 == 0:
        raise ValueError('size is not odd')
    if ndim == 2:
        y, x = np.indices((size,size))
        r = np.sqrt((x - size/2)**2. + (y - size/2)**2.)
    if ndim == 3:
        z, y, x = np.indices((size, size, size))
        r = np.sqrt((x - size/2)**2. + (y - size/2)**2. + (z-size/2)**2.)
    g = np.exp(-r**2. / (2 * sigma**2.))
    #print r[5,5] #This corresponds to r=0, i.e the center
    g /= g.sum()
    return g 

def convolve_mask_fft(input, mask, kernal, ignore=0.50):
    """Convolve masked array. ignore=0.5 means that if the total weight of 
       the weight in the unmasked region is less than 50% then that will
       set to zero """

    input = input.astype(np.float32)
    mask = mask.astype(np.float32)
    kernal = kernal.astype(np.float32)

    input *= mask

    kernal /= kernal.sum()
    c = fftconvolve(input, kernal, mode='same')
    cm = fftconvolve(mask, kernal, mode='same')
    m_cm = cm.copy()
    cm[cm == 0] = 1
    
    c /= cm

    c[m_cm < ignore] = 0.0
  
    return c


def convolve_mask(input, mask, kernal, ignore=0.50):
    """Convolve masked array. ignore=0.5 means that if the total weight of 
       the weight in the unmasked region is less than 50% then that will
       set to zero """

    input = input.astype(np.float32)
    mask = mask.astype(np.float32)
    kernal = kernal.astype(np.float32)

    input *= mask

    kernal /= kernal.sum()
    c = nd.convolve(input, kernal, mode='constant', cval=0.0)
    cm = nd.convolve(mask, kernal, mode='constant', cval=0.0)
    c /= cm

    c[cm < ignore] = 0.0
  
    return c

def convolve_mask_slower(input, mask, kernal, ignore=0.50):
    """mask=1 for good elements and 0 for bad. ignore=0.5 means that
       the number of valid pixels in the kernal is less than 50% then 
       that will set to zero. Notice it is different from convolve_mask"""
    s = np.array(kernal.shape) / 2
    output = np.zeros(input.shape)
    padded_input = np.pad(input, ((s[0],s[1]), (s[0],s[1])), 'constant', constant_values=0)
    padded_mask = np.pad(mask, ((s[0],s[1]), (s[0],s[1])), 'constant', constant_values=0)
    ker_size = kernal.shape[0] * 1.0
    for i in range(s[0], padded_input.shape[0] - s[0]):
        for j in range(s[1], padded_input.shape[1] - s[1]):
            tpad_mask = padded_mask[i-s[0]:i+s[0]+1, j-s[1]:j+s[1]+1]
            mask_n = tpad_mask[tpad_mask==1].shape[0]

            weight = (tpad_mask * kernal).sum()
            if weight == 0 or mask_n/ker_size < ignore:
                output[i-s[0], j-s[1]]  = 0.0
            else:
                output[i-s[0], j-s[1]]  = (padded_input[i-s[0]:i+s[0]+1, j-s[1]:j+s[1]+1] * tpad_mask * kernal).sum() / weight

    return output

def convolve_mask_slow(input, mask, kernal, ignore=0.50):
    """mask=1 for good elements and 0 for bad"""

    output = convolve2d(input*mask, kernal, mode='same', boundary='fill', fillvalue=0)

    s = np.array(kernal.shape) / 2
    weight_array = np.zeros(input.shape)
    padded_input = np.pad(input, ((s[0],s[1]), (s[0],s[1])), 'constant', constant_values=0)
    padded_mask = np.pad(mask, ((s[0],s[1]), (s[0],s[1])), 'constant', constant_values=0)
    ker_size = kernal.shape[0] * 1.0
    for i in range(s[0], padded_input.shape[0] - s[0]):
        for j in range(s[1], padded_input.shape[1] - s[1]):
            tpad_mask = padded_mask[i-s[0]:i+s[0]+1, j-s[1]:j+s[1]+1]
            mask_n = tpad_mask[tpad_mask==1].shape[0]

            weight = (tpad_mask * kernal).sum()
            if weight == 0 or mask_n/ker_size < ignore:
                weight_array[i-s[0], j-s[1]]  = 0.0
            else:
                weight_array[i-s[0], j-s[1]]  = 1 / weight

    print output
    print weight
    output *= weight_array
    return output


if __name__=='__main__':

    image = np.ones(25).reshape(5,5)

    r = random.randint(0, image.ravel().shape[0], 10)

    mask = np.ones(25)
    #mask[r] = 0
    mask = mask.reshape(5,5)

    print mask
    g = Gaussian(1, size=3)

    con_image = convolve_mask(image, mask, g)
    print con_image
    #print g
    #pl.imshow(g)
    #pl.show()


