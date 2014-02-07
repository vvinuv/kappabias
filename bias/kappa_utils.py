import scipy.ndimage as ndimage
import numpy as np
import pylab
import gc
import os
from scipy import fftpack
from matplotlib.colors import LinearSegmentedColormap
import pyfits

def matrc1():
    MatPlotParams = {'axes.titlesize': 15, 'axes.linewidth' : 2.5, 'axes.labelsize': 16, 'xtick.labelsize': 12, 'ytick.labelsize': 12, 'xtick.major.size': 16, 'ytick.major.size' : 16, 'xtick.minor.size': 12, 'ytick.minor.size': 12, 'xtick.major.pad' : 8, 'ytick.major.pad' : 6, 'figure.subplot.left' : 0.1, 'figure.subplot.right' : 0.95, 'figure.subplot.bottom' : 0.1,'figure.subplot.top' : 0.95, 'figure.figsize' : [8,8]}
    pylab.rcParams.update(MatPlotParams)

def bkrdw():
    BkRdW = LinearSegmentedColormap('BkRdW',
                                {'blue':   [(0.0,  0.0, 0.0),
                                            (0.5,  0.0, 0.0),
                                            (1.0,  1.0, 1.0)],

                                 'green': [(0.0,  0.0, 0.0),
                                           (0.25, 0.0, 0.0),
                                           (0.75, 1.0, 1.0),
                                           (1.0,  1.0, 1.0)],

                                 'red':  [(0.0,  0.0, 0.0),
                                          (0.5,  1.0, 1.0),
                                          (1.0,  1.0, 1.0)]} )
    return BkRdW

@np.vectorize
def KS_kernel(ell):
    if abs(ell)==0:
        return 0.+0.j
    else:
        return ell*1./np.conj(ell)


def gamma_to_kappa(shear,dt1,dt2=None):
    """
    simple application of Kaiser-Squires (1995) kernel in fourier
    space to convert complex shear to complex convergence: imaginary
    part of convergence is B-mode.
    """
    if not dt2:
        dt2 = dt1
    N1,N2 = shear.shape

    #convert angles from arcminutes to radians
    dt1 = dt1 * np.pi / 180. / 60.
    dt2 = dt2 * np.pi / 180. / 60.

    #compute k values corresponding to field size
    dk1 = np.pi / N1 / dt1
    dk2 = np.pi / N2 / dt2

    k1 = fftpack.ifftshift( dk1 * (np.arange(2*N1)-N1) )
    k2 = fftpack.ifftshift( dk2 * (np.arange(2*N2)-N2) )

    ipart,rpart = np.meshgrid(k2,k1)
    k = rpart + 1j*ipart

    #compute (inverse, complex conjugate) Kaiser-Squires kernel on this grid
    fourier_map = np.conj( KS_kernel(-k) ) #Inverse of Eq. 43 in Schieder p329
                                           #use KS_kernel(k) to get Eq. 43
    # Also note: pi in D(l) (Eq. 43) cancel with 1/pi in Eqs. 42. That is the 
    # reason why I dont put pi in the Kernal 
    #compute Fourier transform of the shear
    gamma_fft = fftpack.fft2( shear, (2*N1,2*N2) )

    kappa_fft = fourier_map * gamma_fft

    kappa = fftpack.ifft2(kappa_fft)[:N1,:N2]

    return kappa

def kappa_to_gamma(kappa,dt1,dt2=None):
    """
    simple application of Kaiser-Squires (1995) kernel in fourier
    space to convert complex shear to complex convergence: imaginary
    part of convergence is B-mode.
    """
    if not dt2:
        dt2 = dt1
    N1,N2 = kappa.shape

    #convert angles from arcminutes to radians
    dt1 = dt1 * np.pi / 180. / 60.
    dt2 = dt2 * np.pi / 180. / 60.

    #compute k values corresponding to field size
    dk1 = np.pi / N1 / dt1
    dk2 = np.pi / N2 / dt2

    k1 = fftpack.ifftshift( dk1 * (np.arange(2*N1)-N1) )
    k2 = fftpack.ifftshift( dk2 * (np.arange(2*N2)-N2) )

    ipart,rpart = np.meshgrid(k2,k1)
    k = rpart + 1j*ipart

    #compute Kaiser-Squires kernel on this grid. Eq. 43 p. 329
    fourier_map = np.conj( KS_kernel(k) )

    #compute Fourier transform of the kappa
    kappa_fft = fftpack.fft2( kappa, (2*N1,2*N2) )

    gamma_fft = kappa_fft * fourier_map

    gamma = fftpack.ifft2(gamma_fft)[:N1,:N2]

    return gamma


def pixelize_light(ipath, ifile, pix_scale, skip=0, opath='./', 
                   constrain=0, coord=(0,0), bin_ra=None, bin_dec=None):
    """Pixelize the galaxy light instead of the shear. pix_scale in 
       arcmin. The ifile has the format ra dec mag mage 
       (ra dec in degrees) """

    ofile = os.path.join(opath, 'pixelized_%s.npz'%ifile.split('.')[0])
    ifile = os.path.join(ipath, ifile)

    # load ifile
    f = np.loadtxt(ifile, skiprows=skip)
    ra = f[:,0] # deg
    dec = f[:,1] # deg 
    mag = f[:,2]
    mage = f[:,3]

    con = (abs(mag) < 40)
    ra = ra[con]
    dec = dec[con]
    mag = mag[con]
    mage = mage[con]


    if constrain:
        con = (abs(ra-coord[0]) <= constrain) & (abs(dec-coord[1]) <= constrain)
        ra = ra[con]
        dec = dec[con]
        mag = mag[con]
        mage = mage[con]

    flux = 10**(-0.4 * (mag - 15.))

    ra_min, ra_max = ra.min(), ra.max()
    dec_min, dec_max = dec.min(), dec.max()
    if ra_min < 2 and ra_max > 300:
        raw_input("RA include both 0 and 360. The projection doesnt work")
    ra_avg = (ra_max + ra_min) / 2.
    dec_avg = (dec_max + dec_min) / 2.


    #Taking the dec factor in ra
    tra = ra_avg + (ra - ra_avg) * np.cos(dec * np.pi / 180.0)  # deg
    tdec = dec.copy()
    tra_min, tra_max = tra.min(), tra.max()
    tdec_min, tdec_max = tdec.min(), tdec.max()

    #size of the field 
    field_ra = tra_max - tra_min
    field_dec = tdec_max - tdec_min
    field_dec = tdec_max - tdec_min
    tot_area = field_ra * field_dec
    print 'Field area is %2.4f sq. deg'%(tot_area)
    
    if bin_ra is None:
        bin_ra = int(field_ra * 60 / pix_scale) # number of pixels along ra
        bin_dec = int(field_dec * 60 / pix_scale) # number of pixels along dec

    print '%2.2f source per sq arcmin'%(ra.shape[0] / (tot_area * 3600))

    n, dec_b, ra_b = np.histogram2d(tdec, tra, bins=(bin_dec,bin_ra))
    n *= 1.0
    mask = n.copy()
    mask = ndimage.gaussian_filter(mask, 1.2)
    mask[mask > 0] = 1

    nf, dec_b, ra_b = np.histogram2d(tdec,tra,weights=flux, \
                                     bins=(bin_dec,bin_ra))

    #flux_avg = nf / n
    np.savez(ofile, flux=nf, ra=ra_b, dec = dec_b, number=n)

    return ofile, mask#, ra_min, ra_max, dec_min, dec_max, ra_avg



def pixelize_galcount(ipath, ifile, pix_scale, skip=0, opath='./',
                      constrain=0, coord=(0,0), bin_ra=None, bin_dec=None):
    """Pixelize the galaxy number count instead of the shear. 
       pix_scale in arcmin. The ifile has the format ra dec in degrees """

    ofile = os.path.join(opath, 'pixelized_%s.npz'%ifile.split('.')[0])
    ifile = os.path.join(ipath, ifile)

    # load ifile
    f = np.loadtxt(ifile, skiprows=skip)
    ra = f[:,0] # deg
    dec = f[:,1] # deg 

    if constrain:
        con = (abs(ra-coord[0]) <= constrain) & (abs(dec-coord[1]) <= constrain)
        ra = ra[con]
        dec = dec[con]
 
    ra_min, ra_max = ra.min(), ra.max()
    dec_min, dec_max = dec.min(), dec.max()
    if ra.ptp() > 50:
        raw_input("It seems RA include both 0 and 360 or some other ", \
                  "problem, if the input RA width is less than 50 deg. ", \
                  "The projection doesnt work")
    ra_avg = (ra_max + ra_min) / 2.
    dec_avg = (dec_max + dec_min) / 2.

    #print 'Avg ', ra_avg, dec_avg

    #Taking the dec factor in ra
    tra = ra_avg + (ra - ra_avg) * np.cos(dec * np.pi / 180.0)  # deg
    tdec = dec.copy()
    tra_min, tra_max = tra.min(), tra.max()
    tdec_min, tdec_max = tdec.min(), tdec.max()

    #size of the field 
    field_ra = tra_max - tra_min
    field_dec = tdec_max - tdec_min
    tot_area = field_ra * field_dec
    print 'Field area is %2.4f sq. deg'%(tot_area)

    if bin_ra is None:
        bin_ra = int(field_ra * 60 / pix_scale) # number of pixels along ra
        bin_dec = int(field_dec * 60 / pix_scale) # number of pixels along dec

    
    print '%2.2f source per sq arcmin'%(ra.shape[0] / (tot_area * 3600))

    n, dec_b, ra_b = np.histogram2d(tdec, tra, bins=(bin_dec,bin_ra))
    n *= 1.0
    mask = n.copy()
    mask = ndimage.gaussian_filter(mask, 1.2)
    mask[mask > 0] = 1

    #print ra_max - ra_min, field_ra, field_dec, ra_b.shape, dec_b.shape
    #raw_input()
    
    np.savez(ofile, number=n, ra=ra_b, dec = dec_b)

    return ofile, mask#, ra_min, ra_max, dec_min, dec_max, ra_avg

def pixelize_shear_CFHT(ipath, ifile, pix_scale, zmin=0.4, zmax=1.4,
                        skip=0, opath='./', ofile=None,
                        rotate=False, constrain=0, coord=(0,0), 
                        bin_ra=None, bin_dec=None, 
                    col_names=['RA', 'DEC', 'z', 'E1', 'E2', 'W', 'SN', 'Re']):
    """Pixelize the galaxy shear. pix_scale in arcmin. The ifile has 
       the format ra dec (degrees) E1 E2. Rotate=True will rotate the shear
       by 45 deg. It is useful for testing B-mode. col_names tells the column 
       names in the fits file """
  
    if ofile is None:
        ofile = os.path.join(opath, 'pixelized_%s.npz'%ifile.split('.')[0])
    else:
        ofile = os.path.join(opath, ofile)
    ifile = os.path.join(ipath, ifile)

    # load ifile
    hdu = pyfits.open(ifile)
    ra = hdu[1].data.field(col_names[0])
    dec = hdu[1].data.field(col_names[1])
    z = hdu[1].data.field(col_names[2])
    E1 = hdu[1].data.field(col_names[3])
    E2 = hdu[1].data.field(col_names[4])
    try:
        W = hdu[1].data.field(col_names[5])
        SN = hdu[1].data.field(col_names[6])
        Re = hdu[1].data.field(col_names[7])
    except:
        W = np.ones(ra.shape)
        SN = np.zeros(ra.shape)
        Re = np.zeros(ra.shape)
     
    #print 'NRa ', ra.shape, ra.min(), ra.max()
    
    con = (z >= zmin) & (z <= zmax)

    ra = ra[con]
    dec = dec[con]
    E1 = E1[con]
    E2 = E2[con]
    W = W[con]
    z = z[con]
    SN = SN[con]
    Re = Re[con] * 0.187 #pixel to arcsec which is needed for additive correction
    #print 'nRa ', ra.shape, ra.min(), ra.max()

    if rotate:
        E11 = E1 * np.cos(90. * np.pi / 180.) - E2 * np.sin(90. * np.pi / 180.)
        E22 = E1 * np.sin(90. * np.pi / 180.) + E2 * np.cos(90. * np.pi / 180.)
        E1 = E11.copy()
        E2 = E22.copy()

    if constrain:
        con = (abs(ra-coord[0]) <= constrain) & (abs(dec-coord[1]) <= constrain)
        ra = ra[con]
        dec = dec[con]
        E1 = E1[con]
        E2 = E2[con]

    ra_min, ra_max = ra.min(), ra.max()
    dec_min, dec_max = dec.min(), dec.max()
    if ra_min < 2 and ra_max > 300:
        raw_input("RA include both 0 and 360. The projection doesnt work")
    ra_avg = (ra_max + ra_min) / 2.
    dec_avg = (dec_max + dec_min) / 2.
   
    #print 'Avg ', ra_avg, dec_avg

    #Taking the dec factor in ra
    tra = ra_avg + (ra - ra_avg) * np.cos(dec * np.pi / 180.0)  # deg
    tdec = dec.copy()
    tra_min, tra_max = tra.min(), tra.max()
    tdec_min, tdec_max = tdec.min(), tdec.max()

    #size of the field 
    field_ra = tra_max - tra_min
    field_dec = tdec_max - tdec_min
    tot_area = field_ra * field_dec
    print 'Field area is %2.4f sq. deg'%(tot_area)

    if bin_ra is None:
        bin_ra = int(field_ra * 60 / pix_scale) # number of pixels along ra
        bin_dec = int(field_dec * 60 / pix_scale) # number of pixels along dec

    print '%2.2f source per sq arcmin'%(ra.shape[0] / (tot_area * 3600))

    #print bin_ra
    #print bin_ra.shape
    #print 'Pix scale', pix_scale
    if np.all(W == 1):
        m = 0
    else:
        #Additive correction to E2. Eq. 19 Heymans 2012
        F, G, H, r0 = 11.910, 12.715, 2.458, 0.01

        c2 = np.maximum((F * np.log10(SN) - G) / (1 + (Re / r0)**H), 0.0)

        E2 = E2 - c2

        #Multiplicative bias Eq. 17 Heymans 2012
        m = (-0.37 / np.log(SN)) * np.exp(-0.057 * Re * SN)

    Ngal, dec_b, ra_b = np.histogram2d(tdec, tra, bins=(bin_dec,bin_ra))
    print 'size ', Ngal.shape
    Ngal *= 1.0
    mask = Ngal.copy()
    #mask = ndimage.gaussian_filter(mask, 1.2)
    mask[mask > 0] = 1

    Ngal[Ngal == 0] = 1.0

    #Weighted sum of E1 and E2
    NE1, dec_b, ra_b = np.histogram2d(tdec,tra,weights=E1*W,bins=(bin_dec,bin_ra))
    NE2, dec_b, ra_b = np.histogram2d(tdec,tra,weights=E2*W,bins=(bin_dec,bin_ra))

    #Weighted sum of multiplicative bias
    Nm, dec_b, ra_b = np.histogram2d(tdec, tra, weights=(1+m)*W,
                                       bins=(dec_b, ra_b))
    Nm[Nm == 0] = 1.
    #Weighted average of shape 
    epsilon = (NE1 + 1j * NE2) 

    #Correction for multiplicative bias
    #epsilon  = epsilon 

    np.savez(ofile, epsilon=epsilon*mask, ra=ra_b, dec = dec_b, number=Nm, mask=mask)

    return ofile, mask#, ra_min, ra_max, dec_min, dec_max, ra_avg



def pixelize_shear(ipath, ifile, pix_scale, skip=0, opath='./', rotate=False,
                   constrain=0, coord=(0,0), bin_ra=None, bin_dec=None):
    """Pixelize the galaxy shear. pix_scale in arcmin. The ifile has 
       the format ra dec (degrees) g1 g2. Rotate=True will rotate the shear
       by 45 deg. It is useful for testing B-mode """

    ofile = os.path.join(opath, 'pixelized_%s.npz'%ifile.split('.')[0])
    ifile = os.path.join(ipath, ifile)

    # load ifile
    f = np.loadtxt(ifile, skiprows=skip)
    ra = f[:,0] # deg
    dec = f[:,1] # deg 
    g1 = f[:,2]
    g2 = f[:,3]

    if rotate:
        g11 = g1 * np.cos(90. * np.pi / 180.) - g2 * np.sin(90. * np.pi / 180.)
        g22 = g1 * np.sin(90. * np.pi / 180.) + g2 * np.cos(90. * np.pi / 180.)
        g1 = g11.copy()
        g2 = g22.copy()

    if constrain:
        con = (abs(ra-coord[0]) <= constrain) & (abs(dec-coord[1]) <= constrain)
        ra = ra[con]
        dec = dec[con]
        g1 = g1[con]
        g2 = g2[con]

    ra_min, ra_max = ra.min(), ra.max()
    dec_min, dec_max = dec.min(), dec.max()
    if ra_min < 2 and ra_max > 300:
        raw_input("RA include both 0 and 360. The projection doesnt work")
    ra_avg = (ra_max + ra_min) / 2.
    dec_avg = (dec_max + dec_min) / 2.
   
    #print 'Avg ', ra_avg, dec_avg

    #Taking the dec factor in ra
    tra = ra_avg + (ra - ra_avg) * np.cos(dec * np.pi / 180.0)  # deg
    tdec = dec.copy()
    tra_min, tra_max = tra.min(), tra.max()
    tdec_min, tdec_max = tdec.min(), tdec.max()

    #size of the field 
    field_ra = tra_max - tra_min
    field_dec = tdec_max - tdec_min
    tot_area = field_ra * field_dec
    print 'Field area is %2.4f sq. deg'%(tot_area)

    if bin_ra is None:
        bin_ra = int(field_ra * 60 / pix_scale) # number of pixels along ra
        bin_dec = int(field_dec * 60 / pix_scale) # number of pixels along dec

    print '%2.2f source per sq arcmin'%(ra.shape[0] / (tot_area * 3600))

    n, dec_b, ra_b = np.histogram2d(tdec, tra, bins=(bin_dec,bin_ra))
    n *= 1.0
    mask = n.copy()
    mask = ndimage.gaussian_filter(mask, 1.2)
    mask[mask > 0] = 1

    con = (n == 0)
    n[n == 0] = 1.0

    ng1, dec_b, ra_b = np.histogram2d(tdec,tra,weights=g1,bins=(bin_dec,bin_ra))
    ng2, dec_b, ra_b = np.histogram2d(tdec,tra,weights=g2,bins=(bin_dec,bin_ra))

    ng2 = 1.0 * ng2

    gamma = (ng1 + 1j * ng2) / n
    n[con] = 0.0
    gamma[con] = 0.0
    np.savez(ofile, gamma=gamma, ra=ra_b, dec = dec_b, number=n)

    return ofile, mask#, ra_min, ra_max, dec_min, dec_max, ra_avg



if __name__ == '__main__':
    matrc1()
    pix_scale = 1.0 #0.9375 # size of the pixel in arc min
    smooth = 1.5
    ipath = '/home/vinu/Lensing/DES/Kappa_map/NewCoadd'
    
    ifile = 'rxj_i_coadd_fg.txt'
    pixelize_light(ipath, ifile, pix_scale, skip=0, opath='./')

    ifile = 'rxj_i_coadd_fg.txt'
    pixelize_galcount(ipath, ifile, pix_scale, skip=0, opath='./')

    ifile = 'rxj_i_coadd_bg.txt'
    pixelize_shear(ipath, ifile, pix_scale, skip=0, opath='./', rotate=False)
