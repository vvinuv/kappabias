import os
import numpy as np
import pylab as pl
import cosmolopy.distance as cd
import pyfits
from scipy import signal, optimize
import scipy.ndimage as nd
import MyFunc as MyF
from convolution_mask import convolve_mask_fft, Gaussian
import kappa_utils as ku
import minuit
from astropy.stats.funcs import sigma_clip

#from mayavi import mlab



class KappaAmara:

    def __init__(self, ipath, shearfile, galaxyfile, opath, smooth, 
                 zs=1.0, zmin_s=0.4, zmax_s=1.1, zmin_g=0.1, zmax_g=1.1):
        self.shearfile = os.path.join(ipath, shearfile)
        self.galaxyfile = os.path.join(ipath, galaxyfile)
        self.smooth = smooth
        self.zs = zs
        self.zmin_g = zmin_g
        self.zmax_g = zmax_g
        self.zmin_s = zmin_s
        self.zmax_s = zmax_s
        self.initialize()
        #self.delta_rho_3d(bin_ra, bin_dec, bin_z)

    
    def initialize(self):
        self.cosmo = {'omega_M_0':0.3, 'omega_lambda_0':0.7, 
                      'omega_k_0':0.0, 'h':0.72}

        f = pyfits.open(self.galaxyfile)
        d = f[1].data
        f.close()
        self.z = d.field('z') 
        con = (self.z >= self.zmin_g) & (self.z <= self.zmax_g)
        self.ra = d.field('RA')[con] 
        self.dec = d.field('DEC')[con]
        self.kappa_true = np.zeros(self.ra.shape)
        self.z = self.z[con]

        d = 0

    def return_size(self, x, s=3):    
        """Return size of Gaussina kernal"""
        if np.ceil(2*s*x) % 2 == 0:
            size = np.ceil(2*s*x) + 1.
        else:
            size = np.ceil(2*s*x)
        return size
 
    def delta_rho_3d(self, bin_ra, bin_dec, bin_z):

        ra_min, ra_max = self.ra.min(), self.ra.max()
        dec_min, dec_max = self.dec.min(), self.dec.max()
        if ra_min < 2 and ra_max > 300:
            raw_input("RA include both 0 and 360. The projection doesnt work")
        ra_avg = (ra_max + ra_min) / 2.
        dec_avg = (dec_max + dec_min) / 2.

        #print 'Avg ', ra_avg, dec_avg

        #Taking the dec factor in ra
        self.ra = ra_avg + (self.ra - ra_avg) * \
                  np.cos(self.dec * np.pi / 180.0)  # deg
        ra_min, ra_max = self.ra.min(), self.ra.max()

        #size of the field 
        field_ra = ra_max - ra_min
        field_dec = dec_max - dec_min
        tot_area = field_ra * field_dec
        print 'Field area is %2.4f sq. deg'%(tot_area)


        if bin_ra is None:
            bin_ra = self.ra.ptp() * 60.  #ie. 1 arcmin per pixel
        if bin_dec is None:
            bin_dec = self.dec.ptp() * 60.  #ie. 1 arcmin per pixel
        if bin_z is None:
            bin_z = self.z.ptp() / 0.1 # z=0.1 per bin

        self.pixel_scale = self.ra.ptp() * 60. / bin_ra 
        if self.smooth == 0:
            self.sigma = 0.0
            self.kern_size = 1
            self.g_2d = np.array([[1]]) 
            self.g_3d = np.array([[[1]]]) 
        else:
            self.sigma = self.smooth/self.pixel_scale 
            self.kern_size = self.return_size(self.sigma, s=3)
            self.g_2d = Gaussian(self.sigma, size=self.kern_size, ndim=2)
            self.g_3d = Gaussian(self.sigma, size=self.kern_size, ndim=3)

        print 'Pix scale %2.2f arcmin'%self.pixel_scale
        print 'Sigma %2.2f pixels'%self.sigma

        self.N3d, edges = np.histogramdd(np.array([self.z, self.dec, 
                          self.ra]).T, bins=(bin_z, bin_dec, bin_ra))

        self.raedges = edges[2]
        self.decedges = edges[1]
        self.zedges = edges[0]
        self.zavg = (self.zedges[:-1] + self.zedges[1:]) / 2.
        self.raavg = (self.raedges[:-1] + self.raedges[1:]) / 2.
        self.decavg = (self.decedges[:-1] + self.decedges[1:]) / 2.

        # The total galaxies per redshift slice
        N1d, zedge = np.histogram(self.z, bins=self.zedges) 

        # Average per redshift slices. Dividing it by the number of 
        # pixels in RA and DEC directions 
        self.n1d = N1d / (self.N3d.shape[1] * self.N3d.shape[2] * 1.0) 
        self.n1d[self.n1d == 0] = 1
 
        #print bin_ra, bin_dec, bin_z, self.N3d.shape, self.n1d.shape

        # subtracting the average number from each redshift slice.
        self.n3d = self.N3d - self.n1d[:,np.newaxis][:,np.newaxis] 

        #print self.n1d , N1d, self.edges[0]

        #delta = (rho - rho_m) / rho_m
        self.delta3d = self.n3d / self.n1d[:,np.newaxis][:,np.newaxis] 

        #print self.delta3d 

        np.savez('density.npz', raedge=self.raavg, decedge=self.decavg, 
                 zedge=self.zavg, N3d=self.N3d, n1d=self.n1d, 
                 n3d=self.n3d, delta3d=self.delta3d)

    def comoving_d(self):
        self.cosmo = {'omega_M_0':0.3, 'omega_lambda_0':0.7, 
                      'omega_k_0':0.0, 'h':0.72}
        comoving_edges = cd.comoving_distance(self.zedges, **self.cosmo) #Mpc
        comoving_edges /= (1. + self.zedges)
        self.d_c = cd.comoving_distance(self.zavg, **self.cosmo) #Mpc

        #There is some subtilities in this case. When using MICE, the answer
        #makes sense when commenting the following line. When using BCC
        #it requires the following line
        #self.d_c /= (1. + self.zavg)

        #self.d_s = comoving_edges[-1] #source distance
        self.d_s = cd.comoving_distance(self.zs, **self.cosmo) #source distance
        #self.d_s /= (1. + self.zs)
        self.delta_d = comoving_edges[1:] - comoving_edges[:-1]
        
        self.a = 1 / (1 + self.zavg)

    def kappa_predicted(self):
        self.comoving_d()
        c_light = 3e5 #km/s
 
        # Eq. 9 Amara et al.
        constant = ((100. * self.cosmo['h'])**2 * self.cosmo['omega_M_0']) * \
                   (3/2.) * (1/c_light**2)         

        integral_1 = ((self.d_c * (self.d_s - self.d_c) / self.d_s) * \
                     (self.delta_d / self.a))[:,np.newaxis][:,np.newaxis]

        # Smooth the 3d density field and find kappa from that
        self.mask_3d = np.ones(self.delta3d.shape) * self.mask
        self.delta3d_sm = convolve_mask_fft(self.delta3d, \
                                        self.mask_3d, self.g_3d, ignore=0.30)
        self.kappa_pred_sm = constant * np.sum(integral_1 * self.delta3d_sm, \
                                               axis=0)

        # Use unsmoothed density field and generate kappa from that. Later
        # smooth the 2D kappa field
        self.kappa_pred = constant * np.sum(integral_1 * self.delta3d, axis=0)
        self.kappa_pred = convolve_mask_fft(self.kappa_pred, self.mask, \
                                        self.g_2d, ignore=0.30) 

        #print integral_1.shape, self.delta3d.shape, self.kappa_pred.shape

        np.savez('kappa_predicted.npz', kappa=self.kappa_pred)


    def true_values(self, g_to_k=False, e_sign = [-1, -1], 
                    col_names=['RA', 'DEC', 'z', 'E1', 'E2', 'W', 'SN', 'Re']):
        """g_to_k=True implies that create kappa from gamma, otherwise just 
           read kappa directly from the fits table. e_sign tells what is the 
           correct sign for e1 and e2. col_names tells the column names in
           the fits file. It works only with g_to_k=False. Otherwise use 
           difault name for column from the simulation"""
        if g_to_k:
            shearfile1 = os.path.split(self.shearfile)[1].split('.')[0]
            ofile = 'pixelized_%s.npz'%shearfile1
            ofile, mask = ku.pixelize_shear_CFHT('.', self.shearfile, \
                          self.pixel_scale, ofile=ofile, \
                          bin_ra=self.raedges, bin_dec=self.decedges,
                          zmin=self.zmin_s, zmax=self.zmax_s,
                          col_names=col_names)
            f = np.load('pixelized_%s.npz'%shearfile1)
            epsilon = f['epsilon']
            Nm = f['number']
            dt2 = self.pixel_scale
            dt1 = self.pixel_scale
            self.mask = f['mask']
            e1 = convolve_mask_fft(epsilon.real, self.mask, 
                                   self.g_2d, ignore=0.50)
            e2 = convolve_mask_fft(epsilon.imag, self.mask, 
                                   self.g_2d, ignore=0.50)
            Nm = convolve_mask_fft(Nm, self.mask, self.g_2d, ignore=0.50)
            Nm[Nm == 0] = 1

            epsilon = e_sign[0] * e1 + e_sign[1] * 1j * e2
            epsilon /= Nm

            self.kappa_true = ku.gamma_to_kappa(epsilon, dt1, dt2=dt2).real
            self.gamma1_true = epsilon.real 
            self.gamma2_true = epsilon.imag
        else:
            #Reading galaxy file to get the kappa map
            f = pyfits.open(self.galaxyfile)
            d = f[1].data
            f.close()
            z_gal = d.field('Z')
            con = (z_gal >= self.zmin_g) & (z_gal <= self.zmax_g)
            ra_sh = d.field('RA')[con]
            dec_sh = d.field('DEC')[con]
            kappa_true = d.field('KAPPA')[con]
            N, E = np.histogramdd(np.array([dec_sh, ra_sh]).T,
                   bins=(self.decedges, self.raedges))
            self.mask_lens = N.copy()
            Nk, E = np.histogramdd(np.array([dec_sh, ra_sh]).T,
                   bins=(self.decedges, self.raedges), weights=kappa_true)

            N[N == 0] = 1
            self.mask_lens[self.mask_lens > 0] = 1

            self.kappa_true = Nk / (1. * N)

            #Reading source catalog to get the shear field
            f = pyfits.open(self.shearfile)
            d = f[1].data
            f.close()
            z_sh = d.field('Z')
            con = (z_sh >= self.zmin_s) & (z_sh <= self.zmax_s)
            ra_sh = d.field('RA')[con]
            dec_sh = d.field('DEC')[con]
            gamma1_true = d.field('GAMMA1')[con]
            gamma2_true = d.field('GAMMA2')[con]
            z_sh = z_sh[con]
            d = 0

            N, E = np.histogramdd(np.array([dec_sh, ra_sh]).T,
                   bins=(self.decedges, self.raedges))
            self.mask = N.copy()
            Ng1, E = np.histogramdd(np.array([dec_sh, ra_sh]).T,
                   bins=(self.decedges, self.raedges), weights=gamma1_true)
            Ng2, E = np.histogramdd(np.array([dec_sh, ra_sh]).T,
                   bins=(self.decedges, self.raedges), weights=gamma2_true)

            N[N == 0] = 1
            self.mask[self.mask > 0] = 1

            self.gamma1_true = Ng1 / (1. * N)
            self.gamma2_true = Ng2 / (1. * N)

            #Masked convolution
            self.kappa_true = convolve_mask_fft(self.kappa_true, \
                                                self.mask_lens, \
                                                self.g_2d, ignore=0.50)
            self.gamma1_true = convolve_mask_fft(self.gamma1_true, self.mask, \
                                                self.g_2d, ignore=0.50)
            self.gamma2_true = convolve_mask_fft(self.gamma2_true, self.mask, \
                                                self.g_2d, ignore=0.50)


    def gamma_predicted(self):
        """Eq. 26 of Schenider"""
        @np.vectorize 
        def D_kernel(Dx, Dy, Dsq):
            if abs(Dsq)==0:
                return 0., 0.
            else:
                return (Dy**2 - Dx**2) / Dsq**2., (-2 * Dx * Dy) / Dsq**2.
        Dx, Dy = np.mgrid[-10:10:21j, -10:10:21j]
        Dsq = Dx**2 + Dy**2
        #D1 = (Dy**2 - Dx**2) / Dsq**2.
        #D2 = (-2 * Dx * Dy) / Dsq**2.
        D1, D2 = D_kernel(Dx, Dy, Dsq)
        D = D1 + 1j * D2

        #D = -1. / (Dx - 1j * Dy)**2.

        self.gamma_p = signal.convolve2d(self.kappa_pred, \
                                         D, mode='same') / np.pi
        self.gamma_tp = signal.convolve2d(self.kappa_true, \
                                         D, mode='same') / np.pi
   

    def plot():
        a = np.random.randint(0, 100, size=100)
        b = np.random.randint(0, 10, size=100)
        c = np.random.uniform(0, 1, size=100)
        h, e = np.histogramdd(array([a,b,c]).T, bins=(10, 10, 10))

        x,y,z = np.mgrid[0:100:10j, 0:100:10j, 0:1:10j]
        xx = x.ravel()
        yy = y.ravel()
        zz = z.ravel()
        hh = h.ravel()
        mlab.points3d(xx,yy,zz,hh)

        pl.show()

class BiasModeling:

    def __init__(self, g1t, g2t, g1p, g2p, bias_model='linear', bin_no=30, do_plot=False, sigma=1e10, boot_realiz=100, boot_sample=None):
        self.initialize(g1t, g2t, g1p, g2p, bin_no, sigma, do_plot)

        self.binning(valid=None)

        if bias_model == 'linear':
            self.linear_bias()
            self.linear_bias_boot(boot_realiz=boot_realiz, 
                                  boot_sample=boot_sample)
            self.linear_bias_error()
        elif bias_model == 'linear_evolve':
            self.linear_evolve_bias()
        elif bias_model == 'nonlinear':
            self.nonlinear_bias()
        else:
            print 'Unknown bias model. Stopping'
        

    def initialize(self, g1t, g2t, g1p, g2p, bin_no, sigma, do_plot):
        self.g1t = g1t.ravel()
        self.g2t = g2t.ravel()
        self.g1p = g1p.ravel()
        self.g2p = g2p.ravel()
  
        self.gt = abs(self.g1t + 1j*self.g2t)
        self.gp = abs(self.g1p + 1j*self.g2p)

        self.bin1 = np.linspace(self.g1p.min(), self.g1p.max(), bin_no)
        self.bin2 = np.linspace(self.g2p.min(), self.g2p.max(), bin_no)
        self.bin = np.linspace(self.gp.min(), self.gp.max(), bin_no)
        
        self.sigma = sigma
        self.do_plot = do_plot

    def binning(self, valid=None, boot=False):
        #Based on Amara et al. Not that the true value is binned
        #for a fixed predicted value.
        if valid is None:
            valid = np.arange(self.g1t.shape[0])

        self.g1p_b, g1p_be, self.g1t_b, self.g1t_be, N1, B1 = \
        MyF.AvgQ(self.g1p[valid], self.g1t[valid], self.bin1, sigma=self.sigma) 
        self.g2p_b, g2p_be, self.g2t_b, self.g2t_be, N2, B2 = \
        MyF.AvgQ(self.g2p[valid], self.g2t[valid], self.bin2, sigma=self.sigma) 
        self.gp_b, gp_be, self.gt_b, self.gt_be, N, B = \
        MyF.AvgQ(self.gp[valid], self.gt[valid], self.bin, sigma=self.sigma) 


        self.g1t_be[self.g1t_be == 0] = 9999.
        self.g2t_be[self.g2t_be == 0] = 9999.
        self.gt_be[self.gt_be == 0] = 9999.
        N1[N1 == 0] = 1
        N2[N2 == 0] = 1

        #print self.g1p_b, g1p_be, self.g1t_b, self.g1t_be
     
        if self.do_plot and boot is False: 
            N, E = np.histogramdd(np.array([self.g1t, self.g1p]).T, bins=(bin1, bin1))
            pl.subplot(121) 
            pl.contourf(N, origin='lower', extent=[bin1[0], bin1[-1], bin1[0], bin1[-1]])
            #pl.colorbar()
            pl.scatter(self.g1p, self.g1t, s=0.01) 
            pl.scatter(self.g1p[B1==1], self.g1t[B1==1], c='r', s=5.01, edgecolor='') 
            pl.scatter(self.g1p[B1==3], self.g1t[B1==3], c='b', s=5.01, edgecolor='') 
            pl.scatter(self.g1p[B1==7], self.g1t[B1==7], c='g', s=5.01, edgecolor='') 
            #for xx, yy in zip(self.g1p_b, self.g1t_b):
            # print xx, yy

            pl.errorbar(self.g1p_b, self.g1t_b, self.g1t_be/np.sqrt(N1), c='r')
            pl.xlabel(r'$\gamma_1^p$')
            pl.ylabel(r'$\gamma_1^t$')
            pl.xticks([-0.02,-0.015,-0.01,-0.005,0.0,0.005, 0.01, 0.015, 0.02])
            pl.yticks([-0.02,-0.015,-0.01,-0.005,0.0,0.005, 0.01, 0.015, 0.02])
            pl.axis([-0.015, 0.015, -0.015, 0.015])
            #pl.axis([-0.05, 0.05, -0.06, 0.06])
            pl.subplot(122)
            pl.scatter(self.g2p, self.g2t, s=1.01)
            pl.errorbar(self.g2p_b, self.g2t_b, self.g2t_be/np.sqrt(N2), c='r')
            pl.xlabel(r'$\gamma_2^p$')
            pl.ylabel(r'$\gamma_2^t$')
            pl.xticks([-0.02,-0.015,-0.01,-0.005,0.0,0.005, 0.01, 0.015, 0.02])
            pl.yticks([-0.02,-0.015,-0.01,-0.005,0.0,0.005, 0.01, 0.015, 0.02])
            #pl.axis([-0.05, 0.05, -0.06, 0.06])
            pl.axis([-0.015, 0.015, -0.015, 0.015])
            pl.show()
            #self.g1t_be /= np.sqrt(N1)
            #self.g2t_be /= np.sqrt(N2)


    def linear_bias(self, boot=False):
        """Predicted gamma = gamma_g * 1/b. The parameter b[0] used here
           is 1/b not b"""
        
        b_init = [1]
        chi2_1 = lambda b: np.sum(((self.g1t_b - self.g1p_b * b[0]) 
                               / self.g1t_be)**2)
        bias1 = optimize.fmin(chi2_1, b_init)
        chi2_2 = lambda b: np.sum(((self.g2t_b - self.g2p_b * b[0]) 
                               / self.g2t_be)**2)
        bias2 = optimize.fmin(chi2_2, b_init)
        chi2_3 = lambda b: np.sum(((self.gt_b - self.gp_b * b[0]) 
                               / self.gt_be)**2)
        bias = optimize.fmin(chi2_3, b_init)

        if boot is False:
            print 'Bias1 %2.2f'%(1 / bias1)
            print 'Bias2 %2.2f'%(1 / bias2)
            print 'Bias %2.2f'%(1 / bias)

        return 1/bias1, 1/bias2, 1/bias

    def linear_bias_boot(self, boot_realiz=20, boot_sample=None):
        """Bias error using bootstrap"""
        if boot_sample is None:
            boot_sample = self.g1t.shape[0]

        b1_arr, b2_arr, b_arr = [], [], []
        for i in range(boot_realiz):
            print 'Boot sample > %d'%i
            boot_samples = np.floor(np.random.uniform(0,
                                                      self.g1t.shape[0],
                                                      boot_sample))
            boot_samples = boot_samples.astype('int')
            self.binning(valid=boot_samples)

            b1, b2, b = self.linear_bias(boot=True)    
            b1_arr.append(b1)
            b2_arr.append(b2)
            b_arr.append(b)
        b1_arr.sort()
        b2_arr.sort()
        b_arr.sort()

        larg = np.floor(0.16*boot_realiz - 1).astype(int)
        harg = np.floor(0.84*boot_realiz - 1).astype(int)

        b1_med = np.median(b1_arr)
        b1_l, b1_h =  b1_med - b1_arr[larg], b1_arr[harg] - b1_med
                         
        b2_med = np.median(b2_arr)
        b2_l, b2_h =  b2_med - b2_arr[larg], b2_arr[harg] - b2_med

        b_med = np.median(b_arr)
        b_l, b_h =  b_med - b_arr[larg], b_arr[harg] - b_med
  
        print 'b1 (boot) = %2.2f - %2.2f + %2.2f'%(b1_med, b1_l, b1_h) 
        print 'b2 (boot) = %2.2f - %2.2f + %2.2f'%(b2_med, b2_l, b2_h) 
        print 'b (boot) = %2.2f - %2.2f + %2.2f'%(b_med, b_l, b_h) 

    def return_bias(self, chi2):
        m = minuit.Minuit(chi2)
        m.migrad()
        m.hesse()
        b = m.values['b']
        be = m.errors['b']
        bias = 1/b
        bias_e = be/b**2.
        print 'Bias = %2.2f \pm %2.2f'%(bias, bias_e)
        return bias, bias_e 

    def linear_bias_error(self):
        """Predicted gamma = gamma_g * 1/b. The parameter b[0] used here
           is 1/b not b"""
        b_init = [1]
        chi2_1 = lambda b: np.sum(((self.g1t_b - self.g1p_b * b) 
                               / self.g1t_be)**2)
        self.bias1, self.bias1_e = self.return_bias(chi2_1)
        chi2_2 = lambda b: np.sum(((self.g2t_b - self.g2p_b * b) 
                               / self.g2t_be)**2)
        self.bias2, self.bias2_e = self.return_bias(chi2_2)
        chi2_3 = lambda b: np.sum(((self.gt_b - self.gp_b * b) 
                               / self.gt_be)**2)
        self.bias3, self.bias3_e = self.return_bias(chi2_3)


    def linear_evolve_bias(self):
        print 'Not yet'
    def nonlinear_bias(self):
        print 'Not yet'


def linear_bias_kappa(kt, kp):
    kt = kt.ravel()
    kp = kp.ravel()
    kt, con = sigma_clip(kt, 8, 10)
    #kt = kt[con]
    kp = kp[con]
    bin = np.linspace(kp.min(), kp.max(), 30)
    kp_b, kp_be, kt_b, kt_be, N, B =  MyF.AvgQ(kp, kt, bin)
    N[N == 0] = 1
    kt_be /= np.sqrt(N)
    b_init = [1]
    kt_be[kt_be == 0] = 99.0
    chi2 = lambda b: np.sum(((kt_b - kp_b * b[0]) / kt_be)**2)
    bias = optimize.fmin(chi2, b_init)
    print 'Bias ', 1/bias

    
    chi2 = lambda b: np.sum(((kt_b - kp_b * b) / kt_be)**2)
    m = minuit.Minuit(chi2)
    m.migrad()
    m.hesse()
    b = m.values['b']
    be = m.errors['b']
    bias = 1/b
    bias_e = be/b**2.
    print 'b = %2.2f \pm %2.2f'%(bias, bias_e)
    return bias, bias_e

if __name__=='__main__':
   
    ipath = '.'
    ifile = 'Aardvark_v1_0_truth_40.fit'
    opath = '.'
    k = KappaAmara(ipath, ifile, opath)    
    k.delta_rho_3d(50, 50, 10)
    k.kappa_predicted()
