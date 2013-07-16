from pylab import *
from mytools import *
from MyFunc import *
from numpy import *
from scipy import ndimage
import kappa_amara as kk
import pyfits
import os

if __name__=='__main__':


    b_k, be_k = [], []
    b_kg, be_kg = [], []
    b1, b1e = [], []
    b2, b2e = [], []
    b3, b3e = [], []
    smooth_arr = linspace(0, 16, 8)

    fname = 'Aardvark_v1_0_truth_7_12_-10_-15_tmp.fits'
    f = pyfits.open('Aardvark_v1_0_truth_7_12_-10_-15_g_e_mags.fits')
    d = f[1].data
    f.close()
    ra = d.field('RA')
    dec = d.field('DEC')
    z = d.field('Z')
    pz = random.normal(z, scale=0.05) 
    pz[pz < 0] = 0
    k = d.field('KAPPA')
    g1 = d.field('GAMMA1')
    g2 = d.field('GAMMA2')
    e1 = d.field('E1')
    e2 = d.field('E2')
    mg = d.field('Mag_g')
    mr = d.field('Mag_r')
    Mg = d.field('AMag_g')
    Mr = d.field('AMag_r')

    con = (mr < 123) #magnitude cutoff which is mostly S/N cutoff
    #con = (Mr < -20.0) #magnitude cutoff 
    os.system('rm -f %s'%fname)
    write_fits_table(fname, ['RA', 'DEC', 'Z', 'KAPPA', 'GAMMA1', 'GAMMA2', 'E1', 'E2'], [ra[con], dec[con], pz[con], k[con], g1[con], g2[con], e1[con], e2[con]])
 
    for smooth in smooth_arr:
        k = kk.KappaAmara('.', fname, fname, '.', smooth, zs=1.1, zmin_s=0.1, zmax_s=1.1, zmin_g=0.1, zmax_g=1.1)
        k.delta_rho_3d(60, 60, 20)
        #k.true_values(g_to_k=True, e_sign = [-1,-1], col_names=['RA', 'DEC', 'Z', 'GAMMA1', 'GAMMA2', '', '', ''])
        k.true_values(g_to_k=True, e_sign = [-1,-1], col_names=['RA', 'DEC', 'Z', 'E1', 'E2', '', '', ''])
        kfg = k.kappa_true.copy()
        g1 = k.gamma1_true.copy()
        g2 = k.gamma2_true.copy()

        #This is just to average the true values in the simulation
        k.true_values(g_to_k=False)

        #Predicted from the galaxies
        k.kappa_predicted()
        k.gamma_predicted()
        bias, biase = kk.linear_bias_kappa(kfg, k.kappa_pred)
        b_k.append(bias)
        be_k.append(biase)
        bias, biase = kk.linear_bias_kappa(k.kappa_true, k.kappa_pred)
        b_kg.append(bias)
        be_kg.append(biase)
        #kk.linear_bias_kappa(k.kappa_true, k.kappa_pred_sm)
        mask = k.mask

        mask = where(k.gamma1_true == 0, 0, 1)
        mask = ndimage.morphology.binary_erosion(mask, iterations=5)


        g1p = k.gamma_p.real
        g2p = k.gamma_p.imag
        #g1t = -k.gamma1_true
        #g2t = -k.gamma2_true
        g1t = g1.copy()
        g2t = g2.copy()

        m = mask.ravel().astype(bool)
        g1t = nan_to_num(g1t).ravel()[m]
        g2t = nan_to_num(g2t).ravel()[m]
        g1p = nan_to_num(g1p).ravel()[m]
        g2p = nan_to_num(g2p).ravel()[m]


        b = kk.BiasModeling(g1t, g2t, g1p, g2p, bias_model='linear')
        b1.append(b.bias1)
        b1e.append(b.bias1_e)
        b2.append(b.bias2)
        b2e.append(b.bias2_e)
        b3.append(b.bias3)
        b3e.append(b.bias3_e)


    subplot(121)
    errorbar(smooth_arr, b_kg, be_kg, label=r'$\kappa$ from $\gamma$', c='r')
    errorbar(smooth_arr, b_k, be_k, label=r'True $\kappa$', c='k')
    axis([-1, 18, -10, 20])
    legend(loc=0)
    xlabel('Smoothing scale (arcmin)')
    ylabel('Bias')
    subplot(122)
    errorbar(smooth_arr, b1, b1e, label=r'$\gamma_1$', c='r')
    errorbar(smooth_arr, b2, b2e, label=r'$\gamma_2$', c='k')
    errorbar(smooth_arr, b3, b3e, label=r'$\gamma$', c='g')
    legend(loc=0)
    xlabel('Smoothing scale (arcmin)')
    ylabel('Bias')
show()

