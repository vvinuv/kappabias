from pylab import *
from mytools import *
from numpy import *
from scipy import ndimage
import sys
sys.path.append('/home/vinu/scripts/BiasEstimator/bias')
import kappa_amara as kk
import pyfits
import os
import config as c

if __name__=='__main__':


    b_k, be_k = [], []
    b_kg, be_kg = [], []
    bb1, bb1e_l, bb1e_h = [], [], []
    bb2, bb2e_l, bb2e_h = [], [], []
    bb3, bb3e_l, bb3e_h = [], [], []
    b1, b1e = [], []
    b2, b2e = [], []
    b3, b3e = [], []
    smooth_arr = linspace(0, 16, 4)
    #smooth_arr = [9.12]
    #Generate a fits file from given fits table based on the given conditions
    fsource = 'Mice_0_12_0_12_tmp_source_1.fits'
    flens = 'Mice_0_12_0_12_tmp_lens_1.fits'
    f = pyfits.open('../../Mice_0_12_0_12_z_pz_r_g_DM.fits')
    d = f[1].data
    f.close()
    ra = d.field('RA')
    dec = d.field('DEC')
    z = d.field('Z')
    pz = d.field('Pz')
    pz[pz < 0] = 0
    k = d.field('KAPPA')
    g1 = d.field('GAMMA1')
    g2 = d.field('GAMMA2')
    mg = d.field('g')
    mr = d.field('r')
    DM = d.field('DM')
    gmr = mg - mr

    Mr = mr - DM
    #r = random.randint(0, mr.shape[0], 50000)
    #N, E = np.histogramdd(np.array([gmr[r], Mr[r] - DM[r]]).T, bins=(20,20))
    #contourf(N, origin='lower', extent=[Mr[r].min(), Mr[r].max(), gmr[r].min(), gmr[r].max()])

    #scatter(mr[r] - DM[r], gmr[r], c='k', s=0.1) 
    #show()
    #Mg = d.field('AMag_g')
    #Mr = d.field('AMag_r')
    e1 = normal(g1, 0.3, g1.shape)
    e1 = where(abs(e1) >= 1, sign(e1) * mod(e1,1), e1)
    e2 = normal(g2, 0.3, g2.shape)
    e2 = where(abs(e2) >= 1, sign(e2) * mod(e2,1), e2)


    #conl = (gmr > 1.2) & (Mr < -20) #magnitude cutoff which is mostly S/N cutoff
    conl = (gmr <= 1.4)
    cons = (gmr > -9999.0) #magnitude cutoff 
    os.system('rm -f %s %s'%(fsource, flens))
    write_fits_table(flens, ['RA', 'DEC', 'Z', 'KAPPA', 'GAMMA1', 'GAMMA2', 'E1', 'E2'], [ra[conl], dec[conl], z[conl], k[conl], g1[conl], g2[conl], e1[conl], e2[conl]])
    write_fits_table(fsource, ['RA', 'DEC', 'Z', 'KAPPA', 'GAMMA1', 'GAMMA2', 'E1', 'E2'], [ra[cons], dec[cons], z[cons], k[cons], g1[cons], g2[cons], e1[cons], e2[cons]])
 
    for smooth in smooth_arr:
        k = kk.KappaAmara('.', fsource, flens, '.', smooth, zs=c.zs, zmin_s=c.zmin_s, zmax_s=c.zmax_s, zmin_l=c.zmin_l, zmax_l=c.zmax_l)
        k.delta_rho_3d(c.bins, c.bins, c.zbins)

        #Stick to g_to_k=False for now as g_to_k=True works only for CFHT
        k.true_values(g_to_k=False, e_sign = c.e_sign, col_names=['RA', 'DEC', 'Z', 'GAMMA1', 'GAMMA2', '', '', ''])
        #k.true_values(g_to_k=True, e_sign = c.e_sign, col_names=['RA', 'DEC', 'Z', 'E1', 'E2', '', '', ''])
        kfg = k.kappa_true.copy()
        g1 = k.gamma1_true.copy()
        g2 = k.gamma2_true.copy()

        #This is just to average the true values in the simulation
        #k.true_values(g_to_k=False)

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


        b = kk.BiasModeling(g1t, g2t, g1p, g2p, bias_model='linear', do_plot=c.do_plot, sigma=c.sigma, bin_no=c.bin_no, boot_real=c.boot_real, boot_sample=g1t.shape[0]/2)
        bb1.append(b.b1_med)
        bb1e_l.append(b.b1_l)
        bb1e_h.append(b.b1_h)
        bb2.append(b.b2_med)
        bb2e_l.append(b.b2_l)
        bb2e_h.append(b.b2_h)
        bb3.append(b.b_med)
        bb3e_l.append(b.b_l)
        bb3e_h.append(b.b_h)

        b1.append(b.bias1)
        b1e.append(b.bias1_e)
        b2.append(b.bias2)
        b2e.append(b.bias2_e)
        b3.append(b.bias3)
        b3e.append(b.bias3_e)


    savetxt('bias_scale_blue_%d.txt'%c.bins, transpose([smooth_arr, b_kg, be_kg, bb1, bb1e_l, bb1e_h, bb2, bb2e_l, bb2e_h, bb3, bb3e_l, bb3e_h, b1, b1e, b2, b2e, b3, b3e]))
    subplot(131)
    errorbar(smooth_arr, b_kg, be_kg, label=r'$\kappa$ from $\gamma$', c='r')
    errorbar(smooth_arr, b_k, be_k, label=r'True $\kappa$', c='k')
    axis([-1, 18, -10, 20])
    legend(loc=0)
    xlabel('Smoothing scale (arcmin)')
    ylabel('Bias')

    subplot(132)
    errorbar(smooth_arr, bb1, [bb1e_l, bb1e_h], label=r'$\gamma_1$', c='r')
    errorbar(smooth_arr, bb2, [bb2e_l, bb2e_h], label=r'$\gamma_2$', c='k')
    errorbar(smooth_arr, bb3, [bb3e_l, bb3e_h], label=r'$\gamma$', c='g')
    legend(loc=0)
    xlabel('Smoothing scale (arcmin)')
    ylabel('Bias (Bootstrap)')

    subplot(133)
    errorbar(smooth_arr, b1, b1e, label=r'$\gamma_1$', c='r')
    errorbar(smooth_arr, b2, b2e, label=r'$\gamma_2$', c='k')
    errorbar(smooth_arr, b3, b3e, label=r'$\gamma$', c='g')
    legend(loc=0)
    xlabel('Smoothing scale (arcmin)')
    ylabel('Bias (ML)')

show()

