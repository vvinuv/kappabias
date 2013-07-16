from pylab import *
from mytools import whiskerplot
from MyFunc import *
from scipy import ndimage
import bias.kappa_amara as kk
import config as c

if __name__=='__main__':

    k = kk.KappaAmara('.', c.shearfile, c.galaxyfile, '.', c.smooth, zs=c.zs, zmin_s=c.zmin_s, zmax_s=c.zmax_s, zmin_g=c.zmin_g, zmax_g=c.zmax_g)

    k.delta_rho_3d(c.bins, c.bins, c.zbins)
    k.true_values(g_to_k=True, e_sign = c.e_sign, col_names=['RA', 'DEC', 'Z', 'GAMMA1', 'GAMMA2', '', '', ''])
    #k.true_values(g_to_k=True, e_sign = c.e_sign, col_names=['RA', 'DEC', 'Z', 'E1', 'E2', '', '', ''])
    kfg = k.kappa_true.copy()
    g1 = k.gamma1_true.copy()
    g2 = k.gamma2_true.copy()

    k.true_values(g_to_k=False, e_sign = c.e_sign, col_names=['RA', 'DEC', 'Z', 'GAMMA1', 'GAMMA2', '', '', ''])
    #k.true_values(g_to_k=False, e_sign = c.e_sign, col_names=['RA', 'DEC', 'Z', 'E1', 'E2', '', '', ''])
    k.kappa_predicted()
    k.gamma_predicted()
    kk.linear_bias_kappa(kfg[c.ig:-c.ig,c.ig:-c.ig], k.kappa_pred[c.ig:-c.ig,c.ig:-c.ig])
    kk.linear_bias_kappa(k.kappa_true[c.ig:-c.ig,c.ig:-c.ig], k.kappa_pred[c.ig:-c.ig,c.ig:-c.ig])
    kk.linear_bias_kappa(k.kappa_true[c.ig:-c.ig,c.ig:-c.ig], k.kappa_pred_sm[c.ig:-c.ig,c.ig:-c.ig])
    mask = k.mask[c.ig:-c.ig,c.ig:-c.ig]

    #mask = where(k.gamma1_true == 0, 0, 1)
    #mask = ndimage.morphology.binary_erosion(mask, iterations=5)

    figure(1)
    subplot(141)
    imshow(k.kappa_true[c.ig:-c.ig,c.ig:-c.ig], origin='lower')#, vmin=-0.01, vmax=0.03)
    title(r'True $\kappa$')
    colorbar()
    subplot(142)
    imshow(kfg[c.ig:-c.ig,c.ig:-c.ig], origin='lower')#, vmin=-0.01, vmax=0.03)
    title(r'$\kappa$ from noisy shear')
    colorbar()
    subplot(143)
    imshow(k.kappa_pred[c.ig:-c.ig,c.ig:-c.ig], origin='lower')#, vmin=-0.01, vmax=0.03)
    title(r'$\kappa_g$:2D smoothing (%2.1f arcmin)'%smooth)
    colorbar()
    subplot(144)
    imshow(k.kappa_pred_sm[c.ig:-c.ig,c.ig:-c.ig], origin='lower')#, vmin=-0.01, vmax=0.03)
    title(r'$\kappa_g$:3D smoothing (%2.1f arcmin)'%smooth)
    colorbar()


    figure(2)
    subplot(121)
    whiskerplot((k.gamma1_true - 1j* k.gamma2_true)[c.ig:-c.ig,c.ig:-c.ig], scale=4)
    title('True shear')
    #axis([100, 200, 100, 200])
    subplot(122)
    whiskerplot(k.gamma_p[c.ig:-c.ig,c.ig:-c.ig], scale=4)
    #axis([100, 200, 100, 200])
    title('From galaxies')

    figure(3)
    k_1d = k.kappa_true[c.ig:-c.ig,c.ig:-c.ig].ravel()
    bin_k = linspace(k_1d.min(), k_1d.max(), 50)

    scatter(k_1d, k.kappa_pred[c.ig:-c.ig,c.ig:-c.ig].ravel(), s=0.01)
    xavg, xstd, yavg, ystd, N, B = AvgQ(k_1d, k.kappa_pred[c.ig:-c.ig,c.ig:-c.ig].ravel(), bin_k, sigma=2)
    errorbar(xavg, yavg, yerr=ystd/sqrt(N), xerr=xstd, c='r', ls='')


    g1p = k.gamma_p[c.ig:-c.ig,c.ig:-c.ig].real
    g2p = k.gamma_p[c.ig:-c.ig,c.ig:-c.ig].imag
    g1t = k.gamma1_true[c.ig:-c.ig,c.ig:-c.ig]
    g2t = -k.gamma2_true[c.ig:-c.ig,c.ig:-c.ig]

    m = mask.ravel().astype(bool)
    g1t = nan_to_num(g1t).ravel()[m]
    g2t = nan_to_num(g2t).ravel()[m]
    g1p = nan_to_num(g1p).ravel()[m]
    g2p = nan_to_num(g2p).ravel()[m]

    """
    #con = (abs(g1p) <=0.03) & (abs(g1p) <= 0.03)
    bins1 = linspace(g1t.min(), g1t.max(), 10) 
    bins2 = linspace(g2t.min(), g2t.max(), 10) 

    xavg, xstd, yavg, ystd, N = AvgQ(g1t, g1p, bins1)
    for xx, yy in zip(xavg, yavg):
     print xx, yy
    figure(4)
    subplot(121)
    scatter(g1t, g1p, s=0.01)
    errorbar(xavg, yavg, yerr=ystd/sqrt(N), xerr=xstd, c='r', ls='')
    subplot(122)
    scatter(g2t, g2p, s=0.01)
    xavg, xstd, yavg, ystd, N = AvgQ(g2t, g2p, bins2)
    errorbar(xavg, yavg, yerr=ystd/sqrt(N), xerr=xstd, c='r', ls='')
    """

    b = kk.BiasModeling(g1t, g2t, g1p, g2p, bias_model='linear', do_plot=c.do_plot, sigma=c.sigma, bin_no=c.bin_no, boot_real=c.boot_real, boot_sample=c.boot_sample)

    show()



