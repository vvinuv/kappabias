from pylab import *
from mytools import whiskerplot
from MyFunc import *
from scipy import ndimage
import kappa_amara as kk

smooth = 8.5
k = kk.KappaAmara('.', 'Aardvark_v1_0_truth_7_12_-10_-15_g_e.fits', 'Aardvark_v1_0_truth_7_12_-10_-15_g_e.fits', '.', smooth, zs=1.1, zmin_s=0.1, zmax_s=1.1, zmin_g=0.1, zmax_g=1.1)
k.delta_rho_3d(60, 60, 20)
k.true_values(g_to_k=True, e_sign = [-1,-1], col_names=['RA', 'DEC', 'Z', 'E1', 'E2', '', '', ''])
kfg = k.kappa_true.copy()
g1 = k.gamma1_true.copy()
g2 = k.gamma2_true.copy()
k.true_values(g_to_k=False)
k.kappa_predicted()
k.gamma_predicted()
kk.linear_bias_kappa(kfg, k.kappa_pred)
kk.linear_bias_kappa(k.kappa_true, k.kappa_pred)
kk.linear_bias_kappa(k.kappa_true, k.kappa_pred_sm)
mask = k.mask

mask = where(k.gamma1_true == 0, 0, 1)
mask = ndimage.morphology.binary_erosion(mask, iterations=5)

figure(5)
subplot(141)
imshow(k.kappa_true, origin='lower')#, vmin=-0.01, vmax=0.03)
title(r'True $\kappa$')
colorbar()
subplot(142)
imshow(kfg, origin='lower')#, vmin=-0.01, vmax=0.03)
title(r'$\kappa$ from noisy shear')
colorbar()
subplot(143)
imshow(k.kappa_pred, origin='lower')#, vmin=-0.01, vmax=0.03)
title(r'$\kappa_g$:2D smoothing (%2.1f arcmin)'%smooth)
colorbar()
subplot(144)
imshow(k.kappa_pred_sm, origin='lower')#, vmin=-0.01, vmax=0.03)
title(r'$\kappa_g$:3D smoothing (%2.1f arcmin)'%smooth)
colorbar()
show()


figure(2)
subplot(121)
whiskerplot((1*k.gamma1_true-1j* k.gamma2_true), scale=4)
title('True shear')
#axis([100, 200, 100, 200])
subplot(122)
whiskerplot(k.gamma_p, scale=4)
#axis([100, 200, 100, 200])
title('From galaxies')

figure(3)
subplot(121)
imshow(k.kappa_pred, origin='lower')
colorbar()
whiskerplot(k.gamma_p*mask)
title('Predicted')
subplot(122)
imshow(k.kappa_true, origin='lower')
colorbar()
whiskerplot(k.gamma1_true+1j* k.gamma2_true, scale=2)
title('True')

figure(4)

g1p = k.gamma_p.real
g2p = k.gamma_p.imag
g1t = -k.gamma1_true
g2t = -k.gamma2_true

m = mask.ravel().astype(bool)
g1t = nan_to_num(g1t).ravel()[m]
g2t = nan_to_num(g2t).ravel()[m]
g1p = nan_to_num(g1p).ravel()[m]
g2p = nan_to_num(g2p).ravel()[m]


con = (abs(g1p) <=0.03) & (abs(g1p) <= 0.03)
bins1 = linspace(g1t.min(), g1t.max(), 10) 
bins2 = linspace(g2t.min(), g2t.max(), 10) 

xavg, xstd, yavg, ystd, N = AvgQ(g1t, g1p, bins1)
subplot(121)
scatter(g1t, g1p, s=0.01)
errorbar(xavg, yavg, yerr=ystd/sqrt(N), xerr=xstd, c='r', ls='')
subplot(122)
scatter(g2t, g2p, s=0.01)
xavg, xstd, yavg, ystd, N = AvgQ(g2t, g2p, bins2)
print xavg, yavg
errorbar(xavg, yavg, yerr=ystd/sqrt(N), xerr=xstd, c='r', ls='')


b = kk.BiasModeling(g1t, g2t, g1p, g2p, bias_model='linear')
con1 = (abs(b.g1t_be) < 1)
con2 = (abs(b.g2t_be) < 1)
errorbar(b.g1p_b[con1], b.g1t_b[con1], yerr=b.g1t_be[con1], ls='', c='r')
errorbar(b.g2p_b[con2], b.g2t_b[con2], yerr=b.g2t_be[con2], ls='', c='k')
show()


kp = k.kappa_pred.ravel()
kt = k.kappa_true.ravel()
bin = np.linspace(kp.min(), kp.max(), 30) 
kp_b, kp_be, kt_b, kt_be, N1 =  MyF.AvgQ(kp, kt, bin)
scatter(kp, kt, s=0.1)
errorbar(kp_b, kt_b, xerr=kp_be, yerr=kt_be)
show()

