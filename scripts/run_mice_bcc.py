from pylab import *
from mytools import *
import os
import pyfits
from mytools import whiskerplot
from bias.MyFunc import AvgQ
from scipy import ndimage
import sys
sys.path.append('/home/vinu/scripts/BiasEstimator/bias')
import kappa_amara as kk
import config as c

if __name__=='__main__':

fsource = 'Mice_source_3.fits'
flens = 'Mice_lens_3.fits'
f = pyfits.open('../../Mice_0_12_0_12_z_pz_r_g_DM.fits')
d = f[1].data
f.close()

ra = d.field('RA')
dec = d.field('DEC')
z = d.field('Z')
pz = d.field('Pz')
pz[pz < 0] = 0
kappa = d.field('KAPPA')
tg1 = d.field('GAMMA1')
tg2 = d.field('GAMMA2')
mg = d.field('g')
mr = d.field('r')
DM = d.field('DM')
gmr = mg - mr
Mr = mr - DM

e1 = normal(tg1, 0.3, tg1.shape)
e1 = where(abs(e1) >= 1, sign(e1) * mod(e1,1), e1)
e2 = normal(tg2, 0.3, tg2.shape)
e2 = where(abs(e2) >= 1, sign(e2) * mod(e2,1), e2)
conl = (gmr > -9999) #magnitude cutoff which is mostly S/N cutoff
cons = (gmr > -9999.0) #magnitude cutoff 
os.system('rm -f %s %s'%(fsource, flens))

write_fits_table(flens, ['RA', 'DEC', 'Z', 'KAPPA', 'GAMMA1', 'GAMMA2', 'E1', 'E2'], [ra[conl], dec[conl], pz[conl], kappa[conl], tg1[conl], tg2[conl], e1[conl], e2[conl]])
write_fits_table(fsource, ['RA', 'DEC', 'Z', 'KAPPA', 'GAMMA1', 'GAMMA2', 'E1', 'E2'], [ra[cons], dec[cons], pz[cons], kappa[cons], tg1[cons], tg2[cons], e1[cons], e2[cons]])

k = kk.KappaAmara('.', fsource, flens, '.', c.smooth, zs=c.zs, zmin_s=c.zmin_s, zmax_s=c.zmax_s, zmin_l=c.zmin_l, zmax_l=c.zmax_l)

k.delta_rho_3d(c.bins, c.bins, c.zbins)
k.true_values(g_to_k=False, e_sign = c.e_sign, col_names=['RA', 'DEC', 'Z', 'GAMMA1', 'GAMMA2', '', '', ''])
g1 = k.gamma1_true.copy()
g2 = k.gamma2_true.copy()

k.kappa_predicted()
k.gamma_predicted()
kk.linear_bias_kappa(k.kappa_true[c.ig:-c.ig,c.ig:-c.ig], k.kappa_pred[c.ig:-c.ig,c.ig:-c.ig])
kk.linear_bias_kappa(k.kappa_true[c.ig:-c.ig,c.ig:-c.ig], k.kappa_pred_3d[c.ig:-c.ig,c.ig:-c.ig])
mask = k.mask[c.ig:-c.ig,c.ig:-c.ig]

#mask = where(k.gamma1_true == 0, 0, 1)
#mask = ndimage.morphology.binary_erosion(mask, iterations=5)

rcParams.update({'figure.figsize' : [10.5, 2.8]})
figure(1)
clf()
subplot(131)
imshow(k.kappa_true[c.ig:-c.ig,c.ig:-c.ig], origin='lower')#, vmin=-0.03, vmax=0.03)
title(r'True $\kappa$')
xlabel('arcmin')
ylabel('arcmin')
colorbar(shrink=0.8, format='%.2f')
subplot(132)
imshow(k.kappa_pred[c.ig:-c.ig,c.ig:-c.ig], origin='lower')#, vmin=-0.03, vmax=0.03)
title(r'$\kappa_g$:2D smoothing (%2.1f arcmin)'%c.smooth)
xlabel('arcmin')
ylabel('arcmin')
colorbar(shrink=0.8, format='%.2f')
subplot(133)
imshow(k.kappa_pred_3d[c.ig:-c.ig,c.ig:-c.ig], origin='lower')#, vmin=-0.03, vmax=0.03)
title(r'$\kappa_g$:3D smoothing (%2.1f arcmin)'%c.smooth)
xlabel('arcmin')
ylabel('arcmin')
colorbar(shrink=0.8, format='%.2f')
savefig('kappa_true_galaxies_%.1f.png'%c.smooth, bbox_inches='tight')

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
    print type(k_1d), type(k.kappa_pred[c.ig:-c.ig,c.ig:-c.ig].ravel())
    xavg, xstd, yavg, ystd, N, B = AvgQ(k_1d, k.kappa_pred[c.ig:-c.ig,c.ig:-c.ig].ravel(), bin_k, sigma=2)
    errorbar(xavg, yavg, yerr=ystd/sqrt(N), xerr=xstd, c='r', ls='')


g1p = k.gamma_p[c.ig:-c.ig,c.ig:-c.ig].real
g2p = k.gamma_p[c.ig:-c.ig,c.ig:-c.ig].imag
g1t = k.gamma1_true[c.ig:-c.ig,c.ig:-c.ig]
g2t = -k.gamma2_true[c.ig:-c.ig,c.ig:-c.ig]

#m = mask.ravel().astype(bool)
g1t = nan_to_num(g1t).ravel()#[m]
g2t = nan_to_num(g2t).ravel()#[m]
g1p = nan_to_num(g1p).ravel()#[m]
g2p = nan_to_num(g2p).ravel()#[m]

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

    #b = kk.BiasModeling(g1t, g2t, g1p, g2p, bias_model='linear', do_plot=c.do_plot, sigma=c.sigma, bin_no=c.bin_no, boot_real=c.boot_real, boot_sample=c.boot_sample)

    show()



