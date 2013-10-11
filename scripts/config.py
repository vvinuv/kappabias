"""Parameter values to run the scripts"""

smooth = 5.0 #arcmin

bins = 100 #no of bins
zbins = 20 #no of z bins

zs = 1.0 #source redshift
zmin_s = 0.5 #source min z
zmax_s = 1.5 #source max z
zmin_l = 0.2 #lens min z
zmax_l = 0.9 #lens max z

#MICE input file
sourcefile = '../../Mice_0_12_0_12_z_pz_r_g_DM.fits' #../data/Mice_0_12_0_12_tmp.fits'
lensfile = '../../Mice_0_12_0_12_z_pz_r_g_DM.fits' #../data/Mice_0_12_0_12_tmp.fits'

#BCC input files
#sourcefile = '../../Aardvark_v1_0_truth_7_12_-10_-15_g_e_mags.fits'
#lensfile = '../../Aardvark_v1_0_truth_7_12_-10_-15_g_e_mags.fits'

e_sign = [1, -1] #Multiplied by this to the shear
#e_sign = [1, 1] #Multiplied by this to the shear

ig = 10 #These many pixels will be ignored from all the four sides to minimize boundary problems


sigma = 5 #for sigma clipping while averaging
bin_no = 30 #No of shear bins to average 
boot_real = 500 #No of bootstrap realizations
boot_sample = None #No of samples in each bootstrap realization. None implies same number of samples as in the input data. i.e. if there are 100 galaxies in the input then each bootstrap sample will have 100 galaxies

do_plot = False #If it is true then it shows the plot between input shear and predicted shear from galaxy distribution
