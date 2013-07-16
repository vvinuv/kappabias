"""Parameter values to run the scripts"""

smooth = 4.5 #arcmin

bin = 200 #no of bins
zbin = 20 #no of z bins

zs = 1.0 #source redshift
zmin_s = 0.5 #source min z
zmax_s = 1.0 #source max z
zmin_g = 0.2 #lens min z
zmax_g = 0.9 #lens max z

shearfile = '../data/Mice_0_12_0_12_tmp.fits'
galaxyfile = '../data/Mice_0_12_0_12_tmp.fits'

e_sign = [1, -1] #Multiplied by this to the shear

ig = 10 #These many pixels will be ignored from all the four sides to minimize boundary problems


sigma = 5 #for sigma clipping while averaging
bin_no = 30 #No of shear bins to average 
boot_real = 20 #No of bootstrap realizations
boot_sample = None #No of samples in each bootstrap realization. None implies same number of samples as in the input data. i.e. if there are 100 galaxies in the input then each bootstrap sample will have 100 galaxies

do_plot = False #If it is true then it shows the plot between input shear and predicted shear from galaxy distribution