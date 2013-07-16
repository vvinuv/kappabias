import pylab 
import pyfits
import numpy as np
import os


def whiskerplot(shear,dRA=1.,dDEC=1.,scale=5, combine=1,offset=(0,0) ):
    if combine>1:
        s = (combine*int(shear.shape[0]/combine),
             combine*int(shear.shape[1]/combine))
        shear = shear[0:s[0]:combine, 0:s[1]:combine] \
                + shear[1:s[0]:combine, 0:s[1]:combine] \
                + shear[0:s[0]:combine, 1:s[1]:combine] \
                + shear[1:s[0]:combine, 1:s[1]:combine]
        shear *= 0.25

        dRA *= combine
        dDEC *= combine


    theta = shear**0.5
    RA = offset[0] + np.arange(shear.shape[0])*dRA
    DEC = offset[1] + np.arange(shear.shape[1])*dDEC

    pylab.quiver(RA,DEC,
                 theta.real.T,theta.imag.T,
                 pivot = 'middle',
                 headwidth = 0,
                 headlength = 0,
                 headaxislength = 0,
                 scale=scale)
    pylab.xlim(0,shear.shape[0]*dRA)
    pylab.ylim(0,shear.shape[1]*dDEC)
    pylab.xlabel('RA (arcmin)')
    pylab.ylabel('DEC (arcmin)')

def write_fits_table(outfile, keys, data, formats=None):

    os.system('rm -f %s'%outfile)
    if formats is None:
        formats = ['E'] * len(keys)

    cols = []
    for key, d, format in zip(keys, data, formats):
        cols.append(pyfits.Column(name=key, format=format, array=d))


    cols = pyfits.ColDefs(cols)

    tbhdu = pyfits.new_table(cols)

    tbhdu.writeto(outfile)
