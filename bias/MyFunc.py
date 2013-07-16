import MySQLdb as mysql
import sys
from numpy import log10, average, sin, exp, sqrt, std, where, random, median, array, shape, pi, arcsin, arccos, cos, arange, array, isnan 
import numpy as np
import math
import numpy.ma as ma
import minuit
from astropy.stats.funcs import sigma_clip

def linearlsq(x, y):
    """Return slope m and constant c"""
    A = np.vstack([x, np.ones(x.shape[0])]).T
    m, c = np.linalg.lstsq(A, y)[0] 
    return c, m

def bootstrap(x):
    random.seed(1499)
    x = array(x)
    SAMPLE = 100 # No of replica
    NELEMENTS = shape(x)[0] * 3 # Number of elements in each replica of sample
    AVG_rs = [] # The list of averages of random samples
    STD_rs = [] # The list of standard deviations of random samples
    for i in range(SAMPLE):
        rs = x[random.random_integers(0, shape(x)[0] - 1, NELEMENTS)] # The random sample
        AVG_rs.append(average(rs))
	STD_rs.append(std(rs))
    AVG_rs.sort()
    STD_rs.sort()
    Med_AVG_rs = median(AVG_rs)
    Med_STD_rs = median(STD_rs)
    AVG_rs_low = AVG_rs[int(0.025 * SAMPLE - 1)]
    AVG_rs_high = AVG_rs[int(SAMPLE - 0.16 * SAMPLE - 1)]
    STD_rs_low = STD_rs[int(0.025 * SAMPLE - 1)]
    STD_rs_high = STD_rs[int(SAMPLE - 0.16 * SAMPLE - 1)]
    return Med_AVG_rs, AVG_rs_low, AVG_rs_high, Med_STD_rs, STD_rs_low, STD_rs_high
def bootstrapfrac1(x, limit, length):
    x = array(x)
    SAMPLE = 1000 # No of replica
    NELEMENTS = shape(x)[0]  * 3# Number of elements in each replica of sample
    E_frac_rs = [] # The fraction of ellipticals in random samples
    D_frac_rs = [] # The fraction of disk in random samples
    for i in range(SAMPLE):
        rs = x[random.random_integers(0, shape(x)[0] - 1, NELEMENTS)] # The random sample
	E_frac_rs.append(len([ele for ele in rs if ele >= limit]) / (3.0*length))
	D_frac_rs.append(len([ele for ele in rs if ele < limit]) / (3.0*length))
    E_frac_rs.sort()
    D_frac_rs.sort()
    Med_E_frac_rs = median(E_frac_rs)
    Med_D_frac_rs = median(D_frac_rs)
    E_frac_rs_low = E_frac_rs[int(0.025 * SAMPLE - 1)]
    E_frac_rs_high = E_frac_rs[int(SAMPLE - 0.025 * SAMPLE - 1)]
    D_frac_rs_low = D_frac_rs[int(0.025 * SAMPLE - 1)]
    D_frac_rs_high = D_frac_rs[int(SAMPLE - 0.025 * SAMPLE - 1)]
    return Med_E_frac_rs, E_frac_rs_low, E_frac_rs_high, Med_D_frac_rs, D_frac_rs_low, D_frac_rs_high


def bootstrapfrac(x, limit):
    x = array(x)
    SAMPLE = 1000 # No of replica
    NELEMENTS = shape(x)[0]  * 3# Number of elements in each replica of sample
    E_frac_rs = [] # The fraction of ellipticals in random samples
    D_frac_rs = [] # The fraction of disk in random samples
    for i in range(SAMPLE):
        rs = x[random.random_integers(0, shape(x)[0] - 1, NELEMENTS)] # The random sample
        E_frac_rs.append(len([ele for ele in rs if ele >= limit]) / (1.0*len(rs)))
	D_frac_rs.append(len([ele for ele in rs if ele < limit]) / (1.0*len(rs)))
    E_frac_rs.sort()
    D_frac_rs.sort()
    Med_E_frac_rs = median(E_frac_rs)
    Med_D_frac_rs = median(D_frac_rs)
    E_frac_rs_low = E_frac_rs[int(0.025 * SAMPLE - 1)]
    E_frac_rs_high = E_frac_rs[int(SAMPLE - 0.025 * SAMPLE - 1)]
    D_frac_rs_low = D_frac_rs[int(0.025 * SAMPLE - 1)]
    D_frac_rs_high = D_frac_rs[int(SAMPLE - 0.025 * SAMPLE - 1)]
    return Med_E_frac_rs, E_frac_rs_low, E_frac_rs_high, Med_D_frac_rs, D_frac_rs_low, D_frac_rs_high

def bootstrapfracS0(x, limit_l, limit_u):
    x = array(x)
    SAMPLE = 1000 # No of replica
    NELEMENTS = shape(x)[0]  * 3# Number of elements in each replica of sample
    E_frac_rs = [] # The fraction of ellipticals in random samples
    D_frac_rs = [] # The fraction of disk in random samples
    for i in range(SAMPLE):
        rs = x[random.random_integers(0, shape(x)[0] - 1, NELEMENTS)] # The random sample
	E_frac_rs.append(len([ele for ele in rs if ele >= limit_l and  ele <= limit_u]) / (1.0*len(rs)))
	D_frac_rs.append(len([ele for ele in rs if ele < limit_l]) / (1.0*len(rs)))
    E_frac_rs.sort()
    D_frac_rs.sort()
    Med_E_frac_rs = median(E_frac_rs)
    Med_D_frac_rs = median(D_frac_rs)
    E_frac_rs_low = E_frac_rs[int(0.025 * SAMPLE - 1)]
    E_frac_rs_high = E_frac_rs[int(SAMPLE - 0.025 * SAMPLE - 1)]
    D_frac_rs_low = D_frac_rs[int(0.025 * SAMPLE - 1)]
    D_frac_rs_high = D_frac_rs[int(SAMPLE - 0.025 * SAMPLE - 1)]
    return Med_E_frac_rs, E_frac_rs_low, E_frac_rs_high, Med_D_frac_rs, D_frac_rs_low, D_frac_rs_high

def bootstrapfracS01(x, limit_l, limit_u, length):
    x = array(x)
    SAMPLE = 1000 # No of replica
    NELEMENTS = shape(x)[0]  * 3# Number of elements in each replica of sample
    E_frac_rs = [] # The fraction of ellipticals in random samples
    D_frac_rs = [] # The fraction of disk in random samples
    for i in range(SAMPLE):
        rs = x[random.random_integers(0, shape(x)[0] - 1, NELEMENTS)] # The random sample
	E_frac_rs.append(len([ele for ele in rs if ele >= limit_l and  ele <= limit_u]) / (3.0*length))
	D_frac_rs.append(len([ele for ele in rs if ele < limit_l]) / (3.0*length))
    E_frac_rs.sort()
    D_frac_rs.sort()
    Med_E_frac_rs = median(E_frac_rs)
    Med_D_frac_rs = median(D_frac_rs)
    E_frac_rs_low = E_frac_rs[int(0.025 * SAMPLE - 1)]
    E_frac_rs_high = E_frac_rs[int(SAMPLE - 0.025 * SAMPLE - 1)]
    D_frac_rs_low = D_frac_rs[int(0.025 * SAMPLE - 1)]
    D_frac_rs_high = D_frac_rs[int(SAMPLE - 0.025 * SAMPLE - 1)]
    return Med_E_frac_rs, E_frac_rs_low, E_frac_rs_high, Med_D_frac_rs, D_frac_rs_low, D_frac_rs_high



def WriteDb(ParamValues):
    dba = c.database
    pwd = 'cluster'
    usr = c.usr
    tbl = c.table
    try:
        Conn = mysql.connect (host = "localhost",
                                user = "%s" %usr,
                                passwd = "%s" %pwd,
                                db = "%s" %dba) 
    except mysql.Error, e:
        print "Error %d: %s" % (e.args[0], e.args[1])
        sys.exit (1)
    cursor = Conn.cursor()

    DictParamWithValue = {}
    DictParamWithType1 = {}
    AllParams = []
    for dbparam in c.dbparams:
        DBparam = dbparam.split(':')
        DictParamWithType1[DBparam[0]] = 'varchar(500)'
        DictParamWithValue[DBparam[0]] = DBparam[1]
        AllParams.append(DBparam[0])

    if c.decompose:
        DictParamWithType2 = {'Name':'varchar(500)', 'ra':'float', \
                        'dec_':'float',\
                        'z':'float', 'Ie':'float','Ie_err':'float',\
                        're_pixels':'float', 're_err_pixels':'float',\
                        're_kpc':'float', 're_err_kpc':'float' ,'n':'float', \
                       'n_err':'float', 'Avg_Ie':'float', 'Avg_Ie_err':'float',\
                        'eb':'float', 'eb_err':'float', \
                        'Id':'float', 'Id_err':'float', 'rd_pixels':'float',\
                        'rd_err_pixels':'float', 'rd_kpc':'float', \
                        'rd_err_kpc':'float', 'ed':'float', 'ed_err':'float', \
                        'BD':'float', 'BT':'float', 'Point':'float', \
                        'Point_err':'float', 'Pfwhm':'float', \
                        'Pfwhm_kpc':'float', 'chi2nu':'float', \
                        'Goodness':'float', 'run':'int', 'C':'float', \
                        'C_err':'float', 'A':'float', 'A_err':'float', \
                        'S':'float', 'S_err':'float', 'G':'float', 'M':'float',\
                        'distance':'float', 'fit':'int', 'flag':'bigint', \
                        'Comments':'varchar(1000)'}
        ParamToWrite = ['Name','ra','dec_','z', 'Ie','Ie_err','re_pixels',\
                        're_err_pixels', 're_kpc', 're_err_kpc' ,'n', \
                        'n_err', 'Avg_Ie', 'Avg_Ie_err', 'eb', 'eb_err', \
                        'Id', 'Id_err', 'rd_pixels',\
                        'rd_err_pixels', 'rd_kpc', 'rd_err_kpc', \
                        'ed', 'ed_err', 'BD', \
                        'BT', 'Point', 'Point_err', 'Pfwhm', 'Pfwhm_kpc', \
                        'chi2nu', 'Goodness', 'run', 'C', 'C_err', 'A', \
                        'A_err', 'S', 'S_err', 'G', 'M', 'distance', \
                        'fit', 'flag', 'Comments']
        ParamType = ['varchar(500)', 'float', 'float', 'float', 'float',\
                     'float', 'float', 'float', 'float', 'float', 'float',\
                     'float', 'float', 'float', 'float', 'float', 'float',\
                     'float', 'float', 'float', 'float', 'float', 'float',\
                     'float', 'float', 'float', 'float', 'float', 'float',\
                     'float', 'float', 'float', 'int', 'float', 'float',\
                     'float', 'float', 'float', 'float', 'float', 'float',\
                     'float', 'int', 'bigint', 'varchar(500)']
       
    else:
        DictParamWithType2 = {'Name':'varchar(500)', 'ra':'float', \
                        'dec_':'float',\
                        'z':'float', 'C':'float', 'C_err':'float', 'A':'float',\
                        'A_err':'float', 'S':'float', 'S_err':'float',\
                        'G':'float', 'M':'float', 'flag':'bigint', \
                        'Comments':'varchar(500)'}
        ParamToWrite = ['Name','ra','dec_','z', 'C', \
                        'C_err', 'A', 'A_err', 'S', 'S_err', 'G', 'M', \
                        'flag', 'Comments']
        ParamType = ['varchar(500)', 'float', 'float', 'float', 'float',\
                     'float', 'float', 'float','float', 'float', 'float',\
                     'float', 'bigint', 'varchar(500)']
    DictParamWithType = {}  #Dictionary with Type
    DictParamWithType.update(DictParamWithType1)
    DictParamWithType.update(DictParamWithType2)
    ParamValues.append('None')
    ii = 0
    for Param in ParamToWrite:
        DictParamWithValue[Param] = ParamValues[ii]
        ii += 1
    for p in ParamToWrite:
        AllParams.append(p)
    cmd = "CREATE TABLE if not exists %s (" % tbl + ','.join(["%s %s" %(p, \
          DictParamWithType[p]) for p in AllParams]) + ")" 
    cursor.execute(cmd)
    cmd = "INSERT INTO %s values (" % tbl 
    for p in AllParams:
        if DictParamWithType[p] in ('int', 'bigint', 'float'):
            cmd = cmd + str(DictParamWithValue[p]) + ', '
        else:
            cmd = cmd + "'" + str(DictParamWithValue[p]) + "', "
    cmd = str(cmd[:-2]) + ')'
    cursor.execute(cmd)
    cursor.close()
    Conn.close()
#A = ['EDCSNJ1216490-1200091',184.204,-12.0025277778,0.7863,22.76,0.03,1.55,0.04,0.521182261202,0.0134498648052,1.24,0.12,18.8371116745,0.0316672494062,0.47,0.02,21.23,0.01,8.95,0.12,3.00940725017,0.0403495944157,0.35,0.0,0.244343055269,0.196363096362,9999,9999,9999,9999,1.146,0.664,1,3.63747679805,3.88004211969,0.130618587136,9999,0.365522099184,1.4538,0.546744991269,-2.24971293032,25.1297510444,1,1542]
#WriteDb(A)
#CREATE TABLE IF NOT EXISTS book (name char(40), lastname char(40), petname char (40))


def convert_galactic_equa(l, b):
   '''l and b in degrees'''
   r = pi / 180.0
   dec = arcsin(cos(b * r) * sin((l - 33) * r) * sin(62.6 * r) + sin(b * r) * cos(62.6 * r))
   ra  = 282.25 + arcsin((cos(b * r) * sin((l - 33) * r) * cos(62.6 * r) - sin(b * r) * sin(62.6 * r)) / cos(dec)) / r
   dec = dec / r
   return ra, dec

def ReturnBins(x, bins, bintype=1, hardlimit=0, binmax=None):
    '''Return the indices of corresponds to the bins. bins=number of bins. bintype=1 means the bins is the binsize. bintype=0 means bins is the number of bins. hardlimit=0 means the lowe binsize is limited by the minimum number in the array, otherwise it always starts from 0  '''
    if type(bins) == np.ndarray or type(bins) == list:
        return np.digitize(x, bins), bins
    else:
        if hardlimit:
            bmin = 0.0
        else:
            bmin = np.floor(x.min())
        if binmax is None:
            bmax = np.ceil(x.max())
        else:
            bmax = binmax
        if bintype:
            BinN = np.ceil(np.abs(bmax - bmin) / bins) #no of bins
            bins = np.linspace(bmin, bmax, BinN + 1)
        else:
            bins = np.linspace(bmin, bmax, bins + 1)
        return np.digitize(x, bins), bins 




def AvgQ(x, y, bin, bintype=1, hardlimit=0, binmax=None, sigma=1e10):
    '''Average values of scatter plot'''
    def HelpMe(kk, BR, ii):
        if len(kk) > 0:
            return np.average(kk)
        elif len(kk) == 0:
            return (BR[ii-1] + BR[ii])/2.


    x = array(x)
    y = array(y)
    BinNo, BinsReturned = ReturnBins(x, bin, bintype=bintype, hardlimit=hardlimit, binmax=binmax) 
    #print BinNo, BinsReturned
    #BinSize = np.max(BinNo)+1
    BinSize = len(BinsReturned)
    xavg = [HelpMe(x[BinNo == i], BinsReturned, i) for i in range(1, BinSize)]
    xstd = [np.std(x[BinNo == i]) for i in range(1, BinSize)]
    yavg = [np.average(sigma_clip(y[BinNo == i], sigma, 10)[0]) for i in range(1, BinSize)]
    #yavg = [np.average(y[BinNo == i]) for i in range(1, BinSize)]
    ystd = [np.std(y[BinNo == i]) for i in range(1, BinSize)]
    N = [y[BinNo == i].shape[0] for i in range(1, BinSize)] 
    xavg = np.array(xavg)
    xstd = np.array(xstd)
    yavg = np.array(yavg)
    ystd = np.array(ystd)
    N = np.array(N)
    yavg = np.nan_to_num(yavg)
    ystd = np.nan_to_num(ystd)
    xstd = np.nan_to_num(xstd)
    N = np.nan_to_num(N)
    return xavg, xstd, yavg, ystd, N, BinNo

def AvgQBhuv(x, y, bin):
 x = array(x)
 y = array(y)
 yavg = []
 xavg = []
 ystd = []
 N = []
 xmax = x.max()
 if xmax < bin:
  xmax = bin + bin/2.
 for i in arange(bin/2., xmax, bin):
  ysub = y[where(abs(x - i) < bin / 2.0)]
  xsub = x[where(abs(x - i) < bin / 2.0)]
  print xsub
  print average(ysub)
  if isnan(average(ysub)):
#   yavg.append(0)
#   ystd.append(0)
#   xavg.append(i)
#   N.append(0)
   pass
  else:
   yavg.append(average(ysub))
   ystd.append(std(ysub) / 1.0)
   xavg.append(average(xsub))
   N.append(len(ysub))
 xavg = array(xavg)
 yavg = array(yavg)
 ystd = array(ystd)
 N = array(N)
 return xavg, yavg, ystd, N

def AvgQE(x, y, ye, bin, bintype=1, hardlimit=0, binmax=None):
    '''Average values of scatter plot'''
    def HelpMe(kk, BR, ii):
        if len(kk) > 0:
            return np.average(kk)
        elif len(kk) == 0:
            return (BR[ii-1] + BR[ii])/2.


    x = array(x)
    y = array(y)
    w = 1 / array(ye)**2.
    BinNo, BinsReturned = ReturnBins(x, bin, bintype=bintype, hardlimit=hardlimit, binmax=binmax)
    #print BinNo, BinsReturned
    #BinSize = np.max(BinNo)+1
    BinSize = len(BinsReturned)
    xavg = [HelpMe(x[BinNo == i], BinsReturned, i) for i in range(1, BinSize)]
    xstd = [np.std(x[BinNo == i]) for i in range(1, BinSize)]
    yavg = [np.average(y[BinNo == i], weights=w[BinNo == i]) for i in range(1, BinSize)]
    ystd = [np.sqrt(1/np.sum(w[BinNo == i])) for i in range(1, BinSize)]
    N = [y[BinNo == i].shape[0] for i in range(1, BinSize)]
    xavg = np.array(xavg)
    xstd = np.array(xstd)
    yavg = np.array(yavg)
    ystd = np.array(ystd)
    N = np.array(N)
    yavg = np.nan_to_num(yavg)
    ystd = np.nan_to_num(ystd)
    xstd = np.nan_to_num(xstd)
    N = np.nan_to_num(N)
    return xavg, xstd, yavg, ystd, N

def AvgQE_(x, y, yerr, bin):
 '''Average values of scatter plot with error'''
 x = array(x)
 y = array(y)
 yerr = array(yerr)
 yavg = []
 xavg = []
 ystd = []
 xstd = []
 N = [] 
 for i in arange(x.min() + bin/2., x.max(), bin):
  ysub = y[where(abs(x - i) < bin / 2.0)]
  xsub = x[where(abs(x - i) < bin / 2.0)]
  yesub = yerr[where(abs(x - i) < bin / 2.0)]
  yavg.append(average(ysub))
  ystd.append(sqrt(np.sum(yesub**2.)) / (1.0 * len(yesub)))
  xavg.append(average(xsub))
  xstd.append(std(xsub) / 1.0)
  N.append(len(ysub))
 return xavg, yavg, ystd, xstd, N

def AvgQEDisp(x, y, yerr, bin):
 '''Average values of scatter plot with error and return errors include the scatter'''
 if type(bin) is np.ndarray:
  binn = bin * 1.0
 else:
  binn = arange(x.min() + bin/2., x.max(), bin)
 bin = abs(binn[1] - binn[0])
 print bin
 x = array(x)
 y = array(y)
 yerr = array(yerr)
 yavg = []
 xavg = []
 ystd = []
 xstd = []
 N = [] 
 for i in binn:
  ysub = y[where(abs(x - i) < bin / 2.0)]
  xsub = x[where(abs(x - i) < bin / 2.0)]
  yesub = yerr[where(abs(x - i) < bin / 2.0)]
  yavg.append(average(ysub))
  ystd.append(sqrt(std(ysub)**2.0 + (sqrt(np.sum(yesub**2.)) / (1.0 * len(yesub)))**2.0))
  xavg.append(average(xsub))
  xstd.append(std(xsub) / 1.0)
  N.append(len(ysub))
 xavg = array(xavg)
 yavg = array(yavg)
 ystd = array(ystd)
 return xavg, yavg, ystd, xstd, N


def AvgB(b):
 '''Average of histogram bins'''
 return (b[1:] + b[:-1]) / 2.

def NormedErr(a, b, w):
 '''Error for normalized histogram assuming Poisson'''
 b = np.roll(b, -1) - b
 b = b[:-1]
 N = a * b * len(w)
 Nerr = np.sqrt(N)
 Nerr = Nerr / (b * len(w))
 return Nerr

def RaDegToHMS(ra):
    '''Convert to Ra deg to H:M:S'''
    ra = ra / 15.0
    ra1 = int(ra) 
    ra22 = (ra - ra1) * 60.0
    ra2 = int(ra22)
    ra3 = (ra22 - ra2) * 60.0
    return ra1, ra2, ra3

def DecDegToDMS(dec):
    '''Convert to Dec deg to D:M:S'''
    dec1 = int(dec)
    dec22 = abs(dec - dec1) * 60.0
    dec2 = int(dec22) 
    dec3 = (dec22 - dec2) * 60.0
    return dec1, dec2, dec3

def SpheCart1(R, ra, dec):
    '''Convert spherical coordinates to cartesian, given distance,ra,dec. ra and dec in degrees'''
    decd = -1.0 * (dec - 90.0) * pi / 180.0 #Make it to 0 to 180; dec is theta
    rad = ra * pi / 180.0 #ra is phi
    X = R * sin(decd) * cos(rad)
    Y = R * sin(decd) * sin(rad)
    Z = R * cos(decd)
    return X, Y, Z

def SpheCart(R, ra, dec):
    ra = ra * pi / 180.
    dec = dec * pi / 180.
    X = R * cos(dec) * cos(ra)
    Y = R * cos(dec) * sin(ra)
    Z = R * sin(dec)
    return X, Y, Z

def leastsqTmp(x,y, m, full_output=False):
     """find least-squares fit to y = a + bx

     inputs:
       x            sequence of independent variable data
       y            sequence of dependent variable data
       full_output  (optional) return dictionary of all results and stats

     outputs (not using full_output):
       a      y intercept of regression line
       b      coeffecient (slope) of regression line

     full output (using full_output):
       stats  dictionary containing statistics on fit
         key   value
         ---   -----------------------
         a     a in: y = a + bx
         b     a in: y = a + bx
         ap    a in: x = a + b'y
         bp    b in: x = a + b'y
         r2    correlation coeffecient
         var_x variance of x (sigma**2)
         var_y variance of y (sigma**2)
         cov   covariance
         SEa   standard error for a
         SEb   standard error for b
     """
     # notation from  http://mathworld.wolfram.com/CorrelationCoefficient.html
     # and http://mathworld.wolfram.com/LeastSquaresFitting.html

     x=np.asarray(x)
     y=np.asarray(y)
     m=np.asarray(m)
#     x = ma.masked_array(x, m)
#     y = ma.masked_array(y, m)
     n = len(x)
     assert n == len(y)
     mean_x = np.sum(x.astype(np.float))/n
     mean_y = np.sum(y.astype(np.float))/n
     SSxx = np.sum( (x-mean_x)**2 )
     SSyy = np.sum( (y-mean_y)**2 )
     SSxy = np.sum( x*y ) - n*mean_x*mean_y

     # y = a + b x
     # x = ap + bp y

     b = SSxy/SSxx
     bp = SSxy/SSyy

     a = mean_y - b*mean_x


     ap = mean_x - bp*mean_y

     s2 = (SSyy - b*SSxy)/(n-2)
     print s2
     s = math.sqrt(s2)

     SEa = s*math.sqrt( 1/n + mean_x**2/SSxx )
     SEb = s/math.sqrt(SSxx)
     if not full_output:
         return a, b, SEa, SEb

     stats = dict(
         r2 = b*bp,      # correlation coefficient
         var_x = SSxx/n, # variance of x (sigma**2)
         var_y = SSyy/n, # variance of y (sigma**2)
         cov = SSxy/n,   # covariance
         SEa = SEa,      # standard error for a
         SEb = SEb,      # standard error for b
         a = a,          # a in: y = a + bx
         b = b,          # b in: y = a + bx
         ap = ap,        # a' in: x = a' + b'y
         bp = bp,        # b' in: x = a' + b'y
         )
     return stats

def NumRatErr(N1, N2):
    '''Return the ratio N1/N2 and the error assuming N1 and N2 assuming they are  Poissonian'''
    if N1 > 0 and N2 > 0:
        f = N1 / (N2 * 1.0)
        df = f * np.sqrt(1 / (N1 * 1.0) + 1 / (N2 * 1.0))
    else:
        f = 0
        df = 0
    return f, df

def iter_old(x, xe, verbose=1):
    '''The average of a distribution'''
    xavg = average(x)
    N = []
    avgarr = []
    #sigma square
    #print np.std(x), np.average(xe) / 2.
    if len(x) == 1 or np.std(x) <= np.average(xe) / 2.:
        if verbose:
            print 'The individual errors are larger or the order of the scatter'
        sig = 0.0
        sig1 = [0]
        N = [len(x)]
        minN = 0
    else:
        sigsq = arange(0 * np.std(x)**2. / 5., 3 * np.std(x)**2. / 2., \
                       0.01 * np.std(x)**2.) 
        for i in sigsq:
            xavg = average(x, weights=1/(i + xe**2.))
            N.append(sum((x - xavg)**2. / (i + xe**2.)))
            avgarr.append(xavg)
        sig1, N, avgarr = np.sqrt(sigsq), np.array(N), np.array(avgarr)
        tmpN = N-x.shape[0]
        try:
            n1, n2 = np.argmax(tmpN[tmpN < 0]), \
                     np.argmin(tmpN[tmpN > 0])
            # print 'min sigmas ', sig1[tmpN < 0][n1], sig1[tmpN > 0][n2]
            # print 'Nearest N ', tmpN[tmpN < 0][n1], tmpN[tmpN > 0][n2]
            minN = tmpN[tmpN < 0][n1]
            # print np.abs(N-x.shape[0])[n1], np.abs(N-x.shape[0])[n2]
            #the systematic sigma
            sig = np.average([sig1[tmpN < 0][n1], sig1[tmpN > 0][n2]]) 
        except:
            if verbose:
                print 'No convergence. It seems the errors are larger/order \
                      of the scatter. Setting the systematic sigma = 0'
            sig = 0.0
            sig1 = [0]
            N = [len(x)]
            minN = 0
    #print 'Sig > ', sig
    xavg = np.average(x, weights=1/(xe**2. + sig**2.))
    sigma = np.sqrt(1 / np.sum(1/(sig**2. + xe**2.)))
    # return xavg, sigma, sig1, N, minN
    return xavg, sigma

def iter2(x, xe):
    """This should be done"""
    sigsqeff = xe**2.0
    N = x.shape[0] + 9999.0
    while np.abs(N - x.shape[0]) > 1e-1:
     xavg = average(x, weights=1/sigsqeff)
     N = sum((x - xavg)**2. / sigsqeff)
     print N, sigsqeff
     raw_input()
     sigsqeff = np.sum(x - xavg) / N
     sigsys = sigsqeff**2 - xe**2.
     print sigsys
     
def iter(x, xe, verbose=1):
    '''The average of a distribution'''
    w = 1/xe**2.
    xavg = np.average(x, weights=w)
    N = np.sum((x - xavg)**2. / xe**2.)
    #print 'Number ', N, len(x)
    if N <= len(x):
        sigma = np.sqrt(1 / np.sum(1/(xe**2.)))
        if verbose:
            print "The individual errors are larger"
        return xavg, sigma
    else:
        #print 'Hi', (N - len(x)), xe.min(), xe.max()
        syssqarr = arange(xe.min(), 5*std(x), xe.min())
        i = 0
        while (N - len(x)) > 0.01:
            #print syssqarr[i], N, len(x)
            syssq = syssqarr[i]
            w = 1/(xe**2. + syssq)
            xavg, sigma = np.average(x, weights=w), np.sqrt(1 / np.sum(w))
            N = np.sum((x - xavg)**2. / (xe**2.+syssq**2.))
            i += 1
        #print 'sys ', np.sqrt(syssq)
        return xavg, sigma 


def wstd(x,w):
    '''from http://stackoverflow.com/questions/2413522/weighted-std-in-numpy and http://en.wikipedia.org/wiki/Mean_square_weighted_deviation'''
    t = w.sum()
    return (((w*x**2).sum()*t-(w*x).sum()**2)/(t**2-(w**2).sum()))**.5

def weighted_avg_and_std(values, weights):
    """
    Returns the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    variance = np.dot(weights, (values-average)**2)/weights.sum()  # Fast and numerically precise
    return (average, np.sqrt(variance))


def AvgQIter(x, y, ye, bin, bintype=1, hardlimit=0, err='err'):
    '''Find the average and std within different bins using iter method'''
    x = np.asarray(x)
    y = np.asarray(y)
    ye = np.asarray(ye)
    xavg = []
    yavg = []
    xstd = []
    ystd = []
    N = []
    BinNo, BinsReturned = ReturnBins(x, bin, bintype=1, hardlimit=1, binmax=None)
    BinSize = np.max(BinNo) + 1
    for i in range(1, BinSize):
        xsub = x[BinNo == i]
        ysub = y[BinNo == i]
        yesub = ye[BinNo == i]
        if len(ysub) > 0:
            if err=='err':
                yav, sigma = iter(ysub, yesub)
            if err=='noerr':
                yav, sigma = np.average(ysub), np.std(ysub)
            xavg.append(np.average(xsub))
            xstd.append(np.std(xsub))
            yavg.append(yav)
            ystd.append(sigma)
            N.append(len(xsub))
        else:
            pass
    xavg = np.array(xavg)
    yavg = np.array(yavg) 
    xstd = np.array(xstd)
    ystd = np.array(ystd)
    N = np.array(N)
    return xavg, xstd, yavg, ystd, N


def Chi2LineFit(x, y, yerr):
    '''fit a line using chi2 minimization a + b*x. Return a, da, b and db'''
    def f(x, a, b): return a + b*x

    def chi2(a, b):
        c2 = 0.
        for i in range(len(x)):
            c2 += (f(x[i], a, b) - y[i])**2 / yerr[i]**2
        return c2
 
    m = minuit.Minuit(chi2)
    m.printMode = 0
    m.migrad()
    m.hesse()
    return m.values['a'], m.errors['a'], m.values['b'], m.errors['b']#, np.array(m.matrix())



def GenerateRandom(x, histbins=10, number=1):
    '''Given a distribution of x, it generate random value based on that'''

    pdf, bins = np.histogram(x, histbins)
    pdf = pdf / (pdf.sum() * 1.0)
    bins = (bins[1:] + bins[:-1]) / 2.0
    cdf = [np.sum(p for x,p in zip(bins, pdf) if x < i) for i in bins]
    simulated = np.array([np.max([i for r in [random.random()] for i,c in zip(bins, cdf) if c <= r]) for _ in range(number)])
    return simulated


