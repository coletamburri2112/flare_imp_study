from os.path import dirname, join as pjoin
import scipy.io as sio
from scipy.io import readsav
from scipy.io import loadmat
import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import matplotlib.animation as animat
import datetime
from scipy.ndimage import rotate
from astropy.convolution import convolve, Gaussian2DKernel, Tophat2DKernel
from astropy.modeling.models import Gaussian2D
from scipy.spatial.distance import cdist
from scipy.signal import fftconvolve
from itertools import product
import scipy.signal
import matplotlib.dates as mdates

def curve_length(curve):
    """ sum of Euclidean distances between points """
    return np.sum(np.sqrt(np.sum((curve[:-1] - curve[1:])**2,axis=1)))

def datenum_to_datetime(datenum):
    """
    Convert Matlab datenum into Python datetime.
    :param datenum: Date in datenum format
    :return:        Datetime object corresponding to datenum.
    """
    days = datenum % 1
    return datetime.datetime.fromordinal(int(datenum)) + datetime.timedelta(days=days) - datetime.timedelta(days=366)

def datenum(d):
    return 366 + d.toordinal() + (d - datetime.datetime.fromordinal(d.toordinal())).total_seconds()/(24*60*60)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def format_time():
    t = datetime.datetime.now()
    s = t.strftime('%Y-%m-%d %H:%M:%S.%f')
    return s[:-3]

def find_nearest_ind(array, value):
    array = np.asarray(array)
    idx = np.nanargmin(np.abs(array - value))
    return idx

def load_variables(bestflarefile, year, mo, day, sthr, stmin, arnum, xclnum, xcl):

    #load matlab file, get 304 light curves and start/peak/end times for flare
    best304 = sio.loadmat(bestflarefile)

    start304 = best304['starttimes_corr_more'][:,0]
    peak304 = best304['maxtimes_corr_more'][:,0]
    end304 = best304['endtimes_corr_more'][:,0]
    eventindices = best304['starttimes_corr_more'][:,1]
    times304 = best304['event_times_more']
    curves304 = best304['event_curves_more']

    sav_fname=("/Users/owner/Desktop/CU_Research/HMI_files/posfile"+str(year).zfill(4)+str(mo).zfill(2)+str(day).zfill(2)+"_"+str(sthr).zfill(2)+str(stmin).zfill(2)+"_"+str(arnum).zfill(5)+"_"+xcl+str(xclnum)+"_cut08_sat5000.00_brad.sav")
    sav_data = readsav(sav_fname)
    
    aia_cumul8 = sav_data.pos8
    last_cumul8 = aia_cumul8[-1,:,:]
    hmi_dat = sav_data.hmi
    last_mask = last_cumul8*hmi_dat

    aia_step8 = sav_data.inst_pos8

    return best304, start304, peak304, end304, eventindices, times304, curves304, aia_cumul8, aia_step8, last_cumul8, hmi_dat, last_mask

def pos_neg_masking(aia_cumul8, aia_step8, hmi_dat, last_mask):
    hmi_cumul_mask = np.zeros(np.shape(aia_cumul8))
    hmi_cumul_mask1 = np.zeros(np.shape(aia_cumul8))
    for i in range(len(aia_cumul8)):
        frame = np.squeeze(aia_cumul8[i,:,:])
        hmi_cumul_mask[i,:,:] = frame*hmi_dat

    for i in range(len(hmi_cumul_mask)):
        for j in range(len(hmi_cumul_mask[0])):
            for k in range(len(hmi_cumul_mask[1])):
                if hmi_cumul_mask[i,j,k] > 0:
                    hmi_cumul_mask1[i,j,k] = 1
                elif hmi_cumul_mask[i,j,k] < 0:
                    hmi_cumul_mask1[i,j,k] = -1
                else:
                    hmi_cumul_mask1[i,j,k] = 0

    hmi_step_mask = np.zeros(np.shape(aia_step8))
    hmi_step_mask1 = np.zeros(np.shape(aia_step8))

    for i in range(len(aia_step8)):
        frame = np.squeeze(aia_step8[i,:,:])
        hmi_step_mask[i,:,:] = frame*hmi_dat

    for i in range(len(hmi_step_mask)):
        for j in range(len(hmi_step_mask[0])):
            for k in range(len(hmi_step_mask[1])):
                if hmi_step_mask[i,j,k] > 0:
                    hmi_step_mask1[i,j,k] = 1
                elif hmi_step_mask[i,j,k] < 0:
                    hmi_step_mask1[i,j,k] = -1
                else:
                    hmi_step_mask1[i,j,k] = 0
    
    hmi_pos_mask_c = np.zeros(np.shape(hmi_dat))
    hmi_neg_mask_c = np.zeros(np.shape(hmi_dat))

    for i in range(len(hmi_dat)):
        for j in range(len(hmi_dat[0])):
            if last_mask[i,j] > 0:
                hmi_pos_mask_c[i,j] = 1
                hmi_neg_mask_c[i,j] = 0
            elif last_mask[i,j] < 0:
                hmi_pos_mask_c[i,j] = 0
                hmi_neg_mask_c[i,j] = -1
            else:
                hmi_pos_mask_c[i,j] = 0
                hmi_neg_mask_c[i,j] = 0
    
    return hmi_cumul_mask1, hmi_step_mask1, hmi_pos_mask_c, hmi_neg_mask_c

def spur_removal(hmi_neg_mask_c, hmi_pos_mask_c, pos_crit, neg_crit, pt_range):
    neg_rem = np.zeros(np.shape(hmi_neg_mask_c))
    pos_rem = np.zeros(np.shape(hmi_pos_mask_c))

    for i in range(len(neg_rem)):
        for j in range(len(neg_rem[0])):
            n = 0
            if hmi_neg_mask_c[i,j] == -1:
                for k in pt_range:
                    for l in pt_range:
                        if hmi_pos_mask_c[i+k,j-l] == 1:
                            n = n + 1
                if n > neg_crit:
                    neg_rem[i,j] = 0
                else:
                    neg_rem[i,j] = -1
            else:
                neg_rem[i,j] = 0
            
    for i in range(len(pos_rem)):
        for j in range(len(pos_rem[0])):
            n = 0
            if hmi_pos_mask_c[i,j] == 1:
                for k in pt_range:
                    for l in pt_range:
                        if hmi_neg_mask_c[i+k,j-l] == -1:
                            n = n + 1
                if n > pos_crit:
                    pos_rem[i,j] = 0
                else:
                    pos_rem[i,j] = 1
            else:
                pos_rem[i,j] = 0 
    
    return neg_rem, pos_rem

def gauss_conv(pos_rem, neg_rem, sigma = 10):
    
    gauss_kernel = Gaussian2DKernel(sigma)
    hmi_con_pos_c = convolve(pos_rem, gauss_kernel)
    hmi_con_neg_c = convolve(neg_rem, gauss_kernel)
    pil_mask_c = hmi_con_pos_c*hmi_con_neg_c
    
    return hmi_con_pos_c, hmi_con_neg_c, pil_mask_c

def pil_gen(pil_mask_c, hmi_dat, lx=800, ly=800):
    pil_mask_c = -1.0*pil_mask_c
    thresh = 0.05*np.amax(pil_mask_c)
    xc, yc = np.where(pil_mask_c > thresh)

    x = np.linspace(0, lx, lx)
    y = np.linspace(0, ly, ly)
    a, b, c, d, e = np.polyfit(y[yc],x[xc],4)

    ivs = y[yc]

    dvs = a*ivs**4 + b*ivs**3 + c*ivs**2 + d*ivs + e
    
    hmik = hmi_dat/1000
    
    return pil_mask_c, ivs, dvs, hmik

def pos_mask_sep(aia_step8, hmi_dat):
    aia8 = aia_step8
    aia8_pos = np.zeros(np.shape(aia8))
    aia8_neg = np.zeros(np.shape(aia8))

    for i in range(len(aia8)):
        for j in range(len(aia8[0])):
            for k in range(len(aia8[1])):
                if aia8[i,j,k] == 1 and hmi_dat[j,k] > 0:
                    aia8_pos[i,j,k] = 1
                elif aia8[i,j,k] == 1 and hmi_dat[j,k] < 0:
                    aia8_neg[i,j,k] = 1
                    
    return aia8_pos, aia8_neg

def separation(aia8, ivs, dvs, aia8_pos, aia8_neg):
    pil = list(zip(ivs,dvs))

    distpos_med = np.zeros(len(aia8))
    distneg_med = np.zeros(len(aia8))
    distpos_mean = np.zeros(len(aia8))
    distneg_mean = np.zeros(len(aia8))
    
    for i in range(len(aia8)):
        posframe = aia8_pos[i,:,:]
        negframe = aia8_neg[i,:,:]  
        xpos,ypos = np.where(posframe == 1)
        xneg,yneg = np.where(negframe == 1)
        pos_ops = list(zip(ypos,xpos))
        neg_ops = list(zip(yneg,xneg))
        if len(pos_ops) > 0:
            allpos = cdist(pos_ops,pil)
            # set the minimum for each pixel first
            allpos_min = np.amin(allpos,axis=1)
            distpos_med[i] = np.median(allpos_min)
            distpos_mean[i] = np.mean(allpos_min)
        if len(neg_ops) > 0:
            allneg = cdist(neg_ops,pil)
            allneg_min = np.amin(allneg,axis=1)
            distneg_med[i] = np.median(allneg_min)
            distneg_mean[i] = np.mean(allneg_min)
            
    return distpos_med, distpos_mean, distneg_med, distpos_mean
