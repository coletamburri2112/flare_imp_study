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

def spur_removal_sep(hmi_neg_mask_c, hmi_pos_mask_c, pos_crit=3, neg_crit=3, pt_range=[-2,-1,1,2]):
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

def mask_sep(aia_step8, hmi_dat):
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

def mask_elon(aia_cumul8, hmi_dat):
    aia8_a = aia_cumul8
    aia8_pos_2 = np.zeros(np.shape(aia8_a))
    aia8_neg_2 = np.zeros(np.shape(aia8_a))
    
    for i, j, k in np.ndindex(aia8_a.shape):
            if aia8_a[i,j,k] == 1 and hmi_dat[j,k] > 0:
                aia8_pos_2[i,j,k] = 1
            elif aia8_a[i,j,k] == 1 and hmi_dat[j,k] < 0:
                aia8_neg_2[i,j,k] = 1
                
    return aia8_pos_2, aia8_neg_2

def spur_removal_elon(aia8_pos_2, aia8_neg_2, pos_crit=3, neg_crit=3, pt_range=[-2,-1,1,2]):
    neg_rem1 = np.zeros(np.shape(aia8_pos_2))
    pos_rem1 = np.zeros(np.shape(aia8_neg_2))

    for i in range(len(neg_rem1)):
        for j in range(len(neg_rem1[0])-2):
            for k in range(len(neg_rem1[1])-2):
                n = 0
                if aia8_neg_2[i,j,k] == 1:
                    for l in pt_range:
                        for m in pt_range:
                            if aia8_neg_2[i,j+l,k+m] == 1:
                                n = n + 1
                    if (n > neg_crit):
                        neg_rem1[i,j,k] = 1
                    else:
                        neg_rem1[i,j,k] = 0
                else:
                    neg_rem1[i,j,k] = 0
            
    for i in range(len(pos_rem1)):
        for j in range(len(pos_rem1[0])-2):
            for k in range(len(pos_rem1[1])-2):
                n = 0
                if aia8_pos_2[i,j,k] == 1:
                    for l in pt_range:
                        for m in pt_range:
                            if aia8_pos_2[i,j+l,k+m] == 1:
                                n = n + 1
                    if (n > pos_crit):
                        pos_rem1[i,j,k] = 1
                    else:
                        pos_rem1[i,j,k] = 0
                else:
                    pos_rem1[i,j,k] = 0
    
    return neg_rem1, pos_rem1

def lim_pil(ivs, dvs):
    med_x = np.median(ivs)
    med_y = np.median(dvs)
    
    ivs_lim = []
    dvs_lim = []
    
    for i in range(len(ivs)):
        if not (ivs[i] < (med_x - 200)) and not (ivs[i] > (med_x + 200)):
            ivs_lim.append(ivs[i])
            dvs_lim.append(dvs[i])
            
    return ivs_lim, dvs_lim, med_x, med_y

def rib_lim_elon(aia8_pos_2, aia8_neg_2, pos_rem1, neg_rem1, med_x, med_y, ylim0_pos, ylim1_pos, ylim0_neg, ylim1_neg, xlim0_pos, xlim1_pos, xlim0_neg, xlim1_neg):
    aia_pos_rem = np.zeros(np.shape(aia8_pos_2))
    aia_neg_rem = np.zeros(np.shape(aia8_neg_2))
    
    for i in range(len(aia8_neg_2)):
        for j in range(ylim0_neg,ylim1_neg):
            for k in range(xlim0_neg,xlim1_neg):
                if neg_rem1[i,j,k] > 0:
                    aia_neg_rem[i,j,k] = 1
                        
    for i in range(len(aia8_pos_2)):
        for j in range(ylim0_pos,ylim1_pos):
            for k in range(xlim0_pos,xlim1_pos):
                if pos_rem1[i,j,k] > 0:
                    aia_pos_rem[i,j,k] = 1
                    
    return aia_pos_rem, aia_neg_rem

def find_rib_coordinates(aia_pos_rem, aia_neg_rem):
    lr_coord_pos = np.zeros([len(aia_pos_rem),4])        
    lr_coord_neg = np.zeros([len(aia_neg_rem),4])


    for i in range(len(aia_pos_rem)):
        left_x = 0
        left_y = 0
        right_x = 0
        right_y = 0
        for k in range(len(aia_pos_rem[1])):
            for j in range(len(aia_pos_rem[0])):
                if aia_pos_rem[i,j,k] == 1:
                    left_x = k
                    left_y = j
                    break
            if left_x != 0:
                break
        for k in range(len(aia_pos_rem[1])-1,0,-1):
            for j in range(len(aia_pos_rem[0])):
                if aia_pos_rem[i,j,k] == 1:
                    right_x = k
                    right_y = j
                    break
            if right_x != 0:
                break
        lr_coord_pos[i,:] = [left_x,left_y,right_x,right_y]

    for i in range(len(aia_neg_rem)):
        left_x = 0
        left_y = 0
        right_x = 0
        right_y = 0
        for k in range(len(aia_neg_rem[1])):
            for j in range(len(aia_neg_rem[0])):
                if aia_neg_rem[i,j,k] == 1:
                    left_x = k
                    left_y = j
                    break
            if left_x != 0:
                break
        for k in range(len(aia_neg_rem[1])-1,0,-1):
            for j in range(len(aia_neg_rem[0])):
                if aia_neg_rem[i,j,k] == 1:
                    right_x = k
                    right_y = j
                    break
            if right_x != 0:
                break
        lr_coord_neg[i,:] = [left_x,left_y,right_x,right_y]
        
    return lr_coord_neg, lr_coord_pos

def sort_pil(ivs_lim, dvs_lim):
    pil_sort = np.vstack((ivs_lim, dvs_lim)).T
    sortedpil = pil_sort[pil_sort[:,0].argsort()]
    ivs_sort = sortedpil[:,0]
    dvs_sort = sortedpil[:,1]
    
    return ivs_sort, dvs_sort, sortedpil


def elon_dist_arrays(lr_coord_pos, lr_coord_neg, ivs_lim, dvs_lim, ivs_sort, dvs_sort):
    left_pil_dist_pos = np.zeros([len(lr_coord_pos),len(ivs_sort)])
    right_pil_dist_pos = np.zeros([len(lr_coord_pos),len(ivs_sort)])
    left_pil_dist_neg = np.zeros([len(lr_coord_neg),len(ivs_sort)])
    right_pil_dist_neg = np.zeros([len(lr_coord_neg),len(ivs_sort)])
    pil_left_near_neg = np.zeros([len(left_pil_dist_neg),3])
    pil_right_near_neg = np.zeros([len(right_pil_dist_neg),3])
    pil_left_near_pos = np.zeros([len(left_pil_dist_pos),3])
    pil_right_near_pos = np.zeros([len(right_pil_dist_pos),3])
    
    for i in range(len(lr_coord_pos)):
        left_x,left_y,right_x,right_y = lr_coord_pos[i]
        for j in range(len(ivs_sort)):
            left_pil_dist_pos[i,j] = np.sqrt((left_x-ivs_sort[j])**2+(left_y-dvs_sort[j])**2)
            right_pil_dist_pos[i,j] = np.sqrt((right_x-ivs_sort[j])**2+(right_y-dvs_sort[j])**2)
    
    for i in range(len(left_pil_dist_pos)):
        ind = np.where(left_pil_dist_pos[i]==np.min(left_pil_dist_pos[i]))
        pil_left_near_pos[i,:] = [ivs_lim[ind[0][0]],dvs_sort[ind[0][0]],ind[0][0]]
    
    for j in range(len(right_pil_dist_pos)):
        ind = np.where(right_pil_dist_pos[j]==np.min(right_pil_dist_pos[j]))
        pil_right_near_pos[j,:] = [ivs_lim[ind[0][0]],dvs_sort[ind[0][0]],ind[0][0]]
                    

    for i in range(len(lr_coord_neg)):
        left_x,left_y,right_x,right_y = lr_coord_neg[i]
        for j in range(len(ivs_sort)):
            left_pil_dist_neg[i,j] = np.sqrt((left_x-ivs_sort[j])**2+(left_y-dvs_sort[j])**2)
            right_pil_dist_neg[i,j] = np.sqrt((right_x-ivs_sort[j])**2+(right_y-dvs_sort[j])**2)
    
    for i in range(len(left_pil_dist_neg)):
        ind = np.where(left_pil_dist_neg[i]==np.min(left_pil_dist_neg[i]))
        pil_left_near_neg[i,:] = [ivs_lim[ind[0][0]],dvs_sort[ind[0][0]],ind[0][0]]
    
    for j in range(len(right_pil_dist_neg)):
        ind = np.where(right_pil_dist_neg[j]==np.min(right_pil_dist_neg[j]))
        pil_right_near_neg[j,:] = [ivs_lim[ind[0][0]],dvs_sort[ind[0][0]],ind[0][0]]               
        
    return pil_right_near_pos, pil_left_near_pos, pil_right_near_neg, pil_left_near_neg

def elongation(pil_right_near_pos, pil_left_near_pos, pil_right_near_neg, pil_left_near_neg, sortedpil):
    lens_pos = []
    lens_neg = []
    
    for i in range(len(pil_right_near_pos)):
        leftin = int(pil_left_near_pos[i,2])
        rightin = int(pil_right_near_pos[i,2])
        curvei = sortedpil[leftin:rightin,:]
        lens_pos.append(curve_length(curvei))
    
    for i in range(len(pil_right_near_neg)):
        leftin = int(pil_left_near_neg[i,2])
        rightin = int(pil_right_near_neg[i,2])
        curvei = sortedpil[leftin:rightin,:]
        lens_neg.append(curve_length(curvei))
        
    return lens_pos, lens_neg

def convert_to_Mm(lens_pos, dist_pos, lens_neg, dist_neg, conv_f):
    lens_pos_Mm = np.zeros(np.shape(lens_pos))
    lens_neg_Mm = np.zeros(np.shape(lens_neg))
    distpos_Mm = np.zeros(np.shape(dist_pos))
    distneg_Mm = np.zeros(np.shape(dist_neg))
    
    
    for i in range(len(lens_pos)):
        lens_pos_Mm[i] = lens_pos[i]*conv_f
        lens_neg_Mm[i] = lens_neg[i]*conv_f
        distpos_Mm[i] = dist_pos[i]*conv_f
        distneg_Mm[i] = dist_pos[i]*conv_f
        
    dneg_len = np.diff(lens_neg_Mm)/24.
    dpos_len = np.diff(lens_pos_Mm)/24.
    dneg_dist = np.diff(distneg_Mm)/24.
    dpos_dist = np.diff(distpos_Mm)/24.
    
    return lens_pos_Mm, lens_neg_Mm, distpos_Mm, distneg_Mm, dneg_len, dpos_len, dneg_dist, dpos_dist



