from os.path import dirname, join as pjoin
import scipy.io as sio
from scipy.io import readsav
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animat
import datetime
from scipy.spatial.distance import cdist
import scipy.signal
import matplotlib.dates as mdates
from astropy.convolution import convolve, Gaussian2DKernel

def conv_facts():
    """
    Conversion factors for images.

    Returns
    -------
    X : ARR
        Meshgrid of x values for image coordinates.
    Y : TYPE
        Meshgrid of y values for image coordinates.
    conv_f : TYPE
        Conversion factor between pixels and megameters.
    xarr_Mm : TYPE
        x-coordinates, in megameters.
    yarr_Mm : TYPE
        y-coordinates, in megameters.

    """
    pix_to_arcsec = 0.6
    arcsec_to_radians = 1/206265
    radians_to_Mm = 149598
    
    conv_f = pix_to_arcsec*arcsec_to_radians*radians_to_Mm
    
    xarr_Mm = np.zeros(800)
    yarr_Mm = np.zeros(800)
    
    for i in range(800):
        xarr_Mm[i] = (i-400)*conv_f
        yarr_Mm[i] = (i-400)*conv_f
        
    X,Y = np.meshgrid(xarr_Mm,yarr_Mm)
        
    return X, Y, conv_f, xarr_Mm, yarr_Mm
    
def exponential(x,a,b):
    """
    Defines exponential function.

    Parameters
    ----------
    x : float
        Input x value for function.
    a : TYPE
        Amplitude of exponential function.
    b : TYPE
        Second parameter of exponential function.

    Returns
    -------
    float
        Output of exponential function.

    """
    return a * np.exp(b * x )

def exponential_neg(x,a,b):
    """
    Negative amplitude exponential function.

    Parameters
    ----------
    x : float
        Input x value for function.
    a : TYPE
        Amplitude of exponential function.
    b : TYPE
        Second parameter of exponential function.

    Returns
    -------
    float
        Output of exponential function.

    """
    return -a * np.exp(b * x)

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
    ret = datetime.datetime.fromordinal(int(datenum)) + \
        datetime.timedelta(days=days) - datetime.timedelta(days=366)

    return ret

def datenum(d):
    """
    Convert from ordinal to datenum.
    """
    return 366 + d.toordinal() + (d - datetime.datetime.fromordinal(d.toordinal())).total_seconds()/(24*60*60)

def find_nearest(array, value):
    """
    Find nearest value in array to a value.

    Parameters
    ----------
    array : arr
        Array of values to search through.
    value : float
        Value to find the nearest element in array closest to.

    Returns
    -------
    float
        Nearest value in array to "value"

    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def format_time():
    """
    Time formatter.

    Returns
    -------
    string
        Formating for times.

    """
    t = datetime.datetime.now()
    s = t.strftime('%Y-%m-%d %H:%M:%S.%f')
    return s[:-3]

def find_nearest_ind(array, value):
    """
    Find index of element in array closest to value.

    Parameters
    ----------
    array : arr
        Array of values to search through.
    value : float
        Value to find the nearest element in array closest to.

    Returns
    -------
    idx: int
        Index of nearest value in array to "value"
    """
        
    array = np.asarray(array)
    idx = np.nanargmin(np.abs(array - value))
    return idx

def load_variables(bestflarefile, year, mo, day, sthr, stmin, arnum, xclnum,
                   xcl):
    """
    Load variables from HMI and AIA files.

    Parameters
    ----------
    bestflarefile : string
        Path to file containing information about the best-performing flares.
    year : int
        Year of event.
    mo : TYPE
        Month of event.
    day : TYPE
        Day of event.
    sthr : TYPE
        Hour of event start
    stmin : TYPE
        Minute of event start.
    arnum : TYPE
        Active region number.
    xclnum : TYPE
        X-ray class number.
    xcl : TYPE
        X-ray class.

    Returns
    -------
    sav_data_aia : AttrDict
        Dictionary containing all of the saved parameters in the AIA file.
    sav_data : AttrDict
        Dictionary containing all of the saved parameters in the HMI file.
    best304 : dict
        Dictionary containing the SDO/EVE 304 Angstrom data of the 
        best-performing flares in ribbonDB.
    start304 : arr
        Array containing the start times for the flares in best304.
    peak304 : arr
        Array containing the peak times for the flares in best304.
    end304 : arr
        Array containing the end times for the flares in best304.
    eventindices : arr
        Indices of best flares in best304.
    times304 : arr
        Time points for all flares in best304.
    curves304 : arr
        Light curves for all flares in best304.
    aia_cumul8 : arr
        Cumulative ribbon masks from AIA.
    aia_step8 : arr
        Instantaneous ribbon masks from AIA
    last_cumul8 : arr
        The last image in the cumulative mask array.
    hmi_dat : arr
        HMI image prior to the flare, assumed to be the same configuration 
        throughout the flare.
    last_mask : arr
        The last ribbon mask, multiplied by the HMI image for polarity.

    """
    data_dir=pjoin(dirname(sio.__file__),'tests','data')
    #load matlab file, get 304 light curves and start/peak/end times for flare
    best304 = sio.loadmat(bestflarefile)

    start304 = best304['starttimes_corr_more'][:,0]
    peak304 = best304['maxtimes_corr_more'][:,0]
    end304 = best304['endtimes_corr_more'][:,0]
    eventindices = best304['starttimes_corr_more'][:,1]
    times304 = best304['event_times_more']
    curves304 = best304['event_curves_more']

    sav_fname_aia=pjoin(data_dir,"/Users/owner/Desktop/Final_Selection/AIA_Files/aia1600blos"+str(year).zfill(4)+str(mo).zfill(2)+str(day).zfill(2)+"_"+str(sthr).zfill(2)+str(stmin).zfill(2)+"_"+str(arnum).zfill(5)+"_"+xcl+str(xclnum)+".sav")
    sav_data_aia = readsav(sav_fname_aia)
    sav_fname=("/Users/owner/Desktop/CU_Research/HMI_files/posfile"+str(year).zfill(4)+str(mo).zfill(2)+str(day).zfill(2)+"_"+str(sthr).zfill(2)+str(stmin).zfill(2)+"_"+str(arnum).zfill(5)+"_"+xcl+str(xclnum)+"_cut08_sat5000.00_brad.sav")
    sav_data = readsav(sav_fname)
    
    aia_cumul8 = sav_data.pos8
    last_cumul8 = aia_cumul8[-1,:,:]
    hmi_dat = sav_data.hmi
    last_mask = last_cumul8*hmi_dat

    aia_step8 = sav_data.inst_pos8

    return sav_data_aia, sav_data, best304, start304, peak304, end304, \
            eventindices, times304, curves304, aia_cumul8, aia_step8, \
            last_cumul8, hmi_dat, last_mask

def pos_neg_masking(aia_cumul8, aia_step8, hmi_dat, last_mask):
    """
    Masking of positive and negative ribbons according to HMI polarity.

    Parameters
    ----------
    aia_cumul8 : arr
        Cumulative ribbon masks.
    aia_step8 : arr
        Instantaneous ribbon masks.
    hmi_dat : arr
        HMI image prior to the flare, assumed to be the same configuration 
        throughout the flare.        
    last_mask : arr
        The last ribbon mask, multiplied by the HMI image for polarity.

    Returns
    -------
    hmi_cumul_mask1 : arr
        Cumulative magnetic field strength masking estimates for all flare
        images.
    hmi_step_mask1 : arr
        Instantaneous magnetic field strength masking estimates for all flare
        images.
    hmi_pos_mask_c : arr
        Single-frame mask for negative HMI magnetic field, populated with 1. 
    hmi_neg_mask_c : arr
        Single-frame mask for negative HMI magnetic field, populated with -1. 

    """
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

def spur_removal_sep(hmi_neg_mask_c, hmi_pos_mask_c, pos_crit=3, neg_crit=3,
                     pt_range=[-2,-1,1,2]):
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

def separation(aia_step8, ivs, dvs, aia8_pos, aia8_neg):
    pil = list(zip(ivs,dvs))

    distpos_med = np.zeros(len(aia_step8))
    distneg_med = np.zeros(len(aia_step8))
    distpos_mean = np.zeros(len(aia_step8))
    distneg_mean = np.zeros(len(aia_step8))
    
    for i in range(len(aia_step8)):
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
        distneg_Mm[i] = dist_neg[i]*conv_f
        
    dneg_len = np.diff(lens_neg_Mm)/24.
    dpos_len = np.diff(lens_pos_Mm)/24.
    dneg_dist = np.diff(distneg_Mm)/24.
    dpos_dist = np.diff(distpos_Mm)/24.
    
    return lens_pos_Mm, lens_neg_Mm, distpos_Mm, distneg_Mm, dneg_len, dpos_len, dneg_dist, dpos_dist

def prep_304_parameters(sav_data_aia, sav_data, eventindices, flnum, start304, peak304, end304, times304, curves304):
    xlo = sav_data_aia.x1los
    xhi = sav_data_aia.x2los
    ylo = sav_data_aia.y1los
    yhi = sav_data_aia.y2los

    aiadat = sav_data_aia.aia1600
    time = sav_data.tim
    
    nt = len(time)
    nx = aiadat.shape[1]
    ny = aiadat.shape[2]
    t1=str(sav_data.tim[0])
    t2=str(sav_data.tim[-1])
    tst=float(t1[14:15:1])+(float(t1[17:18:1])/60)+(float(t1[20:24:1])/3600)
    tend=float(t2[14:15:1])+(float(t2[17:18:1])/60)+(float(t2[20:24:1])/3600)
    times=np.linspace(tst,tend,nt)

    x = np.linspace(xlo,xhi,nx)
    y = np.linspace(ylo,yhi,ny)
    x,y=np.meshgrid(x,y)
    
    times1600 = np.empty(nt,dtype=datetime.datetime)
    sum1600 = np.empty(nt)
    dn1600 = np.empty(nt)
    
    for i in range(nt):
        timechoi = str(sav_data.tim[i])
        times1600[i] = datetime.datetime.strptime(timechoi[2:21], '20%y-%m-%dT%H:%M:%S')
        dn1600[i] = datenum(times1600[i])
        timestep=aiadat[i,:,:]
        sum1600[i]=timestep.sum()
        
    ind = (np.isclose(eventindices,flnum))
    index = np.where(ind)[0][0]
    
    curve304 = curves304[index]
    time304 = times304[index]
    
    #integrate over all pixels in 1600A line
    for i in range(nt):
        timestep=aiadat[i,:,:]
        sum1600[i]=timestep.sum()
        
    startin = np.where(dn1600==find_nearest(dn1600,start304[ind][0]))
    peakin = np.where(dn1600==find_nearest(dn1600,peak304[ind][0]))
    endin = np.where(dn1600==find_nearest(dn1600,end304[ind][0]))
    
    for i in range(nt):
        timechoi = str(sav_data.tim[i])
        times1600[i] = datetime.datetime.strptime(timechoi[2:21], '20%y-%m-%dT%H:%M:%S')
        
    s304 = find_nearest_ind(time304,min(dn1600))
    e304 = find_nearest_ind(time304,max(dn1600))
    filter_304 = scipy.signal.medfilt(curve304,kernel_size=5)

    med304 = np.median(curve304)
    std304 = np.std(curve304)
    for i in range(len(curve304)):
        if curve304[i] < 0.54:
            curve304[i]='NaN'
        
    timelab = np.empty(nt)
    
    timelabs = range(0,24*len(times),24)
    
    for i in range(len(timelabs)):
        timelab[i] = timelabs[i]/60
            
    return startin, peakin, endin, times, s304, e304, filter_304, med304, std304, timelab, aiadat, nt, dn1600, time304, times1600
    
    
def img_mask(aia8_pos, aia8_neg, aiadat, nt):
    
    # positive and negative masks onto 1600
    #aia8_neg and aia8_pos are the masks for each frame
    
    posrib = np.zeros(np.shape(aia8_pos))
    negrib = np.zeros(np.shape(aia8_neg))
    
    for i in range(len(aia8_pos)):
        posrib[i,:,:] = aia8_pos[i,:,:]*aiadat[i,:,:]
        
    for j in range(len(aia8_neg)):
        negrib[j,:,:] = aia8_neg[j,:,:]*aiadat[j,:,:]
        
    pos1600 = np.empty(nt)
    neg1600 = np.empty(nt)
    
    for i in range(nt):
        timesteppos = posrib[i,:,:]
        pos1600[i]=timesteppos.sum()
        timestepneg = negrib[i,:,:]
        neg1600[i]=timestepneg.sum()
        
        
    return posrib, negrib, pos1600, neg1600

def load_from_file(flnum, pick = True):
    ev = np.load(flnum,allow_pickle=pick)

    dt1600 = ev['dt1600']
    pos1600 = ev['pos1600']
    neg1600 = ev['neg1600']
    time304 = ev['time304']
    filter_304 = ev['filter_304']
    distpos_Mm = ev['distpos_Mm']
    distneg_Mm = ev['distneg_Mm']
    lens_pos_Mm = ev['lens_pos_Mm']
    lens_neg_Mm = ev['lens_neg_Mm']
    ivs = ev['ivs']
    dvs = ev['dvs']
        
    return dt1600, pos1600, neg1600, time304, filter_304, distpos_Mm, distneg_Mm, lens_pos_Mm, lens_neg_Mm, ivs, dvs

def elon_periods(dpos_len, dneg_len):
    elonfiltpos = dpos_len
    elonfiltneg = dneg_len
    elonperiod_start_pos = []
    elonperiod_end_pos = []
    elonperiod_start_neg = []
    elonperiod_end_neg = []
    n = 0
    m = 0
    zer_n = 0
    zer_m = 0
    
    for i in range(len(elonfiltpos)):
        if elonfiltpos[i] > 0:

            n += 1

            if n == 1:
                time = i
            if n > 1 and time not in elonperiod_start_pos:

                elonperiod_start_pos.append(time)
        elif elonfiltpos[i] <= 0:
            
            if n > 1:
                zer_n += 1
                if zer_n > 2:
                    elonperiod_end_pos.append(i)
                    n = 0
                    zer_n = 0
            else:
                n = 0
                continue
                
    for j in range(len(elonfiltneg)):
        if elonfiltneg[j] > 0:
            m += 1
            if m == 1:
                time = j
            if m > 1 and time not in elonperiod_start_neg:
                elonperiod_start_neg.append(time)
        elif elonfiltneg[j] <= 0:
            
            if m > 1:
                zer_m += 1
                if zer_m > 2:
                    elonperiod_end_neg.append(j)
                    m = 0
                    zer_m = 0
            elif zer_m > 1:
                m = 0
                continue
    
    elonperiod_start_pos = list(set(elonperiod_start_pos))
    elonperiod_end_pos = list(set(elonperiod_end_pos))
    elonperiod_start_neg = list(set(elonperiod_start_neg))
    elonperiod_end_neg = list(set(elonperiod_end_neg))
    
    elonperiod_start_pos.sort()
    elonperiod_end_pos.sort()
    elonperiod_start_neg.sort()
    elonperiod_end_neg.sort()
    
    return elonperiod_start_pos, elonperiod_end_pos, elonperiod_start_neg, elonperiod_end_neg

def sep_periods(dpos_dist, dneg_dist, kernel_size=3):
    sepfiltpos = scipy.signal.medfilt(dpos_dist,kernel_size=3)
    sepfiltneg = scipy.signal.medfilt(dneg_dist,kernel_size=3)
    
    sepperiod_start_pos = []
    sepperiod_end_pos = []
    sepperiod_start_neg = []
    sepperiod_end_neg = []
    n = 0
    m = 0
    for i in range(20,len(sepfiltpos)):
        if sepfiltpos[i] > 0:
            n += 1
            if n == 1:
                time = i
            if n > 3 and time not in sepperiod_start_pos:
                sepperiod_start_pos.append(time)
        elif sepfiltpos[i] <= 0:
            if n > 3:
                sepperiod_end_pos.append(i)
                n = 0
            else:
                n = 0
                continue
                
    for i in range(20,len(sepfiltneg)):
        if sepfiltneg[i] > 0:
            m += 1
            if m == 1:
                time = i
            if m > 3 and time not in sepperiod_start_neg:
                sepperiod_start_neg.append(time)
        elif sepfiltneg[i] <= 0:
            if m > 3:
                sepperiod_end_neg.append(i)
                m = 0
            else:
                m = 0
                continue
            
    sepperiod_start_pos = list(set(sepperiod_start_pos))
    sepperiod_end_pos = list(set(sepperiod_end_pos))
    sepperiod_start_neg = list(set(sepperiod_start_neg))
    sepperiod_end_neg = list(set(sepperiod_end_neg))
    
    sepperiod_start_pos.sort()
    sepperiod_end_pos.sort()
    sepperiod_start_neg.sort()
    sepperiod_end_neg.sort()
            
    return sepperiod_start_pos, sepperiod_end_pos, sepperiod_start_neg, sepperiod_end_neg

def prep_times(dn1600, time304):
    dt1600 =[]
    dt304 = []
    for i in range(len(dn1600)):
        dt1600.append(datenum_to_datetime(dn1600[i]))
    
        
    for i in range(len(time304)):
        if np.isnan(time304[i]):
            dt304.append(datenum_to_datetime(time304[0]))
        else:
            dt304.append(datenum_to_datetime(time304[i]))
            
    return dt1600, dt304

# plotting routines

def lc_plot(times, nt, time304, filter_304, s304, e304, dn1600, pos1600, neg1600,
            lens_pos_Mm, lens_neg_Mm, distpos_Mm, distneg_Mm, aiadat, hmi_cumul_mask1,
            dt304, timelab, conv_f, ivs, dvs, year, mo, day, arnum, xcl, xclnum,
            X, Y, xarr_Mm, yarr_Mm, dt1600, flag = 1):
        
    min304=min(filter_304[s304:e304])
    max304=max(filter_304[s304:e304])
    minpos1600=min(pos1600)
    maxpos1600=max(pos1600)
    minneg1600=min(neg1600)
    maxneg1600=max(neg1600)
    
    norm304=(filter_304-min304)/(max304-min304)
    normpos1600=(pos1600-minpos1600)/(maxpos1600-minpos1600)
    normneg1600=(neg1600-minneg1600)/(maxneg1600-minneg1600)
    scalefac = max(pos1600)/max(neg1600)
    #plot 304 line, 1600 line, and figure
    fig= plt.figure(figsize=(25,12))
    
    gs = fig.add_gridspec(9,9)
    ax1 = fig.add_subplot(gs[:, 5:])
    ax2 = fig.add_subplot(gs[0:4, 0:4])
    ax0 = fig.add_subplot(gs[5:, 0:4])
    lns1 = ax0.plot(dn1600[1:],lens_pos_Mm[1:],'-+',c='red',markersize=10,label='Pos. Elongation')
    lns2 = ax0.plot(dn1600[1:],lens_neg_Mm[1:],'-+',c='blue',markersize=10,label='Neg. Elongation')
    ax5 = ax0.twinx()
    ax5.cla()
    lns4 = ax5
    lns5 = ax0.plot(dt1600[25:],distpos_Mm[25:],'-.',c='red',markersize=10,label='Pos. Separation')
    ax0.plot(dt1600[25:],distneg_Mm[25:],'-.',c='blue',markersize=10,label='Neg. Separation')
    
    col1 = ax1.pcolormesh(X,Y,np.log10(aiadat[0,:,:]),cmap='pink',shading='auto')
    col2 = ax1.contour(X,Y,hmi_cumul_mask1[0,:,:],cmap='seismic')
    
    lc304=ax2.plot(dt304,norm304,color='black',linewidth=1,label='Norm. 304$\AA$ Light Curve')
    ax3=ax2.twinx()
    lc1600=ax3.plot(dt1600,normpos1600,linewidth=3,color='red',label='Norm. 1600$\AA$ Light Curve, +')
    lc1600=ax3.plot(dt1600,normneg1600,linewidth=3,color='blue',label='Norm. 1600$\AA$ Light Curve, +')
    ax1.set_title(str(year)+"-"+str(mo)+"-"+str(day)+", AR"+str(arnum)+"\n"+xcl+str(xclnum)+" Class Flare\n",font='Times New Roman',fontsize=25,)#+"I = "+impulsivity+" mW/m$^2$/s")
    ax2.set_title('304$\AA$ and 1600$\AA$ Light Curves',fontsize=25,)
    
    ax0.set_title('Ribbon Separation and Elongation',fontsize=25,)
    ax0.legend(fontsize=15)
    ax0.grid()
    ax2.set_xlim([dn1600[0],dn1600[-1]])
    ax3.set_xlim([dn1600[0],dn1600[-1]])
    ax2.set_xticks([dn1600[0],dn1600[30],dn1600[60],dn1600[90],dn1600[120]])
    ax2.set_xticklabels([timelab[0],timelab[30],timelab[60],timelab[90],timelab[120]])
    ax3.set_xticks([dn1600[0],dn1600[30],dn1600[60],dn1600[90],dn1600[120]])
    ax3.set_xticklabels([timelab[0],timelab[30],timelab[60],timelab[90],timelab[120]])
    ax0.set_xticks([timelab[0],timelab[30],timelab[60],timelab[90],timelab[120]])
    ax0.set_xticklabels([timelab[0],timelab[30],timelab[60],timelab[90],timelab[120]])
    ax0.set_xlim([timelab[0],timelab[-1]])
    ax1.scatter(ivs,dvs, color = 'k', s = 1)
    lines, labels = ax0.get_legend_handles_labels()
    lines2, labels2 = ax5.get_legend_handles_labels()
    ax0.legend(lines + lines2, labels + labels2)
    
    ax5.set_ylim([0,140*conv_f])
    
    def animate(t): 
        ax1.cla()
        ax2.cla()
        ax0.cla()
        ax5 = ax0.twinx()
        ax5.cla()
        col1 = ax1.pcolormesh(X,Y,np.log10(aiadat[t,:,:]),cmap='pink',shading='auto')
        col2 = ax1.contour(X,Y,hmi_cumul_mask1[t,:,:],cmap = 'seismic')
        ax1.set_xlabel('Horizontal Distance from Image Center [Mm]',fontsize=15)
        ax1.set_ylabel('Vertical Distance from Image Center [Mm]',fontsize=15)
        sep = ax0.plot(dt1600[25:],distpos_Mm[25:],'-.',c='red',markersize=10,label='Pos. Separation')
        sep2 = ax0.plot(dt1600[25:],distneg_Mm[25:],'-.',c='blue',markersize=10,label='Neg. Separation')
        ax1.scatter((ivs-400)*conv_f,(dvs-400)*conv_f, color = 'k', s = 1)
        elon = ax5.plot(dt1600[1:],lens_pos_Mm[1:],'-+',c='red',markersize=10,label='Pos. Elongation')
        elon2 = ax5.plot(dt1600[1:],lens_neg_Mm[1:],'-+',c='blue',markersize=10,label='Neg. Elongation')
        ax1.set_xlim([-250*conv_f,250*conv_f])
        ax1.set_ylim([-250*conv_f,250*conv_f])
        lc304=ax2.plot(dt304,norm304,'-x',color='black',linewidth=1,label='304$\AA$')
        ax3=ax2.twinx()
        lc1600=ax3.plot(dt1600,normpos1600,linewidth=3,color='red',label='1600$\AA$, +')
        lc1600=ax3.plot(dt1600,normneg1600,linewidth=3,color='blue',label='1600$\AA$, -')
        ax2.set_xlim([dt1600[0],dt1600[-1]])
        ax2.set_ylim([-0.05,1.05])
        ax3.set_ylim([-0.05,1.05])
    
        myFmt = mdates.DateFormatter('%H:%M')
        ax2.xaxis.set_major_formatter(myFmt)
        ax3.xaxis.set_major_formatter(myFmt)
        ax0.xaxis.set_major_formatter(myFmt)
        ax5.xaxis.set_major_formatter(myFmt)
        textstr = '1600$\AA$ +/- Factor: '+str(round(scalefac,3))
        ax2.text(2*(max(dt1600)-min(dt1600))/5 + min(dt1600),0.1,textstr,fontsize=12,bbox=dict(boxstyle="square", facecolor="white",ec="k", lw=1,pad=0.3))
        ax2.set_xlabel(['Time since 00:00 UT [min], '+year+'-'+mo+'-'+day],fontsize=15)
        ax2.set_xlabel(['Time since 00:00 UT [min], '+year+'-'+mo+'-'+day],fontsize=15)
        ax2.set_ylabel('Norm. Integ. Count, 1600$\AA$',color='purple',fontsize=15)
    
        lines, labels = ax2.get_legend_handles_labels()
        lines2, labels2 = ax3.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2,loc='lower right')
        ax2.grid(linestyle='dashed')
        ax3.grid(linestyle='dashdot')
        ax2.axvline(dt1600[t],linewidth=4,color='black')
        ax0.axvline(dt1600[t],linewidth=4,color='black')
        ax0.axvline(dt1600[t],linewidth=4,color='black')
        ax1.set_title(str(year)+"-"+str(mo)+"-"+str(day)+", AR"+str(arnum)+", "+xcl+str(xclnum)+" Class Flare",fontsize=25)#"+I = "+impulsivity+" mW/m$^2$/s")
        ax2.set_title('304$\AA$ and 1600$\AA$ Light Curves',fontsize=25)
        ax0.set_xlim([dt1600[0],dt1600[-1]])
        ax0.set_xlabel(['Time since 00:00 UT [min], '+year+'-'+mo+'-'+day],fontsize=15)
        ax0.set_ylabel('Separation [Mm]',fontsize=15)
        ax5.set_ylabel('Elongation [Mm]',fontsize=15)
        ax0.set_title('Ribbon Separation and Elongation',fontsize=25,)
        ax0.legend(fontsize=15)
        ax0.grid()
        ax1.text(57,95,str(dt1600[t].hour).zfill(2)+':'+str(dt1600[t].minute).zfill(2)+'.'+str(dt1600[t].second).zfill(2)+' UT',fontsize=20,bbox=dict(boxstyle="square", facecolor="white",ec="k", lw=1,pad=0.3))
        
        lines, labels = ax0.get_legend_handles_labels()
        lines2, labels2 = ax5.get_legend_handles_labels()
        ax0.legend(lines + lines2, labels + labels2,loc='lower right')
    
        ax5.set_ylim([0,140*conv_f])
        return col1, col2, lc304, lc1600, sep, sep2, elon, elon2
    
    if flag == 1:
        ani = animat.FuncAnimation(fig, animate, frames=np.shape(aiadat)[0], interval=20, repeat_delay=0)
    elif flag == 0:
        ani = animat.FuncAnimation(fig, animate, frames=5, interval=20, repeat_delay=0)
    
    ani.save(['/Users/owner/Desktop/'+mo+'_'+day+'_'+year+'.gif'],dpi=200)

    return None

def mask_plotting(X, Y, pos_rem, neg_rem, xarr_Mm, yarr_Mm, flnum):
    fig1,ax1 = plt.subplots(figsize=(6,6))
    ax1.pcolormesh(X,Y,pos_rem,cmap='bwr',vmin=-1, vmax=1)
    ax1.set_title('Positive Mask',font="Times New Roman",fontsize=22,fontweight='bold')
    
    ax1.set_xlim([xarr_Mm[200],xarr_Mm[600]])
    ax1.set_ylim([yarr_Mm[200],yarr_Mm[600]])
    ax1.set_xlabel('Horizontal Distance from Image Center [Mm]',fontsize=17)
    ax1.set_ylabel('Vertical Distance from Image Center [Mm]',fontsize=17)
    
    ax1.tick_params(labelsize=15)
    
    fig2,ax2 = plt.subplots(figsize=(6,6))
    ax2.set_xlabel('Horizontal Distance from Image Center [Mm]',fontsize=17)
    ax2.set_ylabel('Vertical Distance from Image Center [Mm]',fontsize=17)
    ax2.tick_params(labelsize=15)
    ax2.pcolormesh(X,Y,neg_rem,cmap='bwr',vmin=-1, vmax=1)
    
    ax2.set_title('Negative Mask',font="Times New Roman",fontsize=22,fontweight='bold')
    ax2.set_xlim([xarr_Mm[200],xarr_Mm[600]])
    ax2.set_ylim([yarr_Mm[200],yarr_Mm[600]])
    
    fig1.savefig(str(flnum)+'_pos_mask.png')
    fig2.savefig(str(flnum)+'_neg_mask.png')
    
    return None

def convolution_mask_plotting(X, Y, hmi_con_pos_c, hmi_con_neg_c, pil_mask_c,
                              xarr_Mm, yarr_Mm, flnum, xlim=[200,600],
                              ylim=[200,600]):
    fig1,ax1 = plt.subplots(figsize=(6,6))
    ax1.pcolormesh(X,Y,hmi_con_pos_c,cmap='bwr',vmin=-1, vmax=1)
    ax1.set_title('Positive Mask Convolution',font="Times New Roman",fontsize=22,fontweight='bold')
    ax1.set_xlim([xarr_Mm[200],xarr_Mm[600]])
    ax1.set_ylim([yarr_Mm[200],yarr_Mm[600]])
    ax1.set_xlabel('Horizontal Distance from Image Center [Mm]',fontsize=17)
    ax1.set_ylabel('Vertical Distance from Image Center [Mm]',fontsize=17)
    ax1.tick_params(labelsize=15)

    fig2,ax2 = plt.subplots(figsize=(6,6))
    ax2.tick_params(labelsize=15)
    ax2.pcolormesh(X,Y,hmi_con_neg_c,cmap='bwr',vmin=-1, vmax=1)
    ax2.set_xlabel('Horizontal Distance from Image Center [Mm]',fontsize=17)
    ax2.set_ylabel('Vertical Distance from Image Center [Mm]',fontsize=17)
    ax2.set_title('Negative Mask Convolution',font="Times New Roman",fontsize=22,fontweight='bold')
    ax2.set_xlim([xarr_Mm[xlim[0]],xarr_Mm[xlim[1]]])
    ax2.set_ylim([yarr_Mm[ylim[0]],yarr_Mm[ylim[1]]])
    
    
    fig3,ax3 = plt.subplots()
    ax3.pcolormesh(X,Y,pil_mask_c)
    ax3.set_title('Polarity Inversion Line Mask',font="Times New Roman",fontsize=22,fontweight='bold')
    ax3.tick_params(labelsize=15)
    ax3.set_xlim([xarr_Mm[xlim[0]],xarr_Mm[xlim[1]]])
    ax3.set_ylim([yarr_Mm[ylim[0]],yarr_Mm[ylim[1]]])
    
    fig1.savefig(str(flnum)+'_pos_conv_mask.png')
    fig2.savefig(str(flnum)+'_neg_conv_mask.png')    
    fig3.savefig(str(flnum)+'_PIL_conv_mask.png')  
    
    return None

def pil_poly_plot(X, Y, pil_mask_c, hmi_dat, ivs, dvs, conv_f, xarr_Mm, 
                  yarr_Mm, flnum, xlim = [200,600], ylim = [200,600]):
    # Generate the plot
    fig, ax = plt.subplots(figsize=(7,10))
    
    # show color mesh
    ax.pcolormesh(X,Y, pil_mask_c,cmap='hot')
    
    # plot the line
    ax.scatter((ivs-400)*conv_f,(dvs-400)*conv_f, color = 'c', s = 1)
    hmik = hmi_dat/1000
    plt.contour(X,Y,hmik,levels=[-3,-1.8,-.6,.6,1.8,3],cmap='seismic')
    
    ax.set_xlim([xarr_Mm[xlim[0]],xarr_Mm[xlim[1]]])
    ax.set_ylim([yarr_Mm[ylim[0]],yarr_Mm[ylim[1]]])
    ax.set_xticks([-80,-60,-40,-20,0,20,40,60,80])
    ax.set_yticks([-80,-60,-40,-20,0,20,40,60,80])
    cbar=plt.colorbar(orientation='horizontal')
    tick_font_size = 15
    ax.tick_params(labelsize=tick_font_size)
    cbar.ax.tick_params(labelsize=tick_font_size)
    ax.set_xlabel('Horizontal Distance from Image Center [Mm]',fontsize=15)
    ax.set_ylabel('Vertical Distance from Image Center [Mm]',fontsize=15)
        
    cbar.ax.set_xlabel('HMI Contours [kG]',font='Times New Roman',fontsize=17,fontweight='bold')
    ax.set_title('PIL Mask and Polynomial',font='Times New Roman',fontsize=25,fontweight='bold')
    fig.savefig(str(flnum)+'_pilpolyplot.png')
    return None

def ribbon_sep_plot(dist_pos,dist_neg,times,flnum,pltstrt):
    timelab = range(0,24*len(times),24)
    fig,[ax1,ax2] = plt.subplots(2,1,figsize=(13,15))
    ax1.plot(timelab[pltstrt:],dist_pos[pltstrt:],'-+',c='red',markersize=10,label='median')
    ax1.legend(fontsize=15)
    ax1.grid()
    s = str(times[0])
    ax1.set_xlabel('Time [s since '+s[2:-2]+']',font='Times New Roman',fontsize=15)
    ax1.set_ylabel('Cartesian Pixel Distance',font='Times New Roman',fontsize=15)
    ax1.set_title('Positive Ribbon Separation',font='Times New Roman',fontsize=25)
    
    ax2.plot(timelab[pltstrt:],dist_neg[pltstrt:],'-+',c='red',markersize=10,label='median')
    ax2.legend(fontsize=15)
    ax2.grid()
    ax2.set_xlabel('Time [s since '+s[2:-2]+']',font='Times New Roman',fontsize=15)
    ax2.set_ylabel('Cartesian Pixel Distance',font='Times New Roman',fontsize=15)
    ax2.set_title('Negative Ribbon Separation',font='Times New Roman',fontsize=25)

    fig.savefig(str(flnum)+'sep_raw_plt.png')
    
    return None

def ribbon_elon_plot(lens_pos, lens_neg, times, pltstrt, flnum):
    timelab = range(0,24*len(times),24)
    
    fig,ax1 = plt.subplots(figsize=(13,7))
    ax1.plot(timelab[pltstrt:],lens_pos[pltstrt:],'-+',c='red',markersize=10,label='+ Ribbon')
    ax1.plot(timelab[pltstrt:],lens_neg[pltstrt:],'-+',c='blue',markersize=10,label='- Ribbon')
    ax1.legend(fontsize=15)
    ax1.grid()
    s = str(times[0])
    ax1.set_xlabel('Time [s since '+s[2:-2]+']',font='Times New Roman',fontsize=17)
    ax1.set_ylabel('Cartesian Pixel Distance',font='Times New Roman',fontsize=17)
    ax1.set_title('Ribbon Elongation',font='Times New Roman',fontsize=25)

    fig.savefig(str(flnum)+'elon_raw_plt.png') 
    
    return None

def elon_period_plot(dpos_len, dneg_len, times, times1600, lens_pos_Mm, flnum,
                     lens_neg_Mm, elonperiod_start_neg, elonperiod_start_pos,
                     elonperiod_end_neg, elonperiod_end_pos):
    timelab = np.linspace(0,24*len(times),len(times))
    fig,[ax1,ax2,ax3] = plt.subplots(3,1,figsize=(13,20))
    ax3.plot(timelab[1:-1],dpos_len[1:],'-+',c='red',markersize=10,label='+ Ribbon')
    ax3.plot(timelab[1:-1],dneg_len[1:],'-+',c='blue',markersize=10,label='- Ribbon')
    ax3.legend(fontsize=15)
    ax3.grid()
    s = str(times1600[0])
    ax3.set_xlabel('Time [s since '+s[2:13]+', '+s[13:-5]+']',font='Times New Roman',fontsize=17)
    ax3.set_ylabel('Elongation Rate [Mm/sec]',font='Times New Roman',fontsize=17)
    ax3.set_title('Ribbon Elongation Rate',font='Times New Roman',fontsize=25,)
    
    ax1.plot(timelab[1:-1],lens_pos_Mm[1:-1],'-o',c='red',markersize=6,label='mean')
    ax2.plot(timelab[1:-1],lens_neg_Mm[1:-1],'-o',c='blue',markersize=6,label='mean')
    ax1.grid()
    ax1.set_ylabel('Distance [Mm]',font='Times New Roman',fontsize=17)
    ax1.set_title('Ribbon Elongation, Positive Ribbon',font='Times New Roman',fontsize=25,)
    ax2.set_ylabel('Distance [Mm]',font='Times New Roman',fontsize=17)
    ax2.set_title('Ribbon Elongation, Negative Ribbon',font='Times New Roman',fontsize=25,)
    ax2.grid()
    for i,j,k,l in zip(elonperiod_start_pos,elonperiod_end_pos,elonperiod_start_neg,elonperiod_end_neg):
        print(i,j,k,l)
        ax1.axvline(timelab[i],c='green')
        ax1.axvline(timelab[j],c='red')
        ax2.axvline(timelab[k],c='green')
        ax2.axvline(timelab[l],c='red')
        ax1.axvspan(timelab[i], timelab[j], alpha=0.5, color='pink')
        ax2.axvspan(timelab[k], timelab[l], alpha=0.5, color='cyan')
        
    fig.savefig(str(flnum)+'elon_timing_plt.png')
    
    return None

def sep_period_plot(dpos_dist, dneg_dist, times, distpos_Mm, distneg_Mm, flnum,
                    sepperiod_start_pos, sepperiod_end_pos, sepperiod_start_neg,
                    sepperiod_end_neg, indstrt):
    timelab = range(0,24*len(times),24)
    fig,[ax1,ax2,ax3] = plt.subplots(3,1,figsize=(13,20))
    ax3.plot(timelab[indstrt:-1],scipy.signal.medfilt(dpos_dist[indstrt:],kernel_size=3),'-+',c='red',markersize=10,label='+ Ribbon')
    ax3.plot(timelab[indstrt:-1],scipy.signal.medfilt(dneg_dist[indstrt:],kernel_size=3),'-+',c='blue',markersize=10,label='- Ribbon')
    ax3.legend(fontsize=15)
    ax3.grid()
    s = str(times[0])
    ax3.set_xlabel('Time [s since '+s[2:12]+ ', '+s[13:-5]+']',font='Times New Roman',fontsize=17)
    ax3.set_ylabel('Separation Rate [Mm/sec]',font='Times New Roman',fontsize=17)
    ax3.set_title('Ribbon Separation Rate',font='Times New Roman',fontsize=25,)
    
        
    ax1.plot(timelab[indstrt:-1],distpos_Mm[indstrt:-1],'-o',c='red',markersize=6,label='mean')
    ax2.plot(timelab[indstrt:-1],distneg_Mm[indstrt:-1],'-o',c='blue',markersize=6,label='mean')
    ax1.grid()
    #ax1.set_xlabel('Time [s since '+s[2:-2]+']',font='Times New Roman',fontsize=17)
    ax1.set_ylabel('Distance [Mm]',font='Times New Roman',fontsize=17)
    ax1.set_title('Ribbon Separation, Positive Ribbon',font='Times New Roman',fontsize=25,)
    #ax2.set_xlabel('Time [s since '+s[2:-2]+']',font='Times New Roman',fontsize=17)
    ax2.set_ylabel('Distance [Mm]',font='Times New Roman',fontsize=17)
    ax2.set_title('Ribbon Separation, Negative Ribbon',font='Times New Roman',fontsize=25,)
    ax2.grid()
    for i,j,k,l in zip(sepperiod_start_pos,sepperiod_end_pos,sepperiod_start_neg,sepperiod_end_neg):
        ax1.axvline(timelab[i],c='green')
        ax1.axvline(timelab[j],c='red')
        ax2.axvline(timelab[k],c='green')
        ax2.axvline(timelab[l],c='red')
        ax1.axvspan(timelab[i], timelab[j], alpha=0.5, color='pink')
        ax2.axvspan(timelab[k], timelab[l], alpha=0.5, color='cyan')
        
    fig.savefig(str(flnum)+'sep_timing_plt.png')
    
    return None
    
def flux_rec_mod_process(sav_data, dt1600, pos1600, neg1600):
    # process data for reconnection flux, reconnection rate, rise phase exponential modeling
    hmi = sav_data.hmi
    aia8 = sav_data.pos8
    aia8_inst = sav_data.inst_pos8
    aia8_pos = np.zeros(np.shape(aia8))
    aia8_neg = np.zeros(np.shape(aia8))
    aia8_inst_pos = np.zeros(np.shape(aia8_inst))
    aia8_inst_neg = np.zeros(np.shape(aia8_inst))
    xsh,ysh,zsh = aia8.shape
    hmi_dat = sav_data.hmi
    
    for i, j, k in np.ndindex(aia8.shape):
        if aia8[i,j,k] == 1 and hmi_dat[j,k] > 0:
            aia8_pos[i,j,k] = 1
        elif aia8[i,j,k] == 1 and hmi_dat[j,k] < 0:
            aia8_neg[i,j,k] = 1
            
    for i, j, k in np.ndindex(aia8.shape):
        if aia8_inst[i,j,k] == 1 and hmi_dat[j,k] > 0:
            aia8_inst_pos[i,j,k] = 1
        elif aia8_inst[i,j,k] == 1 and hmi_dat[j,k] < 0:
            aia8_inst_neg[i,j,k] = 1    
            
    peak_pos = dt1600[np.argmax(pos1600)]
    peak_neg = dt1600[np.argmax(neg1600)]
    
    return hmi, aia8_pos, aia8_neg, aia8_inst_pos, aia8_inst_neg, peak_pos, peak_neg

def inst_flux_process(aia8_inst_pos, aia8_inst_neg, flnum, conv_f, hmi, dt1600, peak_pos, peak_neg):
    rec_flux_pos_inst = np.zeros(len(aia8_inst_pos))
    rec_flux_neg_inst = np.zeros(len(aia8_inst_neg))
    pos_area_pix_inst = np.zeros(len(aia8_inst_pos))
    neg_area_pix_inst = np.zeros(len(aia8_inst_neg))
    pos_pix_inst = np.zeros(len(aia8_inst_pos))
    neg_pix_inst = np.zeros(len(aia8_inst_neg))
    
    conv_f_cm = conv_f*1e6*100 # conversion factor in cm
    ds2 = conv_f_cm**2 # for each pixel grid
    
    for i in range(len(aia8_inst_pos)):
        pos_mask_inst = aia8_inst_pos[i,:,:]
        neg_mask_inst = aia8_inst_neg[i,:,:]
        
        pos_area_pix_inst[i] = np.sum(pos_mask_inst)
        neg_area_pix_inst[i] = np.sum(neg_mask_inst)
        
        hmi_pos_inst = pos_mask_inst*hmi # in G
        hmi_neg_inst = neg_mask_inst*hmi # in G
        
        pos_pix_inst[i] = np.sum(hmi_pos_inst)
        neg_pix_inst[i] = np.sum(hmi_neg_inst)
        rec_flux_pos_inst[i] = np.sum(hmi_pos_inst)*ds2
        rec_flux_neg_inst[i] = np.sum(hmi_neg_inst)*ds2
    
    fig,ax = plt.subplots(figsize=(10,10))
    ax.scatter(dt1600,rec_flux_pos_inst,c='red',label='+')
    ax.scatter(dt1600,rec_flux_neg_inst,c='blue',label='-')
    ax.grid()
    ax.set_xlabel('Time',font='Times New Roman',fontsize=20)
    ax.axvline(peak_pos,c='red',linestyle=':')
    ax.axvline(peak_neg,c='blue',linestyle = '-.')
    ax.set_ylabel('Reconnection Flux [Mx]',font='Times New Roman',fontsize=20)
    ax.set_title('Reconnection Flux',font='Times New Roman',fontsize=25)
    ax.legend()
    
    fig.savefig(str(flnum)+'_inst_flx.png')
    
    return rec_flux_pos_inst, rec_flux_neg_inst, pos_pix_inst, neg_pix_inst, ds2
    
def cumul_flux_process(aia8_pos, aia8_neg, conv_f, flnum, peak_pos, peak_neg,
                       hmi, dt1600):
    rec_flux_pos = np.zeros(len(aia8_pos))
    rec_flux_neg = np.zeros(len(aia8_neg))
    pos_area_pix = np.zeros(len(aia8_pos))
    neg_area_pix = np.zeros(len(aia8_neg))
    pos_pix = np.zeros(len(aia8_pos))
    neg_pix = np.zeros(len(aia8_neg))
    pos_area = pos_area_pix
    neg_area = neg_area_pix
    
    conv_f_cm = conv_f*1e6*100 # conversion factor in cm
    ds2 = conv_f_cm**2
    for i in range(len(aia8_pos)):
        pos_mask = aia8_pos[i,:,:]
        neg_mask = aia8_neg[i,:,:]
        
        pos_area_pix[i] = np.sum(pos_mask)
        neg_area_pix[i] = np.sum(neg_mask)
        
        hmi_pos = pos_mask*hmi # in G
        hmi_neg = neg_mask*hmi # in G
        
        pos_pix[i] = np.sum(hmi_pos)
        neg_pix[i] = np.sum(hmi_neg)
        rec_flux_pos[i] = np.sum(hmi_pos)*ds2
        rec_flux_neg[i] = np.sum(hmi_neg)*ds2
    
    fig,ax = plt.subplots(figsize=(10,10))
    ax.scatter(dt1600,rec_flux_pos,c='red',label='+')
    ax.scatter(dt1600,rec_flux_neg,c='blue',label='-')
    ax.grid()
    ax.set_xlabel('Time',font='Times New Roman',fontsize=20)
    ax.axvline(peak_pos,c='red',linestyle=':')
    ax.axvline(peak_neg,c='blue',linestyle = '-.')
    ax.set_ylabel('Reconnection Flux [Mx]',font='Times New Roman',fontsize=20)
    ax.set_title('Reconnection Flux',font='Times New Roman',fontsize=25)
    ax.legend()
    
    fig.savefig(str(flnum)+'_cumul_flx.png')
    
    return rec_flux_pos, rec_flux_neg, pos_pix, neg_pix, pos_area_pix, neg_area_pix, ds2,pos_area, neg_area

def exp_curve_fit(exp_ind, pos_pix, neg_pix, exponential, exponential_neg, pos_area, neg_area):
    rise_pos_flx = pos_pix[0:exp_ind]
    rise_neg_flx = neg_pix[0:exp_ind]
    rise_pos_area = pos_area[0:exp_ind]
    rise_neg_area = neg_area[0:exp_ind]

    poptposflx, pcovposflx = scipy.optimize.curve_fit(exponential,range(0,len(rise_pos_flx)),rise_pos_flx)
    poptnegflx, pcovnegflx = scipy.optimize.curve_fit(exponential_neg,range(0,len(rise_neg_flx)),rise_neg_flx)
    poptpos, pcovpos = scipy.optimize.curve_fit(exponential,range(0,len(rise_pos_area)),rise_pos_area)
    poptneg, pcovneg = scipy.optimize.curve_fit(exponential,range(0,len(rise_neg_area)),rise_neg_area)
    
    return poptposflx, pcovposflx, poptnegflx, pcovnegflx, poptpos, poptneg, pcovpos, pcovneg, rise_pos_flx, rise_neg_flx

def exp_curve_plt(dt1600, rec_flux_pos, rec_flux_neg, rise_pos_flx, rise_neg_flx,
                  peak_pos, peak_neg, exp_ind, ds2, exponential, exponential_neg,
                  poptposflx, poptnegflx, flnum):
  
    rise_pos_time = dt1600[0:exp_ind]
    rise_neg_time = dt1600[0:exp_ind]
    
    fig,ax = plt.subplots(figsize=(10,10))
    ax.scatter(dt1600,rec_flux_pos,c='red',label='+')
    ax.scatter(dt1600,rec_flux_neg,c='blue',label='-')
    ax.grid()
    ax.set_xlabel('Time',font='Times New Roman',fontsize=20)
    ax.axvline(peak_pos,c='red',linestyle=':')
    ax.axvline(peak_neg,c='blue',linestyle = '-.')
    ax.set_ylabel('Reconnection Flux [Mx]',font='Times New Roman',fontsize=20)
    ax.set_title('Reconnection Flux',font='Times New Roman',fontsize=25)
    ax.plot(rise_pos_time,ds2*exponential(range(0,len(rise_pos_flx)), *poptposflx), 'r-',label='Exponential Model, +')
    ax.plot(rise_neg_time,ds2*exponential_neg(range(0,len(rise_neg_flx)), *poptnegflx), 'b-',label='Exponential Model, -')
    ax.axvline(dt1600[29])
    ax.legend()
    
    fig.savefig(str(flnum)+'_recflux_model.png')
    
    # now plot log-log of just the impulsive phase
    
    fig2,[ax1,ax2] = plt.subplots(2,1,figsize=(10,20))
    ax1.scatter((dt1600),np.log(rec_flux_pos),c='red')
    ax2.scatter((dt1600),-np.log(-rec_flux_neg),c='blue')
    ax1.grid()
    ax2.grid()
    ax1.set_xlabel('Time',font='Times New Roman',fontsize=20)
    ax2.set_xlabel('Time',font='Times New Roman',fontsize=20)
    #ax.axvline(peak_pos,c='red',linestyle=':')
    #ax.axvline(peak_neg,c='blue',linestyle = '-.')
    ax1.plot((rise_pos_time),np.log(ds2*exponential(range(0,len(rise_pos_flx)), *poptposflx)), 'r-',label='Exponential Model, +')#, label='fit: a = %5.3f, b = %5.3f, c = %5.3f', % tuple(popt))
    ax2.plot((rise_neg_time),-np.log(-ds2*exponential_neg(range(0,len(rise_neg_flx)), *poptnegflx)), 'b-',label='Exponential Model, -')#, label='fit: a = %5.3f, b = %5.3f, c = %5.3f', % tuple(popt))
    
    ax1.set_ylabel(r'Rec. Flx [Mx]',font='Times New Roman',fontsize=20)
    ax1.set_title('Reconnection Flux, Impulsive Phase',font='Times New Roman',fontsize=25)
    ax1.set_xlim(dt1600[0],dt1600[exp_ind])
    ax1.legend(fontsize=15)
    ax2.set_ylabel(r'Rec. Flx [Mx]',font='Times New Roman',fontsize=20)
    ax2.set_title('Reconnection Flux, Impulsive Phase',font='Times New Roman',fontsize=25)
    ax2.set_xlim(dt1600[0],dt1600[exp_ind])
    ax2.legend(fontsize=15)
    
    fig2.savefig(str(flnum)+'_rec_impphase_model.png')

    
    return None

def rib_area_plt(dt1600, poptpos, poptneg, flnum, pos_area_pix, neg_area_pix, peak_pos, peak_neg, exp_ind, exponentialimpdiff = 'no'):
    # cumulative
    pos_area = pos_area_pix
    neg_area = neg_area_pix
    rise_pos_area = pos_area[0:exp_ind]
    rise_neg_area = neg_area[0:exp_ind]
    # plot just the ribbon areas, c = 8
    fig,ax = plt.subplots(figsize=(10,10))
    ax.scatter(dt1600,pos_area,c='red',label='+')
    ax.scatter(dt1600,neg_area,c='blue',label='-')
    rise_pos_time = dt1600[0:exp_ind]
    rise_neg_time = dt1600[0:exp_ind]
    ax.grid()
    ax.set_xlabel('Time',font='Times New Roman',fontsize=20)
    ax.axvline(peak_pos,c='red',linestyle=':')
    ax.axvline(peak_neg,c='blue',linestyle = '-.')
    ax.plot(rise_pos_time,exponential(range(0,len(rise_pos_area)), *poptpos), 'r-',label='Exponential Model, +')#, label='fit: a = %5.3f, b = %5.3f, c = %5.3f', % tuple(popt))
    ax.plot(rise_neg_time,exponential(range(0,len(rise_neg_area)), *poptneg), 'b-',label='Exponential Model, -')#, label='fit: a = %5.3f, b = %5.3f, c = %5.3f', % tuple(popt))
    ax.set_ylabel('Ribbon Area [cm^2]',font='Times New Roman',fontsize=20)
    ax.set_title('Ribbon Area',font='Times New Roman',fontsize=25)
    #if end of modeling region is before end of impulsive phase
    ax.axvline(dt1600[exp_ind])
    ax.legend()
    
    fig.savefig(str(flnum)+'_ribarea_model.png')
    
    #just impulsive region, with log-log
    fig2,[ax1,ax2] = plt.subplots(2,1,figsize=(10,20))
    ax1.scatter((dt1600),np.log(pos_area),c='red')
    ax2.scatter((dt1600),np.log(neg_area),c='blue')
    ax1.grid()
    ax2.grid()
    ax1.set_xlabel('Time',font='Times New Roman',fontsize=20)
    ax2.set_xlabel('Time',font='Times New Roman',fontsize=20)
    ax1.plot((rise_pos_time),np.log(exponential(range(0,len(rise_pos_area)), *poptpos)), 'r-',label='Exponential Model, +')#, label='fit: a = %5.3f, b = %5.3f, c = %5.3f', % tuple(popt))
    ax2.plot((rise_neg_time),np.log(exponential(range(0,len(rise_neg_area)), *poptneg)), 'b-',label='Exponential Model, -')#, label='fit: a = %5.3f, b = %5.3f, c = %5.3f', % tuple(popt))
    
    ax1.set_ylabel('Ribbon Area [cm^2]',font='Times New Roman',fontsize=20)
    ax1.set_title('Ribbon Area, Impulsive Phase',font='Times New Roman',fontsize=25)
    ax1.set_xlim(dt1600[0],dt1600[exp_ind])
    ax1.legend(fontsize=15)
    ax2.set_ylabel('Ribbon Area [cm^2]',font='Times New Roman',fontsize=20)
    ax2.set_title('Ribbon Area, Impulsive Phase',font='Times New Roman',fontsize=25)
    ax2.set_xlim(dt1600[0],dt1600[exp_ind])
    ax2.legend(fontsize=15)
    
    fig2.savefig(str(flnum)+'_impphase_model.png')
    
    return None

# Reconnection rate
def rec_rate(rec_flux_pos, rec_flux_neg, dn1600, dt1600, peak_pos, peak_neg, flnum):
    rec_rate_pos = (np.diff(rec_flux_pos)/(dn1600[1]-dn1600[0]))/3600/24 # Mx/s
    rec_rate_neg = (np.diff(rec_flux_neg)/(dn1600[1]-dn1600[0]))/3600/24 # Mx/s

    fig,ax = plt.subplots(figsize=(10,10))
    ax.scatter(dt1600[1:],rec_rate_pos,c='red',label='+')
    ax.scatter(dt1600[1:],rec_rate_neg,c='blue',label='-')
    ax.grid()
    ax.set_xlabel('Time',font='Times New Roman',fontsize=20)
    ax.axvline(peak_pos,c='red',linestyle=':')
    ax.axvline(peak_neg,c='blue',linestyle = '-.')
    ax.set_ylabel('Reconnection Rate [Mx/s]',font='Times New Roman',fontsize=20)
    ax.set_title('Reconnection Rate',font='Times New Roman',fontsize=25)
    
    fig.savefig(str(flnum)+'_recrate.png')
    return rec_rate_pos, rec_rate_neg

    