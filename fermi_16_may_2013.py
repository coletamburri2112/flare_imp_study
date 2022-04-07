#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 12:10:41 2022

@author: owner
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import readsav
from datetime import datetime
import time

directory = '/Users/owner/Desktop/CU_Research/Fermi_April_2022/'\
            'Fermi_Events_sav/'

instrument = 'n5'
day = '16'
month = 'may'
year = '2013'

dayint = 16
moint = 5
yearint = 2013

low = 16500
high = 18000

ylo = 1e-3
yhi = 100

filename_cspec = directory + 'fermi_' + instrument + '_cspec_bkgd_' + day + \
    month + year + '.sav'
    
cspec_dat = readsav(filename_cspec, python_dict='True')

bksub_cspec = cspec_dat['lc_bksub'][0][0]
raw_cspec = cspec_dat['lc_raw'][0][0]
times = cspec_dat['time']
energies = cspec_dat['ct_energy']

hxrinds = np.where(cspec_dat['ct_energy'] < 300.) and \
    np.where(cspec_dat['ct_energy'] > 25.)
    
cspec_hxr = bksub_cspec[:, hxrinds]
raw_hxr = raw_cspec[:, hxrinds]
cspec_hxr_sum = np.sum(cspec_hxr, axis = 2)
raw_hxr_sum = np.sum(raw_hxr, axis=2)

a = datetime(1970,1,1,0,0,0)
b = datetime(1979,1,1,0,0,0)

err1 = (b-a).total_seconds()

timesadj1 = times + err1

curr = datetime.fromtimestamp(min(timesadj1))
corr = datetime(yearint,moint,dayint,0,0,0)

err2 = (corr-curr).seconds
totsec = (b-a).total_seconds() + err2

timesadj = times + totsec

time.ctime(min(timesadj))
strtimes = []

for i in timesadj:
    strtimes.append(datetime.fromtimestamp(i))
    
flag = 0

for i in range(low, high):
    if cspec_hxr_sum[i] > 0.08:
        flag += 1
    if cspec_hxr_sum[i] < 0.08:
        flag = 0
    if flag > 3:
        startind = i - 3
        break
    
maxind = np.where(raw_hxr_sum[low:high] == max(raw_hxr_sum[low:high]))
fig,ax = plt.subplots(figsize=(15,10))

ax.scatter(strtimes[low:high],np.log10(raw_hxr_sum[low:high]),marker='.',
           label='Raw Cts.')
ax.scatter(strtimes[low:high],np.log10(cspec_hxr_sum[low:high]),marker='.',
           label='Bkgd. Sub. Cts.')
ax.axvline(strtimes[startind],linestyle='--',color='red',label='Start')
ax.axvline(strtimes[maxind[0][0]+low],linestyle='-.',color='black',label='Max')
ax.grid
ax.set_xlabel('Time [DD HH:MM]',font='Times New Roman',fontsize=20)
ax.set_ylabel('Flux [cts/s/cm$^2$/keV]',font='Times New Roman',fontsize=20)
ax.set_title(str(moint)+'-'+str(dayint)+'-'+str(yearint)+' Fermi GBM 25 - 300'\
             'keV Band',font='Times New Roman',fontsize=25)
#ax.set_ylim(np.log10(ylo),np.log10(yhi))
ax.legend(fontsize=15)

fname = '16_may_2013_Fermi.png'

plt.savefig(fname)
