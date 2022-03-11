#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 12:15:45 2022

@author: owner
"""
import fl_funcs

year = 2013
mo = 10
day = 13
sthr = 0
stmin = 12
arnum = 11865
xclnum = 1.7
xcl = 'M'
flnum = 1401

bestflarefile = "/Users/owner/Desktop/CU_Research/MAT_SOURCE/bestperf_more.mat"


print("Loading the data...")

sav_data_aia, sav_data, best304, start304, peak304, end304, eventindices, times304,\
curves304, aia_cumul8, aia_step8, last_cumul8, hmi_dat,\
last_mask = fl_funcs.load_variables(bestflarefile, year, mo, day, sthr, stmin, arnum, 
                                    xclnum, xcl)

X, Y, conv_f, xarr_Mm, yarr_Mm = fl_funcs.conv_facts()

print("Data loaded! Now just some masking and spur removal.")

hmi_cumul_mask1, hmi_step_mask1, hmi_pos_mask_c, hmi_neg_mask_c \
    = fl_funcs.pos_neg_masking(aia_cumul8, aia_step8, hmi_dat, last_mask)

neg_rem, pos_rem = fl_funcs.spur_removal_sep(hmi_neg_mask_c, hmi_pos_mask_c, 
                                             pos_crit = 3, neg_crit = 2)

print("Convolving the HMI images and making the PIL mask.")

hmi_con_pos_c, hmi_con_neg_c, pil_mask_c = fl_funcs.gauss_conv(pos_rem, neg_rem)

pil_mask_c, ivs, dvs, hmik = fl_funcs.pil_gen(pil_mask_c, hmi_dat)

print("Separation values determination.")

aia8_pos, aia8_neg = fl_funcs.mask_sep(aia_step8, hmi_dat)

distpos_med, distpos_mean, distneg_med, distpos_mean \
    = fl_funcs.separation(aia_step8, ivs, dvs, aia8_pos, aia8_neg)
    
print("Elongation values determination.") 

aia8_pos_2, aia8_neg_2 = fl_funcs.mask_elon(aia_cumul8, hmi_dat)

neg_rem1, pos_rem1 = fl_funcs.spur_removal_elon(aia8_pos_2, aia8_neg_2)

ivs_lim, dvs_lim, med_x, med_y = fl_funcs.lim_pil(ivs, dvs)

ylim0_neg = 400
ylim1_neg = int(round(med_y)+100)
ylim0_pos = int(round(med_y)-100)
ylim1_pos = int(round(med_y)+100)
xlim0_neg = 300
xlim1_neg = 400
xlim0_pos = 350
xlim1_pos = int(round(med_y)+100)

aia_pos_rem, aia_neg_rem = fl_funcs.rib_lim_elon(aia8_pos_2, aia8_neg_2,
                                                 pos_rem1, neg_rem1, med_x,
                                                 med_y, ylim0_pos, ylim1_pos,
                                                 ylim0_neg, ylim1_neg,
                                                 xlim0_pos, xlim1_pos,
                                                 xlim0_neg, xlim1_neg)

lr_coord_neg, lr_coord_pos = fl_funcs.find_rib_coordinates(aia_pos_rem,
                                                           aia_neg_rem)

ivs_sort, dvs_sort, sortedpil = fl_funcs.sort_pil(ivs_lim, dvs_lim)
    
pil_right_near_pos, pil_left_near_pos, pil_right_near_neg, pil_left_near_neg \
    = fl_funcs.elon_dist_arrays(lr_coord_pos, lr_coord_neg, ivs_lim, dvs_lim,
                                ivs_sort, dvs_sort)
    
lens_pos, lens_neg = fl_funcs.elongation(pil_right_near_pos, pil_left_near_pos,
                                         pil_right_near_neg, pil_left_near_neg,
                                         sortedpil)

dist_pos = distpos_med
dist_neg = distneg_med

print("Converting separation and elongation to Mm.")

lens_pos_Mm, lens_neg_Mm, distpos_Mm, distneg_Mm, dneg_len, dpos_len, \
    dneg_dist, dpos_dist = fl_funcs.convert_to_Mm(lens_pos, dist_pos, lens_neg,
                                                  dist_neg, conv_f)

print("Loading parameters for 304 and 1600 Angstrom light curves.")

startin, peakin, endin, times, s304, e304, filter_304, med304, std304, \
    timelab, aiadat, nt, dn1600, time304, times1600 \
    = fl_funcs.prep_304_parameters(sav_data_aia, sav_data, eventindices,
                                   flnum, start304, peak304, end304,
                                   times304, curves304)
  
posrib, negrib, pos1600, neg1600 = fl_funcs.img_mask(aia8_pos, aia8_neg, aiadat,
                                                     nt)    

print("Determining the regions of separation and elongation.")  
    
elonperiod_start_pos, elonperiod_end_pos, elonperiod_start_neg, \
    elonperiod_end_neg = fl_funcs.elon_periods(dpos_len, dneg_len)
    
sepperiod_start_pos, sepperiod_end_pos, sepperiod_start_neg, \
    sepperiod_end_neg = fl_funcs.sep_periods(dpos_dist, dneg_dist)
    
dt1600, dt304 = fl_funcs.prep_times(dn1600, time304)

print("Plotting ribbon masks.")

fl_funcs.mask_plotting(X, Y, pos_rem, neg_rem, xarr_Mm, yarr_Mm, flnum)

print("Plotting convolution masks.")

fl_funcs.convolution_mask_plotting(X, Y, hmi_con_pos_c, hmi_con_neg_c, pil_mask_c,
                                   xarr_Mm, yarr_Mm, flnum, xlim=[200,600],
                                   ylim=[200,600])

print("Plotting PIL with representative polynomial.")

fl_funcs.pil_poly_plot(X, Y, pil_mask_c, hmi_dat, ivs, dvs, conv_f, xarr_Mm,
                       yarr_Mm, flnum)

print("Plotting ribbon separation.")

pltstrt = 25

fl_funcs.ribbon_sep_plot(dist_pos, dist_neg, times, flnum, pltstrt)

print("Plotting ribbon elongation.")

pltstrt = 1

fl_funcs.ribbon_elon_plot(lens_pos, lens_neg, times, pltstrt, flnum)

print("Plotting Elongation with Periods")

fl_funcs.elon_period_plot(dpos_len, dneg_len, times, times1600, lens_pos_Mm, 
                          flnum, lens_neg_Mm, elonperiod_start_neg, 
                          elonperiod_start_pos, elonperiod_end_neg, 
                          elonperiod_end_pos)

print("Plotting Separation with Periods")

indstrt = 25
fl_funcs.sep_period_plot(dpos_dist, dneg_dist, times, distpos_Mm, distneg_Mm, flnum,
                    sepperiod_start_pos, sepperiod_end_pos, sepperiod_start_neg,
                    sepperiod_end_neg, indstrt)

print("Processing data for reconnection flux model.")

hmi, aia8_pos, aia8_neg, aia8_inst_pos, aia8_inst_neg, peak_pos, \
    peak_neg = fl_funcs.flux_rec_mod_process(sav_data, dt1600, pos1600, neg1600)
    