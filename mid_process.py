# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 12:15:45 2022

@author: owner
"""
import fl_funcs
from fl_funcs import exponential
from fl_funcs import exponential_neg
import numpy as np

year = 2013
mo = 5
day = 16
sthr = 21
stmin = 36
arnum = 11748
xclnum = 1.3
xcl = 'M'
flnum = 1242
instrument = 'n5'
daystr = '16'
mostr = 'may'
mostrcap = 'May'
yearstr = '2013'
imp = -5.06

bestflarefile = '/Users/owner/Desktop/Oct_2022_Imp/imp_dev/all_and_best_Oct_2022.mat'


print("Loading the data...")

sav_data_aia, sav_data, best304, start304, peak304, end304, eventindices,\
    times304, curves304, aia_cumul8, aia_step8, last_cumul8, hmi_dat,\
    last_mask = fl_funcs.load_variables(bestflarefile, year, mo, day, sthr,
                                        stmin, arnum, xclnum, xcl)

X, Y, conv_f, xarr_Mm, yarr_Mm = fl_funcs.conv_facts()

print("Data loaded! Now just some masking and spur removal.")

hmi_cumul_mask1, hmi_step_mask1, hmi_pos_mask_c, hmi_neg_mask_c \
    = fl_funcs.pos_neg_masking(aia_cumul8, aia_step8, hmi_dat, last_mask)

neg_rem, pos_rem = fl_funcs.spur_removal_sep(hmi_neg_mask_c, hmi_pos_mask_c,
                                             pos_crit=2, neg_crit=2,
                                             jhi=475, jhi2=490)

print("Convolving the HMI images and making the PIL mask.")

hmi_con_pos_c, hmi_con_neg_c, pil_mask_c = fl_funcs.gauss_conv(pos_rem,
                                                               neg_rem,
                                                               sigma=4)

pil_mask_c, ivs, dvs, hmik = fl_funcs.pil_gen(pil_mask_c, hmi_dat)

print("Separation values determination.")

aia8_pos_step, aia8_neg_step = fl_funcs.mask_sep(aia_step8, hmi_dat)

pos_rem0, neg_rem0 = fl_funcs.spur_removal_sep2(aia8_pos_step, aia8_neg_step)

distpos_med, distpos_mean, distneg_med, distpos_mean \
    = fl_funcs.separation(aia_step8, ivs, dvs, pos_rem0, neg_rem0)

print("Elongation values determination.")

aia8_pos_2, aia8_neg_2 = fl_funcs.mask_elon(aia_cumul8, hmi_dat)

neg_rem1, pos_rem1 = fl_funcs.spur_removal_elon(aia8_pos_2, aia8_neg_2)

ivs_lim, dvs_lim, med_x, med_y = fl_funcs.lim_pil(ivs, dvs)

ylim0_neg = 200
ylim1_neg = 600
ylim0_pos = 200
ylim1_pos = 600
xlim0_neg = 200
xlim1_neg = 600
xlim0_pos = 200
xlim1_pos = 600

aia_pos_rem, aia_neg_rem = fl_funcs.rib_lim_elon(aia8_pos_2, aia8_neg_2,
                                                 pos_rem1, neg_rem1, med_x,
                                                 med_y, ylim0_pos, ylim1_pos,
                                                 ylim0_neg, ylim1_neg,
                                                 xlim0_pos, xlim1_pos,
                                                 xlim0_neg, xlim1_neg)

rib_pos_1, rib_pos_2, rib_neg_1, rib_neg_2 = fl_funcs.split_rib(aia_pos_rem,
                                                                aia_neg_rem,
                                                                380, 400)

lr_coord_neg_1, lr_coord_pos_1 = fl_funcs.find_rib_coordinates(rib_pos_1,
                                                               rib_neg_1)

lr_coord_neg_2, lr_coord_pos_2 = fl_funcs.find_rib_coordinates(rib_pos_2,
                                                               rib_neg_2)

ivs_sort, dvs_sort, sortedpil = fl_funcs.sort_pil(ivs_lim, dvs_lim)

pil_right_near_pos_1, pil_left_near_pos_1, pil_right_near_neg_1,\
    pil_left_near_neg_1 = fl_funcs.elon_dist_arrays(lr_coord_pos_1,
                                                    lr_coord_neg_1, ivs_lim,
                                                    dvs_lim, ivs_sort,
                                                    dvs_sort)

pil_right_near_pos_2, pil_left_near_pos_2, pil_right_near_neg_2,\
    pil_left_near_neg_2 = fl_funcs.elon_dist_arrays(lr_coord_pos_2,
                                                    lr_coord_neg_2,
                                                    ivs_lim, dvs_lim, ivs_sort,
                                                    dvs_sort)

lens_pos_1, lens_neg_1 = fl_funcs.elongation(pil_right_near_pos_1,
                                             pil_left_near_pos_1,
                                             pil_right_near_neg_1,
                                             pil_left_near_neg_1,
                                             sortedpil)

lens_pos_2, lens_neg_2 = fl_funcs.elongation(pil_right_near_pos_2,
                                             pil_left_near_pos_2,
                                             pil_right_near_neg_2,
                                             pil_left_near_neg_2,
                                             sortedpil)

lens_pos = [sum(x) for x in zip(lens_pos_1, lens_pos_2)]
lens_neg = [sum(x) for x in zip(lens_neg_1, lens_neg_2)]

dist_pos = distpos_med
dist_neg = distneg_med

print("Converting separation and elongation to Mm.")

lens_pos_Mm, lens_neg_Mm, distpos_Mm, distneg_Mm, dneg_len, dpos_len, \
    dneg_dist, dpos_dist = fl_funcs.convert_to_Mm(lens_pos, dist_pos, lens_neg,
                                                  dist_neg, conv_f)

print("Loading parameters for 304 and 1600 Angstrom light curves.")

startin, peakin, endin, times, s304, e304, filter_304, med304, std304, \
    timelab, aiadat, nt, dn1600, time304, times1600 \
    = fl_funcs.prep_304_1600_parameters(sav_data_aia, sav_data, eventindices,
                                        flnum, start304, peak304, end304,
                                        times304, curves304, outflag=1242)

posrib, negrib, pos1600, neg1600 = fl_funcs.img_mask(aia8_pos_step,
                                                     aia8_neg_step,
                                                     aiadat, nt)

print("Determining the regions of separation and elongation.")

elonperiod_start_pos, elonperiod_end_pos, elonperiod_start_neg, \
    elonperiod_end_neg = fl_funcs.elon_periods(dpos_len, dneg_len)

sepperiod_start_pos, sepperiod_end_pos, sepperiod_start_neg, \
    sepperiod_end_neg = fl_funcs.sep_periods(dpos_dist, dneg_dist, start=1)

dt1600, dt304 = fl_funcs.prep_times(dn1600, time304)

print("Plotting ribbon masks.")

fl_funcs.mask_plotting(X, Y, pos_rem, neg_rem, xarr_Mm, yarr_Mm, flnum)

print("Plotting convolution masks.")

fl_funcs.convolution_mask_plotting(X, Y, hmi_con_pos_c, hmi_con_neg_c,
                                   pil_mask_c, xarr_Mm, yarr_Mm, flnum,
                                   xlim=[200, 600], ylim=[200, 600])

print("Plotting PIL with representative polynomial.")

fl_funcs.pil_poly_plot(X, Y, pil_mask_c, hmi_dat, ivs, dvs, conv_f, xarr_Mm,
                       yarr_Mm, flnum)

print("Plotting ribbon separation.")

pltstrt = 1

fl_funcs.ribbon_sep_plot(dist_pos, dist_neg, times, flnum, pltstrt, dt1600)

print("Plotting ribbon elongation.")

pltstrt = 1

fl_funcs.ribbon_elon_plot(lens_pos, lens_neg, times, pltstrt, flnum, dt1600)

print("Plotting Elongation with Periods")
indstrt = 1
fl_funcs.elon_period_plot(dpos_len, dneg_len, times, times1600, lens_pos_Mm,
                          lens_neg_Mm, flnum, elonperiod_start_neg,
                          elonperiod_start_pos, elonperiod_end_neg,
                          elonperiod_end_pos, indstart=indstrt)

print("Plotting Separation with Periods")

indstrt = 1
fl_funcs.sep_period_plot(dpos_dist, dneg_dist, times, distpos_Mm, distneg_Mm,
                         flnum, sepperiod_start_pos, sepperiod_end_pos,
                         sepperiod_start_neg, sepperiod_end_neg,
                         indstrt=indstrt)

print("Processing data for reconnection flux model.")

hmi, aia8_pos, aia8_neg, aia8_inst_pos, aia8_inst_neg, peak_pos, \
    peak_neg = fl_funcs.flux_rec_mod_process(
        sav_data, dt1600, pos1600, neg1600)

print("Load fluxes and pixel counts.")

rec_flux_pos, rec_flux_neg, pos_pix, neg_pix, pos_area_pix, neg_area_pix, ds2,\
    pos_area, neg_area = fl_funcs.cumul_flux_process(aia8_pos, aia8_neg,
                                                     conv_f, flnum, peak_pos,
                                                     peak_neg, hmi, dt1600)

print("The same, for instantaneous flux.")

rec_flux_pos_inst, rec_flux_neg_inst, pos_pix_inst, neg_pix_inst, \
    ds2 = fl_funcs.inst_flux_process(aia8_inst_pos, aia8_inst_neg, flnum,
                                     conv_f, hmi, dt1600, peak_pos, peak_neg)

print("Reconnection Rate Determination, Plotting.")

rec_rate_pos, rec_rate_neg = fl_funcs.rec_rate(rec_flux_pos, rec_flux_neg,
                                               dn1600, dt1600, peak_pos,
                                               peak_neg, flnum)

exp_ind = np.argmax(pos1600)
exp_ind_area = np.argmax(pos1600)

print("Exponential curve fitting for the fluxes.")

poptposflx, pcovposflx, poptnegflx, pcovnegflx, \
    poptpos, poptneg, pcovpos, pcovneg, rise_pos_flx, \
    rise_neg_flx = fl_funcs.exp_curve_fit(exp_ind, exp_ind_area, pos_pix,
                                          neg_pix, exponential,
                                          exponential_neg, pos_area,
                                          neg_area)

print("Exponential curve plot.")

fl_funcs.exp_curve_plt(dt1600, rec_flux_pos, rec_flux_neg, rise_pos_flx,
                       rise_neg_flx, peak_pos, peak_neg, exp_ind, ds2,
                       exponential, exponential_neg, poptposflx, poptnegflx,
                       flnum)

print("Ribbon Area Plot")

fl_funcs.rib_area_plt(dt1600, poptpos, poptneg, flnum, pos_area_pix,
                      neg_area_pix, peak_pos, peak_neg, exp_ind)

print("Begin determination of shear.")

# Establish limits for ribbons corresponding to shear code.
negylow = ylim0_neg
negyhi = ylim1_neg
negxlow = xlim0_neg
negxhi = xlim1_neg

posylow = ylim0_pos
posyhi = ylim1_pos
posxlow = xlim0_pos
posxhi = xlim1_pos

# Isolate ribbons appropriately for shear analysis
aia_neg_rem_shear, aia_pos_rem_shear = fl_funcs.\
    shear_ribbon_isolation(aia8_neg_step, aia8_pos_step, med_x, med_y,
                           negylow=negylow, negyhi=negyhi, posylow=posylow,
                           posyhi=posyhi, negxlow=negxlow, negxhi=negxhi,
                           posxlow=posxlow, posxhi=posxhi, flag=1)

# Left and right coordinates of positive and negative ribbons
lr_coord_neg_shear, lr_coord_pos_shear = \
    fl_funcs.leftrightshear(aia_pos_rem_shear, aia_neg_rem_shear)

# PIL pixels closest to the left and right coordinates of positive and negative
# ribbons
pil_right_near_pos_shear, pil_left_near_pos_shear, pil_right_near_neg_shear,\
    pil_left_near_neg_shear = fl_funcs.sheardists(lr_coord_pos_shear,
                                                  lr_coord_neg_shear,
                                                  ivs_sort, dvs_sort)

# Guide field to the right and left edges of ribbons
guide_right, guide_left = fl_funcs.guidefieldlen(pil_right_near_pos_shear,
                                                 pil_left_near_pos_shear,
                                                 pil_right_near_neg_shear,
                                                 pil_left_near_neg_shear,
                                                 sortedpil,
                                                 fl_funcs.curve_length)

# Guide field ratio to the right and left edges of ribbons
left_gfr, right_gfr = fl_funcs.gfrcalc(guide_left, guide_right,
                                       distneg_med, distpos_med)

print("Plot guide field ratio proxy based on footpoints.")

# Plot guide field ratio
fl_funcs.plt_gfr(times, right_gfr, left_gfr, flnum, dt1600)

print("Fermi Processing")

raw_hxr_sum, cspec_hxr_sum, fermitimes = fl_funcs.process_fermi(daystr, mostr,
                                                                yearstr,
                                                                instrument,
                                                                day, mo, year,
                                                                low=16500,
                                                                high=18000,
                                                                ylo=1e-3,
                                                                yhi=100)

# Figure for timestamp comparison

indstrt_sep = 1
indstrt_elon = 1
gfr_trans = 7

E_pos, E_neg, E_rat, time_E = fl_funcs.E_field_det(conv_f, distpos_med,
                                                   distneg_med, timelab,
                                                   hmi_dat, pos_rem, neg_rem,
                                                   flnum, dt1600, times,
                                                   startind=gfr_trans)

E_pos = np.append(E_pos, E_pos[-1])
E_neg = np.append(E_neg, E_neg[-1])

shear_ang = fl_funcs.shear_to_angle(times, flnum, dt1600, left_gfr, right_gfr)

quartermaxtim = fl_funcs.quartermaxtime(
    gfr_trans, right_gfr, left_gfr, timelab, fl_funcs.find_nearest_ind, flag=0)

print(quartermaxtim)

muted = fl_funcs.color_muted()
vibrant = fl_funcs.color_vibrant()
medc = fl_funcs.color_medc()
level = 'mid'

fl_funcs.plt_fourpanel(times, right_gfr, left_gfr, flnum, dt1600, time304,
                       filter_304, lens_pos_Mm, lens_neg_Mm, distpos_Mm,
                       distneg_Mm, dt304, timelab, conv_f,
                       elonperiod_start_pos, elonperiod_end_pos,
                       elonperiod_start_neg, elonperiod_end_neg,
                       sepperiod_start_pos, sepperiod_end_pos,
                       sepperiod_start_neg, sepperiod_end_neg, exp_ind,
                       s304, e304, pos1600, neg1600, dn1600, indstrt_elon,
                       indstrt_sep, fermitimes, raw_hxr_sum, cspec_hxr_sum,
                       gfr_trans, E_pos, E_neg, time_E, daystr,mostrcap,yearstr,
                       xcl,xclnum,imp, muted, vibrant, medc, level,low_hxr=16500,
                       high_hxr=18000,  period_flag=0, tick_space=8)

file = 'mid1shear'

np.savez(file, shear_ang, right_gfr, left_gfr)
