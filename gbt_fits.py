from glob import glob
import os, math, time
import sys

import numpy as np

import astropy
from astropy import units as u, constants as const
from astropy.time import Time
from astropy.visualization import quantity_support


import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib as mpl


from scipy.sparse.linalg import eigsh
from scipy.optimize import curve_fit
import scipy.linalg

import load_data as ld
import ds_psr as dsa
import fit_thth as fth
import models_thth as mth
import wsrt_fits as wf
import ththmod as THTH

def plot_fit_results(sp, msp, dic, new_fig=True, ax_y=0.0, myfig=None):
    if new_fig is True:
        fig=plt.figure(figsize=(6,1), dpi=150)
        myfig=fig
    myfig.add_axes([0.0,ax_y,0.15,0.15])
    frame1=plt.gca()
    sp.plot_ds(new_fig=False, lab_mfr=True)
    frame1.axes.get_yaxis().set_ticks([])
    myfig.add_axes([0.2,ax_y,0.15,0.15])
    frame1=plt.gca()
    msp.plot_mds(new_fig=False)
    frame1.axes.get_yaxis().set_ticks([])
    myfig.add_axes([0.4,ax_y,0.15, 0.15])
    frame1=plt.gca()
    msp.plot_me(new_fig=False)
    frame1.axes.get_yaxis().set_ticks([])
    myfig.add_axes([0.6,ax_y,0.15,0.15])
    frame1=plt.gca()
    sp.plot_ss(new_fig=False, tau_lim=[0.0,1.25], vmin=1e3,vmax=5e6, cb=False)
    frame1.axes.get_yaxis().set_ticks([])
    myfig.add_axes([0.8,ax_y,0.15,0.15])
    frame1=plt.gca()
    msp.plot_mss(new_fig=False, cb=False)
    frame1.axes.get_yaxis().set_ticks([])
    myfig.add_axes([1.0,ax_y,0.15,0.15])
    frame1=plt.gca()
    msp.plot_mes(new_fig=False, cb=False)
    frame1.axes.get_yaxis().set_ticks([])
    myfig.add_axes([1.2,ax_y,0.15,0.15])
    frame1=plt.gca()
    mth.plot_etas(dic,new_fig=False)
    frame1.axes.get_yaxis().set_ticks([])
    if new_fig is True:
        plt.show()

def load_new_gbt(mjd, gbt_dir='/mnt/scratch-lustre/gusinskaia/triple_system/2021_GBT_dss/', plot_it=False):
    spec=dsa.load_triple_spectrum(gbt_dir+'%d_GBT_1400_wns_check.npz'%mjd, factor=[1,1], wnoise=True, mean0=True)
    spec.I=np.flip(spec.I, axis=1)
    spec.nI=np.flip(spec.nI, axis=1)
    spec.f=np.flip(spec.f, axis=0)
    spec.I=spec.I/spec.nI
    spec.I[:,0:130]=0.0
    spec.I[:,1900:]=0.0
    spec.I[:,390:420]=0.0
    spec.I[:,510:540]=0.0

    spec_shr=spec.shrink([10,1])
    spec_sel=spec_shr.select(freq_sel=[1150*u.MHz, 1850*u.MHz])
    
    if plot_it is True:
        fig=plt.figure(figsize=(6,2), dpi=150)
        fig.add_axes([0.0,0.0,0.3,1])
        plt.gca()
        spec_sel.plot_ds(new_fig=False)
        plt.colorbar()
        fig.add_axes([0.5,0.0,0.3,1])
        plt.gca()
        spec_sel.plot_ss(new_fig=False, tau_lim=[0.0,1.25], vmin=5e3,vmax=1e7)

    return spec_sel

    

def fit_new_gbt(spec, ntime=1, ntau=512, nfreq=11, npoints=50, freq_start=1200*u.MHz, freq_step=56*u.MHz,
               thth_method='coherent', chi2_method='Nina', reduced=False, save_fig=True):
    if thth_method=='incoherent':
        chi2_method='Eigen'
    mjd_dur=(spec.stend[1]-spec.stend[0])/ntime
    tspecs=[]
    print ('each part is ', mjd_dur*24, 'hour long')
    
    etas=np.empty((ntime,nfreq))
    dveff_ar=np.empty((ntime,nfreq,npoints))
    dveff_chi2=np.empty((ntime,nfreq,npoints))
    dveffs=np.empty((ntime,nfreq))
    e_etas=np.empty((ntime,nfreq))
    e_dveffs=np.empty((ntime,nfreq))
    freqs=np.empty((nfreq))
    times=np.empty((ntime))
    
    for i in range(0,ntime):
        spec_sel_t=spec.select(time_sel=[(spec.stend[0]+i*mjd_dur)*u.d,
                                                   (spec.stend[0]+(i+1)*mjd_dur)*u.d])
        tspecs.append(spec_sel_t)
        times[i]=np.mean(spec_sel_t.mjd.mjd)
        
        
    for k in range(0,len(tspecs)):
        spec_t=tspecs[k]
        fig=plt.figure(figsize=(6,6), dpi=150)
        for i in range(0,nfreq):
            print (k, '=========', i, '==========')

            spec_sel=spec_t.select(freq_sel=[freq_start+freq_step*i, freq_start+freq_step*(i+1)])
            spec_sel.I=spec_sel.I-np.mean(spec_sel.I)
            freqs[i]=np.mean(spec_sel.f.value)
            
            fitdic, my_f, my_mjd, res_dic=fth.daniel_pars_fit(spec_sel, curv_par='dveff', par_lims=[0.25,2.5],
                            edge=1.3,ntau=ntau,d_eff=0.499*u.kpc, npoints=npoints, chi2_method=chi2_method,
                            reduced=reduced, thth_method=thth_method)
            
            if np.isnan(fitdic['eta'] ):
                print ('%.2f MHz, MJD: %.2f,  did not converge'%(my_f.value, my_mjd))
                #etas[k,i],dveffs[k,i], e_etas[k,i], e_dveffs[k,i] = np.nan, np.nan, np.nan, np.nan
                minchi2_dveff=res_dic['par_array'][res_dic['chi2']==res_dic['chi2'].min()][0]
                minchi2_eta=fth.dveff_to_eta(minchi2_dveff, spec_sel)
                
                model_spec=mth.get_models_spec(spec_sel,minchi2_eta,edge=1.25,ntau=512)
                plot_fit_results(spec_sel, model_spec, res_dic, new_fig=False, ax_y=0.2*i, myfig=fig)
                plt.gca()
                plt.title('did not converge')

            else:
                print ('%.2f MHz, MJD: %.2f,  eta: %.2f +/- %.2f, dveff: %.2f +/- %.2f'%(my_f.value, my_mjd,
                                                        fitdic['eta'].value,fitdic['eta_err'].value,
                                                        fitdic['dveff'].value,fitdic['dveff_err'].value))

                model_spec=mth.get_models_spec(spec_sel,fitdic['eta'],edge=1.25,ntau=512)
                plot_fit_results(spec_sel, model_spec, res_dic, new_fig=False, ax_y=0.2*i, myfig=fig)
                

                etas[k,i]=fitdic['eta'].value
                e_etas[k,i]=fitdic['eta_err'].value
                dveffs[k,i]=fitdic['dveff'].value
                e_dveffs[k,i]=fitdic['dveff_err'].value
                dveff_ar[k,i,:]=res_dic['par_array'].value
                dveff_chi2[k,i,:]=res_dic['chi2']
                
        if save_fig is True:
            plt.draw()
            figname='triple_GBT_mjd%.2f_%s_%s.png'%(times[k], thth_method, chi2_method) 
            fig.patch.set_facecolor('w')
            fig.savefig(figname, format='png',bbox_inches='tight',dpi=100)
        plt.show()
            
    res_dic={'etas':etas, 'dveffs':dveffs, 'etas_err':e_etas, 'dveff_e':e_dveffs,
              'freqs':freqs, 'times':times, 'dveff_ar':dveff_ar, 'dveff_chi2':dveff_chi2}
    return res_dic

