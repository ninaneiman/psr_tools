import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os, sys, math, time
from scipy import interpolate
from astropy.time import Time
from astropy.visualization import quantity_support

from astropy import constants as const

import astropy
from matplotlib.colors import LogNorm

import matplotlib as mpl
#import scintools.ththmod as THTH
from scipy.sparse.linalg import eigsh
from scipy.optimize import curve_fit

sys.path.insert(1, '/home/gusinskaia/scintools/scintools')
import ththmod as THTH
np.seterr(divide='ignore', invalid='ignore')



def fit_wsrt_spec(my_spec, figsize=(5,7.5), spec_pieces='Default', etas_pars=[0.5,7.5,0.25],
                      pc7=False, load_model=False, eta_ref=None, ref_freq=None, edge=1.4, time_lim=2.0,
                           save_models=False, wnoise=False,
                    d_eff=1300*u.pc, mean0=True, ind_mean0=True, curv_par='dveff', saveauxname='test_wnoise'):
    if spec_pieces=='Default':
        if pc7 is True:
            spec_pieces=np.array([[1312,1328],[1332,1348],[1352,1368],[1372,1388],[1392,1408],
                      [1412,1428],[1432,1448]])
        else:
            spec_pieces=np.array([[1301,1317],[1321,1337],[1341,1357],[1361,1377],[1381,1397],
                      [1401,1417],[1421,1437],[1441,1457]])
    else:
        spec_pieces=spec_pieces
    res_fit, res_f, res_t, dics_res, all_models =[],[],[],[],[]
    f = open("etas_results_fit_%.2f_%s.txt"%(my_spec.stend[0],saveauxname), "a")
    dics_res_a=[]
    res_f_a=[]

    fig=plt.figure(figsize=figsize, dpi= 70, facecolor='w', edgecolor='k')
    fig.add_axes([0.0,0.0,0.25,1.0])
    shr_spec=my_spec.shrink(factor=[16,1])
    plt.gca()
    shr_spec.plot_ds(new_fig=False)
    for i in range(0,spec_pieces.shape[0]):
        spec_sel=my_spec.select(time_sel=[my_spec.stend[0]*u.d,my_spec.stend[1]*u.d],
                                        freq_sel=[spec_pieces[i,0]*u.MHz,spec_pieces[i,1]*u.MHz])

        if ind_mean0 is True:
            spec_sel.I=spec_sel.I-np.mean(spec_sel.I)
            spec_sel.ss=spec_sel.make_ss(pad_it=True, npad=3)
        fig.add_axes([0.9,0.006+0.125*i,0.25, 0.105])
        frame1=plt.gca()
        spec_sel.plot_ss(new_fig=False, cb=False, fd_lim=[-2.0,2.0], tau_lim=[0.0,1.3],vmin=1e6,vmax=1e9)
        frame1.axes.get_xaxis().set_ticks([])
        frame1.axes.get_yaxis().set_ticks([])
        spec_sel.get_noise()
        if load_model is False:
            fitdic, fit_f, fit_t, dic_res=fth.daniel_pars_fit(spec_sel, etas_pars=etas_pars,
                                                    edge=edge,ntau=512,curv_par=curv_par)
            if np.isnan(fitdic['eta']):
                print ('the fit did not converge, not correct eta, skip this spw')
            else:
                f.write('%.3f, %.3f, %.3f, %.3f, %.3f, %.3f  \n'%(fitdic['eta'].value,
                            fitdic['eta_err'].value,fitdic['dveff'].value, fitdic['dveff_err'].value,
                                                     fit_f.value,fit_t))
                dics_res.append(dic_res)
                dics_res_a.append(dic_res)

        else:
            eta=eta_ref*(ref_freq/spec_sel.f.mean())**2
            eta_err=0.0*eta.unit
            eta_f=np.mean(spec_sel.f)
            eta_t=np.mean(spec_sel.mjd.mjd)

        if np.isfinite(fitdic['eta']):
            res_fit.append(fitdic)
            res_f.append(fit_f)
            res_f_a.append(fit_f)
            res_t.append(fit_t)


            print (fit_f, fit_t)
            print ('eta  : %.3f'%fitdic['eta'].value, fitdic['eta'].unit,
                   'mueff: %.3f'%fitdic['mueff'].value, fitdic['mueff'].unit, 
                   'dveff: %.3f'%fitdic['dveff'].value, fitdic['dveff'].unit)
            print ('e_err: %.3f'%fitdic['eta_err'].value, fitdic['eta'].unit,
                   'm_err: %.3f'%fitdic['mueff_err'].value, fitdic['mueff'].unit, 
                   'v_err: %.3f'%fitdic['dveff_err'].value, fitdic['dveff'].unit)
            #print ('e_err:', fitdic['eta_err'], 'm_err:', fitdic['mueff_err'], 'v_err:',  fitdic['dveff_err'])

            #model_E, model_ds, model_ss, model_field, chi2 = model.get_models(spec_sel, fitdic['eta'],
            #                                                                edge=edge,ntau=512)
            model_spec=mth.get_models_spec(spec_sel, fitdic['eta'],edge=edge,ntau=512)
            
            fig.add_axes([0.3,0.006+0.125*i,0.25, 0.105])
            frame1=plt.gca()
            model_spec.plot_mds(new_fig=False)
            frame1.axes.get_xaxis().set_ticks([])
            frame1.axes.get_yaxis().set_ticks([])
            fig.add_axes([0.6,0.006+0.125*i,0.25, 0.105])
            frame1=plt.gca()
            model_spec.plot_me(new_fig=False)
            frame1.axes.get_xaxis().set_ticks([])
            frame1.axes.get_yaxis().set_ticks([])
            fig.add_axes([1.2,0.006+0.125*i,0.25, 0.105])
            frame1=plt.gca()
            model_spec.plot_mss(new_fig=False, cb=False)
            frame1.axes.get_xaxis().set_ticks([])
            frame1.axes.get_yaxis().set_ticks([])
            fig.add_axes([1.5,0.006+0.125*i,0.25, 0.105])
            frame1=plt.gca()
            model_spec.plot_mes(new_fig=False, cb=False)
            frame1.axes.get_xaxis().set_ticks([])
            frame1.axes.get_yaxis().set_ticks([])
            fig.add_axes([1.95,0.006+0.125*i,0.2, 0.105])
            if save_models is True:
                all_models.append(model_spec)
            if load_model is False:
                frame1=plt.gca()
                mth.plot_etas(dic_res,new_fig=False)
                frame1.axes.get_xaxis().set_ticks([])
                #frame1.axes.get_yaxis().set_ticks([])
                #frame1.axes.set_ylim(0.980,0.998)
                #frame1.axes.set_yscale('log')
                aux_name='fullfit'
            else:
                aux_name='fullmodel_%.1f'%fitdic['eta'].value
        print ('----------')

    fig.add_axes([2.3,0.006+0.125*4,0.3, 0.15])
    frame1=plt.gca()
    chi2sg=np.zeros((8,100))
    for g in range(0,len(dics_res_a)):
        chi2sg[g,:]=dics_res_a[g]['chi2']
        plt.plot(dics_res_a[g]['par_array'],dics_res_a[g]['chi2'], label ='%.2f MHz'%res_f_a[g].value)
        plt.legend(loc=(0.0,1.7))
    fig.add_axes([2.3,0.006+0.125*2,0.3, 0.15])
    frame1=plt.gca()
    plt.plot(dics_res_a[0]['par_array'],chi2sg.mean(0), label='mean')
    plt.plot(dics_res_a[0]['par_array'],np.median(chi2sg, axis=0), label='med')
    plt.legend(loc=(0.0,-1.7))
    
    plt.savefig('triple_%.2f_%s_%s_%s.png'%(my_spec.stend[0],my_spec.tel,aux_name,saveauxname),
                format='png',bbox_inches='tight',dpi=90)
    plt.show()
    f.close()
    return res_fit, res_f, res_t, dics_res, all_models
