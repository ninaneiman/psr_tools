import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os, sys, math, time

from astropy.time import Time
from astropy import units as u
from astropy.visualization import quantity_support

from pint.models import get_model

from astropy import constants as const

import astropy
from matplotlib.colors import LogNorm

import matplotlib as mpl
import scintools.ththmod as THTH
from scipy.sparse.linalg import eigsh
from scipy.optimize import curve_fit
from scipy import interpolate


plt.rcParams['figure.dpi'] = 200
print (astropy.__version__)

import ds_analysis as dsa
import danas_specrum_code as dsc

import importlib
import mc_err_prop as mcep


from scipy.fftpack import fftshift, fft2, ifftshift, ifft2, fftfreq



def angle_from_complex(x, y):
        return 2 * np.arctan( y / ( np.sqrt(x*x + y*y) + x ) )

def err_prop(value_r, value_i, error_r, error_i):


    compreal = mcep.parameter(is_input=True,
                              label='real part',
                              mu=value_r,
                              sigma=error_r,
                              unit=u.dimensionless_unscaled)

    compimag = mcep.parameter(is_input=True,
                              label='imaginary part',
                              mu=value_i,
                              sigma=error_i,
                              unit=u.dimensionless_unscaled)


    theangle = mcep.parameter(label='angle',
                              func=angle_from_complex,
                              inputs=[compreal, compimag],
                              unit=u.rad)

    angle_mean, angle_err_plus, angle_err_minus = mcep.do_mc(theangle, 100000)
    return angle_mean, angle_err_plus, angle_err_minus


def obs_vlbo(obs, vlbidir='/mnt/scratch-lustre/gusinskaia/triple_system/dss_realvlba/',
             i_fl=0, i_fh=None, i_tl=0, i_th=None,savefig=False, namefig='vlbo',
             minmax_t=False, minmax_f=False, minmax_ar=None, minus_mean=True, factor=[1,1],
            plot_ds=True, plot_ss=True, x_lim=10, y_lim=1.4, interp=False):
    triple_ds=np.load('%s%s'%(vlbidir,obs))
    center_frequency=triple_ds['c_fr']
    bw=triple_ds['bw_fr']
    end_mjd=triple_ds['mjd'][1]
    start_mjd=triple_ds['mjd'][0]
    print ('obs MJDs', start_mjd, '-', end_mjd)

    ds=triple_ds['ds']
    if 'WSRT' in obs:
        ds=dsa.shrink_2(ds, factor=factor, size=None)
        
    full_time=(end_mjd-start_mjd)
    ntbin=full_time/ds.shape[0]
    print ('timebin=', ntbin*24.*3600)
    
    a_t_mjd = (np.arange(ds.shape[0]) * ntbin * u.d)+start_mjd*u.d
    a_t = (np.arange(ds.shape[0]) * ntbin * 24.*3600. *u.s)
    if 'AO' in obs:
        a_f = np.linspace(center_frequency+bw/2,center_frequency-bw/2, ds.shape[1]) * u.MHz
    else:
        a_f = np.linspace(center_frequency-bw/2,center_frequency+bw/2, ds.shape[1]) * u.MHz
    
    a_f_all=a_f
    nfbin=(np.amax(a_f)-np.amin(a_f))/a_f.shape[0]
    print ('freqbin=', nfbin)
    if i_fh is not None:
        print ('desired max_freq:',a_f[i_fh])

    at_min=np.amin(a_t_mjd)
    at_max=np.amax(a_t_mjd)
    af_min=np.amin(a_f)
    af_max=np.amax(a_f)
    minmax_dic={'t_min':at_min,'t_max':at_max, 'f_min':af_min, 'f_max':af_max}
    print ('Actual (t,f) edges:',minmax_dic['t_min'],minmax_dic['t_max'],minmax_dic['f_min'],minmax_dic['f_max'])
    
    if i_th is None:
        i_th=len(a_t)
    if i_fh is None:
        i_fh=len(a_f)
    
    if minmax_t is True:
        for i in range(0,len(a_t)):
            dt_min=a_t_mjd[i].value-minmax_ar['t_min'].value
            dt_max=a_t_mjd[i].value-minmax_ar['t_max'].value
            if np.abs(dt_min) <ntbin/2:
                print ('t_min:', i, a_t_mjd[i], minmax_ar['t_min'], dt_min* 24.*3600.*u.s)
                i_tl=i
            if np.abs(dt_max) <ntbin/2:
                print ('t_max:', i, a_t_mjd[i], minmax_ar['t_max'], dt_max* 24.*3600.*u.s)
                i_th=i
            
    if minmax_f is True:
        for i in range(0,len(a_f)):
            df_min=a_f[i].value-minmax_ar['f_min'].value
            df_max=a_f[i].value-minmax_ar['f_max'].value
            if np.abs(df_min) <nfbin.value/2:
                print ('f_min:', i, a_f[i], minmax_ar['f_min'], df_min*u.MHz)
                i_fl=i
            if np.abs(df_max) <nfbin.value/2:
                print ('f_max:',i, a_f[i], minmax_ar['f_max'], df_max*u.MHz)
                i_fh=i
    print ('Arguments of edges: time',i_tl,i_th, 'freq', i_fl,i_fh )
    
    a_f=a_f[i_fl:i_fh+1]
    a_t=a_t[i_tl:i_th+1]
    
    new_at_min=np.amin(a_t_mjd[i_tl:i_th+1])
    new_at_max=np.amax(a_t_mjd[i_tl:i_th+1])
    new_af_min=np.amin(a_f_all[i_fl:i_fh+1])
    new_af_max=np.amax(a_f_all[i_fl:i_fh+1])
    
    new_minmax_dic={'t_min':new_at_min,'t_max':new_at_max, 'f_min':new_af_min, 'f_max':new_af_max,
                    't_len':a_t.shape[0], 'f_len':a_f.shape[0]}
    print ('New (t,f) edges:',new_minmax_dic['t_min'],new_minmax_dic['t_max'],new_minmax_dic['f_min'],
           new_minmax_dic['f_max'])
    
    if 'AO' in obs:
        ds=np.flip(ds, axis=1)
    ds=ds[i_tl:i_th+1,i_fl:i_fh+1]
    if minus_mean is True:
        ds=ds-np.mean(ds)
    
    if interp is True:
        f = interpolate.interp2d(a_f, a_t_mjd[i_tl:i_th+1], ds, kind='linear')
        new_ntbin=(minmax_ar['t_max'].value-minmax_ar['t_min'].value)/minmax_ar['t_len']
        a_t_int = np.arange(minmax_ar['t_min'].value,minmax_ar['t_max'].value, new_ntbin)
        new_nfbin=(minmax_ar['f_max'].value-minmax_ar['f_min'].value)/minmax_ar['f_len']
        a_f_new = np.arange(minmax_ar['f_min'].value,minmax_ar['f_max'].value, new_nfbin)
        
        intp_at_min=np.amin(a_t_int)
        intp_at_max=np.amax(a_t_int)
        intp_af_min=np.amin(a_f_new)
        intp_af_max=np.amax(a_f_new)
        intp_minmax_dic={'t_min':intp_at_min,'t_max':intp_at_max, 'f_min':intp_af_min, 'f_max':intp_af_max,
                        't_len':a_t_int.shape[0], 'f_len':a_f_new.shape[0]}
        print ('interp (t,f) edges:',intp_minmax_dic['t_min'],intp_minmax_dic['t_max'],intp_minmax_dic['f_min'],
               intp_minmax_dic['f_max'], 'ntbin:', new_ntbin*24.*3600, 'nfbin:', new_nfbin )
        
        ds_new = f(a_f_new, a_t_int)
        a_t_new = (np.arange(minmax_ar['t_len']) * new_ntbin * 24.*3600. *u.s)
        DS, f, t=ds_new, a_f_new*u.MHz, a_t_new
    else:
        DS, f, t = ds, a_f, a_t
    

    short_name=obs.split('1400')[0]
    print ('shapes:', DS.shape, f.shape, t.shape)
    if plot_ds is True:
        fig=plt.figure(figsize=(1.5,10.5), dpi= 80, facecolor='w', edgecolor='k')
        dsvmin,dsvmax = np.percentile(DS,[1,99])
        plt.imshow(DS.T, aspect='auto', vmin=dsvmin, vmax=dsvmax, origin='lower')
        if savefig is True:
            plt.savefig('j0337+1715_%%s.png'%(short_name,namefig),format='png', bbox_inches='tight')
        plt.show()
    
    SS, fd, tau=dsa.plot_sec_spectra_daniel(DS,t,f, pad_it=True, npad=3, plot_spectra=plot_ss)
    if plot_ss is True:
        plt.xlim(-x_lim,x_lim)
        plt.ylim(0,y_lim)
    spectrum={'ds':DS, 'f':f, 't':t,'ss':SS,'fd':fd, 'tau':tau, 'name':short_name}
    return spectrum, minmax_dic, new_minmax_dic

def get_cb(rectcb, vmin, vmax):
        rectcb=[0.97, 0.43, 0.015, 0.37]
        ax6=fig.add_axes(rectcb)
        norm2 = mpl.colors.Normalize(vmin=vmin,vmax=vmax)
        cb1 = mpl.colorbar.ColorbarBase(ax6, cmap=plt.cm.get_cmap(cmaps), norm=norm2, orientation='vertical')

def plot_prod(ss0, ss1, figsize=[10,5.5], x_lim=[-7.5,7.5],y_lim=[0.02,0.4],cmap='viridis', save_prod=False):
        fig=plt.figure(figsize=(figsize[0],figsize[1]), dpi= 80, facecolor='w', edgecolor='k')
        rect1=[0.0,0.0,0.2,1]
        ax1 = fig.add_axes(rect1)
        dsvmin,dsvmax = np.percentile(ss0['ds'],[1,99])
        ax1.imshow(ss0['ds'].T, aspect='auto', vmin=dsvmin, vmax=dsvmax, origin='lower',
                   extent=(0,ss0['t'].value[-1]/3600,ss0['f'].value[0],ss0['f'].value[-1]))
        ax1.set_ylabel('MHz')
        ax1.set_xlabel('Time (hr)')
        ax1.set_title(ss0['name'])
        rect2=[0.25,0.0,0.2,1]
        ax2 = fig.add_axes(rect2)
        dsvmin,dsvmax = np.percentile(ss1['ds'],[1,99])
        ax2.imshow(ss1['ds'].T, aspect='auto', vmin=dsvmin, vmax=dsvmax, origin='lower',
                   extent=(y_lim[0],y_lim[1],x_lim[0],x_lim[1]))
        ax2.set_ylabel('MHz')
        ax2.set_xlabel('Time (hr)')
        ax2.set_title(ss1['name'])
        
        rect3=[0.52,0.56,0.45,0.44]
        ax3 = fig.add_axes(rect3)
        ss_min=np.median(np.abs(ss0['ss'])**2)/10
        ss_max=np.max(np.abs(ss0['ss'])**2)*np.exp(-2.7)
        ax3.imshow(np.abs(ss0['ss'])**2,norm=LogNorm(),origin='lower',aspect='auto',
                   extent=dsa.ext_find(ss0['fd'],ss0['tau']),vmax=ss_max, vmin=ss_min)

        ax3.set_xlabel(ss0['tau'].unit.to_string('latex'))
        ax3.set_ylabel(ss0['tau'].unit.to_string('latex'))
        ax3.set_xlim(x_lim[0], x_lim[1])
        ax3.set_ylim(y_lim[0], y_lim[1])
        ax3.set_title(ss0['name'])
        rectcb=[1.0, 0.56, 0.015, 0.44]
        ax6=fig.add_axes(rectcb)
        norm2 = mpl.colors.LogNorm(vmin=ss_min,vmax=ss_max)
        cb1 = mpl.colorbar.ColorbarBase(ax6, cmap=plt.cm.get_cmap('viridis'), norm=norm2, orientation='vertical')
        
        rect4=[0.52,0.0,0.45,0.44]
        ax4 = fig.add_axes(rect4)
        ss_min=np.median(np.abs(ss1['ss'])**2)/10
        ss_max=np.max(np.abs(ss1['ss'])**2)*np.exp(-2.7)
        ax4.imshow(np.abs(ss1['ss'])**2,norm=LogNorm(),origin='lower',aspect='auto',
                   extent=dsa.ext_find(ss1['fd'],ss1['tau']),vmax=ss_max, vmin=ss_min)

        ax4.set_xlabel(ss1['fd'].unit.to_string('latex'))
        ax4.set_ylabel(ss1['tau'].unit.to_string('latex'))
        ax4.set_xlim(x_lim[0], x_lim[1])
        ax4.set_ylim(y_lim[0], y_lim[1])
        ax4.set_title(ss1['name'])
        rectcb2=[1.0, 0.0, 0.015, 0.44]
        ax7=fig.add_axes(rectcb2)
        norm2 = mpl.colors.LogNorm(vmin=ss_min,vmax=ss_max)
        cb2 = mpl.colorbar.ColorbarBase(ax7, cmap=plt.cm.get_cmap('viridis'), norm=norm2, orientation='vertical')
        if save_prod is True:
            plt.savefig('vlbo_prod_%s%s.png'%(ss0['name'],ss1['name']),format='png', bbox_inches='tight')
        return

def plot_cross_spectra(ss0, ss1, figsize=[10,5.5], x_lim=[-7.5,7.5],y_lim=[0.02,0.4],cmap='RdBu', ss_cmap='viridis',
             vm=3.14, save_css=False):
    Icross_single=ss0['ss']*np.conjugate(ss1['ss'])

    fig=plt.figure(figsize=(figsize[0],figsize[1]), dpi= 80, facecolor='w', edgecolor='k')
    
    cz=0.4
    gap=0.15
    cbw=0.015
    
    rect1=[0.0,cz+gap,cz,cz]
    ax1 = fig.add_axes(rect1)
    rectcb1=[cz+cbw, cz+gap, cbw, cz]
    ax5=fig.add_axes(rectcb1)
    
    fd_l, fd_r, tau_b, tau_t=find_edges(Icross_single, ss0['fd'], ss0['tau'], x_lim, y_lim)
    new_Icross=Icross_single[tau_t:tau_b,fd_l:fd_r]
    new_fd=ss0['fd'][fd_l:fd_r]
    new_tau=ss0['tau'][tau_t:tau_b]
    
    css_min=np.median(np.abs(new_Icross))/10
    css_max=np.max(np.abs(new_Icross))*np.exp(-2.7)

    ax1.imshow(np.abs(new_Icross),norm=LogNorm(), extent=dsa.ext_find(new_fd, new_tau),
      aspect='auto',origin='lower', cmap=plt.cm.get_cmap(ss_cmap),vmax=css_max, vmin=css_min)
    ax1.set_title('Power')
    norm1 = mpl.colors.LogNorm(vmin=css_min,vmax=css_max)
    cb1 = mpl.colorbar.ColorbarBase(ax5,cmap=plt.cm.get_cmap(ss_cmap),norm=norm1,orientation='vertical')

    rect2=[cz+gap,cz+gap,cz,cz]
    ax2 = fig.add_axes(rect2)
    rectcb2=[cz*2+gap+cbw, cz+gap, cbw, cz]
    ax6=fig.add_axes(rectcb2)
    ax2.imshow(np.angle(new_Icross), extent=dsa.ext_find(new_fd, new_tau),
      aspect='auto',origin='lower', cmap=cmap, vmin=-vm, vmax=vm)
    ax2.set_title('Phases')
    norm2 = mpl.colors.Normalize(vmin=-vm,vmax=vm)
    cb2 = mpl.colorbar.ColorbarBase(ax6, cmap=cmap, norm=norm2, orientation='vertical')

    rect3=[0.0,0.0,cz,cz]
    ax3 = fig.add_axes(rect3)
    rectcb3=[cz+cbw, 0.0, cbw, cz]
    ax7=fig.add_axes(rectcb3)
    real_mx=np.max([np.abs(np.amax(new_Icross.real)),np.abs(np.amin(new_Icross.real))])/2
    imag_mx=np.max([np.abs(np.amax(new_Icross.imag)),np.abs(np.amin(new_Icross.imag))])/2
    
    #fx_mx=np.abs(np.amax(Icross_single))/2
    #fx_mx=real_mx/2

    ax3.imshow(new_Icross.real, extent=dsa.ext_find(new_fd, new_tau),
               aspect='auto',origin='lower', cmap=cmap,vmin=-real_mx, vmax=real_mx)#, vmin=-fx_mx, vmax=fx_mx)#
    ax3.set_title('Real')
    norm3 = mpl.colors.Normalize(vmin=-real_mx,vmax=real_mx)#vmin=-fx_mx, vmax=fx_mx)#
    cb3 = mpl.colorbar.ColorbarBase(ax7, cmap=cmap, norm=norm3, orientation='vertical')

    rect4=[cz+gap,0.0,cz,cz]
    ax4 = fig.add_axes(rect4)
    rectcb4=[cz*2+gap+cbw, 0.0, cbw, cz]
    ax8=fig.add_axes(rectcb4)

    ax4.imshow(new_Icross.imag, extent=dsa.ext_find(new_fd, new_tau),
               aspect='auto',origin='lower', cmap=cmap, vmin=-imag_mx, vmax=imag_mx)#, vmin=-fx_mx, vmax=fx_mx)#
    ax4.set_title('Imaginary')
    norm4 = mpl.colors.Normalize(vmin=-imag_mx,vmax=imag_mx)#vmin=-fx_mx, vmax=fx_mx)#
    cb4 = mpl.colorbar.ColorbarBase(ax8, cmap=cmap, norm=norm4, orientation='vertical')


    for ax in [ax1,ax2,ax3,ax4]:
        ax.set_xlim(x_lim[0],x_lim[1])
        ax.set_ylim(y_lim[0],y_lim[1])
        ax.set_xlabel(ss1['fd'].unit.to_string('latex'))
        ax.set_ylabel(ss1['tau'].unit.to_string('latex'))

    if save_css is True:
        fig.savefig('vlbo_css_%s%s_%.1f.png'%(ss0['name'],ss1['name'], x_lim[1]),format='png',
                    bbox_inches='tight')
    return

def plot_phase_delay(ss0, ss1, figsize=[10,5.5], x_lim=[-7.5,7.5],y_lim=[0.02,0.4],cmap='RdBu',
             vm=3.14, save_phase=False, print_phase=False):
    Icross_single=ss0['ss']*np.conjugate(ss1['ss'])

    fig=plt.figure(figsize=(figsize[0],figsize[1]), dpi= 80, facecolor='w', edgecolor='k')
    
    height=0.5
    width=0.65
    gap=0.15
    cbw=0.015
    
    rect1=[0.0,height+gap,width,height]
    ax1 = fig.add_axes(rect1)
    rectcb1=[width+cbw, height+gap, cbw, height]
    ax5=fig.add_axes(rectcb1)
        
    fd_l, fd_r, tau_b, tau_t=find_edges(Icross_single, ss0['fd'], ss0['tau'], x_lim, y_lim)
    new_Icross=Icross_single[tau_t:tau_b,fd_l:fd_r]
    new_fd=ss0['fd'][fd_l:fd_r]
    new_tau=ss0['tau'][tau_t:tau_b]
    print ('max phase:', np.amax(np.angle(new_Icross)))
    ax1.imshow(np.angle(new_Icross), extent=dsa.ext_find(new_fd, new_tau),
      aspect='auto',origin='lower', cmap=cmap, vmin=-vm, vmax=vm)
    ax1.set_title('Phases')
    norm2 = mpl.colors.Normalize(vmin=-vm,vmax=vm)
    cb2 = mpl.colorbar.ColorbarBase(ax5, cmap=cmap, norm=norm2, orientation='vertical')
    ax1.set_xlim(x_lim[0],x_lim[1])
    ax1.set_ylim(y_lim[0],y_lim[1])
    ax1.set_xlabel(ss1['fd'].unit.to_string('latex'))
    ax1.set_ylabel(ss1['tau'].unit.to_string('latex'))
    
    
    rect2=[0.0,height+gap/2-2*cbw,width,cbw]
    ax2=fig.add_axes(rect2)
    mean_I_cross, fd_mean,std_I_cross=plot_avg_ang(Icross_single, ss0['fd'], ss0['tau'], fd_lim=x_lim,
                                                   tau_lim=y_lim,plot_it=False)
    new_mean_I_cross=np.array([mean_I_cross,mean_I_cross,mean_I_cross])
    ax2.imshow(np.angle(new_mean_I_cross),aspect='auto', cmap='RdBu',
           extent=(x_lim[0],x_lim[1],0.0,2.0), vmin=-vm, vmax=vm)
    ax2.set_xlim(x_lim[0],x_lim[1])
    ax2.set_yticks([])

    rect3=[0.0, 0.0-cbw, width, height]
    ax3=fig.add_axes(rect3)
    ax3.step(fd_mean, np.angle(mean_I_cross), where='mid', zorder=2, color='grey')
    ang_mx=np.max([np.abs(np.amax(np.angle(mean_I_cross))),np.abs(np.amin(np.angle(mean_I_cross)))])
    ax3.scatter(fd_mean, np.angle(mean_I_cross),c=np.angle(mean_I_cross), cmap='RdBu', s=30, marker='o',
               vmin=-ang_mx, vmax=ang_mx, zorder=1)


    ax3.set_ylim(-ang_mx-0.1,ang_mx+0.1)
    ax3.set_ylabel('Phase')
    ax3.set_xlim(x_lim[0],x_lim[1])

    rectcb2=[width+cbw,0.0-cbw,cbw, height]
    ax7=fig.add_axes(rectcb2)
    norm1 = mpl.colors.Normalize(vmin=-ang_mx, vmax=ang_mx)
    cb1 = mpl.colorbar.ColorbarBase(ax7, cmap=cmap, norm=norm1, orientation='vertical')
    if save_phase is True:
        fig.savefig('vlbo_phase_%s%s_%.1f.png'%(ss0['name'],ss1['name'], x_lim[1]),format='png',
                    bbox_inches='tight')
    if print_phase is True:
        print_phase_shift(mean_I_cross, fd_mean)
    return mean_I_cross, fd_mean, std_I_cross

def get_piece_of_spectrum(spec, freq_ed=None, time_ed=None, plot_ss=False, plot_ds=False,
                          pad_it=True, npad=3, x_lim=7.5,y_lim=1.4):
    #spectrum={'ds':DS, 'f':f, 't':t,'ss':SS,'fd':fd, 'tau':tau}
    DS, t, f= spec['ds'],spec['t'],spec['f']
    if freq_ed is not None:
        DS=spec['ds'][:,freq_ed[0]:freq_ed[1]]
        f=spec['f'][freq_ed[0]:freq_ed[1]]
    if time_ed is not None:
        DS=spec['ds'][time_ed[0]:time_ed[1],:]
        t=spec['t'][time_ed[0]:time_ed[1]]
    if plot_ds is True:
        fig=plt.figure(figsize=(5.5,5.5), dpi= 80, facecolor='w', edgecolor='k')
        dsvmin,dsvmax = np.percentile(DS,[1,99])
        plt.imshow(DS.T, aspect='auto', vmin=dsvmin, vmax=dsvmax, origin='lower')
        plt.show()
    SS, fd, tau=dsa.plot_sec_spectra_daniel(DS,t,f, pad_it=pad_it, npad=npad, plot_spectra=plot_ss)
    if plot_ss is True:
        plt.xlim(-x_lim,x_lim)
        plt.ylim(0,y_lim)
        plt.show()
    spectrum={'ds':DS, 'f':f, 't':t,'ss':SS,'fd':fd, 'tau':tau, 'name':spec['name']+'piece'}
    return spectrum

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def find_edges(I_cross, fd_cross, tau_cross, fd_lim, tau_lim):
    fd_l=find_nearest(fd_cross, fd_lim[0])
    fd_r=find_nearest(fd_cross, fd_lim[1])
    tau_b=np.amax([find_nearest(tau_cross, tau_lim[0]),find_nearest(tau_cross, tau_lim[1])])-1
    tau_t=np.amin([find_nearest(tau_cross, tau_lim[0]),find_nearest(tau_cross, tau_lim[1])])+1
    return fd_l, fd_r, tau_b, tau_t

def plot_avg_ang(I_cross, fd_cross, tau_cross, fd_lim=[-7.5,7.5], tau_lim=[0.02,0.4], plot_it=True):
    
    fd_l, fd_r, tau_b, tau_t=find_edges(I_cross, fd_cross, tau_cross, fd_lim, tau_lim)

    mean_I_cross=np.mean(I_cross[tau_t:tau_b,fd_l:fd_r], axis=0)
    
    f_mn_Ir=np.mean(I_cross.real[tau_t:tau_b,:], axis=0)
    f_mn_Ii=np.mean(I_cross.imag[tau_t:tau_b,:], axis=0)
    #p_mn_Ir=np.mean(I_cross.real[tau_t:tau_b,fd_l:fd_r], axis=0)
    #p_mn_Ii=np.mean(I_cross.imag[tau_t:tau_b,fd_l:fd_r], axis=0)
    
    fd_range=[-10,-2,2,10]
    
    fd_ed=np.array([find_nearest(fd_cross, fd_range[0]),find_nearest(fd_cross, fd_range[1]),
                     find_nearest(fd_cross, fd_range[2]),find_nearest(fd_cross, fd_range[3])])
    
    
    p_mn_Ir=np.mean(I_cross.real[tau_t:tau_b,fd_ed[1]:fd_ed[2]], axis=0)
    p_mn_Ii=np.mean(I_cross.imag[tau_t:tau_b,fd_ed[1]:fd_ed[2]], axis=0)
    
    tipa_error_r=np.mean(np.array([np.std(f_mn_Ir[fd_ed[0]:fd_ed[1]]),np.std(f_mn_Ir[fd_ed[2]:fd_ed[3]])]))
    tipa_error_i=np.mean(np.array([np.std(f_mn_Ii[fd_ed[0]:fd_ed[1]]),np.std(f_mn_Ii[fd_ed[2]:fd_ed[3]])]))
    
    tipa_value_r=np.std(p_mn_Ir)
    tipa_value_i=np.std(p_mn_Ii)
    
    tipa_error=np.array([tipa_value_r,tipa_value_i, tipa_error_r,tipa_error_i])
    fd_ar=fd_cross[fd_l:fd_r]
    
    new_mean_I_cross=np.array([mean_I_cross,mean_I_cross,mean_I_cross])
    ang_mx=np.max([np.abs(np.amax(np.angle(new_mean_I_cross))),np.abs(np.amin(np.angle(new_mean_I_cross)))])
    if plot_it is True:
        fig=plt.figure(figsize=(10, 0.5), dpi= 80, facecolor='w', edgecolor='k')
        plt.imshow(np.angle(new_mean_I_cross),aspect='auto', cmap='RdBu',
               extent=(fd_cross[fd_l].value,fd_cross[fd_r].value,0.0,2.0), vmin=-ang_mx, vmax=ang_mx)
        plt.colorbar()
        plt.xlim(fd_lim[0],fd_lim[1])
        plt.ylabel('Just pixels')
        plt.show()
    return mean_I_cross, fd_ar, tipa_error

def plot_phase_shift(mean_I_cross, fd_mean):
    fig=plt.figure(figsize=(10, 5), dpi= 80, facecolor='w', edgecolor='k')
    plt.step(fd_mean, np.angle(mean_I_cross), where='mid', zorder=2, color='grey')
    ang_mx=np.max([np.abs(np.amax(np.angle(mean_I_cross))),np.abs(np.amin(np.angle(mean_I_cross)))])
    plt.scatter(fd_mean, np.angle(mean_I_cross),c=np.angle(mean_I_cross), cmap='RdBu', s=30, marker='o',
               vmin=-ang_mx, vmax=ang_mx, zorder=1)
    plt.colorbar()
    plt.ylim(-ang_mx-0.1,ang_mx+0.1)
    plt.ylabel('Phase')

def print_phase_shift(mean_I_cross, fd_mean):
    I_ang=np.angle(mean_I_cross)
    ang_min=np.amin(I_ang)
    ang_max=np.amax(I_ang)
    i_min=find_nearest(I_ang, ang_min)
    i_max=find_nearest(I_ang, ang_max)
    
    fd_min=fd_mean[i_min]
    fd_max=fd_mean[i_max]
    pi=math.pi
    if i_min < i_max:
        print ('Peak phases:',ang_min*u.rad.to(u.deg),ang_max*u.rad.to(u.deg),'deg; or',ang_min/pi,ang_max/pi,'pi')
        diff=ang_min-ang_max
        print ('maybe phase difference:',diff*u.rad.to(u.deg),'deg; or',diff/pi,'pi')
        print ('Doppler:', fd_min, fd_max)
    else:
        print ('Peak phases:',ang_max*u.rad.to(u.deg),ang_min*u.rad.to(u.deg),'deg; or',ang_max/pi,ang_min/pi,'pi')
        diff=ang_max-ang_min
        print ('maybe phase difference:',diff*u.rad.to(u.deg),'deg; or',diff/pi,'pi')
        print ('Doppler:', fd_max, fd_min)

def get_cross_sec_spec(ss0, ss1, figsize=[10,5.5], x_lim=[-7.5,7.5],y_lim=[0.02,0.4],cmap='RdBu',
                       spec_cmap='viridis',vm=3.14,plot_products=True,save_prod=False,plot_css=True,save_css=False,
                      plot_phase=True,print_phase=False, save_phase=False, crop_spec=False):
    if plot_products is True:
        plot_prod(ss0, ss1, figsize=figsize, x_lim=x_lim,y_lim=y_lim, save_prod=save_prod, cmap=spec_cmap)
        
    Icross_single=ss0['ss']*np.conjugate(ss1['ss'])
    
    if plot_css is True:
        plot_cross_spectra(ss0, ss1, figsize=figsize, x_lim=x_lim,y_lim=y_lim,cmap=cmap, ss_cmap=spec_cmap,
             vm=vm, save_css=save_css)
        
    if plot_phase is True:
        mean_I_cross, fd_mean, std_I_cross=plot_phase_delay(ss0, ss1, figsize=figsize, x_lim=x_lim,y_lim=y_lim,
                        cmap=cmap,vm=vm, save_phase=save_phase, print_phase=print_phase)
    if crop_spec is True:
        fd_l, fd_r, tau_b, tau_t=find_edges(Icross_single, ss0['fd'], ss0['tau'], x_lim, y_lim)
        Icross_single=Icross_single[tau_t:tau_b,fd_l:fd_r]
        fd=ss0['fd'][fd_l:fd_r]
        tau=ss0['tau'][tau_t:tau_b]
    else:
        fd=ss0['fd']
        tau=ss0['tau']
    
    
    return Icross_single, fd, tau

from scipy.odr import *

def fit_slope(x,y,x_err,y_err, print_info=False):

    # Define a function (quadratic in our case) to fit the data with.
    def f(B, x):
        '''Linear function y = m*x + b'''
        return B[0]*x + B[1]

    # Create a model for fitting.
    linear = Model(f)

    # Create a RealData object using our initiated data from above.
    data = RealData(x, y, sx=x_err, sy=y_err)

    # Set up ODR with the model and data.
    myodr = ODR(data, linear, beta0=[1.0, 1.0])
    myoutput = myodr.run()
    #myoutput.pprint()
    slope=myoutput.beta
    err_slope=myoutput.sd_beta

    sme1=slope[1]-err_slope[1]
    spe1=slope[1]+err_slope[1]
    sme0=slope[0]-err_slope[0]
    spe0=slope[0]+err_slope[0]
    fit_mean=slope[0]*(spe1-(sme1))/(spe0-(sme0))+slope[1]
    low_value = slope[0]*len(y)+slope[1]
    
    if print_info is True:
        print('slope:', slope)
        print('error:', err_slope)
        print('fit_mean:', fit_mean)
        print('low value:', low_value)
        
    return slope, err_slope


def get_phase_shift_fit(ss0,ss1,fd_lim=[-1.3,1.0],tau_lim=[0.03,0.35],figsize=[10,5.5], color='b'):
    Icross_single=ss0['ss']*np.conjugate(ss1['ss'])
    mean_I_cross, fd_mean,std_I_cross=plot_avg_ang(Icross_single, ss0['fd'], ss0['tau'], fd_lim=[-2,2],
                                                   tau_lim=tau_lim,plot_it=False)
    plt.plot(fd_mean,np.angle(mean_I_cross),  color='lightgrey', zorder=0, alpha=0.7)
    mean_I_cross, fd_mean,std_I_cross=plot_avg_ang(Icross_single, ss0['fd'], ss0['tau'], fd_lim=fd_lim,
                                                   tau_lim=tau_lim,plot_it=False)
    v, ep, em=err_prop(std_I_cross[0], std_I_cross[1], std_I_cross[2], std_I_cross[3])
    mean_err=np.mean(np.array([ep.value,em.value]))
    ph_err_fac=mean_err/v
    fd_bin=((np.amax(fd_mean)-np.amin(fd_mean))/len(fd_mean))/2
    fd_err=np.ones((len(fd_mean)))*fd_bin.value
    Iang_err=np.array(np.angle(mean_I_cross)*ph_err_fac)

    plt.errorbar(np.array(fd_mean),np.array(np.angle(mean_I_cross)),
                 yerr=Iang_err, xerr=fd_err,ls='none', marker='o', color=color, zorder=1)
    slope, err_slope=fit_slope(x=np.array(fd_mean.value),y=np.array(np.angle(mean_I_cross)),
                                                                    x_err=fd_err,y_err=Iang_err)
    model=slope[0]*fd_mean.value+slope[1]
    plt.plot(fd_mean, model, color=color, ls=':', zorder=2)
    return slope, err_slope


def plot_slopes(ss0,ss1, tau_lim=[0.01,0.4], fd_lim=[-1.5,1.5], ranges=[0.01,0.2], number=10,y_lim=[-1e10,2e10]):
    colors=plt.cm.viridis(np.linspace(0.0,1.0, number))
    tau_bin=(ranges[1]-ranges[0])/number
    fig=plt.figure(figsize=(10.5,5.5), dpi= 80, facecolor='w', edgecolor='k')
    slopes=[]
    slope_errs=[]
    
    for i in range(0,number):
        tau_lim=[ranges[0]+tau_bin*i,ranges[1]]
        slope, err_slope=get_phase_shift_fit(ss0,ss1,fd_lim=fd_lim,tau_lim=tau_lim,
                                             figsize=[10,5.5], color=colors[i])
        slopes.append(slope[0])
        slope_errs.append(err_slope[0])
    print (np.mean(slopes), '+', np.amax(slopes)-np.mean(slopes), '-', np.mean(slopes)-np.amin(slopes))
    plt.xlim(-2,2)
    slope=[np.mean(slopes),np.amax(slopes)-np.mean(slopes),np.mean(slopes)-np.amin(slopes)]


    fig=plt.figure(figsize=(10.5,10.5), dpi= 80, facecolor='w', edgecolor='k')
    for i in range(0,number):
        fd_lim=[-2,2]
        tau_lim=[ranges[0]+tau_bin*i,ranges[1]]
        Icross_single=ss0['ss']*np.conjugate(ss1['ss'])
        mean_I_cross, fd_mean, std_I_cross= plot_avg_ang(Icross_single, ss0['fd'], ss0['tau'], fd_lim=fd_lim,
                                                         tau_lim=tau_lim, plot_it=False)

        plt.plot(fd_mean,mean_I_cross.real, label='%.2f'%tau_lim[0],  color=colors[i])#, label='%.2f'%ylim_0)
        plt.plot(fd_mean,mean_I_cross.imag,color=colors[i], ls=':')
        plt.legend(loc=(1.0,0.25))
    plt.ylim(y_lim[0],y_lim[1])
    return slope


