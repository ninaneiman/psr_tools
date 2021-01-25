import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
#from pathlib import Path
#import pulsarbat as pb
#from time import process_time_ns
from glob import glob
import os, sys, math, time

from astropy.time import Time
from astropy.visualization import quantity_support

from astropy import constants as const

import astropy
from matplotlib.colors import LogNorm

import matplotlib as mpl
import scintools.ththmod as THTH
from scipy.sparse.linalg import eigsh
from scipy.optimize import curve_fit


import manage_data as md


plt.rcParams['figure.dpi'] = 200
print (astropy.__version__)



###Daniel's functions:
##Function for making extent tuple for plotting
def ext_find(x,y):
    dx=np.diff(x).mean()
    dy=np.diff(y).mean()
    ext=[(x[0]-dx/2).value,(x[-1]+dx/2).value,(y[0]-dy/2).value,(y[-1]+dy/2).value]
    return(ext)

##Curvature evolution with freuquency
def eta_func(f0,A):
    return(A/(f0**2))

def give_step(start, end):
    step=(end-start)/21
    return step


def plot_sec_spectra_daniel(ds, a_t, a_f, parabola=False, etas=np.arange(1,20,2),
                     plot_half=True, pad_it=False, npad=3, vmax=None, vmin=None, plot_spectra=True):
    if pad_it is True:
        ds_pad=np.pad(ds.T,((0,npad*ds.T.shape[0]),(0,npad*ds.T.shape[1])),mode='constant',
                         constant_values=ds.T.mean())
        ds=ds_pad.T
        fd=THTH.fft_axis(a_t,u.mHz,npad)
        tau=THTH.fft_axis(a_f,u.us,npad)
    else:
        fd=THTH.fft_axis(a_t,u.mHz)
        tau=THTH.fft_axis(a_f,u.us)

    SS=np.fft.fftshift(np.fft.fft2(ds.T))
    SS_ext=ext_find(fd,tau)
    if vmin is None:
        SS_min=np.median(np.abs(SS)**2)/10
    else:
        SS_min=vmin
    if vmax is None:
        SS_max=np.max(np.abs(SS)**2)*np.exp(-2.7)
    else:
        SS_max=vmax
    if plot_spectra is True:
        fig=plt.figure(figsize=(17, 4), dpi= 80, facecolor='w', edgecolor='k')
        plt.imshow(np.abs(SS)**2,norm=LogNorm(),origin='lower',aspect='auto',extent=SS_ext,
                  vmax=SS_max, vmin=SS_min)
        plt.colorbar()    

        plt.xlabel(fd.unit.to_string('latex'))
        plt.ylabel(tau.unit.to_string('latex'))
        plt.xlim(fd[0].value, fd[-1].value)

        if parabola is True:
            etas=etas*u.us/u.mHz**2
            for eta in etas:
                plt.plot(fd,(eta)*fd**2, ls='-', zorder=3)
 
        if plot_half is True:
            plt.ylim(0, tau[-1].value)
        else:
            plt.ylim(tau[0].value, tau[-1].value)

    return SS, fd, tau


def PlotFunc_nina(dspec,time,freq,SS,fd,tau,
            edges,eta_fit,eta_sig,etas,measure,etas_fit,fit_res,
            tau_lim=None,fd_lim=None,method='eigenvalue'):
    '''
    Plotting script to look at invidivual chunks
    Arguments
    dspec -- 2D numpy array containing the dynamic spectrum
    time -- 1D numpy array of the dynamic spectrum time bins (with units)
    freq -- 1D numpy array of the dynamic spectrum frequency channels (with units)
    SS -- 2D numpy array of the conjugate spectrum
    fd -- 1D numpy array of the SS fd bins (with units)
    tau -- 1D numpy array of the SS tau bins (with units)
    edges -- 1D numpy array with the bin edges for theta-theta
    eta_fit -- Best fit curvature
    eta_sig -- Error on best fir curvature
    etas -- 1D numpy array of curvatures searched over
    measure -- 1D numpy array with largest eigenvalue (method = 'eigenvalue') or chisq value (method = 'chisq') for etas
    etas_fit -- Subarray of etas used for fitting
    fit_res -- Fit parameters for parabola at extremum
    tau_lim -- Largest tau value for SS plots
    method -- Either 'eigenvalue' or 'chisq' depending on how curvature was found
    '''
    if fd_lim is None:
        fd_lim=min(2*edges.max(),fd.max().value)
    if np.isnan(eta_fit):
        eta=etas.mean()
        thth_red, thth2_red, recov, model, edges_red,w,V = THTH.modeler(SS, tau, fd, etas.mean(), edges)
    else:
        eta=eta_fit
        thth_red, thth2_red, recov, model, edges_red,w,V = THTH.modeler(SS, tau, fd, eta_fit, edges)
    ththE_red=thth_red*0
    ththE_red[ththE_red.shape[0]//2,:]=np.conjugate(V)*np.sqrt(w)
    ##Map back to time/frequency space
    recov_E=THTH.rev_map(ththE_red,tau,fd,eta,edges_red,isdspec = False)
    model_E=np.fft.ifft2(np.fft.ifftshift(recov_E))[:dspec.shape[0],:dspec.shape[1]]
    model_E*=(dspec.shape[0]*dspec.shape[1]/4)
    model_E[dspec>0]=np.sqrt(dspec[dspec>0])*np.exp(1j*np.angle(model_E[dspec>0]))
    model_E=np.pad(model_E,
                    (   (0,SS.shape[0]-model_E.shape[0]),
                        (0,SS.shape[1]-model_E.shape[1])),
                    mode='constant',
                    constant_values=0)
    recov_E=np.abs(np.fft.fftshift(np.fft.fft2(model_E)))**2
    model_E=model_E[:dspec.shape[0],:dspec.shape[1]]
    N_E=recov_E[:recov_E.shape[0]//4,:].mean()
    
    SS_ext=ext_find(fd,tau)
    SS_min=np.median(np.abs(SS)**2)/10
    SS_max=np.max(np.abs(2*recov)**2)*np.exp(-2.7)
    
    
    thth_min=np.median(np.abs(thth_red))/10
    thth_max=np.max(np.abs(thth_red))

    grid=plt.GridSpec(5,2)
    plt.figure(figsize=(8,20))
    plt.subplot(grid[0,0])
    plt.imshow(dspec,
            aspect='auto',
            extent=ext_find(time.to(u.min),freq),
            origin='lower')#,vmin=0,vmax=dspec.max())
    plt.xlabel('Time (min)')
    plt.ylabel('Freq (MHz)')
    plt.title('Data Dynamic Spectrum')
    plt.subplot(grid[0,1])
    plt.imshow(model[:dspec.shape[0],:dspec.shape[1]],
            aspect='auto',
            extent=ext_find(time.to(u.min),freq),
            origin='lower')#,vmin=0,vmax=dspec.max())
    plt.xlabel('Time (min)')
    plt.ylabel('Freq (MHz)')
    plt.title('Model Dynamic Spectrum')
    plt.subplot(grid[1,0])
    plt.imshow(np.abs(SS)**2,
            norm=LogNorm(),
            origin='lower',
            aspect='auto',
            extent=SS_ext,
            vmin=SS_min,vmax=SS_max)
    plt.xlim((-fd_lim,fd_lim))
    plt.ylim((0,tau_lim))
    plt.xlabel(r'$f_D$ (mHz)')
    plt.ylabel(r'$\tau$ (us)')
    plt.title('Data Secondary Spectrum')
    plt.subplot(grid[1,1])
    plt.imshow(np.abs(recov)**2,
            norm=LogNorm(),
            origin='lower',
            aspect='auto',
            extent=SS_ext,
            vmin=SS_min,vmax=SS_max)
    plt.xlim((-fd_lim,fd_lim))
    plt.ylim((0,tau_lim))
    plt.xlabel(r'$f_D$ (mHz)')
    plt.ylabel(r'$\tau$ (us)')
    plt.title('Model Secondary Spectrum')
    
    plt.subplot(grid[2,0])
    plt.imshow(np.abs(thth_red)**2,
            norm=LogNorm(),
            origin='lower',
            aspect='auto',
            extent=[edges_red[0],edges_red[-1],edges_red[0],edges_red[-1]],
            vmin=thth_min,vmax=thth_max**2.)
    plt.xlabel(r'$\theta_1$')
    plt.ylabel(r'$\theta_2$')
    plt.title(r'Data $\theta-\theta$')
    plt.subplot(grid[2,1])
    
    plt.imshow(np.abs(thth2_red)**2,
            norm=LogNorm(),
            origin='lower',
            aspect='auto',
            extent=[edges_red[0],edges_red[-1],edges_red[0],edges_red[-1]],
            vmin=thth_min,vmax=thth_max**2.)
    plt.xlabel(r'$\theta_1$')
    plt.ylabel(r'$\theta_2$')
    plt.title(r'Data $\theta-\theta$')
    plt.subplot(grid[3,:])
    plt.plot(etas,measure)
    if not np.isnan(eta_fit):
        exp_fit = int(('%.0e' % eta_fit.value)[2:])
        exp_err = int(('%.0e' % eta_sig.value)[2:])
        fmt = "{:.%se}" % (exp_fit - exp_err)
        fit_string = fmt.format(eta_fit.value)[:2 + exp_fit - exp_err]
        err_string = '0%s' % fmt.format(10**(exp_fit) + eta_sig.value)[1:]
        
        plt.plot(etas_fit,
            THTH.chi_par(etas_fit.value, *fit_res),
            label=r'$\eta$ = %s $\pm$ %s $s^3$' % (fit_string, err_string))
        plt.legend()
    if method == 'eigenvalue':
        plt.title('Eigenvalue Search')
        plt.ylabel(r'Largest Eigenvalue')
    else:
        plt.title('Chisquare Search')
        plt.ylabel(r'$\chi^2$')
    plt.xlabel(r'$\eta$ ($s^3$)')
    plt.subplot(grid[4,0])
    plt.imshow(np.angle(model_E),
            cmap='twilight',
            aspect='auto',
            extent=ext_find(time.to(u.min),freq),
            origin='lower',
            vmin=-np.pi,vmax=np.pi)
    plt.xlabel('Time (min)')
    plt.ylabel('Freq (MHz)')
    plt.title('Recovered Phases')
    plt.subplot(grid[4,1])
    plt.imshow(recov_E,
            norm=LogNorm(),
            origin='lower',
            aspect='auto',
            extent=ext_find(fd,tau),
            vmin=N_E)
    plt.xlim((-fd_lim,fd_lim))
    plt.ylim((0,tau_lim))
    plt.xlabel(r'$f_D$ (mHz)')
    plt.ylabel(r'$\tau$ (us)')
    plt.title('Recovered Wavefield')
    plt.colorbar()
    plt.tight_layout()


def PlotFunc_nina_save(dspec,time,freq,SS,fd,tau,
            edges,eta_fit,eta_sig,etas,measure,etas_fit,fit_res,
            tau_lim=None,fd_lim=None,method='eigenvalue'):
    '''
    Plotting script to look at invidivual chunks
    Arguments
    dspec -- 2D numpy array containing the dynamic spectrum
    time -- 1D numpy array of the dynamic spectrum time bins (with units)
    freq -- 1D numpy array of the dynamic spectrum frequency channels (with units)
    SS -- 2D numpy array of the conjugate spectrum
    fd -- 1D numpy array of the SS fd bins (with units)
    tau -- 1D numpy array of the SS tau bins (with units)
    edges -- 1D numpy array with the bin edges for theta-theta
    eta_fit -- Best fit curvature
    eta_sig -- Error on best fir curvature
    etas -- 1D numpy array of curvatures searched over
    measure -- 1D numpy array with largest eigenvalue (method = 'eigenvalue') or chisq value (method = 'chisq') 
    for etas
    etas_fit -- Subarray of etas used for fitting
    fit_res -- Fit parameters for parabola at extremum
    tau_lim -- Largest tau value for SS plots
    method -- Either 'eigenvalue' or 'chisq' depending on how curvature was found
    '''
    if fd_lim is None:
        fd_lim=min(2*edges.max(),fd.max().value)
    if np.isnan(eta_fit):
        eta=etas.mean()
        thth_red, thth2_red, recov, model, edges_red,w,V = THTH.modeler(SS, tau, fd, etas.mean(), edges)
    else:
        eta=eta_fit
        thth_red, thth2_red, recov, model, edges_red,w,V = THTH.modeler(SS, tau, fd, eta_fit, edges)
    ththE_red=thth_red*0
    ththE_red[ththE_red.shape[0]//2,:]=np.conjugate(V)*np.sqrt(w)
    ##Map back to time/frequency space
    recov_E=THTH.rev_map(ththE_red,tau,fd,eta,edges_red,isdspec = False)
    model_E=np.fft.ifft2(np.fft.ifftshift(recov_E))[:dspec.shape[0],:dspec.shape[1]]
    model_E*=(dspec.shape[0]*dspec.shape[1]/4)
    model_E[dspec>0]=np.sqrt(dspec[dspec>0])*np.exp(1j*np.angle(model_E[dspec>0]))
    model_E=np.pad(model_E,
                    (   (0,SS.shape[0]-model_E.shape[0]),
                        (0,SS.shape[1]-model_E.shape[1])),
                    mode='constant',
                    constant_values=0)
    recov_E=np.abs(np.fft.fftshift(np.fft.fft2(model_E)))**2
    model_E=model_E[:dspec.shape[0],:dspec.shape[1]]
    N_E=recov_E[:recov_E.shape[0]//4,:].mean()

    SS_ext=ext_find(fd,tau)
    SS_min=np.median(np.abs(SS)**2)/10
    SS_max=np.max(np.abs(2*recov)**2)*np.exp(-2.7)


    thth_min=np.median(np.abs(thth_red))/10
    thth_max=np.max(np.abs(thth_red))

    
    dic={'ds':dspec, 'time':time, 'freq':freq, 'ds_model':model[:dspec.shape[0],:dspec.shape[1]],
               'ss':np.abs(SS)**2, 'SS_min':SS_min,'SS_max':SS_max, 'ss_model':np.abs(recov)**2,
              'thth':np.abs(thth_red)**2,'ed_red':edges_red, 'thth_min':thth_min,'thth_max':thth_max**2.,
              'thth_model':np.abs(thth2_red)**2, 'eta_fit':etas_fit,'fit_res':fit_res,
              'phases':np.angle(model_E),'field':recov_E,'N_E':N_E}
    return dic


###############
#Nina's functions:

def get_ready_ds(data, fac=4,size=[1024,9600,8], ntbin_init=2.684354782104492, freq_range=[312.9375,350.4375],
                    onp=[1,4], offp=[5,8]):
    data=md.shrink_3(data,factor=[fac,1,1], size=size)
    ntbin=ntbin_init*fac
    lenght=data.shape[0]*ntbin
    print (lenght/60., 'min')
    a_t = (np.arange(data.shape[0]) * ntbin * u.s)
    a_f = np.linspace(freq_range[0],freq_range[1], data.shape[1]) * u.MHz
    ds=md.get_dyn_spectra(data, on=onp, off=offp, fig_width=3)
    return ds,a_t,a_f


def get_ds_piece(ds,a_t,a_f,fr=None,tm=None):
    if fr is None:
        fr=[0,len(ds.shape[1])]
    else:
        fr=fr
    if tm is None:
        tm=[0,len(ds.shape[0])]
    else:
        tm=tm
    n_ds=ds[tm[0]:tm[1],fr[0]:fr[1]]
    n_t = a_t[tm[0]:tm[1]]
    n_f = a_f[fr[0]:fr[1]]
    return n_ds, n_t, n_f


def work_on_piece(n_ds,n_t, n_f,etas_pars=[1e4,2e5,0.1e5], edge=.4, plot_tt=True, npad=3, ntau=512,
                 vmin=1e8, vmax=1e15, xlims=None):
    dsvmin,dsvmax = np.percentile(n_ds,[1,99])
    plt.imshow(n_ds, aspect='auto', vmin=dsvmin, vmax=dsvmax)
    plt.show()
    sec, fd, tau=plot_sec_spectra_daniel(n_ds, n_t, n_f, parabola=False, etas=np.arange(1,20,2),
                     plot_half=True, pad_it=False, npad=3, vmax=vmax, vmin=vmin)
    if xlims is not None:
        plt.xlim(xlims[0],xlims[1])
    plt.show()
    ##Pad before forming secondary spectrum
    ds_pad=np.pad(n_ds.T,((0,npad*n_ds.T.shape[0]),(0,npad*n_ds.T.shape[1])),mode='constant',
                     constant_values=n_ds.T.mean())
    p_sec, p_fd, p_tau =plot_sec_spectra_daniel(n_ds, n_t, n_f, parabola=False, etas=np.arange(1,20,2),
                     plot_half=True, pad_it=True, npad=npad, vmax=vmax, vmin=vmin)
    if xlims is not None:
        plt.xlim(xlims[0],xlims[1])
    plt.show()
    etas_init=np.arange(etas_pars[0],etas_pars[1],etas_pars[2])
    sss, ffd, ttau =plot_sec_spectra_daniel(n_ds, n_t, n_f, parabola=True,
                                                etas=np.arange(etas_pars[0],etas_pars[1],etas_pars[2]),
                     plot_half=True, pad_it=True, npad=npad, vmax=vmax, vmin=vmin)
    if xlims is not None:
        plt.xlim(xlims[0],xlims[1])
    plt.show()
    edges=np.linspace(-edge,edge,ntau)
    etas=etas_init*u.us/u.mHz**2
    if plot_tt is True:
        dic, rows=math.modf(len(etas_init)/3)
        raws=int(rows)
        fig, axs = plt.subplots(raws, 3,figsize=(5,10))
        i=0
        k=0
        for eta in etas:
            thth_red,edges_red=THTH.thth_redmap(p_sec, p_tau, p_fd, eta, edges)
            j=np.mod(i, 3)
            axs[k, j].imshow(np.abs(thth_red),
                           norm=LogNorm(),
                           extent=[edges_red[0],edges_red[-1],edges_red[0],edges_red[-1]],
                           origin='lower')
            axs[k, j].set_xlabel(r'$\theta_1$', fontsize=5)
            axs[k, j].set_ylabel(r'$\theta_2$',fontsize=5)
            axs[k, j].set_title(r'$\eta$ =%s $s^3$' %eta.to_value(u.s**3), fontsize=5)
            axs[k, j].tick_params(length=2, width=1, which='major', labelsize=5, colors='k')
            axs[k, j].tick_params(length=1, width=1, which='minor', labelsize=5, colors='k')


            #axs[k, j].colorbar()
            #plt.show()
            if j==2:
                k=k+1
            i=i+1
        for l in range(0,np.mod(len(etas_init),3)+1):
            
            axs[raws-1, 2-l].imshow(np.abs(thth_red),
                           norm=LogNorm(),
                           extent=[edges_red[0],edges_red[-1],edges_red[0],edges_red[-1]],
                           origin='lower')
            axs[raws-1, 2-l].set_xlabel(r'$\theta_1$', fontsize=5)
            axs[raws-1, 2-l].set_ylabel(r'$\theta_2$',fontsize=5)
            axs[raws-1, 2-l].set_title(r'$\eta$ =%s $s^3$' %eta.to_value(u.s**3), fontsize=5)
            axs[raws-1, 2-l].tick_params(length=2, width=1, which='major', labelsize=5, colors='k')
            axs[raws-1, 2-l].tick_params(length=1, width=1, which='minor', labelsize=5, colors='k')
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=-0.6, hspace=0.3)
        plt.show()
    return p_sec, p_fd, p_tau, etas_init



def work_on_piece_woplots(n_ds,n_t, n_f,etas_pars=[1e4,2e5,0.1e5], edge=.4, npad=3, ntau=512):
    
    ds_pad=np.pad(n_ds.T,((0,npad*n_ds.T.shape[0]),(0,npad*n_ds.T.shape[1])),mode='constant',
                     constant_values=n_ds.T.mean())
    ds=ds_pad.T
    p_fd=THTH.fft_axis(n_t,u.mHz,npad)
    p_tau=THTH.fft_axis(n_f,u.us,npad)

    p_sec=np.fft.fftshift(np.fft.fft2(ds.T))
    
    etas_init=np.arange(etas_pars[0],etas_pars[1],etas_pars[2])
    edges=np.linspace(-edge,edge,ntau)
    etas=etas_init*u.us/u.mHz**2
    return p_sec, p_fd, p_tau, etas_init



def get_daniel_eigenv_fit(ds, a_t,a_f, n_ds, n_t, n_f, p_sec, p_fd, p_tau, etas_init,edge=.4,
                           ntau=512,yed=30,xed=15):
    ##Define range of curvatures to search
    eta_low=np.amin(etas_init)*u.us/(u.mHz**2)
    eta_high=np.amax(etas_init)*u.us/(u.mHz**2)
    ##Estimate noise in dynamic spectrum
    temp=np.fft.fftshift(np.abs(np.fft.fft2(ds.T)/np.sqrt(ds.T.shape[0]*ds.T.shape[1]))**2)
    N=np.sqrt(temp[:temp.shape[0]//6,:temp.shape[1]//6].mean())

    ##Setup for chisq search
    etas2=np.linspace(eta_low.value,eta_high.value,100)*eta_low.unit
    eigs=np.zeros(etas2.shape[0])

    ##Determine chisq for each curvature
    mask=np.ones(n_ds.T.shape,dtype=bool)
    
    ##Bin Edges for thth plot 
    ##(should be symmetric about 0 and contain an even number of points)
    edges=np.linspace(-edge,edge,512)
    
    for i in range(etas2.shape[0]):
        eta=etas2[i]
        ##Fits a model generated by taking the outer product of the dominate eigenvector
        ##(in theta-theta space) and then mapping back to the dynamic spectrum
        eigs[i]=THTH.Eval_calc(p_sec, p_tau, p_fd, eta, edges)

    ##Fit for a parabola around the minimum
    e_min=etas2[eigs==eigs.max()][0]
    etas_fit=etas2[np.abs(etas2-e_min)<.1*e_min]
    eigs_fit=eigs[np.abs(etas2-e_min)<.1*e_min]
    C=eigs_fit.max()
    x0=etas_fit[eigs_fit==C][0].value
    A=(eigs_fit[0]-C)/((etas_fit[0].value-x0)**2)
    popt,pcov=curve_fit(THTH.chi_par,etas_fit.value,eigs_fit,p0=np.array([A,x0,C]))
    eta_fit=popt[1]*etas2.unit
    eta_sig=np.sqrt(-(eigs_fit-THTH.chi_par(etas_fit.value,*popt)).std()/popt[0])*etas2.unit

    PlotFunc_nina(n_ds.T,n_t,n_f,p_sec,p_fd,p_tau,
                edges,eta_fit,eta_sig,etas2,eigs,etas_fit,popt,tau_lim=yed, fd_lim=xed)
    return eta_fit, eta_sig, np.mean(n_f)

def get_daniel_chi2_fit(ds, a_t,a_f, n_ds, n_t, n_f, p_sec, p_fd, p_tau, etas_init,edge=.4,ntau=512,yed=30,xed=15,save=False):
    ##Define range of curvatures to search
    eta_low=np.amin(etas_init)*u.us/(u.mHz**2)
    eta_high=np.amax(etas_init)*u.us/(u.mHz**2)
    ##Estimate noise in dynamic spectrum
    temp=np.fft.fftshift(np.abs(np.fft.fft2(ds.T)/np.sqrt(ds.T.shape[0]*ds.T.shape[1]))**2)
    N=np.sqrt(temp[:temp.shape[0]//6,:temp.shape[1]//6].mean())
    ##Setup for chisq search
    etas2=np.linspace(eta_low.value,eta_high.value,100)*eta_low.unit
    chisq=np.zeros(etas2.shape[0])
    edges=np.linspace(-edge,edge,ntau)
    ##Determine chisq for each curvature
    mask=np.ones(n_ds.T.shape,dtype=bool)
    #THTH.chisq_calc(dspec2,SS, tau, fd, eta, edges,mask,N)
    for i in range(etas2.shape[0]):
        eta=etas2[i]
        ##Fits a model generated by taking the outer product of the dominate eigenvector
        ##(in theta-theta space) and then mapping back to the dynamic spectrum
        chisq[i]=THTH.chisq_calc(n_ds.T,p_sec, p_tau, p_fd, eta, edges,mask,N)

    ##Fit for a parabola around the minimum
    e_min=etas2[chisq==chisq.min()][0]
    etas_fit=etas2[np.abs(etas2-e_min)<.1*e_min]
    chisq_fit=chisq[np.abs(etas2-e_min)<.1*e_min]
    C=chisq_fit.min()
    x0=etas_fit[chisq_fit==C][0].value
    A=(chisq_fit[0]-C)/((etas_fit[0].value-x0)**2)
    popt,pcov=curve_fit(THTH.chi_par,etas_fit.value,chisq_fit,p0=np.array([A,x0,C]))
    eta_fit=popt[1]*etas2.unit
    eta_sig=np.sqrt((chisq_fit-THTH.chi_par(etas_fit.value,*popt)).std()/popt[0])*etas2.unit
    #chisq[i]=THTH.chisq_calc(dspec2,SS, tau, fd, eta, edges,mask,N)
    #THTH.PlotFunc(n_ds.T,n_t,n_f,p_sec,p_fd,p_tau,
    #        edges,eta_fit,eta_sig,etas2,chisq,etas_fit,popt,tau_lim=yed,method='chisq')


    PlotFunc_nina(n_ds.T,n_t,n_f,p_sec,p_fd,p_tau,
            edges,eta_fit,eta_sig,etas2,chisq,etas_fit,popt,tau_lim=yed, fd_lim=xed,method='chisq')
    dic=0.0
    if save is True:
        dic=PlotFunc_nina_save(n_ds.T,n_t,n_f,p_sec,p_fd,p_tau,
            edges,eta_fit,eta_sig,etas2,chisq,etas_fit,popt,tau_lim=yed, fd_lim=xed,method='chisq')
    return eta_fit, eta_sig, np.mean(n_f), dic

###############################
####Data manage funtions

def factor_of_two(x, n=5, odd=True):
    x=int(x)
    if (x%2!=0) and (odd is True):
        x=x-1
    os.system("factor %d > numout.txt"%x)
    arr0=np.genfromtxt("numout.txt")
    arr=arr0[1:len(arr0)]
    twos=arr[(arr[:]==2)]
    N=len(twos)

    while (N < n):
        os.system("factor %d > numout.txt"%x)
        arr0=np.genfromtxt("numout.txt")
        arr=arr0[1:-1]
        twos=arr[(arr[:]==2)]
        N=len(twos)
        x=x-2
    print (arr)
    return x

def shrink_2(array, factor=[1,1], size=None):
    #ideally need error function for factor to always be factor of 2
    a_shape=np.shape(array)
    if size is None:
        size=[]
        for j in range(0,len(factor)):
            if factor[j]==1:
                print (array.shape[j])
                size.append(array.shape[j])
            else:
                size.append(factor_of_two(array.shape[j],n=int(math.log(factor[j],2)),odd=True))
        size=np.array(size)
        print (size)
    else:
        size=size

    new_data=np.zeros((int(size[0]/factor[0]),array.shape[1]))
    print (new_data.shape)
    for i in range(0,size[0],factor[0]):
        k=int(i/factor[0])-1
        new_data[k,:]=np.mean(array[i:i+factor[0],:],axis=0)
    array=new_data

    nnew_data=np.zeros((array.shape[0],int(size[1]/factor[1])))
    for i in range(0,size[1],factor[1]):
        k=int(i/factor[1])
        nnew_data[:,k]=np.mean(array[:,i:i+factor[1]],axis=1)
    array=nnew_data
    new_shape=np.shape(nnew_data)
    print ('shapes', a_shape, new_shape)
    return nnew_data

###############################
#velovity function


def fr_cos(x,vel,Pb_d,T_asc0):
    fun_cos=vel*np.sin(2.*math.pi*((x-T_asc0)/Pb_d))
    return fun_cos

def orb_phase(x,Pb_d,T_asc0):
    o_ph=np.sin(2.*math.pi*(((x-T_asc0)/Pb_d)))
    return o_ph

def vel_calc(mjd_start,mjd_end, T_asc0=55920.40771662, Pb_d=1.6293932, ab_x=1.92379, print_output=True,t_range=0):
    a_x=ab_x*const.c.to('cm/s').value
    Pb_s=Pb_d*(24.0*3600.0)#sec
    
    vel=(2*math.pi*a_x/Pb_s)/1e5
    
    tascs=[]
    T_ascn=T_asc0
    obs_dur=mjd_end-mjd_start
    for i in range(0,250):
        T_ascn=T_ascn+Pb_d
        tascs.append(T_ascn)

    for i in range(0,len(tascs)):
        if np.abs(mjd_start - tascs[i]) <0.8:
            ctasc=tascs[i]
    cos_diff=np.abs(fr_cos(mjd_start,vel,Pb_d,T_asc0)-fr_cos(mjd_end,vel,Pb_d,T_asc0))
    
    x=np.arange(mjd_start-Pb_d-t_range,mjd_end+Pb_d+t_range,0.005)
    xobs=np.arange(mjd_start,mjd_end,0.005)
    x_amp=fr_cos(x,vel,Pb_d,T_asc0)
    xobs_amp=fr_cos(xobs,vel,Pb_d,T_asc0)
    if print_output is True:
        print ('Maximum velocity')
        print (vel,'km/s')
        print ('Observation epoch:')
        print (mjd_start,'-',mjd_end, '; %.2f hours'%(obs_dur*24.))
        print ('Closest_tasc:')
        print (ctasc)
        print ('Delta velocity:')
        print ('st.: %.3f'%fr_cos(mjd_start,vel,Pb_d,T_asc0), 'end: %.2f'%fr_cos(mjd_end,vel,Pb_d,T_asc0),
           'diff: %.2f'%cos_diff)
    return x, x_amp, xobs, xobs_amp
    

def get_vel_function(mjd_start,mjd_end, T_asc0=55920.40771662, Pb_d=1.6293932, ab_x=1.92379):
    a_x=ab_x*const.c.to('cm/s').value
    Pb_s=Pb_d*(24.0*3600.0)#sec
    
    vel=(2*math.pi*a_x/Pb_s)/1e5
    
    x=np.arange(mjd_start,mjd_end,0.005)
    x_amp=fr_cos(x,vel,Pb_d,T_asc0)
    return x, x_amp

def get_vel_model(mjd_start,mjd_end, s=0.5, pm=29.7):
    xi, xi_amp=get_vel_function(mjd_start,mjd_end, T_asc0=55920.40771662, Pb_d=1.6293932, ab_x=1.92379)
    xo, xo_amp=get_vel_function(mjd_start,mjd_end, T_asc0=56233.93512,Pb_d=327.25685, ab_x=117.992)
    xe, xe_amp=get_vel_function(mjd_start,mjd_end, T_asc0=55698.9+365.25/4.,Pb_d=365.25, ab_x=499.00478)
    s_fac=(1.-s)/s
    func_1=s_fac*(xi_amp+xo_amp+pm)+xe_amp
    func_2=s_fac*(xi_amp+xo_amp+pm)-xe_amp
    
    func_3=-s_fac*(xi_amp+xo_amp+pm)+xe_amp
    func_4=-s_fac*(xi_amp+xo_amp+pm)-xe_amp
    
    return xi, func_1, func_2, func_3, func_4

def get_vel(eta,nu=1800*u.MHz,s=0.5, d_psr=1300*u.pc):
    c=const.c
    d_eff=((1.-s)/s)*d_psr
    v_eff=((c*d_eff)/(eta*2.*nu**2.))**0.5
    return  v_eff.decompose().to('km/s')


###Plotting functions

fig=plt.figure(figsize=(6, 4), dpi= 80, facecolor='w', edgecolor='k')
rect1=[0.0,0.0,1.0,1.0]


def plot_vels_woax(data,fig=fig,rect=[0.0,-0.4,1.0,0.3], print_output=False, x_lim=0.05, t_range=0.0):
    ax1 = fig.add_axes(rect)
    o_clr=np.array([241, 196, 15])/255.
    o_clr2=np.array([247, 220, 111])/255.
    i_clr=np.array([231, 76, 60])/255.
    i_clr2=np.array([255, 171, 145])/255.
    e_clr=np.array([46, 134, 193])/255.
    e_clr2=np.array([144, 202, 249])/255.
    
    full_time=(data['mjd'][1]-data['mjd'][0])*(24.*3600.)
    ntbin=full_time/data['ds'].shape[0]
    a_t = (np.arange(data['ds'].shape[0]) * ntbin * u.s)
    
    timax=(np.amax(a_t)/3600)/11.37
    
    xi, xi_amp, xiobs, xiobs_amp=vel_calc(data['mjd'][0],data['mjd'][1],print_output=print_output, t_range=t_range)
    ax1.plot(xi,xi_amp, lw=2, color=i_clr, ls=':', alpha=0.4)
    ax1.plot(xiobs,xiobs_amp, lw=2, color=i_clr)
    ax1.text(xiobs[0], xiobs_amp[0], 'inner', color=i_clr)
    ax1.plot(xi,-xi_amp, lw=2, color=i_clr, ls=':', alpha=0.4)
    ax1.plot(xiobs,-xiobs_amp, lw=2, color=i_clr2)
    #ax1.text(xiobs[0], -xiobs_amp[0], 'inner', color=i_clr2)
    xo, xo_amp, xoobs, xoobs_amp=vel_calc(data['mjd'][0],data['mjd'][1], T_asc0=56233.93512,
                                          Pb_d=327.25685, ab_x=117.992,print_output=print_output)
    ax1.text(xoobs[0], xoobs_amp[0], 'outer', color=o_clr)
    ax1.plot(xo,xo_amp, lw=2, color=o_clr, ls=':', alpha=0.5)
    ax1.plot(xoobs,xoobs_amp, lw=2, color=o_clr)
    #ax1.text(xoobs[0], -xoobs_amp[0], 'outer', color=o_clr2)
    ax1.plot(xo,-xo_amp, lw=2, color=o_clr2, ls=':', alpha=0.5)
    ax1.plot(xoobs,-xoobs_amp, lw=2, color=o_clr2)
    xe, xe_amp, xeobs, xeobs_amp=vel_calc(data['mjd'][0],data['mjd'][1], T_asc0=55698.9+365.25/4.,
                                          Pb_d=365.25, ab_x=499.00478,print_output=print_output)
    ax1.plot(xe,-xe_amp, lw=2, color=e_clr, ls=':', alpha=0.4)
    ax1.plot(xeobs,-xeobs_amp, lw=2, color=e_clr)
    ax1.plot(xeobs,xeobs_amp, lw=2, color=e_clr2)
    ax1.plot(xe,xe_amp, lw=2, color=e_clr2, ls=':', alpha=0.4)
    ax1.text(xeobs[0], -xeobs_amp[0], 'Earth', color=e_clr)
    #ax1.text(xeobs[0], xeobs_amp[0], 'Earth', color=e_clr2)
    ax1.set_xlim(data['mjd'][0]-x_lim,data['mjd'][1]+x_lim)
    v1=np.mean(xiobs_amp)+np.mean(xoobs_amp)-np.mean(xeobs_amp)
    v2=np.mean(xiobs_amp)+np.mean(xoobs_amp)+np.mean(xeobs_amp)
    ax1.set_title('i: %.2f; o: %.2f; E: %.2f'%(np.mean(xiobs_amp),np.mean(xoobs_amp),
                                                   np.abs(np.mean(xeobs_amp))))
    ax1.set_xlabel('MJD')
    ax1.set_ylabel('Velocity (km/s)')




def plot_ds_triple_wa_ax(filenpz='/mnt/scratch-lustre/gusinskaia/triple_system/5602579_AO_1400_ds.npz',
                         fig=fig, rect=[0.0,0.0,1.0,1.0], ylims=[1150,1850], factor=[8,1]):
    ax = fig.add_axes(rect)
    triple_ds=np.load(filenpz)
    print (triple_ds['ds'].shape)
    ds=triple_ds['ds']
    if 'WSRT' in filenpz:
        ds=shrink_2(ds, factor=factor, size=None)

    end_mjd=triple_ds['mjd'][1]
    start_mjd=triple_ds['mjd'][0]
    center_frequency=triple_ds['c_fr']
    bw=triple_ds['bw_fr']
    vmin,vmax = np.percentile(triple_ds['ds'],[1,99])

    full_time=(end_mjd-start_mjd)*(24.*3600.)
    ntbin=full_time/ds.shape[0]
    a_t = (np.arange(ds.shape[0]) * ntbin * u.s)
    if 'AO' in filenpz:
        a_f = np.linspace(center_frequency+bw/2,center_frequency-bw/2, ds.shape[1]) * u.MHz
        ax.imshow(ds[:,:].T,extent=(0,(end_mjd-start_mjd)*24,center_frequency+bw/2,center_frequency-bw/2),
               vmin=vmin, vmax=vmax, aspect='auto')
    else:
        a_f = np.linspace(center_frequency-bw/2,center_frequency+bw/2, ds.shape[1]) * u.MHz
        ax.imshow(ds[:,:].T,extent=(0,(end_mjd-start_mjd)*24,center_frequency-bw/2,center_frequency+bw/2),
               vmin=vmin, vmax=vmax, aspect='auto', origin='lower')
    if ylims is not None:
        ax.set_ylim(ylims[0],ylims[1])
    ax.set_ylabel('Frequency (MHz)')
    ax.set_xlabel('Time (hr)')
    return ds, a_t, a_f


def plot_sec_woax(ds, a_t, a_f, lim=6,fig=fig,rect=[0.0,-0.8,1.0,0.3],v_min=None,v_max=None, tau_lim=None):
    ax2 = fig.add_axes(rect)
    fd=THTH.fft_axis(a_t,u.mHz)
    tau=THTH.fft_axis(a_f,u.us)

    SS=np.fft.fftshift(np.fft.fft2(ds.T))
    if v_min or v_max is None:
        v_min,v_max = np.percentile(np.abs(SS)**2,[3,99.95])
        print (v_min,v_max)

    ax2.imshow(np.abs(SS)**2,norm=LogNorm(),origin='lower',aspect='auto',extent=ext_find(fd,tau),
              vmax=v_max, vmin=v_min)

    ax2.set_xlabel(fd.unit.to_string('latex'))
    ax2.set_ylabel(tau.unit.to_string('latex'))
    ax2.set_xlim(-lim,lim)
    if tau_lim is None:
        plt.ylim(0, tau[-1].value)
    else:
        plt.ylim(0, tau_lim)
    return SS, fd, tau

def plot_part(ds, a_t, a_f, fr, tm, v_min=1e8,v_max=1e15, lim=6,fig=fig,rect=[0.0,-0.4,1.0,0.3]):
    fig=plt.figure(figsize=(6, 4), dpi= 80, facecolor='w', edgecolor='k')
    vmin,vmax = np.percentile(ds,[1,99])
    plt.imshow(ds[tm[0]:tm[1],fr[0]:fr[1]].T,vmin=vmin, vmax=vmax, aspect='auto', origin='lower')
    plt.show()
    n_ds,n_t,n_f=get_ds_piece(ds,a_t,a_f,fr=fr,tm=tm)
    sec, fd, tau=plot_sec_spectra_daniel(n_ds, n_t, n_f, parabola=False, etas=np.arange(1,20,2),
                     plot_half=True, pad_it=False, npad=3, vmin=v_min,vmax=v_max)
    plt.xlim(-lim,lim)
    plt.show()
    npad=4
    ds_pad=np.pad(n_ds.T,((0,npad*n_ds.T.shape[0]),(0,npad*n_ds.T.shape[1])),mode='constant',
                     constant_values=n_ds.T.mean())
    p_sec, p_fd, p_tau =plot_sec_spectra_daniel(n_ds, n_t, n_f, parabola=False, etas=np.arange(1,20,2),
                     plot_half=True, pad_it=True, npad=npad, vmin=v_min,vmax=v_max)
    plt.xlim(-lim,lim)
    plt.show()


def plot_ds_triple(filenpz='/mnt/scratch-lustre/gusinskaia/triple_system/5602579_AO_1400_ds.npz', figsize=[4,9],
                  plot_orb=False, ylims=[1150,1850],factor=[8,1], plot_ss=True):
    triple_ds=np.load(filenpz)
    
    full_time=(triple_ds['mjd'][1]-triple_ds['mjd'][0])*(24.*3600.)
    ntbin=full_time/triple_ds['ds'].shape[0]
    a_t = (np.arange(triple_ds['ds'].shape[0]) * ntbin * u.s)
    
    timax=(np.amax(a_t)/3600)/11.37
    print (timax.value)
    fig=plt.figure(figsize=(figsize[0]*timax.value, figsize[1]), dpi= 80, facecolor='w', edgecolor='k')
    rect=[0.0,0.0,1.0,1.0]
    ds, a_t, a_f=plot_ds_triple_wa_ax(filenpz=filenpz, fig=fig, rect=rect, ylims=ylims, factor=factor)
    if plot_orb is True:
        plot_vels_woax(triple_ds,fig=fig,rect=[0.0,-0.4,1.0,0.3])
    if plot_ss is True:
        ss, fd, tau=plot_sec_woax(ds=ds, a_t=a_t, a_f=a_f, fig=fig,rect=[0.0,-0.7,1.0,0.2])
    return ds, a_t, a_f


