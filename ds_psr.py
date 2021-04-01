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
from scipy.sparse.linalg import eigsh
from scipy.optimize import curve_fit
import load_data as ld
import ththmod as THTH
#np.seterr(divide='ignore', invalid='ignore')


def load_triple_spectrum(name='/mnt/scratch-lustre/gusinskaia/triple_system/5602579_AO_1400_ds.npz',factor=[1,1], tel='Undefined', psr='PSRJ0337+1715', pad_it=True, npad=3, mean0=True, wnoise=True):
    ''' load  data from npz file into Spec object using load_data module
    Takes:
    filenpz - name of the npz file to load data from
    factor - [#int, #int], where #int is always a factor of 2. 
    factor difined the number by which to downsample 2D ds data (see shrink_2 function)
    tel - Observatpry on which the data was taken (if known)
    psr - pulsar name of the object (if known)
    pad_it  - whether to pad the secondary spectra with zeros
    npad - (int) number of paded zeros 
    mean0 - set mean of the data to 0 (i.e. whether to subtract mean)
    wnoise - whether to load noise data (which is present or not)
    Returns:
    Spec object'''

    triple_ds=np.load(name)
    ds,a_t,a_f, ns=ld.load_triple(name, factor=factor, mean0=mean0, wnoise=wnoise)
    if tel == 'Undefined':
        spt_tel=name.split('_')
        if "AO" in spt_tel:
            tel="AO"
        if "WSRT" in spt_tel:
            tel="WSRT"
        if "GBT" in spt_tel:
            tel="GBT"
    if wnoise is False:
        ns_info='no noise'
    else:
        ns_info='w noise'
    my_spec=Spec(I=ds,t=a_t,f=a_f,stend=triple_ds['mjd'], nI=ns, tel=tel, psr=psr,pad_it=pad_it, npad=npad, ns_info=ns_info)
    return my_spec


##Function for making extent tuple for plotting
def ext_find(x,y):
    '''function from Daniels code'''
    dx=np.diff(x).mean()
    dy=np.diff(y).mean()
    ext=[(x[0]-dx/2).value,(x[-1]+dx/2).value,(y[0]-dy/2).value,(y[-1]+dy/2).value]
    return(ext)


def fun_shrink_ds(ds,t,f,factor=[1,1]):
    '''Downsamples ds, t, and f by ant integer factors'''
    new_ds=ld.shrink_any_2(ds, factor=factor)
    new_t=ld.shrink_any_1(t, factor[0])
    new_f=ld.shrink_any_1(f, factor[1])
    return new_ds, new_t, new_f


def fun_plot_ds_woaxis(ds, t, f, fig=plt.figure(figsize=(3,8), dpi=150), rect=[0.0,0.0,1.0,1.0]):
    '''function that plots ds inside a given axis. The purpose is to avoid wasting time setting parameters right.
    it takes:
    ds - dynamic spectrum (2D array)
    t, f - time and frequency axes (astropy quantity with units)
    fig - figure to which to add an axis
    rect - rectangular parameters of the axis'''

    ax=fig.add_axes(rect)
    vmin,vmax = np.percentile(ds,[1,99])
    ax.imshow(ds.T,extent=(0,(t[-1].value-t[0].value)/3600.,np.amin(f).value,np.amax(f).value),
           vmin=vmin, vmax=vmax, aspect='auto', origin='lower')
    ax.set_xlabel('Time (hr)')
    ax.set_ylabel('Frequency (%s)'%f.unit.to_string('latex'))

def fun_plot_ds(ds, t, f, new_fig=True, figsize=(3,8), dpi=150, vmin=None, vmax=None):
    '''function that plots ds. The purpose is to avoid wasting time setting parameters right.
    it takes:
    ds - dynamic spectrum (2D array)
    t, f - time and frequency axes (astropy quantity with units)
    newfig - whether or not one wishes/needs to create new plt.figure()
    dpi - resolution of a newly created figure
    vmin, vmax - value limits of ds for plt.imshow()'''

    if new_fig is True:
    	plt.figure(figsize=figsize, dpi=dpi)
    if (vmin is None) and (vmax is None):
        vmin,vmax = np.percentile(ds,[1,99])
    plt.imshow(ds.T,extent=(0,(t[-1].to(u.hr).value-t[0].to(u.hr).value),np.amin(f).value,np.amax(f).value),
           vmin=vmin, vmax=vmax, aspect='auto', origin='lower')
    plt.xlabel('Time (hr)')
    plt.ylabel('Frequency (%s)'%f.unit.to_string('latex'))

def fun_make_ss(ds, t, f, pad_it=True, npad=3):
    '''Makes secondary spectrun from dynamic spectrum using Daniels funtions
    Takes:
    ds - dynamic spectrum (2D array)
    t, f - time and frequency axes (astropy quantity with units)
    pad_it  - whether to pad the secondary spectra with zeros
    npad - (int) number of paded zeros
    Returns:
    Is - secodary specrum array
    tau, fd  - delay and dopler shift
    '''
    if pad_it is True:
        ds_pad=np.pad(ds.T,((0,npad*ds.T.shape[0]),(0,npad*ds.T.shape[1])),mode='constant',
                        constant_values=ds.T.mean())
        ds=ds_pad.T
        fd=THTH.fft_axis(t,u.mHz,npad)
        tau=THTH.fft_axis(f,u.us,npad)
        Is=np.fft.fftshift(np.fft.fft2(ds.T))
    else:
        fd=THTH.fft_axis(t,u.mHz)
        tau=THTH.fft_axis(f,u.us)
        Is=np.fft.fftshift(np.fft.fft2(ds.T))    
    return Is, tau, fd


def fun_plot_ss(Is, tau, fd, fd_lim=[-1.5, 1.5], tau_lim=[0.0,1.4], vmin=None, vmax=None, new_fig=True, figsize=(3,2), dpi=150, cb=True):
    ''' Plots secondary spectrum
    Takes:
    Is  - complex array- secodary specrum array
    tau, df - delay and dopler shift
    fd_lim, tau_lim - [float, float]  - limit of dopler shift and delay in plot
    vmin, vmax - value limits of ds for plt.imshow()

    new_fig - whether or not one wishes/needs to create new plt.figure()
    figsize - (float, float) - size of the newly created figure
    dpi - resolution of a newly created figure
    cb - whether or not to show the colorbar
    '''

    SS_ext=ext_find(fd,tau)
    if vmin is None:
        vmin=np.median(np.abs(Is[(Is!=0)])**2)*2
    if vmax is None:
        vmax=np.max(np.abs(Is[(Is!=0)])**2)*np.exp(-2.7)/10
    if tau_lim is None:
        tau_lim=np.array([tau[0].value,tau[-1].value])
    if fd_lim is None:
        fd_lim=np.array([fd[0].value,fd[-1].value])
    if new_fig is True:
        fig=plt.figure(figsize=figsize, dpi=dpi, facecolor='w', edgecolor='k')
    plt.imshow(np.abs(Is)**2,norm=LogNorm(vmax=vmax, vmin=vmin),aspect='auto',extent=SS_ext, origin='lower')
    if cb is True:
        plt.colorbar()

    plt.xlabel(fd.unit.to_string('latex'))
    plt.ylabel(tau.unit.to_string('latex'))
    plt.xlim(fd_lim[0], fd_lim[1])
    plt.ylim(tau_lim[0], tau_lim[1])

def find_nearest(array, value):
    '''Finds index of the nearest to value number in an array
    Takes: 
    array - (np.array(dtype=float)) array with numbers
    value - (float) - number to which the find the index of the closes number in an array
    Returns:
    idx (int)''' 
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

#not sure this function is in use - could delete 
def find_edges(I_cross, fd_cross, tau_cross, fd_lim, tau_lim):
    fd_l=find_nearest(fd_cross, fd_lim[0])
    fd_r=find_nearest(fd_cross, fd_lim[1])
    tau_b=np.amax([find_nearest(tau_cross, tau_lim[0]),find_nearest(tau_cross, tau_lim[1])])-1
    tau_t=np.amin([find_nearest(tau_cross, tau_lim[0]),find_nearest(tau_cross, tau_lim[1])])+1
    return fd_l, fd_r, tau_b, tau_t


def crop_array(my_array, idx=None, values=None):
    '''Crops array in time and frequency (given by values of time and/or frequency) or by indexes of the array
    Takes:
    array -  (1D array), (astropy quantity with units)
    values - (astropy units quantites) [q,q] - min and max of values to cut the array with. 
    idx - [int,int] - index edges to crop array to
    Returns:
    res_array - cropped 1D array
    res_idx - indexes to which it was cropped
    '''
    if idx is not None and values is not None:
        raise ValueError("Both selection chosen! Choose one: based either on index or value!")
    elif values is not None:
        if (values[0].unit != my_array.unit) or (values[0].unit != my_array.unit):
            raise ValueError("Unit of an array and selection do not match!")
        res_i0=find_nearest(my_array.value, values[0].value)
        res_i1=find_nearest(my_array.value, values[1].value)
        res_idx=np.sort(np.array([res_i0,res_i1]))
    elif idx is not None:
        res_idx=np.sort(np.array([idx[0],idx[1]]))
    else:
        res_idx=np.array([0,len(my_array)])
    res_array=my_array[res_idx[0]:res_idx[1]]
    res_idx=[res_idx[0], res_idx[1]]
    return res_array, res_idx
    
def fun_select(ds,t,f, time_sel=None, freq_sel=None, freq_idx=None, time_idx=None, ns=None):
    '''Crops array in time and frequency (given by values of time and/or frequency) or by indexes of the array
    Takes:
    ds - dynamic spectrum (2D array)
    t, f -  time and frequency axes (astropy quantity with units)
    time_sel, freq_sel - (astropy units quantites) [q,q] - min and max of time and frequency to cut the array with. 
    time_idx, freq_idx - [int,int] - index edges in time and freq axes to crop array to
    ns - noise spectrum (if present)
    Returns:
    I_sel, t_sel, f_sel - cropped ds, time and frequency
    t_idx, f_idx - what?
    nI_sel - cropped ns
    '''
    if (freq_sel is not None) or (freq_idx is not None):
        f_sel, f_idx = crop_array(f, idx=freq_idx, values=freq_sel)
    else:
        f_idx=[0,len(f)]
        f_sel=f[f_idx[0]:f_idx[1]]
    if (time_sel is not None) or (time_idx is not None):
        t_sel, t_idx = crop_array(t, idx=time_idx, values=time_sel)
    else:
        t_idx=[0,len(t)]
        t_sel=t[t_idx[0]:t_idx[1]]
    I_sel = ds[t_idx[0]:t_idx[1],f_idx[0]:f_idx[1]]
    if ns is None:
        nI_sel=np.random.normal(size=np.shape(I_sel))*np.std(I_sel)/6
    else:
        nI_sel=ns[t_idx[0]:t_idx[1],f_idx[0]:f_idx[1]]
    return I_sel, t_sel, f_sel, t_idx, f_idx, nI_sel

def fun_interp(ds, mjd, f, t_ed, f_ed, t_len, f_len, ns=None):
    '''Interpolates ds to a new time and frequency axis
    Takes:
    ds - dynamic spectrum (2D array)
    mjd - time axis in mjd (astropy time object)
    f - frequency axis (astropy quantity with units)
    t_ed - (q -astropy units quantites) [q,q] - edges of the desired time axis to be interpolated to
    t_len - (int) - size of the desired time axis to be interpolated to 
    f_ed - (q -astropy units quantites) [q,q] - edges of the desired frequency axis to be interpolated to
    f_len - (int) - size of the desired frequency axis to be interpolated to 

    ns - noise spectrum (if present)
    Returns:
    I_new - interpolated ds
    t_sec_new - new time axis (q)
    f_new - new frequency axis
    t_int - new time axis (float)
    nI_new - interpolated ns
    '''
    fun=interpolate.interp2d(f, mjd, ds, kind='linear')
    new_ntbin=(t_ed[1].value-t_ed[0].value)/(t_len)
    t_int = np.arange(t_ed[0].value,t_ed[1].value, new_ntbin)
    new_nfbin=(f_ed[1].value-f_ed[0].value)/(f_len-1)
    f_new = np.arange(f_ed[0].value,f_ed[1].value+new_nfbin, new_nfbin)
    I_new = fun(f_new, t_int)
    if ns is None:
        nI_new=np.random.normal(size=np.shape(I_new))*np.std(I_new)/6
    else:
        nfun=interpolate.interp2d(f, mjd, ns, kind='linear')
        nI_new=nfun(f_new,t_int)
    t_sec_new = (np.arange(t_len) * new_ntbin * 24.*3600. *u.s)
    return I_new, f_new, t_sec_new, t_int, nI_new



##Curvature evolution with freuquency
def eta_func(f0,A):
    '''Daniels function'''
    return(A/(f0**2))

class SecSpec(object):
    '''Seconday spectum object'''
    def __init__(self, ds, Is, tau, fd):
        self.ds = ds
        self.Is = Is
        self.tau = tau
        self.fd = fd
    def __repr__(self):
        return "<Secondary spectrum>"


class Spec(object):
    '''Spectrum object'''
    def __init__(self, I, t, f, stend=[0.0,1.0], nI=None, tel='Unknown', psr='PSRJ0337+1715',pad_it=True, npad=3, ns_info='no noise', subbands=None):
        '''class to manipilate dynamic spectra
        Usually initiated from npz file using function load_triple_spectra,
        but it can aslo be cropped from other Spec object
        or constracted manualy. While initiated it automatically creates secondary spectra, thus requiring decision about padding.
  
        I - dynamic spectra (2D float array)
        t, f - time and frequency axes (astropy.units quantities)
        stend - start and end of the observation in mjd (astropy.Time object)
        nI - array of noise in dynamic spectra

        tel - Observatory with which the data was obtained (if known)
        psr - pulsar name of the object (if known)
        pad_it  - whether to pad the secondary spectra with zeros
        npad - (int) number of paded zeros 
        ns_info - whether noise data is present or not
        '''
        self.I = I
        self.t = t
        self.f = f
        self.nsinfo=ns_info
        if nI is None:
            self.nI=np.random.normal(size=np.shape(self.I))*np.std(self.I)/6
            self.nsinfo='no noise'
        else:
            self.nI=nI
        self.stend=stend
        
        if subbands is None:
            self.subbands=np.array([[1301,1317],[1321,1337],[1341,1357],[1361,1377],[1381,1397],
                      [1401,1417],[1421,1437],[1441,1457]])
        else:
            self.subbands=subbands

        full_time=(self.stend[1]-self.stend[0])
        ntbin=full_time/len(self.t)
        a_t_mjd = (np.arange(len(self.t)) * ntbin * u.d)+self.stend[0]*u.d
        self.mjd=Time(a_t_mjd, format='mjd', scale='utc')
                   
        self.tel=tel
        self.psr=psr
        self.ss = self.make_ss(pad_it, npad)
    def __repr__(self):
        times=(self.stend[1]-self.stend[0])*24.
        
        mjd_c=(self.stend[1]+self.stend[0])/2.
        return "<Dynamic spectrum: Dur: %.2f hr, Freq: %.2f - %2.f MHz, MJD: %.2f, PSR: %s, Tel: %s, %s>"%(times,np.amin(self.f).value, np.amax(self.f).value, mjd_c, self.psr, self.tel, self.nsinfo)
    def plot_ds(self, new_fig=True, figsize=(3,8), dpi=150, vmin=None, vmax=None):
        '''Plots ds with pre-defined settings. (see fun_plot_ds)'''
        fun_plot_ds(self.I, self.t, self.f,new_fig=new_fig,figsize=figsize, dpi= dpi, vmin=vmin, vmax=vmax)
    def plot_nds(self, new_fig=True, figsize=(3,8), dpi=150, vmin=None, vmax=None):
        '''Plots noise of the dynamic spectra same way as plot_ds. (see plot_ds)'''
        fun_plot_ds(self.nI, self.t, self.f,new_fig=new_fig,figsize=figsize, dpi= dpi, vmin=vmin, vmax=vmax)
        
    def make_ss(self, pad_it=True, npad=3):
        '''Makes secondary spectra and  loads it to SecSpec object'''
        Is, tau, fd=fun_make_ss(self.I, self.t, self.f, pad_it=pad_it, npad=npad)
        return SecSpec(self, Is, tau, fd)   

 
    def plot_ss(self,fd_lim=[-1.5, 1.5], tau_lim=[0.0,1.4], vmin=None, vmax=None, new_fig=True, figsize=(3,2), dpi=150, cb=True):
        '''Plots secondary spectra with pre-defined plotting settings'''
        fun_plot_ss(self.ss.Is, self.ss.tau, self.ss.fd, fd_lim=fd_lim, tau_lim=tau_lim, vmin=vmin, vmax=vmax, new_fig=new_fig,
                figsize=figsize, dpi=dpi, cb=cb)

    def select(self, time_sel=None, freq_sel=None, freq_idx=None, time_idx=None, pad_it=True, npad=3):
        '''Crops a piece of ds as a sepatate Spec object'''
        if (time_sel is not None) and (time_sel[0].unit == u.s):
            time_axis=self.t
            I_sel, sec_sel, f_sel, t_idx, f_idx, nI_sel=fun_select(self.I,time_axis,self.f, time_sel=time_sel,
                              freq_sel=freq_sel, freq_idx=freq_idx,time_idx=time_idx, ns=self.nI)
            mjd_sel=self.mjd.mjd[t_idx[0]:t_idx[1]]*u.d
            mjd_bin=mjd_sel[1]-mjd_sel[0]
            my_stend=[mjd_sel[0].value,mjd_sel[-1].value+mjd_bin.value]
            I_sel = self.I[t_idx[0]:t_idx[1],f_idx[0]:f_idx[1]]
            nI_sel = self.nI[t_idx[0]:t_idx[1],f_idx[0]:f_idx[1]]

        if (time_sel is not None) and (time_sel[0].unit == u.d):
            time_axis=self.mjd.mjd * u.d
            I_sel, mjd_sel, f_sel, t_idx, f_idx, nI_sel=fun_select(self.I,time_axis,self.f, time_sel=time_sel,
                              freq_sel=freq_sel, freq_idx=freq_idx,time_idx=time_idx, ns=self.nI)
            sec_sel=self.t[t_idx[0]:t_idx[1]]
            mjd_bin=mjd_sel[1]-mjd_sel[0]
            my_stend=[mjd_sel[0].value,mjd_sel[-1].value+mjd_bin.value]
            I_sel = self.I[t_idx[0]:t_idx[1],f_idx[0]:f_idx[1]]
            nI_sel = self.nI[t_idx[0]:t_idx[1],f_idx[0]:f_idx[1]]

        if (time_sel is None) and (time_idx is None):
            time_axis=self.t
            I_sel, mjd_sel, f_sel, t_idx, f_idx, nI_sel=fun_select(self.I,time_axis,self.f, time_sel=time_sel,
                              freq_sel=freq_sel, freq_idx=freq_idx,time_idx=time_idx, ns=self.nI)
            I_sel = self.I[:,f_idx[0]:f_idx[1]]
            nI_sel = self.nI[:,f_idx[0]:f_idx[1]]
            my_stend=self.stend
            sec_sel=self.t

        return Spec(I=I_sel, t=sec_sel, f=f_sel, stend=my_stend, nI=nI_sel,tel=self.tel, psr=self.psr, pad_it=pad_it, npad=npad, ns_info=self.nsinfo, subbands=self.subbands)

    def shrink(self, factor=[1,1], pad_it=True, npad=3):
        ds, t, f=fun_shrink_ds(self.I, self.t, self.f, factor=factor)
        ns=ld.shrink_any_2(self.nI, factor=factor)
        mjd=ld.shrink_any_1(self.mjd.mjd, factor=factor[0])
        mjd_bin=mjd[1]-mjd[0]
        return Spec(I=ds, t=t, f=f, stend=[mjd[0],mjd[-1]+mjd_bin], nI=ns,tel=self.tel, psr=self.psr, pad_it=pad_it, npad=npad, ns_info=self.nsinfo, subbands=self.subbands)

    def interp(self, t_ed, f_ed, t_len, f_len, pad_it=True, npad=3):
        '''Interpolates given ds to a new grid in time and frequency as well as new range'''
        I_new, f_new, t_sec_new, t_int, nI_new=fun_interp(self.I, self.mjd.mjd, self.f,
                                     t_ed=t_ed, f_ed=f_ed, t_len=t_len, f_len=f_len, ns=self.nI)
        new_spec=Spec(I=I_new, t=t_sec_new, f=f_new*u.MHz, stend=[t_ed[0].value,t_ed[1].value], nI=nI_new,
                        tel=self.tel, psr=self.psr, pad_it=pad_it, npad=npad)
        return new_spec

    def get_noise(self):
        '''Calculates noise level in ds using Daniel method'''
        temp=np.fft.fftshift(np.abs(np.fft.fft2(self.I.T)/np.sqrt(self.I.T.shape[0]*self.I.T.shape[1]))**2)
        N=np.sqrt(temp[:temp.shape[0]//6,:temp.shape[1]//6].mean())
        return N 


