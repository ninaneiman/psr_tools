import numpy as np
import matplotlib.pyplot as plt
import os, sys, math, time
from astropy.time import Time
from astropy.time import TimeDelta
from astropy.io import ascii
from glob import glob
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from astropy.io import fits
import math
import pickle
from astropy import units as u
import matplotlib as mpl

import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec
from matplotlib.font_manager import FontProperties

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

def shrink(array, factor=[1,1,1,1], size=None):
    #ideally need error function for factor to always be factor of 2
    if size is None:
        size=[]
        for j in range(0,4):
            if factor[j]==1:
                size.append(array.shape[j])
            else:
                size.append(factor_of_two(array.shape[j],n=int(math.log(factor[j],2)),odd=True))
        size=np.array(size)
        print (size)
    else:
        size=size
    
    
    new_data=np.zeros((int(size[0]/factor[0]),array.shape[1],array.shape[2],array.shape[3]))
    print (new_data.shape)
    for i in range(0,size[0]-factor[0],factor[0]):
        k=int(i/factor[0])
        new_data[k,:,:,:]=np.mean(array[i:i+factor[0],:,:,:],axis=0)
    array=new_data
    
    nnew_data=np.zeros((array.shape[0],int(size[1]/factor[1]),array.shape[2],array.shape[3]))
    for i in range(0,size[1],factor[1]):
        k=int(i/factor[1])
        nnew_data[:,k,:,:]=np.mean(array[:,i:i+factor[1],:,:],axis=1)
    array=nnew_data
    
    nnnew_data=np.zeros((array.shape[0],array.shape[1],int(size[2]/factor[2]),array.shape[3]))
    for i in range(0,size[2],factor[2]):
        k=int(i/factor[2])
        nnnew_data[:,:,k,:]=np.mean(array[:,:,i:i+factor[2],:],axis=2)
    array=nnnew_data
    
    nnnnew_data=np.zeros((array.shape[0],array.shape[1],array.shape[2],int(size[3]/factor[3])))
    for i in range(0,size[3],factor[3]):
        k=int(i/factor[3])
        nnnnew_data[:,:,:,k]=np.mean(array[:,:,:,i:i+factor[3]],axis=3)
    return nnnnew_data



def shrink_3(array, factor=[1,1,1], size=None):
    #ideally need error function for factor to always be factor of 2
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
    
    
    new_data=np.zeros((int(size[0]/factor[0]),array.shape[1],array.shape[2]))
    print (new_data.shape)
    for i in range(0,size[0],factor[0]):
        k=int(i/factor[0])
        new_data[k,:,:]=np.mean(array[i:i+factor[0],:,:],axis=0)
    array=new_data
    
    nnew_data=np.zeros((array.shape[0],int(size[1]/factor[1]),array.shape[2]))
    for i in range(0,size[1],factor[1]):
        k=int(i/factor[1])
        nnew_data[:,k,:]=np.mean(array[:,i:i+factor[1],:],axis=1)
    array=nnew_data
    
    nnnew_data=np.zeros((array.shape[0],array.shape[1],int(size[2]/factor[2])))
    for i in range(0,size[2],factor[2]):
        k=int(i/factor[2])
        nnnew_data[:,:,k]=np.mean(array[:,:,i:i+factor[2]],axis=2)
    array=nnnew_data
    
    return nnnew_data

def load_and_scale(npy_file, ntbin=5, scale_it=False, factor=[1,1,1], size=None):
    in_prof=np.load(npy_file)
    print (in_prof.shape)
    print (ntbin*in_prof.shape[0]/60, 'min')
    if scale_it is True:
        new_prof=shrink_3(in_prof, factor=factor, size=size)
        in_prof=new_prof
        print ('new lenght:', ntbin*in_prof.shape[0]*factor[0]/60, 'min')
    return in_prof


def get_profile(in_prof, time_axis=0, freq_axis=1, rm_rfi=True, plot_it=True, figsize=[15,5]):
    print (in_prof.shape)
    if plot_it is True:
        rect1 = [0.05, 0.05, 0.45, 0.95]
        rect2 = [0.525, 0.05, 0.45, 0.95]
        fig=plt.figure(figsize=(figsize[0], figsize[1]), dpi= 80, facecolor='w', edgecolor='k')
        ax2 = fig.add_axes(rect2)
        ax1 = fig.add_axes(rect1)

    if rm_rfi is True:
        b_f = in_prof.mean(time_axis)  # Mean over time
        #b = b[..., 0] + b[..., 3]  # Sum two polarized intensities
        med_f = np.median(b_f, axis=1, keepdims=True)
        c_f = b_f/med_f - 1  # divide by median for each frequency to reduce RFI
        b_t = in_prof.mean(freq_axis)  # Mean over time
        #b = b[..., 0] + b[..., 3]  # Sum two polarized intensities
        med_t = np.median(b_t, axis=1, keepdims=True)
        c_t = b_t/med_t - 1  # divide by median for each frequency to reduce RFI
    else:
        c_t=in_prof.mean(freq_axis)
        c_f=in_prof.mean(time_axis)
    if plot_it is True: 
        ax1.imshow(c_f.T, aspect='auto')
        ax2.imshow(c_t.T, aspect='auto')#, vmin=-1e-3, vmax=10e-4)
    return c_t, c_f


def get_dyn_spectra(in_prof, on=[8,13], off=[0,6], off2=None, on2=None, lim=None, plot_it=True,
                    rotate=True, fig_width=4):
    # Get off-pulse flux as a function of frequency, by averaging over phase
    # ranges that do not have pulsar signal
    off_p = in_prof[:, :, off[0]:off[1]].mean(2)
    if off2 is not None:
        off_p = (off_p+in_prof[:, :, off2[0]:off2[1]].mean(2))/2.
    on_p = in_prof[:, :, on[0]:on[1]].mean(2)
    if on2 is not None:
        on_p = (on_p+in_prof[:, :, on2[0]:on2[1]].mean(2))/2.
    cln = on_p/off_p - 1

    print('cln shape:', cln.shape)
    if lim is None:
        j1810_x=cln
    else:
        j1810_x=cln[lim[0]:lim[1],:]
    if plot_it is True:
        if rotate is True:
            fig=plt.figure(figsize=(15, fig_width), dpi= 80, facecolor='w', edgecolor='k')
            plt.imshow(j1810_x, aspect='auto')
        else:
            fig=plt.figure(figsize=(fig_width, 15), dpi= 80, facecolor='w', edgecolor='k')
            plt.imshow(j1810_x.T, aspect='auto')
    print('x shape:', j1810_x.shape)
    return j1810_x



def plot_sec_spectra(j1810_x, ntbin=5,freq_range=[306.6875+6.75,356.6875-6.75], parabola=False, etas=np.arange(1,20,2),
                     plot_half=True, vmax=-1, vmin=-5):
    lenght=j1810_x.shape[0]*ntbin
    print (lenght/60., 'min')
    j1810_t = (np.arange(j1810_x.shape[0]) * lenght/ntbin * u.s).to(u.hr)
    j1810_f = np.linspace(freq_range[0], freq_range[1], j1810_x.shape[1]) * u.MHz
    
    j1810_sec = np.fft.fft2(j1810_x)
    j1810_sec /= j1810_sec[0, 0] #--> means sec=sec/sec[0,0]
    j1810_sec = np.fft.fftshift(j1810_sec)

    j1810_tau = np.fft.fftfreq(j1810_x.shape[-1], j1810_f[1]-j1810_f[0]).to(u.us)
    j1810_tau = np.fft.fftshift(j1810_tau)*u.us
    j1810_fd = np.fft.fftfreq(j1810_x.shape[-2], j1810_t[1]-j1810_t[0]).to(u.mHz)
    j1810_fd = np.fft.fftshift(j1810_fd)*u.mHz# << j1012_fd.unit
    j1810_p = np.abs(j1810_sec)**2
    print (np.amin(np.log10(j1810_p)),np.amax(np.log10(j1810_p)))

    fig=plt.figure(figsize=(17, 8), dpi= 80, facecolor='w', edgecolor='k')
    plt.imshow(np.log10(j1810_p).T, aspect='auto', vmax=vmax, vmin=vmin, cmap='viridis', origin=0,
               extent=(j1810_fd[0].value, j1810_fd[-1].value, j1810_tau[0].value, j1810_tau[-1].value))
    plt.xlabel(j1810_fd.unit.to_string('latex'))
    plt.ylabel(j1810_tau.unit.to_string('latex'))
    plt.colorbar()
    
    if parabola is True:
        etas=etas*u.us/u.mHz**2
        for eta in etas:
            plt.plot(j1810_fd,(eta)*j1810_fd**2, ls='-', zorder=3)

    plt.xlim(j1810_fd[0].value, j1810_fd[-1].value)
    if plot_half is True:
        plt.ylim(0, j1810_tau[-1].value)
    else:
        plt.ylim(j1810_tau[0].value, j1810_tau[-1].value)
        
    return j1810_sec, j1810_fd, j1810_tau



