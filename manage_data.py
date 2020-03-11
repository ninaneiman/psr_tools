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

plt.rcParams['figure.dpi'] = 200


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
    for i in range(0,size[0],factor[0]):
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

def get_cln_ds(new_data, on1=[4,9], on2=[14,16], off1=[1,4], off2=[9,14], phase_axis=2):
    off1 = new_data[:, :, off1[0]:off1[1]].mean(phase_axis)
    off2 = new_data[:, :, off2[0]:off2[1]].mean(phase_axis)
    off = (off1+off2)/2.
    # Get on-pulse flux
    on1 = new_data[:, :,on1[0]:on1[1]].mean(phase_axis)
    on2 = new_data[:, :,on2[0]:on2[1]].mean(phase_axis)
    on = (on1+on2)/2.
    cln = on/off-1
    return cln

def get_cp(nopp_data, a=1, rm_rfi=True):
    if rm_rfi is True:
        b_p = nopp_data.mean(a)
        med_p = np.median(b_p, axis=1, keepdims=True)
        cp = b_p/med_p - 1
    else:
        cp=nopp_data.mean(a)
    return cp

def get_data_cp(data, a=1, rm_rfi=True):
    nopp_data=get_nopp_data(data)
    cp=get_cp(nopp_data, a=a, rm_rfi=rm_rfi)
    return cp

def get_nopp_data(data):
    nopp_data=data.sum(1)
    nopp_data=nopp_data-np.amin(nopp_data)
    return nopp_data

def get_lc(c_t, regime='all'):
    if regime == 'all':
        ct=c_t.mean(1)
    if regime == 'on':
        on1 = c_t[:,4:9].mean(1)
        on2 = c_t[:,14:16].mean(1)
        ct = (on1+on2)/2.
    if regime == 'off':
        off1 = c_t[:, 1:4].mean(1)
        off2 = c_t[:, 9:14].mean(1)
        ct = (off1+off2)/2.
    return ct


def get_ct3(in_prof,part=2,crop=10, regime='all', rm_rfi=True):
    cts=[]
    for j in range(0,part):
        part_crop=int(in_prof.shape[1]/crop)
        part_split=int((in_prof.shape[1]-2*part_crop)/part)
        part_ghz=1+(1./crop)+j*((1-(2.*1./crop))/part)
        c_t=get_cp(in_prof[:,j*part_split+part_crop:(j+1)*part_split+part_crop,:], a=1, rm_rfi=rm_rfi)
        cts.append(c_t)
    return cts


#######################################################
#PLOTTING FUNCTIONS


def plot_lc3(in_prof,part=2,crop=10,regime='all', ax=plt, colors=None, rm_rfi=True, time_range=[0,122.58]):
    if colors is None:
        colors=[np.array([57, 73, 171])/255.,np.array([33, 150, 243])/255., np.array([128, 222, 234])/255.]
    else:
        colors=colors
    cts=get_ct3(in_prof,part=part,crop=crop, regime='all')
    for j in range(0,part):
        part_ghz=1+(1./crop)+j*((1-(2.*1./crop))/part)
        c_t=cts[j]
        t = np.linspace(time_range[0],time_range[1], c_t.shape[0])*u.min
        ct=get_lc(c_t, regime=regime)
        ax.plot(t,ct, label='%.2f-%.2f GHz'%(part_ghz,part_ghz+(1-(2.*1./crop))/part), color=colors[j])  
        ax.legend(loc=(0.0,0.8), ncol=1)
        ax.set_ylabel('Arbitrary flux (%s)'%regime)
        ax.set_xlabel('Time (min)')


def plot_profile(data, time_axis=0, freq_axis=1, rm_rfi=True, figsize=[2.5,7.5], min1=0,
                 plot_ds=False, axis_units=False, factor=1, ntbin=5, freq_band=[1,2], time_range=[0,122.58],
                plot_lc=False, regime_lc='all'):
    in_prof=get_nopp_data(data)
    print (in_prof.shape)
    if axis_units is True:
        t = np.linspace(time_range[0],time_range[1], in_prof.shape[time_axis])*u.min
        f = np.linspace(freq_band[0], freq_band[1], in_prof.shape[freq_axis]) * u.GHz
    rect1 = [0.0, 0.05, 0.14, 0.95]
    rect2=[0.245, 0.05, 0.14, 0.95]
    fig=plt.figure(figsize=(figsize[0], figsize[1]), dpi= 80, facecolor='w', edgecolor='k')
    if plot_ds is True:
        rect3=[0.49, 0.05, 0.41, 0.95]
        ax3 = fig.add_axes(rect3)
    if plot_lc is True:
        rect4=[0.0, -0.37, 0.41, 0.3]
        ax4 = fig.add_axes(rect4)
        rect5=[0.49, -0.37, 0.41, 0.3]
        ax5 = fig.add_axes(rect5)
    ax2 = fig.add_axes(rect2)
    ax1 = fig.add_axes(rect1)
    
    c_t=get_cp(in_prof, a=freq_axis, rm_rfi=rm_rfi)
    c_f=get_cp(in_prof, a=time_axis, rm_rfi=rm_rfi)
    
    if plot_ds is True:
        cln=get_cln_ds(in_prof)
        if axis_units is True:
            ax3.imshow(cln.T, aspect='auto',extent=(t[0].value, t[-1].value, f[-1].value, f[0].value), cmap='binary')
            ax3.set_xlabel('Time (min)')
            ax3.set_ylabel('Frequency (GHz)')
        else:
            ax3.imshow(cln.T, aspect='auto', cmap='binary')
    if plot_lc is True:
        t = np.linspace(time_range[0],time_range[1], c_t.shape[0])*u.min
        ax4.plot(t,c_t.mean(1), label='all')
        ax4.plot(t,get_lc(c_t, regime='on'), label='on')
        ax4.plot(t,get_lc(c_t, regime='off'), ls=':', label='off')
        ax4.legend(loc=(0.01,0.87), ncol=3)
        ax4.set_xlabel('Time (min)')
        ax4.set_ylabel('Arbitrary flux')
        plot_lc3(in_prof, ax=ax5, regime=regime_lc, time_range=time_range)
    if axis_units is True:
        ax1.imshow(c_f, aspect='auto',extent=(0,c_f.shape[1], f[-1].value, f[0].value),cmap='coolwarm')
        ax2.imshow(c_t, aspect='auto',extent=(0,c_f.shape[1], t[-1].value, t[0].value),cmap='coolwarm')
        ax1.set_xlabel('psr phase bins')
        ax1.set_ylabel('Frequency (GHz)')
        ax2.set_xlabel('psr phase bins')
        ax2.set_ylabel('Time (min)')
    else:
        ax1.imshow(c_f, aspect='auto',cmap='coolwarm')
        ax2.imshow(c_t, aspect='auto',cmap='coolwarm')#, vmin=-1e-3, vmax=10e-4)
    return


def plot_parted_ct(datas,crop=10,prt=3,figsize=[9,7], wd=0.15, cmap='viridis', colors=None):
    fig=plt.figure(figsize=(figsize[0], figsize[1]), dpi= 80, facecolor='w', edgecolor='k') 


    wd=wd
    gap=wd*0.08
    axes_ct=[]
    for i in range(len(datas)*prt, 0, -1):
        rect=[(i-1)*(wd+gap),0.05,wd,0.95]
        axes_ct.append(fig.add_axes(rect))

    axes_lc=[]
    for i in range(len(datas),0,-1):
        rect = [(i-1)*prt*(wd+gap), -0.37, prt*(wd+gap)-gap, 0.3]
        axes_lc.append(fig.add_axes(rect))

    for i in range(0,3):
        j=prt-1
        axs=axes_lc[i]
        new_data=get_nopp_data(datas[i])
        plot_lc3(new_data, crop=crop, ax=axs, regime='on', part=prt, colors=colors)
        cts=get_ct3(new_data,part=prt,crop=10, regime='all')
        for ax in axes_ct[i*prt:(i+1)*prt]:
            t = np.linspace(0,128.58, cts[j].shape[0])*u.min
            part_ghz=1+(1./crop)+j*((1-(2.*1./crop))/prt)
            ax.imshow(cts[j], aspect='auto',interpolation='none', cmap=cmap, extent=(0,16, t[-1].value, t[0].value))
            ax.set_xlabel('%.2f-%.2f GHz'%(part_ghz,part_ghz+(1-(2.*1./crop))/prt))
            ax.set_ylabel('Time (min)')
            j=j-1

    axes_cf=[]
    for i in range(len(datas),0,-1):
        rect = [(i-1)*prt*(wd+gap), 1.07, prt*(wd+gap)-gap, 0.2]
        axes_cf.append(fig.add_axes(rect))

    for i in range(0,len(datas)):
        axss=axes_cf[i]
        part11=int(datas[i].shape[2]/crop)
        f = np.linspace(1+1./11.,2-1./11., 16, datas[i].shape[2]-2*part11)*u.GHz
        axss.imshow(get_data_cp(datas[i], a=0).T[:,part11:-part11], aspect='auto',interpolation='none', cmap=cmap,
                   extent=(f[0].value, f[-1].value,0,16))
        axss.set_xlabel('Frequency (GHz)')



def plot_image(image, trans=True, mloc=50, figsize=[5,10], cmap='viridis'):
    rect1=[0.0, 0.0, 1.0, 1.0]
    fig=plt.figure(figsize=(figsize[0], figsize[1]), dpi= 80, facecolor='w', edgecolor='k')
    ax = fig.add_axes(rect1)
    if trans is True:
        ax.imshow(image.T, aspect='auto', cmap=cmap)
    else:
        ax.imshow(image, aspect='auto', cmap=cmap)
    ax.yaxis.set_major_locator(MultipleLocator(mloc))
    ax.yaxis.set_minor_locator(MultipleLocator(mloc))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))


def plot_off(data, mloc=50, figsize=[5,10]):
    new_data=data.sum(1)
    off1 = new_data[:, :, 1:4].mean(2)
    off2 = new_data[:, :, 9:14].mean(2)
    off = (off1+off2)/2.
    plot_image(off, trans=True, mloc=mloc, figsize=figsize)



###############################################
#RFI REMOVING FUNCTIONS
###############################################


def remove_rfi(dd_data,n, do_max=True, factor=1):
    data=np.copy(dd_data)
    pixels=[]
    for x in range (0,n):
        new_data=data.sum(1)
        new_data=new_data-np.amin(new_data)
        off=get_cln_ds(new_data)
        max_arguments=[]
        max_values=[]
        min_arguments=[]
        min_values=[]
        for i in range(0,off.shape[0]):
            max_arguments.append([i,np.argmax(off[i,:])])
            max_values.append(np.amax(off[i,:]))
            min_arguments.append([i,np.argmin(off[i,:])])
            min_values.append(np.amin(off[i,:]))
        coord_max=max_arguments[np.argmax(max_values)]
        coord_min=min_arguments[np.argmin(min_values)]
        #print ('max:', coord_max)
        #print ('min:', coord_min)
        data[coord_min[0],:,coord_min[1],:]=0.0
        pixels.append(coord_min)
        pixels.append(coord_max)
        if do_max is True:
            data[coord_max[0],:,coord_max[1],:]=0.0
    dd_data[:,:,:int(710/factor),:]=data[:,:,:int(710/factor),:]
    dd_data[:,:,int(1074/factor):,:]=data[:,:,int(1074/factor):,:]
    return dd_data, pixels


def remove_jumps(rows, threshold=0.000075):
    bad_channels=[]
    for j in range (1,len(rows)-1):
        if (rows[j]>0 and rows[j+1]>0) or (rows[j]<0 and rows[j+1]<0):
            delta = np.abs(np.abs(rows[j+1]) - np.abs(rows[j]))
        else:
            delta = np.abs(rows[j+1]-rows[j])
        if (rows[len(rows)-j-1]>0 and rows[len(rows)-j]>0) or (rows[len(rows)-j-1]<0 and rows[len(rows)-j]<0):
            delta_2=np.abs(np.abs(rows[len(rows)-j-1]) - np.abs(rows[len(rows)-j]))
        else:
            delta_2=np.abs(rows[len(rows)-j-1] - rows[len(rows)-j])
        if (delta >  threshold):
                bad_channels.append(j+1)
        if (delta_2 >  threshold):
                bad_channels.append(len(rows)-j-1)
    return bad_channels



def remove_channel_rfi(data,factor=1,threshold_f=0.0005, nn=12):
    dd_data=np.copy(data)
    ntbin=int(dd_data.shape[0]/nn)
    all_bad_channels=[]
    for x in range(0,nn):
        print (x, 'range:',ntbin*x,ntbin*(x+1))
        new_data=dd_data.sum(1)
        off1 = new_data[:, :, 1:4].mean(2)
        off2 = new_data[:, :, 9:14].mean(2)
        off = (off1+off2)/2.
        ds=off[ntbin*x:ntbin*(x+1),:]
        Bandpass=ds.mean(0)
        plt.plot(Bandpass)
        plt.show()
        bad_channels=[]
        bad_channels=remove_jumps(Bandpass,threshold=threshold_f/factor)
        new_bad_channels=[]
        for i in bad_channels:
            if (i< int(710/factor)) or (i>int(1074/factor)):
                new_bad_channels.append(i)
                all_bad_channels.append(i)
                plt.plot(ds[:,i])
        plt.show()        
        print (np.array(new_bad_channels))
        print ('this channels are deleted:', len(new_bad_channels))
        for i in new_bad_channels:
            dd_data[:,:,i,:]=0.0
    return dd_data, all_bad_channels


def remove_frequency_rfi(data,factor=1,threshold_f=0.0005, theshold_t=0.05, nn=12):
    dd_data=np.copy(data)
    ntbin=int(dd_data.shape[0]/nn)
    all_bad_pixels=[]
    for x in range(0,nn):
        print (x, 'range:',ntbin*x,ntbin*(x+1))
        new_data=dd_data.sum(1)
        off1 = new_data[:, :, 1:4].mean(2)
        off2 = new_data[:, :, 9:14].mean(2)
        off = (off1+off2)/2.
        ds=off[ntbin*x:ntbin*(x+1),:]
        Bandpass=ds.mean(0)
        plt.plot(Bandpass)
        plt.show()
        bad_channels=[]
        bad_channels=remove_jumps(Bandpass,threshold=threshold_f/factor)
        new_bad_channels=[]
        for i in bad_channels:
            if (i< int(710/factor)) or (i>int(1074/factor)):
                new_bad_channels.append(i)
                plt.plot(ds[:,i])
        print  (np.array(new_bad_channels))
        print ('range:',ntbin*x,ntbin*(x+1), '; how much channels are bad:', len(bad_channels))
        plt.show()

        bad_pixels=[]
        for i in new_bad_channels:
            bad_times=remove_jumps(ds[:,i], threshold=theshold_t/factor)
            for j in bad_times:
                bad_pixels.append([j,i])
                all_bad_pixels.append([j+ntbin*x,i])
        print ('range:',ntbin*x,ntbin*(x+1),'; this much bad pixels in these %d channels:'%len(bad_channels),
               len( bad_pixels))
        for j in bad_pixels:
            dd_data[j[0]+ntbin*x,:,j[1],:]=0.0
    return dd_data, all_bad_pixels


def remove_time_rfi(data,factor=1,threshold_t=0.0002, threshold_f=0.005, nn=12):
    dd_data=np.copy(data)
    ntbin=int(dd_data.shape[2]/nn)
    all_bad_pixels=[]
    for x in range(0,nn):
        print (x, 'range:',ntbin*x,ntbin*(x+1))
        new_data=dd_data.sum(1)
        off1 = new_data[:, :, 1:4].mean(2)
        off2 = new_data[:, :, 9:14].mean(2)
        off = (off1+off2)/2.
        ds=off[:,ntbin*x:ntbin*(x+1)]


        Lightcurve=ds.mean(1)
        plt.plot(Lightcurve)
        plt.show()

        bad_times=[]
        bad_times=remove_jumps(Lightcurve,threshold=threshold_t/factor)
        for i in bad_times:
                plt.plot(ds[i,:])
        print ('range:',ntbin*x,ntbin*(x+1), '; how much times are bad:', len(bad_times))
        plt.show()

        bad_pixels=[]
        for i in bad_times:
            bad_channels=remove_jumps(ds[i,:], threshold=threshold_f/factor)
            new_bad_channels=[]
            for p in bad_channels:
                if (p+ntbin*x< 710/factor) or (p+ntbin*x>1074/factor):
                    new_bad_channels.append(p)
            for j in new_bad_channels:
                bad_pixels.append([i,j])
                all_bad_pixels.append([i,j+ntbin*x]) 
        print ('range:',ntbin*x,ntbin*(x+1),'; this much bad pixels in these %d times:'%len(bad_times),
               len( bad_pixels))
        for j in bad_pixels:
            dd_data[j[0],:,j[1]+ntbin*x,:]=0.0
    return dd_data, all_bad_pixels
