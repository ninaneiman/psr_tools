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

def factor_of_two(x, n=5, odd=True):                                                                                                                                                    
    '''finds closest smaller number with enough (defined by n) factors of 2 
    takes: any number
    returns: easy factorable number by 2'''
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
    '''downsamples 2D-data by a fixed factors of 2
    takes: 2D-array (array) and factors by which to downsampe that array (factor) in each axis
    size - the desired size in each axis (if known)
    returns: downsampled 2D array'''
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

    return nnew_data

def shrink(array, factor=[1,1,1,1], size=None):
    '''downsamples 4D-data by a fixed factors of 2
    takes: 4D-array (array) and factors by which to downsampe that array (factor) in each axis
    size - the desired size in each axis(if known)
    returns: downsampled 4D array'''
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


def load_triple(filenpz='/mnt/scratch-lustre/gusinskaia/triple_system/5602579_AO_1400_ds.npz',factor=[1,1], mean0=True, wnoise=True):
    '''loads ds data from npz file, downsamples it, subtructs mean (if applied) and loads noise data if present (and applied)
    Takes:
    filenpz - name of the npz file to load data from
    factor - [#int, #int], where #int is always a factor of 2. 
    factor difined the number by which to downsample 2D ds data (see shrink_2 function)
    mean0 - set mean of the data to 0 (i.e. whether to subtract mean)
    wnoise - whether to load noise data (which is present or not)
    Returns:
    ds - dynamic spectra (2D array)
    t - time axis (astropy units quantity)
    f - frequency axis (astropy units quantity)
    ns - noise of dynamic spectra (2D array)'''
    triple_ds=np.load(filenpz)
    print (triple_ds['ds'].shape)
    ds=triple_ds['ds']
    if wnoise is True:
        ns=triple_ds['noise']
    else:
        ns=np.random.normal(size=np.shape(ds))*np.std(ds)/6
    if 'WSRT' in filenpz:
        ds=shrink_2(ds, factor=factor, size=None)
        ns=shrink_2(ns, factor=factor, size=None)/np.sqrt(factor[0]*factor[1])
    if mean0 is True:
        ds=ds-ds.mean()
        ns=ns-ds.mean()
    end_mjd=triple_ds['mjd'][1]
    start_mjd=triple_ds['mjd'][0]
    center_frequency=triple_ds['c_fr']
    bw=triple_ds['bw_fr']

    full_time=(end_mjd-start_mjd)*(24.*3600.)
    ntbin=full_time/ds.shape[0]
    a_t = (np.arange(ds.shape[0]) * ntbin * u.s)
    a_t = (np.arange(ds.shape[0]) * ntbin * u.s)
    if 'AO' in filenpz:
        a_f = np.linspace(center_frequency+bw/2,center_frequency-bw/2, ds.shape[1]) * u.MHz
        ds=np.flip(ds, axis=1)
        ns=np.flip(ns, axis=1)
    else:
        a_f = np.linspace(center_frequency-bw/2,center_frequency+bw/2, ds.shape[1]) * u.MHz
    return ds, a_t, a_f, ns
