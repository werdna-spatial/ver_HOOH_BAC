# -*- coding: utf-8 -*-
import sys
import argparse
import cartopy
import pandas as pd
import xarray as xr
import datetime
import numpy as np
import os 
import geopandas as gpd
from shapely.geometry import Point
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.gridspec as gridspec
import dask.dataframe as dd
import pathlib
from pylr2 import regress2

import cartopy.feature as cfeature
import cartopy.crs as ccrs
from scipy import stats
from matplotlib import gridspec


import figcom

def main():
    
    PD_base=figcom.base
    PD_out=figcom.out
    PD_tag=figcom.tag
    trace_title=str('TRAC09_DOC')
    trace_name='TRAC09'
    trace_label=str('Yr Mean Dep Int concentration  (mol C/m^2)')
    filename=str(PD_tag)+str('m_')+str(trace_title)+str('_YM_DI.png')
    file_out=PD_out.joinpath(filename)
    
    NC_file_list=[\
        # '3d.PP.Primary.v4c.nc',\
        # '3d.TRAC01.DIC.v4c.nc',\
        # '3d.TRAC02.NH4.v4c.nc',\
        # '3d.TRAC03.NO2.v4c.nc',\
        # '3d.TRAC04.NO3.v4c.nc',\
        # '3d.TRAC05.PO4.v4c.nc',\
        # '3d.TRAC06.SiO2.v4c.nc',\
        #'3d.TRAC07.HOOH.v4c.nc',\
        # '3d.TRAC08.FeT.v4c.nc',\
        '3d.TRAC09.DOC.v4c.nc',\
        # '3d.TRAC10.DON.v4c.nc',\
        # '3d.TRAC11.DOP.v4c.nc',\
        # '3d.TRAC12.DOFe.v4c.nc',\
        # '3d.TRAC13.POC.v4c.nc',\
        # '3d.TRAC14.PON.v4c.nc',\
        # '3d.TRAC15.POP.v4c.nc',\
        # '3d.TRAC16.POSi.v4c.nc',\
        # '3d.TRAC17.POFe.v4c.nc',\
        # '3d.TRAC18.PIC.v4c.nc',\
        # '3d.TRAC19.ALK.v4c.nc',\
        # '3d.TRAC20.O2.v4c.nc',\
        # '3d.TRAC21.Diatom01.v4c.nc',\
        # '3d.TRAC22.Euk01.v4c.nc',\
        #'3d.TRAC23.Syn01.v4c.nc',\
        #'3d.TRAC24.Pro01.v4c.nc',\
        # '3d.TRAC25.Tricho01.v4c.nc',\
        # '3d.TRAC26.Cocco01.v4c.nc',\
        # zoo1 consumes Pro01
        #'3d.TRAC27.Zoo01.v4c.nc',\
        # zoo2 consumes Syn01
        #'3d.TRAC28.Zoo02.v4c.nc',\
        # '3d.TRAC29.Zoo03.v4c.nc',\
        # '3d.TRAC30.Zoo04.v4c.nc',\
        # '3d.TRAC31.Chl1.v4c.nc',\
        # '3d.TRAC32.Chl2.v4c.nc',\
        # '3d.TRAC33.Chl3.v4c.nc',\
        # '3d.TRAC34.Chl4.v4c.nc',\
        # '3d.TRAC35.Chl5.v4c.nc',\
        #'3d.UTK_holl.holling.v4c.nc',\
        #'3d.UTK_2ZPP.holling.v4c.nc',\
        #'3d.UTK_C.C.v4c.nc',\
        #'3d.UTK_D.D.v4c.nc',\
        'grid.v4c.nc'\
        ]
    PD_NC=pathlib.Path.joinpath(PD_base,'NC_trace')
    os.chdir(PD_NC)
    print(os.getcwd())
    NC_file=NC_file_list
    #L_tracer=L_tracer_list[i]
    ds_tracers=xr.open_mfdataset(NC_file,  combine='by_coords', parallel=False,chunks={'T':1,'Z':1})
    #
    #
    ds_tracers['DepHeight']=xr.zeros_like(ds_tracers['RC'])
    for index in range(ds_tracers.Nr):
        d=2*(ds_tracers.RL[index].values-ds_tracers.RC[index].values)
        ds_tracers['DepHeight'][index]=d
        #print(index,d)
    ds_tracers['DepHeight'].values
    #
    # limit time
    #time - use  last year 3
    # fix date units on NC side before read
    #find *.nc -maxdepth 1 -type f -exec  ncatted -O -a units,T,o,c,'seconds since 2000-01-01' {} \;
    #
    # there is sometimes a single day in the last year
    #
    print('Limit year  :: start', flush=True, file = sys.stdout)
    #
    tempdate=ds_tracers.T.max()
    #YEAR_index=int(tempdate.T.dt.year)
    YEAR_index=int(2009)
    ds_tracers=ds_tracers.isel(T=(ds_tracers.T.dt.year == YEAR_index))
    #ds_tracers=ds_tracers.isel(T=(ds_tracers.T.dt.year >= 2008))
    #
    #limit depth and sum
    print('Limit depth  :: start', flush=True, file = sys.stdout)
    # z is  positive depth starting at 5
    #ds_tracers=ds_tracers.isel(Z=(ds_tracers.Z > (1.0) ))
    #ds_tracers=ds_tracers.sum(dim='Z')
    #
    ds_tracers['bathy_mask']=ds_tracers['Depth'].where(ds_tracers['Depth'] != 0)  
    #ds_tracers=ds_tracers.where(ds_tracers['Depth'] != 0)
    ds_tracers=ds_tracers.where(ds_tracers['Depth']>1, drop=True)
    #
    sec_in_day=86400
    mole_in_mmol=0.0001
    trace_title=str('TRAC09_DOC')
    trace_name='TRAC09'
    trace_label=str('Yr Mean Dep Int concentration  (mol C/m^2)')
    #
    ds_tracers['trace']=ds_tracers[trace_name]*mole_in_mmol
    ds_tracers['sa_trace']=ds_tracers['trace']*ds_tracers['DepHeight']
    ds_tracers['sum_sa_trace']=ds_tracers['sa_trace'].sum(dim=["Z"])
    ds_tracers['Msum_sa_trace']=ds_tracers['sum_sa_trace'].mean(dim=["T"])
    ds_tracers['LMsum_sa_trace']=np.log10(ds_tracers['Msum_sa_trace'])
    #
    x=ds_tracers['Msum_sa_trace'].values.ravel()
    #print('HOOH :: '+str(x.size)+' '+str(np.nanmax(x))+' '+str(np.nanmin(x)), flush=True, file = sys.stdout)
    print(str(trace_name)+' :: '+str(x.size)+' '+str(np.nanmax(x))+' '+str(np.nanmin(x))+' '+str(np.nanpercentile(x,90))+' '+str(np.nanpercentile(x,50))+' '+str(np.nanpercentile(x,10)), flush=True, file = sys.stdout)
    #
    map_layer=ds_tracers['Msum_sa_trace']
    fig, axs = plt.subplots(nrows=1,ncols=1,subplot_kw={'projection': ccrs.EckertIV()} , figsize=(18,12) )
    #
    #cmap = plt.get_cmap('bwr')
    cmap = plt.get_cmap('viridis')
    axs.set_global()
    axs.coastlines()
    #axs.add_feature(cartopy.feature.LAND, zorder=0, edgecolor='black')
    #axs.stock_img()
    minrange=0
    maxrange=0.15
    #
    #axs.set_title('AXS',fontsize=30)
    plt.suptitle(str(trace_title),fontsize=30)
    im=map_layer.plot.pcolormesh( x='X', y='Y',
                                               ax=axs,
                                               transform=ccrs.PlateCarree(),
                                               vmin= minrange,
                                               vmax= maxrange,
                                               add_colorbar=False,
                                               zorder=10, 
                                               cmap=cmap)
    
    #
    cbar_a = fig.colorbar(im, ax=axs, shrink=0.8, location='right')
    cbar_a.set_label(str(trace_label))
    #
    fig.savefig(file_out,dpi=300, transparent=True)
    ########################
    print('EXIT  :: EXIT', flush=True, file = sys.stdout)
   
    
   
#
if __name__ == "__main__":
    #Initialize
    
    main()
