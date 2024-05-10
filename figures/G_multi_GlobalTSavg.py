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
    #
    #lat=49.5
    #long=355.5    
    #
    # aloha
    lat=22.5
    long=202.5    
    PD_base=figcom.base
    PD_out=figcom.out
    PD_tag=figcom.tag
    filename=str(PD_tag)+str('multi_GlobalProdSum.png')
    file_out=PD_out.joinpath(filename)
    PD_NC=pathlib.Path.joinpath(PD_base,'NC_trace')
    
    NC_file_list=[\
        # '3d.PP.Primary.v4c.nc',\
        # '3d.TRAC01.DIC.v4c.nc',\
        '3d.TRAC02.NH4.v4c.nc',\
        # '3d.TRAC03.NO2.v4c.nc',\
        # '3d.TRAC04.NO3.v4c.nc',\
        # '3d.TRAC05.PO4.v4c.nc',\
        # '3d.TRAC06.SiO2.v4c.nc',\
        '3d.TRAC07.HOOH.v4c.nc',\
        # '3d.TRAC08.FeT.v4c.nc',\
        # '3d.TRAC09.DOC.v4c.nc',\
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
        '3d.TRAC21.Diatom01.v4c.nc',\
        '3d.TRAC22.Euk01.v4c.nc',\
        '3d.TRAC23.Syn01.v4c.nc',\
        '3d.TRAC24.Pro01.v4c.nc',\
        '3d.TRAC25.Tricho01.v4c.nc',\
        '3d.TRAC26.Cocco01.v4c.nc',\
        '3d.TRAC27.Zoo01.v4c.nc',\
        '3d.TRAC28.Zoo02.v4c.nc',\
        '3d.TRAC29.Zoo03.v4c.nc',\
        '3d.TRAC30.Zoo04.v4c.nc',\
        # '3d.TRAC31.Chl1.v4c.nc',\
        # '3d.TRAC32.Chl2.v4c.nc',\
        # '3d.TRAC33.Chl3.v4c.nc',\
        # '3d.TRAC34.Chl4.v4c.nc',\
        # '3d.TRAC35.Chl5.v4c.nc',\
        'grid.v4c.nc'\
        ]
#
    
    # os.chdir(PD_NC)
    # print(os.getcwd())
    # NC_file=NC_file_list
    #
    #####
    #
    fig, axs = plt.subplots(nrows=12,ncols=1, figsize=(10,20), sharex='all' )
    #plt.subplots_adjust(wspace=0.01,hspace=0.01)
    fig.suptitle(str(PD_base), fontweight ="bold")
    # i=0
    # ii=0
    # jj=0
    # HOOH_min=-0.7
    # HOOH_max=0.0
    # SYN_min=0.75
    # SYN_max=1.5
    # PRO_min=0.75
    # PRO_max=1.5
    #
    print('Model :: '+str(PD_NC), flush=True, file = sys.stdout)
   # print('index :: '+str(i)+' '+str(ii)+' '+str(jj), flush=True, file = sys.stdout)
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
    #tempdate=ds_tracers.T.max()
    #YEAR_index=int(tempdate.T.dt.year)
    YEAR_index=int(2009)
    ds_tracers=ds_tracers.isel(T=(ds_tracers.T.dt.year == YEAR_index))
    #
    #limit depth and sum
    print('Limit depth  :: start', flush=True, file = sys.stdout)
    #ds_tracers=ds_tracers.isel(Z=(ds_tracers.Z > (min(ext_deps_all)-1) ) )
    #ds_tracers=ds_tracers.sum(dim='Z')
    #
    ds_tracers['bathy_mask']=ds_tracers['Depth'].where(ds_tracers['Depth'] != 0)  
    ds_tracers=ds_tracers.where(ds_tracers['Depth'] != 0)
    ##################
    # # #
    tracer_tuple=(\
        [0,'nh4','TRAC02',-9999.9999],\
        [1,'hooh','TRAC07',-9999.9999],\
        [2,'dia','TRAC21',-9999.9999],\
        [3,'euk','TRAC22',-9999.9999],\
        [4,'syn','TRAC23',-9999.9999],\
        [5,'pro','TRAC24',-9999.9999],\
        [6,'tri','TRAC25',-9999.9999],\
        [7,'coc','TRAC26',-9999.9999],\
        [8,'zoo1','TRAC27',-9999.9999],\
        [9,'zoo2','TRAC28',-9999.9999],\
        [10,'zoo3','TRAC29',-9999.9999],\
        [11,'zoo4','TRAC30',-9999.9999],\
        )
    for tup in tracer_tuple:
        print("process: "+str(tup[1])) 
        tr_ref=str(tup[1])
        nc_ref=str(tup[2])
        ds_tracers[tr_ref]=ds_tracers[nc_ref]*ds_tracers['DepHeight']
        ds_tracers[tr_ref]=ds_tracers[tr_ref].sum(dim=["Z"])
        ds_tracers[tr_ref]=ds_tracers[tr_ref].mean(dim=["X","Y"])
        
    for tup in tracer_tuple:
        print("graph: "+str(tup[1]))
        index=tup[0]
        tr_ref=str(tup[1])
        mean_value=-9999.0
        mean_value=ds_tracers[tr_ref].mean(dim=["T"]).values
        tup[3]=float(mean_value)
        mean_str='annual Mean of Global Mean Int. Depth:'+str(mean_value)
        print(mean_str)
        axs[index].plot(ds_tracers.T.dt.dayofyear ,ds_tracers[tr_ref].values, linestyle='solid', color='green' , linewidth=2, markersize=12,label=str(tr_ref))  
    
        axs[index].hlines(mean_value,xmin=0,xmax=360, colors='red', linestyles='-')
        axs[index].annotate(mean_str,
                xy=(0.1, 0.1), xycoords='axes fraction',
                horizontalalignment='left', verticalalignment='center',
                color='k',fontsize=10)
        axs[index].annotate(str(tr_ref),
                xy=(-0.15, .5), xycoords='axes fraction',
                horizontalalignment='center', verticalalignment='center',
                rotation=90,
                color='k',fontsize=20)
    #
    for tup in tracer_tuple:
        print(tup)
    #   
    fig.savefig(file_out,dpi=300)
    ########################
    print('EXIT  :: EXIT', flush=True, file = sys.stdout)
   
    
   
#
if __name__ == "__main__":
    #Initialize
    
    main()
