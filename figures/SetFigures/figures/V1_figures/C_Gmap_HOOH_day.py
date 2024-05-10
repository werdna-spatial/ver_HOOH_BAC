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

def main():
    NC_file_list=[\
        'Zgud.TRAC21.c01.v4c.nc',\
        'Zgud.TRAC22.c02.v4c.nc',\
        'Zgud.TRAC23.c03.v4c.nc',\
        'Zgud.TRAC24.c04.v4c.nc',\
        'Zgud.TRAC25.c05.v4c.nc',\
        'Zgud.TRAC07.HOOH.v4c.nc',\
        'Zgud.TRAC02.NH4.v4c.nc',\
        'volume.v4c.nc', \
        'grid.v4c.nc'          \
        ]
    #
    PD_NC_list=[]    
    PD_NC_list.append(pathlib.Path('/lustre/isaac/proj/UTK0105/Hack_Session/gudb/verification/HOOH/detox_comp/run_HOOH_nodetox_509550/NC_trace/'))
    PD_NC_list.append(pathlib.Path('/lustre/isaac/proj/UTK0105/Hack_Session/gudb/verification/HOOH/detox_comp/run_HOOH_detox_509552/NC_trace/'))
    
    #PD_NC_list.append(pathlib.Path(''))
    #PD_NC_list.append(pathlib.Path(''))
    #
    for i in range(72):
        extract_date_index=i
        print(extract_date_index)
        #fig, axs = plt.subplots(nrows=3,ncols=2,subplot_kw={'projection': ccrs.Mollweide()} , figsize=(18,10) )
        fig, axs = plt.subplots(nrows=3,ncols=2,subplot_kw={'projection': ccrs.EckertIV()} , figsize=(18,10) )
        #fig.suptitle('Single Mean over CELL Z(0-160m) and T(1 yr)', fontweight ="bold")
        i=0
        ii=0
        jj=0
        HOOH_min=13
        HOOH_max=14.5
        NH4_min=9
        NH4_max=11
        axs_list=[[0,0],[0,1],[1,0],[1,1]]
        PD_NC=PD_NC_list[0]
        for PD_NC in PD_NC_list:
            print('Model :: '+str(PD_NC), flush=True, file = sys.stdout)
            #ii=axs_list[i][0]
            #jj=axs_list[i][1]
            print('index :: '+str(i)+' '+str(ii)+' '+str(jj), flush=True, file = sys.stdout)
            os.chdir(PD_NC)
            NC_file=NC_file_list
            #L_tracer=L_tracer_list[i]
            ds_tracers=xr.open_mfdataset(NC_file,  combine='by_coords', parallel=False,chunks={'T':1,'Z':1})
            #
            # limit time
            #time - use  last year 3 years
            #print('Limit year  :: start', flush=True, file = sys.stdout)
            tempdate=ds_tracers.T[extract_date_index]
            #YEAR_index=int(tempdate.T.dt.year)
            ds_tracers=ds_tracers.isel(T=(ds_tracers.T == tempdate))
            ds_tracers=ds_tracers.sum(dim=['T'])
            #
            #
            ds_tracers['vol_HOOH']=ds_tracers['TRAC07']*ds_tracers['vol']
            ds_tracers['vol_HOOH_sumZ']=ds_tracers['vol_HOOH'].sum(dim=["Z"])
            #ds_tracers['vol_HOOH_mTsZ']=ds_tracers['vol_HOOH_sumZ'].mean(dim=["T"])
            ds_tracers['log_vol_HOOH_sZ']=np.log10(ds_tracers['vol_HOOH_sumZ'])
            x=ds_tracers['log_vol_HOOH_sZ'].values.ravel()
            #print('HOOH :: '+str(x.size)+' '+str(np.nanmax(x))+' '+str(np.nanmin(x)), flush=True, file = sys.stdout)
            print('HOOH :: '+str(x.size)+' '+str(np.nanmax(x))+' '+str(np.nanmin(x))+' '+str(np.nanpercentile(x,90))+' '+str(np.nanpercentile(x,50))+' '+str(np.nanpercentile(x,10)), flush=True, file = sys.stdout)
            #
            #
            axs[i,0].set_global()
            axs[i,0].coastlines()
            cmap = plt.get_cmap('Purples')
            im_a=ds_tracers['log_vol_HOOH_sZ'].plot.pcolormesh( x='X', y='Y',
                                                       ax=axs[i,0],
                                                       transform=ccrs.PlateCarree(),
                                                       vmin= HOOH_min,
                                                       vmax= HOOH_max,
                                                       add_colorbar=False,
                                                       zorder=0, 
                                                       cmap=cmap)
            #
            ds_tracers['vol_NH4']=ds_tracers['TRAC02']*ds_tracers['vol']
            ds_tracers['vol_NH4_sumZ']=ds_tracers['vol_NH4'].sum(dim=["Z"])
            #ds_tracers['vol_NH4_mTsZ']=ds_tracers['vol_NH4_sumZ'].mean(dim=["T"])
            ds_tracers['log_vol_NH4_sZ']=np.log10(ds_tracers['vol_NH4_sumZ'])
            x=ds_tracers['log_vol_NH4_sZ'].values.ravel()
            #print('HOOH :: '+str(x.size)+' '+str(np.nanmax(x))+' '+str(np.nanmin(x)), flush=True, file = sys.stdout)
            print('NH4 :: '+str(x.size)+' '+str(np.nanmax(x))+' '+str(np.nanmin(x))+' '+str(np.nanpercentile(x,90))+' '+str(np.nanpercentile(x,50))+' '+str(np.nanpercentile(x,10)), flush=True, file = sys.stdout)
            #
            #
            axs[i,1].set_global()
            axs[i,1].coastlines()
            cmap = plt.get_cmap('Greens')
            im_b=ds_tracers['log_vol_NH4_sZ'].plot.pcolormesh( x='X', y='Y',
                                                       ax=axs[i,1],
                                                       transform=ccrs.PlateCarree(),
                                                       vmin= NH4_min,
                                                       vmax= NH4_max,
                                                       add_colorbar=False,
                                                       zorder=0, 
                                                       cmap=cmap)
            #
            if i==0:
                prev_HOOH=ds_tracers['log_vol_HOOH_sZ']
                prev_NH4=ds_tracers['log_vol_NH4_sZ']
            i=i+1
            #
            #
        #
        axs[i,0].set_global()
        axs[i,0].coastlines()
        cmap = plt.get_cmap('seismic')
        diff_HOOH=prev_HOOH-ds_tracers['log_vol_HOOH_sZ']
        im_c=diff_HOOH.plot.pcolormesh( x='X', y='Y',
                                                   ax=axs[i,0],
                                                   transform=ccrs.PlateCarree(),
                                                   vmin= -0.35,
                                                   vmax= 0.35,
                                                   add_colorbar=False,
                                                   zorder=0, 
                                                   cmap=cmap)
        axs[i,1].set_global()
        axs[i,1].coastlines()
        cmap = plt.get_cmap('seismic')
        diff_NH4=prev_NH4-ds_tracers['log_vol_NH4_sZ']
        im_d=diff_NH4.plot.pcolormesh( x='X', y='Y',
                                                   ax=axs[i,1],
                                                   transform=ccrs.PlateCarree(),
                                                   vmin= -0.1,
                                                   vmax= 0.1,
                                                   add_colorbar=False,
                                                   zorder=0, 
                                                   cmap=cmap)
        #
        #    
        cbar_a = fig.colorbar(im_a, ax=axs[:2, 0], shrink=0.7, location='bottom')
        cbar_a.set_label('log10 HOOH(mmol)')
        cbar_b = fig.colorbar(im_b, ax=axs[:2, 1], shrink=0.7, location='bottom')
        cbar_b.set_label('log10 NH4(mmol)')
        cbar_c = fig.colorbar(im_c, ax=axs[2, 0], shrink=0.7, location='bottom')
        cbar_c.set_label('log10 Diff HOOH(mmol)')
        cbar_d = fig.colorbar(im_d, ax=axs[2, 1], shrink=0.7, location='bottom')
        cbar_d.set_label('log10 Diff NH4(mmol)')
        #
        axs[0,0].annotate(str(tempdate.values),
                xy=(0.01, 1.25), xycoords='axes fraction',
                horizontalalignment='center', verticalalignment='center',
                color='k',fontsize=10)
        axs[0,0].annotate('HOOH',
                xy=(0.5, 1.15), xycoords='axes fraction',
                horizontalalignment='center', verticalalignment='center',
                color='k',fontsize=20)
        axs[0,1].annotate('NH4',
                xy=(0.5, 1.15), xycoords='axes fraction',
                horizontalalignment='center', verticalalignment='center',
                color='k',fontsize=20)
        axs[0,0].annotate('Prod',
                xy=(-0.15, .5), xycoords='axes fraction',
                horizontalalignment='center', verticalalignment='center',
                rotation=90,
                color='k',fontsize=20)
        axs[1,0].annotate('Detox',
                xy=(-0.15, .5), xycoords='axes fraction',
                horizontalalignment='center', verticalalignment='center',
                rotation=90,
                color='k',fontsize=20)
        axs[2,0].annotate('Prod-Detox',
                xy=(-0.15, .5), xycoords='axes fraction',
                horizontalalignment='center', verticalalignment='center',
                rotation=90,
                color='k',fontsize=20)
        #fig.tight_layout()
        #
        PD_out=pathlib.Path('/lustre/isaac/proj/UTK0105/Hack_Session/gudb/verification/HOOH/detox_comp/results/')
        #
        filename=str('C_Gmap')+str('_mean_HOOH_NH4_day')+str(extract_date_index)+str('.png')
        file_out=PD_out.joinpath(filename)
        fig.savefig(file_out,dpi=300)
    ########################
    print('EXIT  :: EXIT', flush=True, file = sys.stdout)
   
    
   
#
if __name__ == "__main__":
    #Initialize
    
    main()
