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
    #filename=str(PD_tag)+str('_mapSep_HOOH_Syn_Pro')+str('.png')
    #file_out=PD_out.joinpath(filename)
    
    NC_file_list=[\
        # '3d.PP.Primary.v4c.nc',\
        # '3d.TRAC01.DIC.v4c.nc',\
        # '3d.TRAC02.NH4.v4c.nc',\
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
        # '3d.TRAC21.Diatom01.v4c.nc',\
        # '3d.TRAC22.Euk01.v4c.nc',\
        '3d.TRAC23.Syn01.v4c.nc',\
        '3d.TRAC24.Pro01.v4c.nc',\
        # '3d.TRAC25.Tricho01.v4c.nc',\
        # '3d.TRAC26.Cocco01.v4c.nc',\
        # '3d.TRAC27.Zoo01.v4c.nc',\
        # '3d.TRAC28.Zoo02.v4c.nc',\
        # '3d.TRAC29.Zoo03.v4c.nc',\
        # '3d.TRAC30.Zoo04.v4c.nc',\
        # '3d.TRAC31.Chl1.v4c.nc',\
        # '3d.TRAC32.Chl2.v4c.nc',\
        # '3d.TRAC33.Chl3.v4c.nc',\
        # '3d.TRAC34.Chl4.v4c.nc',\
        # '3d.TRAC35.Chl5.v4c.nc',\
        'grid.v4c.nc'\
        ]
    #    
    PD_NC_list=[]    
    PD_NC_list.append(pathlib.Path.joinpath(PD_base,'Nimp/NC_trace'))
    PD_NC_list.append(pathlib.Path.joinpath(PD_base,'Kpro/NC_trace'))
    PD_NC_list.append(pathlib.Path.joinpath(PD_base,'Spro/NC_trace'))
    #
    #####
    #
    #fig, axs = plt.subplots(nrows=3,ncols=3,subplot_kw={'projection': ccrs.EckertIV()} , figsize=(18,10) )
    #plt.subplots_adjust(wspace=0.01,hspace=0.01)
    #fig.suptitle('Single Mean over CELL Z(0-160m) and T(1 yr)', fontweight ="bold")
    i=0
    ii=0
    jj=0
    HOOH_min=0.0
    HOOH_max=2.5
    SYN_min=0.0
    SYN_max=32.0
    PRO_min=0.0
    PRO_max=32.0
    #
    PD_NC=PD_NC_list[0]
    for PD_NC in PD_NC_list:
        print('Model :: '+str(PD_NC), flush=True, file = sys.stdout)
        print('index :: '+str(i)+' '+str(ii)+' '+str(jj), flush=True, file = sys.stdout)
        os.chdir(PD_NC)
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
        #
        #limit depth and sum
        print('Limit depth  :: start', flush=True, file = sys.stdout)
        #ds_tracers=ds_tracers.isel(Z=(ds_tracers.Z > (min(ext_deps_all)-1) ) )
        #ds_tracers=ds_tracers.sum(dim='Z')
        #
        ds_tracers['bathy_mask']=ds_tracers['Depth'].where(ds_tracers['Depth'] != 0)  
        ds_tracers=ds_tracers.where(ds_tracers['Depth'] != 0)
        ##################
        ##################
        ds_tracers['hooh']=ds_tracers['TRAC07']
        ds_tracers['syn']=ds_tracers['TRAC23']
        ds_tracers['pro']=ds_tracers['TRAC24']
        #
        ds_tracers['sa_hooh']=ds_tracers['hooh']*ds_tracers['DepHeight']
        ds_tracers['sa_syn']=ds_tracers['syn']*ds_tracers['DepHeight']
        ds_tracers['sa_pro']=ds_tracers['pro']*ds_tracers['DepHeight']
        #
        ds_tracers['sum_sa_hooh']=ds_tracers['sa_hooh'].sum(dim=["Z"])
        ds_tracers['sum_sa_syn'] =ds_tracers['sa_syn' ].sum(dim=["Z"])
        ds_tracers['sum_sa_pro'] =ds_tracers['sa_pro' ].sum(dim=["Z"])
        #
        ds_tracers['Msum_sa_hooh']=ds_tracers['sum_sa_hooh'].mean(dim=["T"])
        ds_tracers['Msum_sa_syn'] =ds_tracers['sum_sa_syn' ].mean(dim=["T"])
        ds_tracers['Msum_sa_pro'] =ds_tracers['sum_sa_pro' ].mean(dim=["T"])
        #
        #ds_tracers['LMsum_sa_hooh']=np.log10(ds_tracers['Msum_sa_hooh'])
        #ds_tracers['LMsum_sa_syn'] =np.log10(ds_tracers['Msum_sa_syn' ])
        #ds_tracers['LMsum_sa_pro'] =np.log10(ds_tracers['Msum_sa_pro' ])
        #
        #
        x=ds_tracers['Msum_sa_hooh'].values.ravel()
        #print('HOOH :: '+str(x.size)+' '+str(np.nanmax(x))+' '+str(np.nanmin(x)), flush=True, file = sys.stdout)
        print('HOOH :: '+str(x.size)+' '+str(np.nanmax(x))+' '+str(np.nanmin(x))+' '+str(np.nanpercentile(x,90))+' '+str(np.nanpercentile(x,50))+' '+str(np.nanpercentile(x,10)), flush=True, file = sys.stdout)
        #
        x=ds_tracers['Msum_sa_syn'].values.ravel()
        #print('HOOH :: '+str(x.size)+' '+str(np.nanmax(x))+' '+str(np.nanmin(x)), flush=True, file = sys.stdout)
        print('SYN :: '+str(x.size)+' '+str(np.nanmax(x))+' '+str(np.nanmin(x))+' '+str(np.nanpercentile(x,90))+' '+str(np.nanpercentile(x,50))+' '+str(np.nanpercentile(x,10)), flush=True, file = sys.stdout)
        #
        x=ds_tracers['Msum_sa_pro'].values.ravel()
        #print('HOOH :: '+str(x.size)+' '+str(np.nanmax(x))+' '+str(np.nanmin(x)), flush=True, file = sys.stdout)
        print('PRO :: '+str(x.size)+' '+str(np.nanmax(x))+' '+str(np.nanmin(x))+' '+str(np.nanpercentile(x,90))+' '+str(np.nanpercentile(x,50))+' '+str(np.nanpercentile(x,10)), flush=True, file = sys.stdout)
        #
        #
        fig, axs = plt.subplots(nrows=1,ncols=3,subplot_kw={'projection': ccrs.EckertIV()} , figsize=(18,10) )
        plt.subplots_adjust(wspace=0.01,hspace=0.01)
        #
        axs[0].set_global()
        axs[0].coastlines()
        cmap = plt.get_cmap('Purples')
        im_a=ds_tracers['Msum_sa_hooh'].plot.pcolormesh( x='X', y='Y',
                                                   ax=axs[0],
                                                   transform=ccrs.PlateCarree(),
                                                   vmin= HOOH_min,
                                                   vmax= HOOH_max,
                                                   add_colorbar=False,
                                                   zorder=0, 
                                                   cmap=cmap)
        #
        #
        axs[1].set_global()
        axs[1].coastlines()
        cmap = plt.get_cmap('Greens')
        im_b=ds_tracers['Msum_sa_syn'].plot.pcolormesh( x='X', y='Y',
                                                   ax=axs[1],
                                                   transform=ccrs.PlateCarree(),
                                                   vmin= SYN_min,
                                                   vmax= SYN_max,
                                                   add_colorbar=False,
                                                   zorder=0, 
                                                   cmap=cmap)
        ##
        axs[2].set_global()
        axs[2].coastlines()
        cmap = plt.get_cmap('Greens')
        im_c=ds_tracers['Msum_sa_pro'].plot.pcolormesh( x='X', y='Y',
                                                   ax=axs[2],
                                                   transform=ccrs.PlateCarree(),
                                                   vmin= PRO_min,
                                                   vmax= PRO_max,
                                                   add_colorbar=False,
                                                   zorder=0, 
                                                   cmap=cmap)
        #
        
        filename=str(PD_tag)+str('_mapSep_HOOH_Syn_Pro')+str('.png')
        file_out=PD_out.joinpath(filename)
                #
    #
    #
    #    
        cbar_a = fig.colorbar(im_a, ax=axs[0], shrink=0.7, location='bottom')
        cbar_a.set_label('HOOH(mmol)')
        cbar_b = fig.colorbar(im_b, ax=axs[1], shrink=0.7, location='bottom')
        cbar_b.set_label('Syn(mmol)')
        cbar_c = fig.colorbar(im_c, ax=axs[2], shrink=0.7, location='bottom')
        cbar_c.set_label('Pro(mmol)')
        # cbar_d = fig.colorbar(im_d, ax=axs[2, 1], shrink=0.7, location='bottom')
        # cbar_d.set_label('log10 Diff NH4(mmol)')
        # #
        axs[0].annotate('HOOH',
                xy=(0.5, 1.15), xycoords='axes fraction',
                horizontalalignment='center', verticalalignment='center',
                color='k',fontsize=20)
        axs[1].annotate('Syn',
                xy=(0.5, 1.15), xycoords='axes fraction',
                horizontalalignment='center', verticalalignment='center',
                color='k',fontsize=20)
        axs[2].annotate('Pro',
                xy=(0.5, 1.15), xycoords='axes fraction',
                horizontalalignment='center', verticalalignment='center',
                color='k',fontsize=20)
        if i==0:
            filename=str(PD_tag)+str('_map_NIMP_HOOH_Syn_Pro')+str('.png')
            axs[0].annotate('No Impact',
                    xy=(-0.15, .5), xycoords='axes fraction',
                    horizontalalignment='center', verticalalignment='center',
                    rotation=90,
                    color='k',fontsize=20)
        elif i==1:
            filename=str(PD_tag)+str('_map_KPRO_HOOH_Syn_Pro')+str('.png')
            axs[0].annotate('Kill Pro',
                    xy=(-0.15, .5), xycoords='axes fraction',
                    horizontalalignment='center', verticalalignment='center',
                    rotation=90,
                    color='k',fontsize=20)
        elif i==2:
            filename=str(PD_tag)+str('_map_SPRO_HOOH_Syn_Pro')+str('.png')
            axs[0].annotate('Save Pro',
                    xy=(-0.15, .5), xycoords='axes fraction',
                    horizontalalignment='center', verticalalignment='center',
                    rotation=90,
                    color='k',fontsize=20)
        #
        #
        file_out=PD_out.joinpath(filename)
        fig.savefig(file_out,dpi=300)
        #
        i=i+1
    ########################
    print('EXIT  :: EXIT', flush=True, file = sys.stdout)
   
    
   
#
if __name__ == "__main__":
    #Initialize
    
    main()
