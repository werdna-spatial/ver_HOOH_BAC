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
    filename=str(PD_tag)+str('_loc_tracer_')+str(lat)+str('_')+str(long)+str('.png')
    file_out=PD_out.joinpath(filename)
    
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
    PD_NC_list=[]    
    PD_NC_list.append(pathlib.Path.joinpath(PD_base,'Nimp/NC_trace'))
    PD_NC_list.append(pathlib.Path.joinpath(PD_base,'Kpro/NC_trace'))
    PD_NC_list.append(pathlib.Path.joinpath(PD_base,'Spro/NC_trace'))
    #
    #####
    #
    fig, axs = plt.subplots(nrows=12,ncols=3, figsize=(18,18), sharex='all' )
    #plt.subplots_adjust(wspace=0.01,hspace=0.01)
    #fig.suptitle('Single Mean over CELL Z(0-160m) and T(1 yr)', fontweight ="bold")
    i=0
    ii=0
    jj=0
    HOOH_min=-0.7
    HOOH_max=0.0
    SYN_min=0.75
    SYN_max=1.5
    PRO_min=0.75
    PRO_max=1.5
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
        # Limit Location
       
        ds_tracers=ds_tracers.where(ds_tracers.Y == lat,drop=True)
        ds_tracers=ds_tracers.where(ds_tracers.X == long,drop=True)
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
        ds_tracers['nh4']=ds_tracers['TRAC02']*ds_tracers['DepHeight']
        ds_tracers['hooh']=ds_tracers['TRAC07']*ds_tracers['DepHeight']
        ds_tracers['dia']=ds_tracers['TRAC21']*ds_tracers['DepHeight']
        ds_tracers['euk']=ds_tracers['TRAC22']*ds_tracers['DepHeight']
        ds_tracers['syn']=ds_tracers['TRAC23']*ds_tracers['DepHeight']
        ds_tracers['pro']=ds_tracers['TRAC24']*ds_tracers['DepHeight']
        ds_tracers['tri']=ds_tracers['TRAC25']*ds_tracers['DepHeight']
        ds_tracers['coc']=ds_tracers['TRAC26']*ds_tracers['DepHeight']
        ds_tracers['zoo1']=ds_tracers['TRAC27']*ds_tracers['DepHeight']
        ds_tracers['zoo2']=ds_tracers['TRAC28']*ds_tracers['DepHeight']
        ds_tracers['zoo3']=ds_tracers['TRAC29']*ds_tracers['DepHeight']
        ds_tracers['zoo4']=ds_tracers['TRAC30']*ds_tracers['DepHeight']
        #
        ds_tracers['nh4']=ds_tracers['nh4'].sum(dim=["Z","X","Y"])
        ds_tracers['hooh']=ds_tracers['hooh'].sum(dim=["Z","X","Y"])
        ds_tracers['dia']=ds_tracers['dia'].sum(dim=["Z","X","Y"])
        ds_tracers['euk']=ds_tracers['euk'].sum(dim=["Z","X","Y"])
        ds_tracers['syn']=ds_tracers['syn'].sum(dim=["Z","X","Y"])
        ds_tracers['pro']=ds_tracers['pro'].sum(dim=["Z","X","Y"])
        ds_tracers['tri']=ds_tracers['tri'].sum(dim=["Z","X","Y"])
        ds_tracers['coc']=ds_tracers['coc'].sum(dim=["Z","X","Y"])
        ds_tracers['zoo1']=ds_tracers['zoo1'].sum(dim=["Z","X","Y"])
        ds_tracers['zoo2']=ds_tracers['zoo2'].sum(dim=["Z","X","Y"])
        ds_tracers['zoo3']=ds_tracers['zoo3'].sum(dim=["Z","X","Y"])
        ds_tracers['zoo4']=ds_tracers['zoo4'].sum(dim=["Z","X","Y"])
        #
        #
        #
        # x=ds_tracers['hooh'].values.ravel()
        # #print('HOOH :: '+str(x.size)+' '+str(np.nanmax(x))+' '+str(np.nanmin(x)), flush=True, file = sys.stdout)
        # print('HOOH :: '+str(x.size)+' '+str(np.nanmax(x))+' '+str(np.nanmin(x))+' '+str(np.nanpercentile(x,90))+' '+str(np.nanpercentile(x,50))+' '+str(np.nanpercentile(x,10)), flush=True, file = sys.stdout)
        # #
        # x=ds_tracers['syn'].values.ravel()
        # #print('HOOH :: '+str(x.size)+' '+str(np.nanmax(x))+' '+str(np.nanmin(x)), flush=True, file = sys.stdout)
        # print('SYN :: '+str(x.size)+' '+str(np.nanmax(x))+' '+str(np.nanmin(x))+' '+str(np.nanpercentile(x,90))+' '+str(np.nanpercentile(x,50))+' '+str(np.nanpercentile(x,10)), flush=True, file = sys.stdout)
        # #
        # x=ds_tracers['pro'].values.ravel()
        # #print('HOOH :: '+str(x.size)+' '+str(np.nanmax(x))+' '+str(np.nanmin(x)), flush=True, file = sys.stdout)
        # print('PRO :: '+str(x.size)+' '+str(np.nanmax(x))+' '+str(np.nanmin(x))+' '+str(np.nanpercentile(x,90))+' '+str(np.nanpercentile(x,50))+' '+str(np.nanpercentile(x,10)), flush=True, file = sys.stdout)
        # #
        #
        
        p0=axs[0,i].plot(ds_tracers.T.dt.dayofyear ,ds_tracers['nh4'].values, linestyle='solid', color='green' , linewidth=2, markersize=12,label=str('NH4'))  
        p1=axs[1,i].plot(ds_tracers.T.dt.dayofyear ,ds_tracers['hooh'].values, linestyle='solid', color='green' , linewidth=2, markersize=12,label=str('hooh'))  
        p2=axs[2,i].plot(ds_tracers.T.dt.dayofyear ,ds_tracers['dia'].values, linestyle='solid', color='green' , linewidth=2, markersize=12,label=str('dia'))  
        p3=axs[3,i].plot(ds_tracers.T.dt.dayofyear ,ds_tracers['euk'].values, linestyle='solid', color='green' , linewidth=2, markersize=12,label=str('euk'))  
        p4=axs[4,i].plot(ds_tracers.T.dt.dayofyear ,ds_tracers['syn'].values, linestyle='solid', color='green' , linewidth=2, markersize=12,label=str('syn'))  
        p5=axs[5,i].plot(ds_tracers.T.dt.dayofyear ,ds_tracers['pro'].values, linestyle='solid', color='green' , linewidth=2, markersize=12,label=str('pro'))  
        p6=axs[6,i].plot(ds_tracers.T.dt.dayofyear ,ds_tracers['tri'].values, linestyle='solid', color='green' , linewidth=2, markersize=12,label=str('tri'))  
        p7=axs[7,i].plot(ds_tracers.T.dt.dayofyear ,ds_tracers['coc'].values, linestyle='solid', color='green' , linewidth=2, markersize=12,label=str('coc'))  
        p8=axs[8,i].plot(ds_tracers.T.dt.dayofyear ,ds_tracers['zoo1'].values, linestyle='solid', color='green' , linewidth=2, markersize=12,label=str('zoo1'))  
        p9=axs[9,i].plot(ds_tracers.T.dt.dayofyear ,ds_tracers['zoo2'].values, linestyle='solid', color='green' , linewidth=2, markersize=12,label=str('zoo2'))  
        p10=axs[10,i].plot(ds_tracers.T.dt.dayofyear ,ds_tracers['zoo3'].values, linestyle='solid', color='green' , linewidth=2, markersize=12,label=str('zoo3'))  
        p11=axs[11,i].plot(ds_tracers.T.dt.dayofyear ,ds_tracers['zoo4'].values, linestyle='solid', color='green' , linewidth=2, markersize=12,label=str('zoo4'))  
        #
        #
        i=i+1
        #
        #
    #
    #
    #    
    # cbar_a = fig.colorbar(im_a, ax=axs[2, 0], shrink=0.7, location='bottom')
    # cbar_a.set_label('log10 HOOH(mmol)')
    # cbar_b = fig.colorbar(im_b, ax=axs[2, 1], shrink=0.7, location='bottom')
    # cbar_b.set_label('log10 Syn(mmol)')
    # cbar_c = fig.colorbar(im_c, ax=axs[2, 2], shrink=0.7, location='bottom')
    # cbar_c.set_label('log10 Pro(mmol)')
    # # cbar_d = fig.colorbar(im_d, ax=axs[2, 1], shrink=0.7, location='bottom')
    # # cbar_d.set_label('log10 Diff NH4(mmol)')
    # # #
    axs[0,0].annotate('No Impact',
            xy=(0.5, 1.15), xycoords='axes fraction',
            horizontalalignment='center', verticalalignment='center',
            color='k',fontsize=20)
    axs[0,1].annotate('Kill Pro',
            xy=(0.5, 1.15), xycoords='axes fraction',
            horizontalalignment='center', verticalalignment='center',
            color='k',fontsize=20)
    axs[0,2].annotate('Save Pro',
            xy=(0.5, 1.15), xycoords='axes fraction',
            horizontalalignment='center', verticalalignment='center',
            color='k',fontsize=20)
    #
    #
    axs[0,0].annotate('NH4',
            xy=(-0.15, .5), xycoords='axes fraction',
            horizontalalignment='center', verticalalignment='center',
            rotation=90,
            color='k',fontsize=20)
    axs[1,0].annotate('HOOH',
            xy=(-0.15, .5), xycoords='axes fraction',
            horizontalalignment='center', verticalalignment='center',
            rotation=90,
            color='k',fontsize=20)
    axs[2,0].annotate('Dia',
            xy=(-0.15, .5), xycoords='axes fraction',
            horizontalalignment='center', verticalalignment='center',
            rotation=90,
            color='k',fontsize=20)
    axs[3,0].annotate('Euk',
            xy=(-0.15, .5), xycoords='axes fraction',
            horizontalalignment='center', verticalalignment='center',
            rotation=90,
            color='k',fontsize=20)
    axs[4,0].annotate('Syn',
            xy=(-0.15, .5), xycoords='axes fraction',
            horizontalalignment='center', verticalalignment='center',
            rotation=90,
            color='k',fontsize=20)
    axs[5,0].annotate('Pro',
            xy=(-0.15, .5), xycoords='axes fraction',
            horizontalalignment='center', verticalalignment='center',
            rotation=90,
            color='k',fontsize=20)
    axs[6,0].annotate('tri',
            xy=(-0.15, .5), xycoords='axes fraction',
            horizontalalignment='center', verticalalignment='center',
            rotation=90,
            color='k',fontsize=20)
    axs[7,0].annotate('coc',
            xy=(-0.15, .5), xycoords='axes fraction',
            horizontalalignment='center', verticalalignment='center',
            rotation=90,
            color='k',fontsize=20)
    axs[8,0].annotate('zoo1',
            xy=(-0.15, .5), xycoords='axes fraction',
            horizontalalignment='center', verticalalignment='center',
            rotation=90,
            color='k',fontsize=20)
    axs[9,0].annotate('zoo2',
            xy=(-0.15, .5), xycoords='axes fraction',
            horizontalalignment='center', verticalalignment='center',
            rotation=90,
            color='k',fontsize=20)
    axs[10,0].annotate('zoo3',
            xy=(-0.15, .5), xycoords='axes fraction',
            horizontalalignment='center', verticalalignment='center',
            rotation=90,
            color='k',fontsize=20)
    axs[11,0].annotate('zoo4',
            xy=(-0.15, .5), xycoords='axes fraction',
            horizontalalignment='center', verticalalignment='center',
            rotation=90,
            color='k',fontsize=20)
   
    #
    
   
    fig.savefig(file_out,dpi=300)
    ########################
    print('EXIT  :: EXIT', flush=True, file = sys.stdout)
   
    
   
#
if __name__ == "__main__":
    #Initialize
    
    main()
