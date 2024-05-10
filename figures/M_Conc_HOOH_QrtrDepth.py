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
    filename=str(PD_tag)+str('map')+str('_Hconc_QtrDep.png')
    file_out=PD_out.joinpath(filename)
    
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
        # zoo1 consumes Pro01
        '3d.TRAC27.Zoo01.v4c.nc',\
        # zoo2 consumes Syn01
        '3d.TRAC28.Zoo02.v4c.nc',\
        # '3d.TRAC29.Zoo03.v4c.nc',\
        # '3d.TRAC30.Zoo04.v4c.nc',\
        # '3d.TRAC31.Chl1.v4c.nc',\
        # '3d.TRAC32.Chl2.v4c.nc',\
        # '3d.TRAC33.Chl3.v4c.nc',\
        # '3d.TRAC34.Chl4.v4c.nc',\
        # '3d.TRAC35.Chl5.v4c.nc',\
        '3d.UTK_holl.holling.v4c.nc',\
        '3d.UTK_2ZPP.holling.v4c.nc',\
        '3d.UTK_C.C.v4c.nc',\
        '3d.UTK_D.D.v4c.nc',\
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
    #YEAR_index=int(2009)
    #ds_tracers=ds_tracers.isel(T=(ds_tracers.T.dt.year == YEAR_index))
    ds_tracers=ds_tracers.isel(T=(ds_tracers.T.dt.year >= 2008))
    #
    #limit depth and sum
    print('Limit depth  :: start', flush=True, file = sys.stdout)
    #ds_tracers=ds_tracers.isel(Z=(ds_tracers.Z > (min(ext_deps_all)-1) ) )
    #ds_tracers=ds_tracers.sum(dim='Z')
    #
    ds_tracers['bathy_mask']=ds_tracers['Depth'].where(ds_tracers['Depth'] != 0)  
    ds_tracers=ds_tracers.where(ds_tracers['Depth'] != 0)
    #
    
    ##################
    ##################
    ds_tracers['hooh']=ds_tracers['TRAC07']
    ds_tracers['syn']=ds_tracers['TRAC23']
    ds_tracers['pro']=ds_tracers['TRAC24']
    ds_tracers['Bdetox']=ds_tracers['UTK_C']
    ds_tracers['PHOOH']=ds_tracers['UTK_D']
    #
    ds_tracers['sa_hooh']=ds_tracers['hooh']*ds_tracers['DepHeight']
    ds_tracers['sa_syn']=ds_tracers['syn']*ds_tracers['DepHeight']
    ds_tracers['sa_pro']=ds_tracers['pro']*ds_tracers['DepHeight']
    ds_tracers['sa_Bdetox']=ds_tracers['Bdetox']*ds_tracers['DepHeight']
    ds_tracers['sa_PHOOH']=ds_tracers['PHOOH']*ds_tracers['DepHeight']
    #
    ds_tracers['sum_sa_hooh']=ds_tracers['sa_hooh'].sum(dim=["Z"])
    ds_tracers['sum_sa_syn'] =ds_tracers['sa_syn' ].sum(dim=["Z"])
    ds_tracers['sum_sa_pro'] =ds_tracers['sa_pro' ].sum(dim=["Z"])
    ds_tracers['sum_sa_Bdetox'] =ds_tracers['sa_Bdetox' ].sum(dim=["Z"])
    ds_tracers['sum_sa_PHOOH'] =ds_tracers['sa_PHOOH' ].sum(dim=["Z"])
    #
    # ds_tracers['Msum_sa_hooh']=ds_tracers['sum_sa_hooh'].mean(dim=["T"])
    # ds_tracers['Msum_sa_syn'] =ds_tracers['sum_sa_syn' ].mean(dim=["T"])
    # ds_tracers['Msum_sa_pro'] =ds_tracers['sum_sa_pro' ].mean(dim=["T"])
    # ds_tracers['Msum_sa_Bdetox'] =ds_tracers['sum_sa_Bdetox' ].mean(dim=["T"])
    # ds_tracers['Msum_sa_PHOOH'] =ds_tracers['sum_sa_PHOOH' ].mean(dim=["T"])
    # #
    # #
    # ds_tracers['LMsum_sa_hooh']=np.log10(ds_tracers['Msum_sa_hooh'])
    # ds_tracers['LMsum_sa_syn'] =np.log10(ds_tracers['Msum_sa_syn' ])
    # ds_tracers['LMsum_sa_pro'] =np.log10(ds_tracers['Msum_sa_pro' ])
    # ds_tracers['LMsum_sa_Bdetox'] =np.log10(ds_tracers['Msum_sa_Bdetox' ])
    # ds_tracers['LMsum_sa_PHOOH'] =np.log10(ds_tracers['Msum_sa_PHOOH' ])
    #
    #
    sec_in_day=86400
    mole_in_mmol=0.001
    Gminrange=0.0
    Gmaxrange=1.0e-06
    #maxrange=0.002*sec_in_day
    fig, axs = plt.subplots(nrows=4,ncols=3,subplot_kw={'projection': ccrs.EckertIV()} , figsize=(18,12) )
    #
    qtr=0
    for qtr in range(4):
        print(qtr)
        if (qtr==0):
            ds_qtr=ds_tracers.sel(T=slice("2008-12-01", "2009-02-30"))
            title=str("DJF")
        elif (qtr==1):
            ds_qtr=ds_tracers.sel(T=slice("2008-03-01", "2009-05-30"))
            title=str("MAM")    
        elif (qtr==2):
            ds_qtr=ds_tracers.sel(T=slice("2008-06-01", "2009-08-30"))
            title=str("JJA")  
        elif (qtr==3):
            ds_qtr=ds_tracers.sel(T=slice("2008-09-01", "2009-11-30"))
            title=str("SON")  
        #
        axs[qtr,0].annotate(str(title),
                xy=(-0.1, .5), xycoords='axes fraction',
                horizontalalignment='center', verticalalignment='center',
                rotation=90,
                color='k',fontsize=20)
        #axs[qtr,0].set_ylabel(title)
        #
        ds_qtr['MAP'] =ds_qtr['hooh' ].mean(dim=["T"])
        #
        for layer in range(3):
            if(layer==0):
                minrange=Gminrange
                maxrange=Gmaxrange                
            elif (layer==1):
                minrange=Gminrange
                maxrange=Gmaxrange    
                #maxrange=0.75*Gmaxrange
            elif (layer==2):
                minrange=Gminrange
                maxrange=Gmaxrange    
                #maxrange=0.5*Gmaxrange
            #print("layer : "+str(layer))
            map_layer=ds_qtr['MAP'].isel(Z=layer)*sec_in_day*mole_in_mmol
            x=map_layer.values.ravel()
            #print('HOOH :: '+str(x.size)+' '+str(np.nanmax(x))+' '+str(np.nanmin(x)), flush=True, file = sys.stdout)
            print(str(title)+str("_")+str(layer)+' :: '+str(x.size)+' '+str(np.nanmax(x))+' '+str(np.nanmin(x))+' '+str(np.nanpercentile(x,90))+' '+str(np.nanpercentile(x,50))+' '+str(np.nanpercentile(x,10)), flush=True, file = sys.stdout)
            print("Graph min,max : "+str(minrange)+" , "+str(maxrange))
            #
            axs[qtr,layer].set_global()
            axs[qtr,layer].coastlines()
            #cmap = plt.get_cmap('plasma')
            cmap = plt.get_cmap('Purples')
            PHOOH_min=0
            PHOOH_max=0.05
            im=map_layer.plot.pcolormesh( x='X', y='Y',
                                                       ax=axs[qtr,layer],
                                                       transform=ccrs.PlateCarree(),
                                                       vmin= minrange,
                                                       vmax= maxrange,
                                                       add_colorbar=False,
                                                       zorder=0, 
                                                       cmap=cmap)
            if (qtr==0):
                cbar_a = fig.colorbar(im, ax=axs[:,layer], shrink=0.8, location='right')
                cbar_a.set_label('HOOH Concentration (mmol HOOH/m^3)')
    #
    #
    #
    #cbar_a = fig.colorbar(im, ax=axs[:,:], shrink=0.8, location='right')
    #cbar_a.set_label('HOOH Production (mmol HOOH/m^3/day)')
   
    #
    
    fig.savefig(file_out,dpi=300)
    ########################
    print('EXIT  :: EXIT', flush=True, file = sys.stdout)
   
    
   
#
if __name__ == "__main__":
    #Initialize
    
    main()
