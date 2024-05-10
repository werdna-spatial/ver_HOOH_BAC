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
    NC_file_list = [\
        #        'Zgud.TRAC21.c01.v4c.nc',\
        #        'Zgud.TRAC22.c02.v4c.nc',\
        #        'Zgud.TRAC23.c03.v4c.nc',\
        #        'Zgud.TRAC24.c04.v4c.nc',\
        #        'Zgud.TRAC25.c05.v4c.nc',\
        'Zgud.TRAC07.HOOH.v4c.nc',\
        'Zgud.TRAC02.NH4.v4c.nc',\
        'volume.v4c.nc', \
        'grid.v4c.nc'          \
    ]
    #
    PD_NC_list = []
    PD_NC_list.append(pathlib.Path(
        '/lustre/isaac/proj/UTK0105/Hack_Session/gudb/verification/HOOH/detox_comp/run_HOOH_nodetox_509550/NC_trace/'))
    PD_NC_list.append(pathlib.Path(
        '/lustre/isaac/proj/UTK0105/Hack_Session/gudb/verification/HOOH/detox_comp/run_HOOH_detox_509552/NC_trace/'))
    PD_NC_list.append(pathlib.Path(
        '/lustre/isaac/proj/UTK0105/Hack_Session/gudb/verification/HOOH/detox_comp/run_HOOH_tradeoff_521650/NC_trace/'))
    #PD_NC_list.append(pathlib.Path(''))
    #PD_NC_list.append(pathlib.Path(''))
    #

    #fig, axs = plt.subplots(nrows=1,ncols=2,subplot_kw={'projection': ccrs.Mollweide()} )
    #fig.suptitle('Single Mean over CELL Z(0-160m) and T(1 yr)', fontweight ="bold")
    NC_file = NC_file_list
    #
    PD_NC = PD_NC_list[0]
    os.chdir(PD_NC)
    ds_nodetox = xr.open_mfdataset(
        NC_file,  combine='by_coords', parallel=False, chunks={'T': 1, 'Z': 1})
    # limit time
    #time - use  last year 3 years
    #print('Limit year  :: start', flush=True, file = sys.stdout)
    tempdate=ds_nodetox.T.min()
    YEAR_index=int(tempdate.T.dt.year)
    ds_nodetox=ds_nodetox.isel(T=(ds_nodetox.T.dt.year == YEAR_index))
    #    
    #
    PD_NC = PD_NC_list[1]
    os.chdir(PD_NC)
    ds_detox = xr.open_mfdataset(
        NC_file,  combine='by_coords', parallel=False, chunks={'T': 1, 'Z': 1})
   
    # limit to a year
    ds_detox=ds_detox.isel(T=(ds_detox.T.dt.year == YEAR_index))
    #
    #
    PD_NC = PD_NC_list[2]
    os.chdir(PD_NC)
    ds_tradeoff = xr.open_mfdataset(
        NC_file,  combine='by_coords', parallel=False, chunks={'T': 1, 'Z': 1})
    # limit to a year
    ds_tradeoff=ds_tradeoff.isel(T=(ds_tradeoff.T.dt.year == YEAR_index))
    #
    #
    ds_nodetox['vol_HOOH'] = ds_nodetox['TRAC07']*ds_nodetox['vol']
    ds_nodetox['vol_HOOH_sumZ'] = ds_nodetox['vol_HOOH'].sum(dim=["Z"])
    ds_nodetox['vol_HOOH_mXYsZ'] = ds_nodetox['vol_HOOH_sumZ'].mean(dim=["X", "Y"])
    #
    ds_detox['vol_HOOH'] = ds_detox['TRAC07']*ds_detox['vol']
    ds_detox['vol_HOOH_sumZ'] = ds_detox['vol_HOOH'].sum(dim=["Z"])
    ds_detox['vol_HOOH_mXYsZ'] = ds_detox['vol_HOOH_sumZ'].mean(dim=["X", "Y"])
    #
    ds_tradeoff['vol_HOOH'] = ds_tradeoff['TRAC07']*ds_tradeoff['vol']
    ds_tradeoff['vol_HOOH_sumZ'] = ds_tradeoff['vol_HOOH'].sum(dim=["Z"])
    ds_tradeoff['vol_HOOH_mXYsZ'] = ds_tradeoff['vol_HOOH_sumZ'].mean(dim=["X", "Y"])
    # plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(ds_nodetox['T.dayofyear'].values, ds_nodetox['vol_HOOH_mXYsZ'].values, color='tab:blue' , linestyle='dashed', linewidth=2, markersize=12,label='Production')
    ax.plot(ds_detox['T.dayofyear'].values, ds_detox['vol_HOOH_mXYsZ'].values, color='tab:red' , linestyle='dotted', linewidth=2, markersize=12,label='Detoxify')
    ax.plot(ds_tradeoff['T.dayofyear'].values, ds_tradeoff['vol_HOOH_mXYsZ'].values, color='tab:green' , linestyle='dashdot', linewidth=2, markersize=12,label='tradeoff')
    
    ax.legend()
    ax.set_title('HOOH Global Mean Time Series')
    plt.show()
    #
    PD_out=pathlib.Path('/lustre/isaac/proj/UTK0105/Hack_Session/gudb/verification/HOOH/detox_comp/results/')
    #
    filename=str('C3_Glin')+str('_meanXY_HOOH.png')
    file_out=PD_out.joinpath(filename)
    fig.savefig(file_out,dpi=300)
    
    
    ########################
    print('EXIT  :: EXIT', flush=True, file = sys.stdout)
   
    
   
#
if __name__ == "__main__":
    #Initialize
    
    main()
