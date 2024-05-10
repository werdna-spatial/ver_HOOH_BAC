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
#
#  ffmpeg -framerate 3  -start_number 10000 -i Mov_HOOH%05d.png   -pix_fmt yuv420p -qscale:v 2  Mov_HOOH_3_v4.mp4
#
def main():
    NC_file_list=[\
        #'Zgud.TRAC21.c01.v4c.nc',\
        #'Zgud.TRAC22.c02.v4c.nc',\
        #'Zgud.TRAC23.c03.v4c.nc',\
        #'Zgud.TRAC24.c04.v4c.nc',\
        #'Zgud.TRAC25.c05.v4c.nc',\
        'Zgud.TRAC07.HOOH.v4c.nc',\
        #'Zgud.TRAC02.NH4.v4c.nc',\
        'volume.v4c.nc', \
        'grid.v4c.nc'          \
        ]
    #
    PD_NC_list=[]   
    PD_NC_list.append(pathlib.Path('/lustre/isaac/proj/UTK0105/Hack_Session/gudb/verification/HOOH/run_HOOH_detox_512835/NC_trace/'))    
    PD_NC_list.append(pathlib.Path('/lustre/isaac/proj/UTK0105/Hack_Session/gudb/verification/HOOH/detox_comp/run_HOOH_nodetox_509550/NC_trace/'))
    PD_NC_list.append(pathlib.Path('/lustre/isaac/proj/UTK0105/Hack_Session/gudb/verification/HOOH/detox_comp/run_HOOH_detox_509552/NC_trace/'))
    
    #PD_NC_list.append(pathlib.Path(''))
    #PD_NC_list.append(pathlib.Path(''))
    #
    PD_NC=PD_NC_list[0]
    
    print('Model :: '+str(PD_NC), flush=True, file = sys.stdout)
    #ii=axs_list[i][0]
    #jj=axs_list[i][1]
    os.chdir(PD_NC)
    NC_file=NC_file_list
    #L_tracer=L_tracer_list[i]
    ds_tracers=xr.open_mfdataset(NC_file,  combine='by_coords', parallel=False,chunks={'T':1,'Z':1})
    #
    # limit time
    #time - use  last year 3 years
    print('Limit year  :: start', flush=True, file = sys.stdout)
    tempdate=ds_tracers.T.max()
    YEAR_index=int(tempdate.T.dt.year)
    ds_tracers=ds_tracers.isel(T=(ds_tracers.T.dt.year == YEAR_index))
    #
    # sum depth for HOOH
    ds_tracers['vol_HOOH_sZ'] = (ds_tracers['TRAC07']*ds_tracers['vol']).sum(dim=["Z"])
    ds_tracers['vol_HOOH_sZ']=ds_tracers['vol_HOOH_sZ'].where(ds_tracers['vol_HOOH_sZ']>1000)
    ds_tracers['log_vol_HOOH_sZ']=np.log10(ds_tracers['vol_HOOH_sZ'])
    #ds_tracers['vol_HOOH_sumZ'] = ds_detox['vol_HOOH'].sum(dim=["Z"])
    #
    xlist=[]
    #HOOH_min=13.0
    #HOOH_max=14.4
    HOOH_min=10**13.0
    HOOH_max=10**14.4
    timeindex=10000
    time=ds_tracers.T[36]
    for time in ds_tracers.T:
        print(time)
        ds_anal=ds_tracers.isel(T=(ds_tracers.T == time))
        fig, axs = plt.subplots(nrows=1,ncols=1,subplot_kw={'projection': ccrs.EckertIV()} , figsize=(19.20,10.80) )
        
        #x=ds_anal['log_vol_HOOH_sZ'].values.ravel()
        x=ds_anal['vol_HOOH_sZ'].values.ravel()
        #print('HOOH :: '+str(x.size)+' '+str(np.nanmax(x))+' '+str(np.nanmin(x)), flush=True, file = sys.stdout)
        print(str(time.values)+' '+str(x.size)+' '+str(np.nanmax(x))+' '+str(np.nanmin(x))+' '+str(np.nanpercentile(x,90))+' '+str(np.nanpercentile(x,50))+' '+str(np.nanpercentile(x,10)), flush=True, file = sys.stdout)
        temp=[str(time.values),str(x.size),str(np.nanmax(x)),str(np.nanmin(x)),str(np.nanpercentile(x,90)),str(np.nanpercentile(x,50)),str(np.nanpercentile(x,10))]
        
        xlist.append(temp)
        #
        #
        #axs.set_extent([-80,80, 0, 360], crs=ccrs.Geodetic())
        axs.set_global()
        axs.stock_img()
        axs.coastlines()
        cmap = plt.get_cmap('plasma')
        #im_a=ds_anal['log_vol_HOOH_sZ'][0].plot.pcolormesh( x='X', y='Y',
        im_a=ds_anal['vol_HOOH_sZ'][0].plot.pcolormesh( x='X', y='Y',
                                                   ax=axs,
                                                   transform=ccrs.PlateCarree(),
                                                   vmin= HOOH_min,
                                                   vmax= HOOH_max,
                                                   add_colorbar=False,
                                                   zorder=0, 
                                                   cmap=cmap)
        #
        cbar_a = fig.colorbar(im_a, ax=axs, shrink=0.7, location='right')
        #cbar_a.set_label('$log_{10}$ HOOH (mmol)')
        cbar_a.set_label('HOOH (mmol)')
        #
        # axs.annotate(str(time.values),
        #         xy=(0.01, 1.15), xycoords='axes fraction',
        #         horizontalalignment='center', verticalalignment='center',
        #         color='k',fontsize=10)
        #
        #plt.show()
        #
        PD_out=pathlib.Path('/lustre/isaac/proj/UTK0105/Hack_Session/gudb/verification/HOOH/detox_comp/movie/')
        #
        filename=str('Mov_HOOH_log_')+str(timeindex)+".png"
        filename=str('Mov_HOOH_lin_')+str(timeindex)+".png"
        file_out=PD_out.joinpath(filename)
        fig.savefig(file_out,dpi=100)
        #
        timeindex=timeindex+1
    ########################
    PD_out=pathlib.Path('/lustre/isaac/proj/UTK0105/Hack_Session/gudb/verification/HOOH/detox_comp/movie/')
    #filename=str('Mov_HOOH_log_maxmin')+".csv"
    filename=str('Mov_HOOH_lin_maxmin')+".csv"
    file_out=PD_out.joinpath(filename)
    df_xlist = pd.DataFrame(xlist) 
    df_xlist.to_csv(file_out) 
    
    #
    
    print('EXIT  :: EXIT', flush=True, file = sys.stdout)
     
      
     
  #
if __name__ == "__main__":
    #Initialize
    
    main()      
        