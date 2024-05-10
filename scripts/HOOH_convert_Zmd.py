# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 16:32:21 2022

@author: eric
"""

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
import matplotlib.colors as colors
import matplotlib.cbook as cbook
from matplotlib import cm

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

    PD_NC=pathlib.Path('./')
    os.chdir(PD_NC)
    #
    ds_grid=xr.open_mfdataset('grid.v4c.nc',  combine='by_coords', parallel=False,chunks={'T':1,'Z':1})
    depth_values=np.array(ds_grid.Z.values)
    pre='3d'
    post='Z3d'


    files=[\
        '.PP.Primary.v4c.nc',
        '.TRAC01.DIC.v4c.nc',\
        '.TRAC02.NH4.v4c.nc',\
        '.TRAC03.NO2.v4c.nc',\
        '.TRAC04.NO3.v4c.nc',\
        '.TRAC05.PO4.v4c.nc',\
        '.TRAC06.SiO2.v4c.nc',\
        '.TRAC07.HOOH.v4c.nc',\
        '.TRAC08.FeT.v4c.nc',\
        '.TRAC09.DOC.v4c.nc',\
        '.TRAC10.DON.v4c.nc',\
        '.TRAC11.DOP.v4c.nc',\
        '.TRAC12.DOFe.v4c.nc',\
        '.TRAC13.POC.v4c.nc',\
        '.TRAC14.PON.v4c.nc',\
        '.TRAC15.POP.v4c.nc',\
        '.TRAC16.POSi.v4c.nc',\
        '.TRAC17.POFe.v4c.nc',\
        '.TRAC18.PIC.v4c.nc',\
        '.TRAC19.ALK.v4c.nc',\
        '.TRAC20.O2.v4c.nc',\
        '.TRAC21.Diatom01.v4c.nc',\
        '.TRAC22.Euk01.v4c.nc',\
        '.TRAC23.Syn01.v4c.nc',\
        '.TRAC24.Pro01.v4c.nc',\
        '.TRAC25.Tricho01.v4c.nc',\
        '.TRAC26.Cocco01.v4c.nc',\
        '.TRAC27.Zoo01.v4c.nc',\
        '.TRAC28.Zoo02.v4c.nc',\
        '.TRAC29.Zoo03.v4c.nc',\
        '.TRAC30.Zoo04.v4c.nc',\
        '.TRAC31.Chl1.v4c.nc',\
        '.TRAC32.Chl2.v4c.nc',\
        '.TRAC33.Chl3.v4c.nc',\
        '.TRAC34.Chl4.v4c.nc',\
        '.TRAC35.Chl5.v4c.nc'\
        ]
    
    for F in files:
        print(F,PD_NC)
        I_name=pre+F
        print(I_name)
        O_name=post+F
        print(O_name)
        ds_tracers=xr.open_mfdataset(I_name,  combine='by_coords', parallel=False,chunks={'T':1,'Z':1})
        ds_tracersZ=ds_tracers.rename({str('Zmd000023'):str('Z')})
        ds_tracersZ=ds_tracersZ.assign_coords(Z=depth_values)
        ds_tracersZ.to_netcdf(path=O_name)
        ds_tracers.close()
        ds_tracersZ.close()
    ########################
    #
    
    #
    print('EXIT  :: EXIT', flush=True, file = sys.stdout)
   
    
   
#
if __name__ == "__main__":
    #Initialize
    
    main()
