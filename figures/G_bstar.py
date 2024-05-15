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
import csv
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from scipy import stats
from matplotlib import gridspec
from glob import glob

#import figcom
def get_value_darwintxt(file,var_string):
    chk_next=False
    save_l_no=-9999
    save_line=str('-9999')
    with open(file, 'r') as fp:
        for l_no, line in enumerate(fp):
            # search string
            if chk_next:
                #print(line)
                if line[1].isspace():
                    #print('indent found')
                    save_line=str(save_line)+str(line.rstrip())
                    chk_next=True
                else:
                    #print('string found in a file')
                    #print('Line Number:', save_l_no)
                    #print('Line:', save_line)
                    # don't look for next lines
                    break
            elif var_string in line:
                #print(line)
                save_l_no=l_no
                save_line=line.rstrip()
                chk_next=True
    
    csv_line= save_line.replace('=',',').rstrip()
    csv_line= csv_line.replace(' ','')
    return(csv_line)



def main():
    #
   
    #PD_base=figcom.base
    #PD_out=figcom.out
    #PD_tag=figcom.tag
    dep_layer=1
    grp_out=pathlib.Path(str('/lustre/isaac/scratch/ecarr/runs/Bstar/figures'))
    filename=str('G')+str('_bstar_pro_'+str(dep_layer)+'.png')
    file_out=grp_out.joinpath(filename)
    #
    filename_csv=str('bstar_values_'+str(dep_layer)+'.csv')
    file_csv=grp_out.joinpath(filename_csv)
    #
    #find $(pwd) -maxdepth 1 -type d -not -path '*/\.*' | sort
    # 
    Dir_list=glob('/lustre/isaac/scratch/ecarr/runs/Bstar/run_*')
    # Dir_list=[\
    #     '/lustre/isaac/scratch/ecarr/runs/vDarwin/run_vDarwin_1d_0328_1396057',\
    #     '/lustre/isaac/scratch/ecarr/runs/vDarwin/run_vDarwin_1d_0328_1397802',\
    #     '/lustre/isaac/scratch/ecarr/runs/vDarwin/run_vDarwin_1d_0328_1404701',\
    #     '/lustre/isaac/scratch/ecarr/runs/vDarwin/run_vDarwin_1d_0328_1404764',\
    #     '/lustre/isaac/scratch/ecarr/runs/vDarwin/run_vDarwin_1d_0328_1404833',\
    #     '/lustre/isaac/scratch/ecarr/runs/vDarwin/run_vDarwin_1d_0328_1404847',
    #     ]
    # #
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
    graph_collect=[]
    df_gc = pd.DataFrame(\
                columns=[\
                'dirpath',\
                'HOOH_ABIOTIC_DETOX',\
                'HOOH_BACTERIA_PHI',\
                'HOOH_BACTERIA_STAR',\
                'HOOH_QUANTUM_YIELD',\
                'HOOH_GRP_KDAM',\
                'HOOH_GRP_PHI',\
                'meanHOOH',\
                'meanPRO',\
                'meanSYN',\
                'depth'\
                ])
    csv_file=grp_out.joinpath('bstar_data.csv')
    with open(csv_file, 'w', newline='') as csvfile:    
        dirpath=Dir_list[0]
        i=0    
        for dirpath in Dir_list:
            print(dirpath)
            var_string=str('dirpath')
            df_gc.loc[i, var_string] = str(dirpath)
            #
            PD_base=pathlib.Path(str(dirpath))
            #filename_traits=str('gud_traits.txt')
            filename_traits=str('gud_params.txt')
            file_traits=PD_base.joinpath(filename_traits)
            #        
            var_string=str('HOOH_ABIOTIC_DETOX')
            ret_line=get_value_darwintxt(file_traits,var_string)
            Var=ret_line.split(',')
            print(Var)
            df_gc.loc[i, var_string] = Var[1]
            #
            var_string=str('HOOH_BACTERIA_PHI')
            ret_line=get_value_darwintxt(file_traits,var_string)
            Var=ret_line.split(',')
            print(Var)
            df_gc.loc[i, var_string] = Var[1]
            #
            var_string=str('HOOH_BACTERIA_STAR')
            ret_line=get_value_darwintxt(file_traits,var_string)
            Var=ret_line.split(',')
            print(Var)
            df_gc.loc[i, var_string] = Var[1]
            #
            var_string=str('HOOH_QUANTUM_YIELD')
            ret_line=get_value_darwintxt(file_traits,var_string)
            Var=ret_line.split(',')
            print(Var)
            df_gc.loc[i, var_string] = Var[1]  
            #
            var_string=str('HOOH_GRP_KDAM')
            ret_line=get_value_darwintxt(file_traits,var_string)
            Var=ret_line.split(',')
            print(Var)
            df_gc.loc[i, var_string] = Var            
            #
            var_string=str('HOOH_GRP_PHI')
            ret_line=get_value_darwintxt(file_traits,var_string)
            Var=ret_line.split(',')
            print(Var)
            df_gc.loc[i, var_string] = Var
            #
            #
            #    
            PD_NC=pathlib.Path.joinpath(PD_base,'NC_trace')
            os.chdir(PD_NC)
            print(os.getcwd())
            NC_file=NC_file_list
            #
            ds_tracers=xr.open_mfdataset(NC_file,  combine='by_coords', parallel=False,chunks={'T':1,'Z':1})
            #
            # ds_tracers['DepHeight']=xr.zeros_like(ds_tracers['RC'])
            # for index in range(ds_tracers.Nr):
            #     d=2*(ds_tracers.RL[index].values-ds_tracers.RC[index].values)
            #     ds_tracers['DepHeight'][index]=d
            #     #print(index,d)
            # ds_tracers['DepHeight'].values
            #
            print('Limit year  :: start', flush=True, file = sys.stdout)
            #
            tempdate=ds_tracers.T.max()
            YEAR_index=int(2009)
            ds_tracers=ds_tracers.isel(T=(ds_tracers.T.dt.year == YEAR_index))            
            #limit depth and sum
            print('Limit depth  :: start', flush=True, file = sys.stdout)
            ds_tracers=ds_tracers.isel(Z=dep_layer)
            #
            mole_in_mmol=1/1000.0
            cubic_m_in_liter=1/1000.0
            SYN_cellconv=float(8e10)
            PRO_cellconv=float(2.3e11)
            #
            ds_tracers['hooh']=ds_tracers['TRAC07']*mole_in_mmol*cubic_m_in_liter
            ds_tracers['syn']=ds_tracers['TRAC23']*mole_in_mmol*cubic_m_in_liter
            ds_tracers['pro']=ds_tracers['TRAC24']*mole_in_mmol*cubic_m_in_liter
            #
            var_string=str('meanHOOH')
            ds_tracers[var_string]=ds_tracers['hooh'].mean(dim=["T","X","Y"])
            df_gc.loc[i, var_string] = float(ds_tracers[var_string].values)
            #
            var_string=str('meanSYN')
            ds_tracers[var_string] =ds_tracers['syn' ].mean(dim=["T","X","Y"])*SYN_cellconv
            df_gc.loc[i, var_string] = float(ds_tracers[var_string].values)
            #
            var_string=str('meanPRO')
            ds_tracers[var_string] =ds_tracers['pro' ].mean(dim=["T","X","Y"])* PRO_cellconv
            df_gc.loc[i, var_string] = float(ds_tracers[var_string].values)
            #
            var_string=str('depth')
            df_gc.loc[i, var_string] = float(ds_tracers.Z.values)
            #
            i=i+1    
        #
        df_gc=df_gc.sort_values(by=['HOOH_BACTERIA_STAR'])
        df_gc['dirpath']=df_gc['dirpath'].astype('string')
        df_gc['HOOH_ABIOTIC_DETOX']=df_gc['HOOH_ABIOTIC_DETOX'].astype(float)
        df_gc['HOOH_BACTERIA_PHI']=df_gc['HOOH_BACTERIA_PHI'].astype(float)
        df_gc['HOOH_BACTERIA_STAR']=df_gc['HOOH_BACTERIA_STAR'].astype(float)
        df_gc['HOOH_QUANTUM_YIELD']=df_gc['HOOH_QUANTUM_YIELD'].astype(float)
        #df_gc['HOOH_GRP_KDAM']=df_gc['HOOH_GRP_KDAM'].astype(float)
        #df_gc['HOOH_GRP_PHI']=df_gc['HOOH_GRP_PHI'].astype(float)
        df_gc['meanHOOH']=df_gc['meanHOOH'].astype(float)
        df_gc['meanPRO']=df_gc['meanPRO'].astype(float)
        df_gc['meanSYN']=df_gc['meanSYN'].astype(float)
        df_gc['depth']=df_gc['depth'].astype(float)
        df_gc.to_csv(csvfile)
    ###########
    #
    #  Create Graph
    #
    # graph_collect.append(info)
    # df_info=pd.DataFrame(\
    #     [[dirpath,Vabs[2],Vabs[4],meanSUS,meanINF,meanRES,meanZOO,meanVIR,meanVG]],\
    #     columns= df_gc.columns\
    #     )
    # df_gc=pd.concat([df_gc,df_info], ignore_index=True)
    #######################
    #
    # get no impact value
    NI_BASE=pathlib.Path(str('/lustre/isaac/scratch/ecarr/runs/Qx2_set/run_HOOH_1483768'))
    NI_NC=pathlib.Path.joinpath(NI_BASE,'NC_trace')
    os.chdir(NI_NC)
    print(os.getcwd())
    NC_file=NC_file_list
    ds_tracers=xr.open_mfdataset(NC_file,  combine='by_coords', parallel=False,chunks={'T':1,'Z':1})
    #
    #
    print('Limit year  :: start', flush=True, file = sys.stdout)
    YEAR_index=int(2009)
    ds_tracers=ds_tracers.isel(T=(ds_tracers.T.dt.year == YEAR_index))            
    #limit depth and sum
    print('Limit depth  :: start', flush=True, file = sys.stdout)
    ds_tracers=ds_tracers.isel(Z=dep_layer)
    #
    mole_in_mmol=1/1000.0
    cubic_m_in_liter=1/1000.0
    
    ds_tracers['hooh']=ds_tracers['TRAC07']*mole_in_mmol*cubic_m_in_liter
    ds_tracers['syn']=ds_tracers['TRAC23']*mole_in_mmol*cubic_m_in_liter
    ds_tracers['pro']=ds_tracers['TRAC24']*mole_in_mmol*cubic_m_in_liter
    #
    var_string=str('meanHOOH')
    ds_tracers[var_string]=ds_tracers['hooh'].mean(dim=["T","X","Y"])
    #df_gc.loc[i, var_string] = float(ds_tracers[var_string].values)
    #
    var_string=str('meanSYN')
    ds_tracers[var_string] =ds_tracers['syn' ].mean(dim=["T","X","Y"])*SYN_cellconv
    #df_gc.loc[i, var_string] = float(ds_tracers[var_string].values)
    #
    var_string=str('meanPRO')
    ds_tracers[var_string] =ds_tracers['pro' ].mean(dim=["T","X","Y"])*PRO_cellconv
    #df_gc.loc[i, var_string] = float(ds_tracers[var_string].values)
    ######
    #
    # get AS  abiotic and Syn phi detox
    AS_BASE=pathlib.Path(str('/lustre/isaac/scratch/ecarr/runs/Bstar/AS_run_HOOH_BAC_1523959/'))
    AS_NC=pathlib.Path.joinpath(AS_BASE,'NC_trace')
    os.chdir(AS_NC)
    print(os.getcwd())
    NC_file=NC_file_list
    ds_as=xr.open_mfdataset(NC_file,  combine='by_coords', parallel=False,chunks={'T':1,'Z':1})
    #
    #
    print('Limit year  :: start', flush=True, file = sys.stdout)
    YEAR_index=int(2009)
    ds_as=ds_as.isel(T=(ds_as.T.dt.year == YEAR_index))            
    #limit depth and sum
    print('Limit depth  :: start', flush=True, file = sys.stdout)
    ds_as=ds_as.isel(Z=dep_layer)
    #
    mole_in_mmol=1/1000.0
    cubic_m_in_liter=1/1000.0
    ds_as['hooh']=ds_as['TRAC07']*mole_in_mmol*cubic_m_in_liter
    ds_as['syn']=ds_as['TRAC23']*mole_in_mmol*cubic_m_in_liter
    ds_as['pro']=ds_as['TRAC24']*mole_in_mmol*cubic_m_in_liter
    #
    var_string=str('meanHOOH')
    ds_as[var_string]=ds_as['hooh'].mean(dim=["T","X","Y"])
    #df_gc.loc[i, var_string] = float(ds_tracers[var_string].values)
    #
    var_string=str('meanSYN')
    ds_as[var_string] =ds_as['syn' ].mean(dim=["T","X","Y"])*SYN_cellconv
    #df_gc.loc[i, var_string] = float(ds_tracers[var_string].values)
    #
    var_string=str('meanPRO')
    ds_as[var_string] =ds_as['pro' ].mean(dim=["T","X","Y"])*PRO_cellconv
    #df_gc.loc[i, var_string] = float(ds_tracers[var_string].values)
    #
    #########
    # Pro   
    fig,axs =  plt.subplots(1, 1, sharex='all') 
    plt.suptitle(str("Pro : Bstar Bacteria Detox at Depth: "+str(float(ds_tracers.Z.values))) ,fontsize=10) 
    # #, fontsize=20
   
    axs.hlines(y=float(ds_tracers['meanPRO'].values),\
        xmin=float(df_gc['HOOH_BACTERIA_STAR'].min()),\
        xmax=float(df_gc['HOOH_BACTERIA_STAR'].max()),\
        linewidth=2,\
        color='r',\
        label=str('Pro under No Impact')\
        )
    axs.hlines(y=float(ds_as['meanPRO'].values),\
        xmin=float(df_gc['HOOH_BACTERIA_STAR'].min()),\
        xmax=float(df_gc['HOOH_BACTERIA_STAR'].max()),\
        linewidth=2,\
        color='c',\
        label=str('Pro under Phi Syn only')\
         )
    axs.scatter(df_gc['HOOH_BACTERIA_STAR'].values,df_gc['meanPRO'].values,\
              #linestyle='solid',\
              color='green' ,\
              #linewidth=2,\
              marker='*',\
              s=20**2,\
              label=str('Phi_SYN & Phi_Bac*Bstar')
              )  
    # #
    # # axs.annotate(str('log10(mean(V/G)))=')+str(logmeanVG),
    # #         xy=(0.05, 0.95), xycoords='axes fraction',
    # #         horizontalalignment='center', verticalalignment='center',
    # #         color='k',fontsize=20)
    # #
    #axs.legend(loc='center left') 
    axs.legend(loc='best') 
    axs.set_xlabel('Bstar (cell/ml)')
    axs.set_ylabel('Pro mean[X,Y,T]  (cell/ml)')
    #
    filename=str('G')+str('_bstar_pro_'+str(dep_layer)+'.png')
    file_out=grp_out.joinpath(filename)
    fig.savefig(file_out,dpi=300)
    #
    #########
    # Syn   
    fig,axs =  plt.subplots(1, 1, sharex='all') 
    plt.suptitle(str("Syn : Bstar Bacteria Detox at Depth: "+str(float(ds_tracers.Z.values))) ,fontsize=10) 
    # #, fontsize=20
   
    axs.hlines(y=float(ds_tracers['meanSYN'].values),\
        xmin=float(df_gc['HOOH_BACTERIA_STAR'].min()),\
        xmax=float(df_gc['HOOH_BACTERIA_STAR'].max()),\
        linewidth=2,\
        color='r',\
        label=str('Syn under No Impact')\
        )
    axs.hlines(y=float(ds_as['meanSYN'].values),\
        xmin=float(df_gc['HOOH_BACTERIA_STAR'].min()),\
        xmax=float(df_gc['HOOH_BACTERIA_STAR'].max()),\
        linewidth=2,\
        color='c',\
        label=str('SYN under Phi Syn only')\
        )
    
    axs.scatter(df_gc['HOOH_BACTERIA_STAR'].values,df_gc['meanSYN'].values,\
              #linestyle='solid',\
              color='blue' ,\
              #linewidth=2,\
              marker='*',\
              s=20**2,\
              label=str('Phi_SYN & Phi_Bac*Bstar')
              )  
    # #
    # # axs.annotate(str('log10(mean(V/G)))=')+str(logmeanVG),
    # #         xy=(0.05, 0.95), xycoords='axes fraction',
    # #         horizontalalignment='center', verticalalignment='center',
    # #         color='k',fontsize=20)
    # #
    #axs.legend(loc='center left') 
    axs.legend(loc='best') 
    axs.set_xlabel('Bstar (cell/ml)')
    axs.set_ylabel('Syn mean[X,Y,T]  (cell/ml)')
    #
    filename=str('G')+str('_bstar_syn_'+str(dep_layer)+'.png')
    file_out=grp_out.joinpath(filename)
    fig.savefig(file_out,dpi=300)
    #
    #########
    # HOOH   
    fig,axs =  plt.subplots(1, 1, sharex='all') 
    plt.suptitle(str("HOOH : Bstar Bacteria Detox at Depth: "+str(float(ds_tracers.Z.values))) ,fontsize=10) 
    # #, fontsize=20
   
    # axs.hlines(y=float(ds_tracers['meanHOOH'].values),\
    #     xmin=float(df_gc['HOOH_BACTERIA_STAR'].min()),\
    #     xmax=float(df_gc['HOOH_BACTERIA_STAR'].max()),\
    #     linewidth=2,\
    #     color='r',\
    #     label=str('HOOH under No Impact')\
    #     )
    axs.hlines(y=float(ds_as['meanHOOH'].values),\
        xmin=float(df_gc['HOOH_BACTERIA_STAR'].min()),\
        xmax=float(df_gc['HOOH_BACTERIA_STAR'].max()),\
        linewidth=2,\
        color='c',\
        label=str('HOOH under Phi Syn only')\
        )
    
    axs.scatter(df_gc['HOOH_BACTERIA_STAR'].values,df_gc['meanHOOH'].values,\
              #linestyle='solid',\
              color='purple' ,\
              #linewidth=2,\
              marker='*',\
              s=20**2,\
              label=str('Phi_SYN & Phi_Bac*Bstar')
              )  
    # #
    # # axs.annotate(str('log10(mean(V/G)))=')+str(logmeanVG),
    # #         xy=(0.05, 0.95), xycoords='axes fraction',
    # #         horizontalalignment='center', verticalalignment='center',
    # #         color='k',fontsize=20)
    # #
    #axs.legend(loc='center left') 
    axs.legend(loc='best') 
    axs.set_xlabel('Bstar (cell/ml)')
    axs.set_ylabel('HOOH mean[X,Y,T]  (mol/l)')
    #
    filename=str('G')+str('_bstar_hooh_'+str(dep_layer)+'.png')
    file_out=grp_out.joinpath(filename)
    fig.savefig(file_out,dpi=300)
    #
    #
    print('EXIT  :: EXIT', flush=True, file = sys.stdout)
   
    
   
#
if __name__ == "__main__":
    #Initialize
    
    main()
