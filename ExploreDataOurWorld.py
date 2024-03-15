#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 11:28:35 2021

@author: llopez
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
# import plotly.graph_objects as go
# from ipywidgets import widgets, interactive
import seaborn as sns # Optional, will only affect the color of bars and the grid
# import matplotlib.font_manager
from matplotlib import style
style.use('ggplot') or plt.style.use('ggplot')

# Preprocesado y modelado
# ==============================================================================
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import scale
# Configuración warnings
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')
# import dash
# import dash_bootstrap_components as dbc
# import dash_core_components as dcc
# import dash_html_components as html
# from dash.dependencies import Input, Output, State
# import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from statsmodels.nonparametric.smoothers_lowess import lowess

# -------------------------------------------------------------------------------
#                   EXTRAPOLATION FUNCTION FOR NOCY DATA
# -------------------------------------------------------------------------------

def find_local_minima(x):
   from scipy.ndimage import rank_filter
   x_erode = rank_filter(x, -0, size=180,mode='mirror')
   return x_erode == x

def SplSplitFitIntervals2(Data_i,atribute):
    
    # from scipy import signal

    # df_orig = pd.read_csv('datasets/elecequip.csv', parse_dates=['date'], index_col='date')

    df_orig=Data_i
    # 1. Moving Average (Esto es opcional, creo)
    # df_ma = df_orig[atribute].rolling(3, center=True, closed='both').mean()
    # df_orig[atribute]=df_ma
    # df_orig[atribute] = df_orig[atribute].fillna(0)
    
    # 2. Loess Smoothing (5% and 15%)
    # df_loess_5 = pd.DataFrame(lowess(df_orig[atribute], np.arange(len(df_orig[atribute])), 
                                     # frac=0.05)[:, 1], index=df_orig.index, columns=[atribute])
    # df_loess_15 = pd.DataFrame(lowess(df_orig[atribute], np.arange(len(df_orig[atribute])), 
    #                                     frac=0.15)[:, 1], index=df_orig.index, columns=[atribute])
    df_loess_15 = pd.DataFrame(lowess(df_orig[atribute], np.arange(len(df_orig[atribute])), 
                                     frac=0.05)[:, 1], index=df_orig.index, columns=[atribute])
    
    # Plot
    # smooth_d2 =np.gradient(df_loess_15.values.reshape([len(df_loess_15)]))
    min_peakind = find_local_minima(df_loess_15.values.reshape([len(df_loess_15)]))

    # find switching points
    # infls = np.where(np.diff(np.sign(smooth_d2)))[0]
    infls = np.asarray(np.where(min_peakind==True),dtype=int)[0]
    # infls = infls.reshape([len(infls)])
    init=0
    # fin=len(s)
    pos=0
    if len(infls)>0:        
        for i in infls:
            length=i-init
            if length < 120:
                infls = np.delete(infls, pos)
                pos=pos-1
    
            pos=pos+1
            init=i
        
        # if infls[len(infls)-1]-len(df_loess_15)<=120:
        #     infls[len(infls)-1]=len(df_loess_15)
        # else:
        #     infls=np.append(infls,len(df_loess_15)-1)   
    # infls=np.append(infls,len(df_loess_15)-1)    
    if len(infls)==0:
        infls=np.append(infls,0)   
        infls=np.append(infls,len(df_loess_15)-1)   
        

    # fig, axes = plt.subplots(4,1, figsize=(7, 7), sharex=True, dpi=120)
    # df_orig[atribute].plot(ax=axes[0], color='k', title='Original Series')
    # for i, infl in enumerate(infls, 1):
    #     plt.axvline(x=infl, color='k', label=f'Inflection Point {i}')
    # df_loess_5[atribute].plot(ax=axes[1], title='Loess Smoothed 5%')
    # df_loess_15[atribute].plot(ax=axes[2], title='Loess Smoothed 15%')
    # df_ma.plot(ax=axes[3], title='Moving Average (3)')
    # fig.suptitle('How to Smoothen a Time Series', y=0.95, fontsize=14)
    # plt.show()
    return infls
        
# -------------------------------------------------------------------------------
#                   EXTRAPOLATION FUNCTION FOR NOCY DATA
# -------------------------------------------------------------------------------
def SplitFitIntervals(Data):
    Contries=Data['location'].unique()
    Dict = {"location":[],"intervals":[]};
    # atribute='New Cases'
    atribute='Active'
    for i in Contries:
        # x=Data[Data['location']==i][['New Cases']]
        x= Data[Data['location']==i]
        # x=Data[Data['location']==i][['Total Cases']]
        # x=Data[Data['location']==i][['Active']]
        intervals=SplSplitFitIntervals2(x,atribute)
        Dict['location'].append(i)
        Dict['intervals'].append(intervals)
    DF=pd.DataFrame.from_dict(Dict)
    return DF
    
# -------------------------------------------------------------------------------
#                   EXTRAPOLATION FUNCTION FOR NOCY DATA
# -------------------------------------------------------------------------------
# def GetIntervals(s):
#     # smooth
#     # Datai=Data[Data['location']=='Brazil']
#     # Cases=Datai['Active']
#     # s=Cases
#     s=s.values.astype(float)
#     # S=np.random.randn(1000,len(s))+s
#     s2=s.cumsum()
#     s2=s2/max(s2)
#     # s2=s
#     smooth = gaussian_filter1d(s2,120,mode='mirror',order=4,cval=7.0, truncate=10.0)
#     # smooth = gaussian_filter1d(s2,200,mode='mirror',order=3,cval=8.0, truncate=10.0)
#     # smooth = gaussian_filter1d(s2,110,mode='nearest',order=3,cval=0.0, truncate=10.0)
#     # smooth = gaussian_filter1d(s,90,mode='nearest',order=3,cval=0.0, truncate=10.0)
#     # smooth = savgol_filter(s2, int(len(s)/2), 3) # window size 51, polynomial order 3
#     smooth = (smooth-min(smooth))/max(smooth)


#     # compute second derivative
#     smooth_d2 = np.gradient(np.gradient(smooth))

#     # find switching points
#     infls = np.where(np.diff(np.sign(smooth_d2)))[0]
#     infls=np.insert(infls,0, 0)
#     init=0
#     # fin=len(s)
#     pos=0
#     for i in infls:
#         length=i-init
#         if length <180:
#             # infls[pos]=infls[pos]+ (120-length)
#             infls = np.delete(infls, pos)
#             pos=pos-1
#             # infls[pos]=infls[pos]+ 90
#             # infls = np.delete(infls, pos-1)
#             # infls[pos]=infls[pos]+ 90
#         pos=pos+1
#         init=i
#     infls=np.append(infls,len(s)-1)    
#     # if pos>0:
#     #     length=init-fin
#     #     if length <10:
#     #         infls = np.delete(infls, pos-1)
  
#     # # plot results
#     # plt.plot(s/max(s), label='New Cases')
#     # # plt.plot(-(s/max(s)), label='New Cases inverted')
#     # # plt.plot(s/max(s)-((s/max(s))), label='New Cases')
#     # # plt.plot(S.mean(axis=0), label='Noisy Data')
#     # # plt.plot(s2, label='Total Cases')
#     # plt.plot(smooth, label='Smoothed Cases')
#     # # plt.plot(smooth_d2 / np.max(smooth_d2), label='Second Derivative (scaled)')
#     # for i, infl in enumerate(infls, 1):
#     #     plt.axvline(x=infl, color='k', label=f'Inflection Point {i}')
#     # plt.legend(bbox_to_anchor=(1.55, 1.0))
#     # infls = np.unique(infls)
#     return infls
# -------------------------------------------------------------------------------
#                   EXTRAPOLATION FUNCTION FOR NOCY DATA
# -------------------------------------------------------------------------------
# def GetIntervals(s):
#     length=len(s)
#     days=180
#     chuncks=int(length/days)
#     infls=np.zeros(chuncks).astype(int)
#     for i in np.arange(1, chuncks-1):
#         infls[i]=infls[i-1]+days
#     return infls

# -------------------------------------------------------------------------------
#                   EXTRAPOLATION FUNCTION FOR NOCY DATA
# -------------------------------------------------------------------------------
def GetRecovered(c,d):
    c=c.reshape(len(c))
    d=d.reshape(len(d))
    r=np.zeros(len(d))
    t=np.arange(len(d))
    posd=d>0
    delay=(posd==False).sum()
    if delay <14:
        delay=14
    for i in t[delay:len(t)]:
        r[i]=c[i-delay]-d[i-delay]
        if r[i]== np.nan or r[i]<0:
            r[i]=0
    # locations = np.where(np.diff(s.reshape(len(s))) != 0)[0] + 1
    # result = np.split(s.reshape(len(s)), locations)
    # Max=0
    # x=0
    # for i in np.arange(len(result)):
    #     size=len(result[i])
    #     x+=len(result[i])
    #     if size>=Max:
    #         Max=size
    #         chunck=result[i]
    #         Posi=x-len(result[i])-1
    #         Posf=x
          
    # t = np.arange(len(s)).reshape(len(s))
    # f = interp1d(t[0:Posi], s[0:Posi],kind='linear', fill_value='extrapolate')
    # sn=f(t)
    # fig = plt.figure(facecolor='w')
    # ax = fig.add_subplot(111, axisbelow=True)
    # ax.plot(t,c, 'ob', alpha=0.5, lw=4, label='Reported')
    # ax.plot(t,d, 'ok', alpha=0.5, lw=4, label='Deaths')
    # ax.plot(t,r, 'oy', alpha=0.5, lw=4, label='Recovered')
    # ax.set_xlabel('Time /days')
    # ax.set_ylabel('%')
    # ax.set_ylim(bottom=0)
    # ax.yaxis.set_tick_params(length=0)
    # ax.xaxis.set_tick_params(length=0)
    # ax.set_title(State)
    # plt.xticks(rotation=45, ha='right')
    # plt.autoscale(enable=True, axis='x', tight=True)
    # legend = ax.legend()
    # legend.get_frame().set_alpha(0.5)
    # for spine in ('top', 'right', 'bottom', 'left'):
    #     ax.spines[spine].set_visible(True)
    # plt.show()
    
    
    return r.reshape(len(r),1)
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

def GetDataVariants():
    RawDataVariants=pd.read_csv('https://mendel3.bii.a-star.edu.sg/aa6c31f8-886b-481d-99f6-52caee98e4d3?raw=true')
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

def GetDataSets(Ploting,date,From,To):
    """
    This function don't take any argument as entry, just download from repositories online
    the data abput cases, deaths, recovred and vaccinated popoulation
    Return a dictionary with each counttry time series with the next information:
        Name of the country
       Edpidemic Data : 'Time','Active','Confirmed','Deaths','Recovered'
       Vaccination data 'Dates vaccination','Vaccines per millon'
    """
    
    RawData=pd.read_csv('https://github.com/owid/covid-19-data/blob/master/public/data/owid-covid-data.csv?raw=true',
        header=0)
    RawData.date=pd.to_datetime(RawData.date)
    RawData=RawData[RawData['date']>=From]
    DatRawDataa=RawData[RawData['date']<=To]
    RawData=RawData[RawData.date>=pd.to_datetime(date)]
    Atributes=RawData.columns
    Continents=RawData['continent'].unique()
    Continents=[x for x in Continents if str(x) != 'nan']
    ColumnsDF=['continent','location','Date','New Cases','Total Cases','Total Deaths',
               'Recovered','New Vaccinated','Vaccination with 1 dose','Fully vaccinated'
               ,'Booster vaccinated','Population','Population Density','Total Vaccinations']
    
    FullData=pd.DataFrame(columns=ColumnsDF)
    #----------- Creation of directories to save data-------------------------
    path0 = os.getcwd()
    path1='/Our World Data Dataframe'
    subpaths=list(['/Dynamics'])
    path=path0+path1
    try:
        os.mkdir(path)
    except OSError:
        print ("Directory %s already exist" % path)
    else:
        print ("Successfully created the directory %s " % path)
    for p in subpaths:
        subpath=path+p
        try:
            os.mkdir(subpath)
        except OSError:
            print ("Directory %s already exist" % subpath)
        else:
            print ("Successfully created the directory %s " % subpath)
    # ------------------------------------------------------------------------    
        
    for i in Continents:
        print('Continent: ' + i)
        Data=RawData[RawData['continent']==i]
        Countries=Data['location'].unique()
        # -------------------Create directory for continent j-----------------
        for p in subpaths:
            subpath=path+p+'/'+i
            try:
                os.mkdir(subpath)
            except OSError:
                print ("Directory %s already exist" % subpath)
            else:
                print ("Successfully created the directory %s " % subpath)
        # --------------------------------------------------------------------
        for j in Countries:
            print('Country: ' + j)
            Data_i=Data[Data['location']==j]
            if len(Data_i)>1:
                time=pd.to_datetime( Data_i['date'])
                TotalCases=Data_i['total_cases']
                Total_Deaths=Data_i['total_deaths']
                New_cases=Data_i['new_cases_smoothed']
                Population= Data_i['population']
                Density=Data_i['population_density']
                
                # Recovered=TotalCases-Total_Deaths-New_cases
                Recovered=pd.DataFrame(GetRecovered(Data_i['total_cases'].values,Data_i['total_deaths'].values)).squeeze()
                # -----------Filling the gaps of the vaccinated population----------
                New_vaccinations=Data_i['new_vaccinations_smoothed'].replace([0,np.nan], method='ffill')
                New_vaccinations=New_vaccinations.replace([np.nan], 0)
                
                Fully_vaccinated=Data_i['people_fully_vaccinated_per_hundred'].replace([0,np.nan], method='ffill')
                Fully_vaccinated=Fully_vaccinated.replace([np.nan], 0)
                
                One_dose_vaccination=Data_i['people_vaccinated_per_hundred'].replace([0,np.nan], method='ffill')
                One_dose_vaccination=One_dose_vaccination.replace([np.nan], 0)
                
                Booster_vaccination=Data_i['total_boosters_per_hundred'].replace([0,np.nan], method='ffill')
                Booster_vaccination=Booster_vaccination.replace([np.nan], 0)
                
                Density=Density.replace([np.nan], 0)
                
                Recovered=Recovered.replace([np.nan], 0)
                Total_Deaths=Total_Deaths.replace([np.nan], 0)
                New_cases=New_cases.replace([np.nan], 0)
                TotalCases=TotalCases.replace([np.nan], 0)
                Active = TotalCases.values-Recovered.values-Total_Deaths.values
                Active[Active <0] = 0
                # Active = TotalCases.items-Recovered.items-Total_Deaths.items
                # -----------Calculate real number of vaccinated population---------
                Booster_vaccination=Booster_vaccination*Data_i['population'].iloc[0]/100
                Fully_vaccinated=(Fully_vaccinated*Data_i['population'].iloc[0]/100)-Booster_vaccination
                One_dose_vaccination=(One_dose_vaccination*Data_i['population'].iloc[0]/100)-Fully_vaccinated-Booster_vaccination
                
                total_vaccinations=Data_i['total_vaccinations'].replace([0,np.nan], method='ffill')
                total_vaccinations=total_vaccinations.replace([np.nan], 0)
                
                # Booster_vaccination=Booster_vaccination
                # Fully_vaccinated=Fully_vaccinated-Booster_vaccination
                # One_dose_vaccination=One_dose_vaccination-Fully_vaccinated
                FigName='Our World in Data/Plots/Continents/'+i+'/'+j
                # -------------------Saving Data-----------------------------------
                DataCountry = {'Date': list(time),
                    'New Cases': list(New_cases),
                    'Total Cases': list(TotalCases), 
                    'Total Deaths': list(Total_Deaths), 
                    'Active':list(Active),  
                    'Recovered' : list(Recovered),
                    'New Vaccinated':list(New_vaccinations),
                    'Vaccination with 1 dose':list(One_dose_vaccination),
                    'Fully vaccinated':list(Fully_vaccinated),
                    'Booster vaccinated':list(Booster_vaccination) ,
                    'Population':list(Population) ,
                    'Population Density':list(Density),
                    'Total Vaccinations':list(total_vaccinations)
                    }
            
                DataCountry= pd.DataFrame(DataCountry)
                DataCountry.insert(0, 'continent', i)
                DataCountry.insert(1, 'location', j)
                # fig, axs = plt.subplots(2)
                # fig.suptitle('Vertically stacked subplots')
                # axs[0].plot(x, y)
                # axs[1].plot(x, -y)
                # infls=GetIntervals(DataCountry['New Cases']) 
                # infls=GetIntervals(DataCountry['Active'])
                # infls=SplSplitFitIntervals2(DataCountry,'New Cases')
                infls=SplSplitFitIntervals2(DataCountry,'Active')
                time=DataCountry['Date']
                # -------------------Plot Data for Country-------------------------
                if Ploting == True:
                    FigName='Our World Data Dataframe/Dynamics'+'/'+i+'/'+j
                    fig,axs = plt.subplots(3)
                    # ax = fig.add_subplot(111, axisbelow=True)
                    axs[0].plot(time,TotalCases, 'r', alpha=0.5, lw=4, label='Total Cases')
                    # axs[0].plot(time,Recovered, 'ob', alpha=0.5, lw=4, label='Recovered')
                    axs[0].plot(time,Total_Deaths, 'k', alpha=0.5, lw=4, label='Total Deaths')
                    for infl in infls:
                        axs[0].axvline(x=time[infl], color='k')
                    axs[0].legend(shadow=True, fancybox=True)
                    axs[0].legend(shadow=True, fancybox=True)
                    axs[0].set_title(j)
        
                    
                    axs[1].plot(time,New_cases, 'g', alpha=0.5, lw=4, label='New cases')
                    # axs[1].plot(time,Active, 'ob', alpha=0.5, lw=4, label='Active')
                    for infl in infls:
                        axs[1].axvline(x=time[infl], color='k')
                    axs[1].legend(shadow=True, fancybox=True)
                    
                    axs[2].plot(time,Active, 'b', alpha=0.5, lw=4, label='Active')
                    # axs[2].plot(time,New_vaccinations, '-b', alpha=0.5, lw=4, label='New Vaccinated')
                    # axs[2].plot(time,One_dose_vaccination, '-k', alpha=0.5, lw=4, label='1 Dose vaccination')
                    # axs[2].plot(time,Fully_vaccinated, '-r', alpha=0.5, lw=4, label='Fully vaccinated')
                    # axs[2].plot(time,Booster_vaccination, '-m', alpha=0.5, lw=4, label='Booster vaccinated')
                    axs[2].legend(shadow=True, fancybox=True)
                    
                    plt.show()
                    fig.savefig(FigName,dpi=600)
                    plt.close(fig)
                # -----------------generating final Data Frame--------------------
                FullData=FullData.append(DataCountry)
            
    FullData['Date']=pd.to_datetime(FullData['Date'])
    return FullData,RawData
# ------------------------------------------------------------------------------
#                  PLOTING THINGS FROM THE DATA SET
# ------------------------------------------------------------------------------
def Plots(RawData):
    """
    

    Parameters
    ----------
    RawData : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # Country = widgets.Dropdown(
    # options=['All'] + list(RawData['location'].unique()),
    # value='All',
    # description='Country:',
    # )

    # def plotit(area, start_year, end_year):
    #     """
    #     Filters and plot the dataframe as a stacked bar chart of count of Male versus Women
    
    #     Args:
    #     -----
    #         * area (str): the area to filter on, or "All" to display all Areas
    
    #         * start_year, end_year (int, as float): the start and ends years, inclusive
    
    #         Note: the dataframe to plot is globally defined here as `df`
    
    #     Returns:
    #     --------
    #         A matplotlib stacked bar chart
    
    #     """
    #     if start_year > end_year:
    #         print("You must select a start year that is prior to end year")
    #     else:
    #         df2 = RawData.copy()
    #         if area != 'All':
    #             df2 = df2[df2.Area == area]
    
    #         # Filter between min and max years (inclusive)
    #         df2 = df2[(df2.Year >= start_year) & (df2.Year <= end_year)]
    
    
    #         # Plot it (only if there's data to plot)
    #         if len(df2) > 0:
    #             df2.groupby(['Year', 'Sex']).sum()['Count'].unstack().plot(kind='bar', stacked=True, title="Area = {}".format(area))
    #             plt.show();
    #         else:
    #             print("No data to show for current selection")
# ------------------------------------------------------------------------------
#                  SELECT DATA FROM CONTINENT
# ------------------------------------------------------------------------------
def SelectContinent(Data,continent):
    DataC=Data[Data['continent']==continent]
    return DataC
# ------------------------------------------------------------------------------
#                  SELECT DATA FROM CONTINENT
# ------------------------------------------------------------------------------
def SelectCountry(Data,country):
    DataC=Data[Data['location']==country]
    return DataC

# ------------------------------------------------------------------------------
#                  PERFORM PCA
# ------------------------------------------------------------------------------
def PrincipalComp(Data,location):
    Data=Data._get_numeric_data()
    Data=Data.replace(np.nan, 0)
    Data.info()
    print('----------------------')
    print('Variables mean')
    print('----------------------')
    Data.mean(axis=0)
    print('-------------------------')
    print('Variables Variance')
    print('-------------------------')
    Data.var(axis=0)
    # Entrenamiento modelo PCA con escalado de los datos
    # ==============================================================================
    pca_pipe = make_pipeline(StandardScaler(), PCA())
    pca_pipe.fit(Data)
    
    # Se extrae el modelo entrenado del pipeline
    modelo_pca = pca_pipe.named_steps['pca']
    # Se combierte el array a dataframe para añadir nombres a los ejes.
    PCADataFrame=pd.DataFrame(
        data    = modelo_pca.components_,
        columns = Data.columns,
        index   = ['PC'+ str(i+1) for i in np.arange(len(Data.columns))]
    )
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 2))
    componentes = modelo_pca.components_
    plt.imshow(componentes.T, cmap='viridis', aspect='auto')
    plt.yticks(range(len(Data.columns)), Data.columns)
    plt.xticks(range(len(Data.columns)), np.arange(modelo_pca.n_components_) + 1)
    plt.title(location)
    plt.grid(False)
    
    plt.colorbar();
    # Porcentaje de varianza explicada por cada componente
    # ==============================================================================
    print('----------------------------------------------------')
    print('Porcentaje de varianza explicada por cada componente')
    print('----------------------------------------------------')
    print(modelo_pca.explained_variance_ratio_)
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
    ax.bar(
        x      = np.arange(modelo_pca.n_components_) + 1,
        height = modelo_pca.explained_variance_ratio_
    )
    
    for x, y in zip(np.arange(len(Data.columns)) + 1, modelo_pca.explained_variance_ratio_):
        label = round(y, 2)
        ax.annotate(
            label,
            (x,y),
            textcoords="offset points",
            xytext=(0,10),
            ha='center'
        )
    
    ax.set_xticks(np.arange(modelo_pca.n_components_) + 1)
    ax.set_ylim(0, 1.1)
    ax.set_title('Porcentaje de varianza explicada por cada componente: '+location)
    ax.set_xlabel('Componente principal')
    ax.set_ylabel('Por. varianza explicada');
    
    # Porcentaje de varianza explicada acumulada
    # ==============================================================================
    prop_varianza_acum = modelo_pca.explained_variance_ratio_.cumsum()
    print('------------------------------------------')
    print('Porcentaje de varianza explicada acumulada')
    print('------------------------------------------')
    print(prop_varianza_acum)
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
    ax.plot(
        np.arange(len(Data.columns)) + 1,
        prop_varianza_acum,
        marker = 'o'
    )
    
    for x, y in zip(np.arange(len(Data.columns)) + 1, prop_varianza_acum):
        label = round(y, 2)
        ax.annotate(
            label,
            (x,y),
            textcoords="offset points",
            xytext=(0,10),
            ha='center'
        )
        
    ax.set_ylim(0, 1.1)
    ax.set_xticks(np.arange(modelo_pca.n_components_) + 1)
    ax.set_title('Porcentaje de varianza explicada acumulada: '+location)
    ax.set_xlabel('Componente principal')
    ax.set_ylabel('Por. varianza acumulada');
    return PCADataFrame
# ------------------------------------------------------------------------------
#                 SEASONALITY
# ------------------------------------------------------------------------------
def seasonality(Data,year,location,atributes):
    Data['date']=pd.to_datetime(Data['date'])
    Data['Month']=Data['date'].dt.month
    Data['Year']=Data['date'].dt.year
    Data=Data[Data['Year']<=year[len(year)-1]]
    Data=Data[Data['Year']>=year[0]]
    fig, axes = plt.subplots(len(atributes), 1, figsize=(11, 10), sharex=True)
    for name, ax in zip(atributes,axes):
        sns.boxplot(data=Data, x='Month', y=name, ax=ax,showfliers = False)
        ax.set_ylabel('Number')
        ax.set_title(name + ' ' + location)
        # Remove the automatic x-axis label from all but the bottom subplot
        if ax != axes[-1]:
            ax.set_xlabel('')
# -----------------------------------------------------------------------------
#                 DIST PLOT
# -----------------------------------------------------------------------------
def Distribution(Data,year,atribute):
    Data['date']=pd.to_datetime(Data['date'])
    Data['Month']=Data['date'].dt.month
    Data['Year']=Data['date'].dt.year
    Data=Data[Data['Year']<=year[len(year)-1]]
    Data=Data[Data['Year']>=year[0]]
    
    
    fig, axes = plt.subplots(2, 1, figsize=(18, 10))

    sns.jointplot(data = Data,x = "location", y = atribute[0],ax=axes[0,0])
    sns.jointplot(data = Data,x = "location", y = atribute[1],ax=axes[1,0])
    # f = plt.figure(1)
    # for J in [JP1, JP2]:
    #     for A in J.fig.axes:
    #         f._axstack.add(f._make_key(A), A)
    # fig, axes = plt.subplots(len(atribute), 1, figsize=(11, 10), sharex=True)
    # for name, ax in zip(atribute,axes):
    #     sns.jointplot(data = Data,x = "location", y = name, ax=ax)
    #     ax.set_ylabel('Value')
    #     ax.set_title(Data['continent'].unique().item())
    #     # Remove the automatic x-axis label from all but the bottom subplot
    #     if ax != axes[-1]:
    #         ax.set_xlabel('')
    #subplots size adjustment

# -----------------------------------------------------------------------------
#                 DIST PLOT
# -----------------------------------------------------------------------------
def PlotCountryDynamics(Data,Country,From,To):
    Data=Data[Data['location']==Country]
    Data=Data[Data['Date']>=From]
    Data=Data[Data['Date']<=To]
    Data=Data.reset_index(drop=True)
    Data['Date']=pd.to_datetime(Data['Date'])    
    time=Data['Date']
    TotalCases=Data['Total Cases']
    Total_Deaths=Data['Total Deaths']
    New_cases= Data['New Cases']
    One_dose_vaccination=Data['Vaccination with 1 dose']
    Fully_vaccinated=Data['Fully vaccinated']
    Booster_vaccination=Data['Booster vaccinated']
    Total_Vaccination=Data['Total Vaccinations']
    N=Data['Population']
    Active=Data['Active']
    Recovered=Data['Recovered']
    Data['Active']=Data['Active']-(Data['Active'].mean())
    Data['Active']=Data['Active']/Data['Active'].max()
   
    infls=SplSplitFitIntervals2(Data,'Active') 
    # infls=GetIntervals(Data['Total Cases'])  
    # infls=GetIntervals(Data['New Cases']) 
    

    fig,axs = plt.subplots(2,facecolor='w',edgecolor='k')
    # fig = plt.subplots(2,figsize=(10, 6),constrained_layout=True)
    # axs = fig.subplots(111,facecolor='w', axisbelow=True)
    axs[0].plot(time,TotalCases, 'or', alpha=0.5, lw=4, label='Total Cases')
    axs[0].plot(time,Recovered, 'ob', alpha=0.5, lw=4, label='Recovered')
    # axs[0].plot(time,New_cases.cumsum(), '*b', alpha=0.5, lw=4, label='Total Cases control')
    axs[0].plot(time,Total_Deaths, 'ok', alpha=0.5, lw=4, label='Total Deaths')
    for infl in infls:
        # if infl>= len(time):
        axs[0].axvline(x=time[infl], color='k')
    axs[0].legend(shadow=True, fancybox=True)
    axs[0].legend(shadow=True, fancybox=True)
    axs[0].set_title(Country)

    
    axs[1].plot(time,New_cases, 'og', alpha=0.5, lw=4, label='New cases')
    axs[1].plot(time,Active, 'ob', alpha=0.5, lw=4, label='Active')
    for infl in infls:
        # if infl>= len(time):
        axs[1].axvline(x=time[infl], color='k')
    axs[1].legend(shadow=True, fancybox=True)
    
    # axs[2].plot(time,Active, 'ob', alpha=0.5, lw=4, label='Active')
    # axs[2].plot(time,N, '-b', alpha=1, lw=4, label='Population')
    # axs[2].plot(time,One_dose_vaccination, '-k', alpha=0.5, lw=4, label='1 Dose vaccination')
    # axs[2].plot(time,Fully_vaccinated, '-r', alpha=0.5, lw=4, label='Fully vaccinated')
    # axs[2].plot(time,Booster_vaccination, '-m', alpha=0.5, lw=4, label='Booster vaccinated')
    # axs[2].plot(time,One_dose_vaccination+Fully_vaccinated+Booster_vaccination, 'r', alpha=1, lw=4, label='Total Vaccinated')
    # axs[2].plot(time,Total_Vaccination, 'k', alpha=1, lw=4, label='Total Vaccination Real')

    # axs[2].legend(shadow=True, fancybox=False)
    
    plt.show()
    


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
From=pd.to_datetime('11-01-2021')
To=pd.to_datetime('12-01-2022')
Date='01-01-2019'

Data,RawData=GetDataSets(Ploting=False, date=Date,From=From,To=To)
Data=Data[Data['Date']>=pd.to_datetime(Date)]
RawData['date']=pd.to_datetime(RawData['date'])
RawData=RawData[RawData['date']>=pd.to_datetime(Date)]

path='Our World Data Dataframe/'

file='FilteredDataOmicron.csv'
Data.to_csv(path+file,sep=',')
# 
file='RawDataOmicron.csv'
RawData.to_csv(path+file,sep=',')

file='Intervalsomicron.csv'
Intervals=SplitFitIntervals(Data)
Intervals.to_csv(path+file,sep=',')
# RawData.info()
# RawData.shape
# Description=RawData.describe()

Atributes=RawData.columns
# Continents=RawData.continent.unique()
Columns=['iso_code', 'continent', 'location', 'date', 'total_cases', 'new_cases',
       'total_deaths', 'new_deaths',
       'total_cases_per_million',
       'new_cases_per_million', 
       'total_deaths_per_million', 'new_deaths_per_million',
       'reproduction_rate', 'icu_patients',
       'icu_patients_per_million', 'hosp_patients',
       'hosp_patients_per_million', 'weekly_icu_admissions',
       'weekly_icu_admissions_per_million', 'weekly_hosp_admissions',
       'weekly_hosp_admissions_per_million','total_vaccinations',
       'people_vaccinated', 'people_fully_vaccinated', 'total_boosters',
       'new_vaccinations', 
       'total_vaccinations_per_hundred', 'people_vaccinated_per_hundred',
       'people_fully_vaccinated_per_hundred', 'total_boosters_per_hundred',
       'stringency_index',
       'population', 'population_density', 'median_age', 'aged_65_older',
       'aged_70_older', 'gdp_per_capita', 'extreme_poverty',
       'cardiovasc_death_rate', 'diabetes_prevalence', 'female_smokers',
       'male_smokers', 'handwashing_facilities', 'hospital_beds_per_thousand',
       'life_expectancy', 'human_development_index',
       'excess_mortality_cumulative_absolute', 'excess_mortality_cumulative',
       'excess_mortality', 'excess_mortality_cumulative_per_million']
DataSubSet=RawData[np.intersect1d(RawData.columns, Columns)]

Continents=list(RawData['continent'].unique())
Continents= [x for x in Continents if str(x) != 'nan']
# Continents=Continents[np.isnan(Continents)]

# Continent='Asia'


# RawData[['total_cases_per_million']].plot.hist(bins = 100, title = 'total cases per million')

# RawData[['median_age']].plot.hist(bins = 20, title = 'median age')

# plt.figure(figsize= (10,10), dpi=100)
# sns.heatmap(RawData.corr())

# DataContinent=SelectContinent(Data, 'Europe')
# DataContinent=SelectContinent(RawData, 'Europe')
# PCA_Data=PrincipalComp(DataSubSet)

years=[2020,2021,2022]
# years=[2020]

Atr=['new_cases', 'new_deaths', 'hosp_patients']
Atr=['new_cases', 'new_deaths']
Atr2=['median_age', 'gdp_per_capita']
Atr2=['median_age']
# sns.jointplot(data=DataSubSet, x="reproduction_rate", y="median_age", hue="continent")
# sns.jointplot(data=DataSubSet, x="reproduction_rate", y="gdp_per_capita", hue="continent")
path='/home/leonardo/Imágenes/DataNew/'
# # sns.heatmap(DataSubSet.corr())
for Continent in Continents:
    DataContinent=SelectContinent(DataSubSet, Continent)
    # seasonality(DataContinent,years,Continent,Atr)
    # plt.savefig(path+Continent+"Seasonality.png") 
    # 
    # Distribution(DataContinent,years,Atr2)
    # sns.jointplot(x = "location", y = 'median_age', data = DataContinent)
    # PCA_Data=PrincipalComp(DataContinent,Continent)
    # plt.figure(figsize = (15,8))
    # kwargs={'rotation':45}
    # sns.jointplot(data = DataContinent,x = "location", y = 'median_age')
    # locs, labels = plt.xticks()
    # plt.setp(labels, rotation=45)
    # ax.set_xticklabels(ax.get_xticklabels(),rotation = 30)
    # joint_plot = sns.jointplot(data = DataContinent,x = "location", y = 'gdp_per_capita')
    # fig = joint_plot.get_figure()
    # sns.jointplot(data = DataContinent,x = "location", y = 'gdp_per_capita')
    # fig = sns.jointplot(data=DataContinent, x="reproduction_rate", y="median_age", hue="continent")
    # fig.savefig(path+Continent+"R0VsAge.png") 
    # # fig.close()
    # fig = sns.jointplot(data=DataContinent, x="reproduction_rate", y="gdp_per_capita", hue="continent")
    # fig.savefig(path+Continent+"R0VsGDP.png") 
    # # fig.close()
    # plt.figure(figsize =(20,30))
    # ax = plt.axes()
    # sns.heatmap(DataContinent.corr())
    # ax.set_title(Continent)
    # plt.xticks(rotation=90)
    # plt.yticks(rotation=25)
    # plt.show()
    # fig = sns.displot(data=DataContinent, x="reproduction_rate", y="gdp_per_capita", kind="kde")
    # fig.savefig(path+Continent+"R0VsGDPDist.png") 
    # fig = sns.displot(data=DataContinent, x="reproduction_rate", y="median_age", kind="kde")
    # fig.savefig(path+Continent+"R0VsAgeDist.png") 
    # fig.close()

# # sns.heatmap(DataContinent.corr())
# # fig, axes = plt.subplots(len(Atr2), 1, figsize=(11, 10), sharex=True)
# # for name, ax in zip(Atr2,axes):
# #     print(axes)


# from scipy.stats import norm
# mean=Data['Population Density'].mean()
# std=Data['Population Density'].std()
# x=Data['Population Density'].values
# print(norm.cdf(x, mean, std))

