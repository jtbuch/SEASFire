import numpy as np
import random
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats, interpolate
from scipy.optimize import minimize
from scipy.special import gamma
from sklearn.linear_model import LinearRegression
from math import factorial
import itertools
from copy import deepcopy

import netCDF4 # module that reads in .nc files (built on top of HDF5 format)
import pandas as pd
import geopandas as gpd
from geopandas.tools import sjoin
import xarray as xr
import rioxarray
import xesmf as xe # for regridding
import pickle # for saving and loading models
from pathlib import Path

# Date and time related libraries
from dateutil.relativedelta import relativedelta
from calendar import monthrange
import datetime

from shapely.geometry import Point, mapping
from shapely.geometry.polygon import Polygon
from pyproj import CRS, Transformer # for transforming projected coordinates to elliptical coordinates
import tensorflow as tf
import tensorflow_probability as tfp
tfd= tfp.distributions

#self-libraries
from fire_utils import *
from ml_utils import *

DATADIR= '../data/seasonal'

def clim_xarr_init(clim_df, input_var_arr, scaling_flag, tstart_mon, trend_mons, start_mon, end_mon, xarr_end_date= '2023-05-01'):
    
    '''
    Function to initialize clim_xarr for climatology and trend calculations

    clim_df: dataframe containing predictor variables
    input_var_arr: array of input predictors
    scaling_flag: type of normalization to be applied to input predictors
    tot_months: total number of months in the dataset
    tstart_mon: starting month of the trend period
    trend_mons: number of months in the trend period
    start_mon: starting month of the seasonal cycle period
    end_mon: ending month of the seasonal cycle period
    xarr_end_date: end date of the xarray
    
    # input_var_arr --> array of input predictors 
    # scaling_flag --> type of normalization to be applied to input predictors
    '''

    tot_months= len(pd.date_range(start='1952-01-01', end= xarr_end_date, freq='MS'))
    clim_xarr= xarray.Dataset(
                data_vars= dict(
                    Tmax= (["time", "Y", "X"], clim_df['Tmax'].values.reshape(tot_months, 208, 155)),
                    Solar= (["time", "Y", "X"], clim_df['Solar'].values.reshape(tot_months, 208, 155)),
                    VPD= (["time", "Y", "X"], clim_df['VPD'].values.reshape(tot_months, 208, 155)),
                    Tmin= (["time", "Y", "X"], clim_df['Tmin'].values.reshape(tot_months, 208, 155)),
                    Prec= (["time", "Y", "X"], clim_df['Prec'].values.reshape(tot_months, 208, 155)),
                    RH= (["time", "Y", "X"], clim_df['RH'].values.reshape(tot_months, 208, 155)),
                    SM_0_100cm= (["time", "Y", "X"], clim_df['SM_0_100cm'].values.reshape(tot_months, 208, 155)),
                    PDSI= (["time", "Y", "X"], clim_df['PDSI'].values.reshape(tot_months, 208, 155)),
                    FFWI_max7= (["time", "Y", "X"], clim_df['FFWI_max7'].values.reshape(tot_months, 208, 155)),
                    CAPE= (["time", "Y", "X"], clim_df['CAPE'].values.reshape(tot_months, 208, 155)),),
                coords=dict(
                    X=(["X"], np.linspace(0, 154, 155, dtype= np.int64)),
                    Y=(["Y"], np.linspace(0, 207, 208, dtype= np.int64)),
                    time= (["time"], pd.date_range(start='1952-01-01', end= xarr_end_date, freq='MS')),),) #np.linspace(0, tot_months- 1, tot_months, dtype= np.int64) 
    
    for input_var in input_var_arr:
        if scaling_flag == 'trend':
            result = clim_xarr[input_var][tstart_mon:trend_mons, :, :].polyfit(dim = "time", deg = 1)
            trend= result.polyfit_coefficients.sel(degree= 1).values
            intercept= result.polyfit_coefficients.sel(degree= 0).values
            date_ns_arr= np.array(clim_xarr.time - np.datetime64('1952-01-01'))/np.timedelta64(1, 'ns')

            # multiply trend and intercept with time to get a (857, 208, 155) dimension array
            trend= np.kron(trend[np.newaxis, :, :], date_ns_arr[:, np.newaxis, np.newaxis])
            intercept= np.tile(intercept, (tot_months, 1, 1))
            clim_xarr[input_var + '_trend']= xarray.DataArray(trend + intercept, dims=('time', 'Y', 'X'), coords={'time': clim_xarr.time.values, \
                                                                                                                'Y': clim_xarr.Y.values, 'X': clim_xarr.X.values})
            detrended_arr= clim_xarr[input_var] - clim_xarr[input_var + '_trend']
            clim_xarr[input_var]= detrended_arr/detrended_arr.std(dim= 'time')
            clim_xarr[input_var + '_std']= xarray.DataArray(np.tile(detrended_arr.std(dim= 'time'), (tot_months, 1, 1)), dims=('time', 'Y', 'X'), \
                                                                                                        coords={'time': clim_xarr.time.values, 'Y': clim_xarr.Y.values, 'X': clim_xarr.X.values})
        elif scaling_flag == 'trend_w_seasonal_cycle':
            clim_xarr= clim_xarr.sel(time= (clim_xarr.time.dt.month >= start_mon) & (clim_xarr.time.dt.month <= end_mon))
            target_mon_arr= np.arange(start_mon, end_mon + 1, 1)
            ivar_mon_groups= clim_xarr[input_var][tstart_mon:trend_mons, :, :].groupby('time.month')
            tot_mon_groups= clim_xarr[input_var].groupby('time.month')
            trend_arr= []
            intercept_arr= []
            #detrended_std_xarr= []

            for ivar in target_mon_arr: #ivar_mon_groups.groups.keys():
                result= ivar_mon_groups[ivar].polyfit(dim = "time", deg = 1)
                trend_arr.append(result.polyfit_coefficients.sel(degree= 1).values) 
                intercept_arr.append(result.polyfit_coefficients.sel(degree= 0).values)

                date_ns_arr= np.array((tot_mon_groups[target_mon_arr[0]].time - np.datetime64('1952-01-01'))/np.timedelta64(1, 'ns'))
                trend_xarr= xarray.DataArray(data= np.kron(np.array(trend_arr)[0, :, :], date_ns_arr[:, np.newaxis, np.newaxis]), dims= ['time', 'Y', 'X'], \
                                                                                                        coords= dict(time= tot_mon_groups[target_mon_arr[0]].time.values, Y= clim_xarr.Y.values, X= clim_xarr.X.values))
                intercept_xarr= xarray.DataArray(data= np.kron(np.array(intercept_arr)[0, :, :], np.ones((len(tot_mon_groups[target_mon_arr[0]].time), 1, 1))), dims= ['time', 'Y', 'X'], \
                                                                                                        coords= dict(time= tot_mon_groups[target_mon_arr[0]].time.values, Y= clim_xarr.Y.values, X= clim_xarr.X.values))
                #detrended_std_xarr= xarray.DataArray(np.kron((tot_mon_groups[1] - trend_xarr - intercept_xarr).std(dim= 'time'), np.ones((len(tot_mon_groups[1].time), 1, 1))), dims= ['time', 'Y', 'X'], \
                #                                                                                        coords= dict(time= tot_mon_groups[1].time.values, Y= clim_xarr.Y.values, X= clim_xarr.X.values))
                                                                                            
            for i in range(len(target_mon_arr)- 1):
                date_ns_arr= np.array((tot_mon_groups[target_mon_arr[i+1]].time - np.datetime64('1952-01-01'))/np.timedelta64(1, 'ns'))
                tmptrend_xarr= xarray.DataArray(np.kron(np.array(trend_arr)[i, :, :], date_ns_arr[:, np.newaxis, np.newaxis]), dims= ['time', 'Y', 'X'], \
                                                                                        coords= dict(time= tot_mon_groups[target_mon_arr[i+1]].time.values, Y= clim_xarr.Y.values, X= clim_xarr.X.values))
                tmpintercept_xarr= xarray.DataArray(np.kron(np.array(intercept_arr)[i, :, :], np.ones((len(tot_mon_groups[target_mon_arr[i+1]].time), 1, 1))), dims= ['time', 'Y', 'X'], \
                                                                                        coords= dict(time= tot_mon_groups[target_mon_arr[i+1]].time.values, Y= clim_xarr.Y.values, X= clim_xarr.X.values))
                tmpdetrended_std_xarr= xarray.DataArray(np.kron((tot_mon_groups[target_mon_arr[i+1]] - tmptrend_xarr - tmpintercept_xarr).std(dim= 'time'), np.ones((len(tot_mon_groups[target_mon_arr[i+1]].time), 1, 1))), dims= ['time', 'Y', 'X'], \
                                                                                        coords= dict(time= tot_mon_groups[target_mon_arr[i+1]].time.values, Y= clim_xarr.Y.values, X= clim_xarr.X.values))
                trend_xarr= xarray.concat([trend_xarr, tmptrend_xarr], dim= 'time')
                intercept_xarr= xarray.concat([intercept_xarr, tmpintercept_xarr], dim= 'time')
                #detrended_std_xarr= xarray.concat([detrended_std_xarr, tmpdetrended_std_xarr], dim= 'time')

                trend_xarr= trend_xarr.sortby('time')
                intercept_xarr= intercept_xarr.sortby('time')
                #detrended_std_xarr= detrended_std_xarr.sortby('time')

            clim_xarr[input_var + '_trend']= trend_xarr + intercept_xarr
            detrended_std_xarr= (clim_xarr[input_var] - clim_xarr[input_var + '_trend']).std(dim= 'time') 
            clim_xarr[input_var + '_std']= xarray.DataArray(np.tile(detrended_std_xarr, (len(clim_xarr.time.values), 1, 1)), dims=('time', 'Y', 'X'), \
                                                                                    coords={'time': clim_xarr.time.values, 'Y': clim_xarr.Y.values, 'X': clim_xarr.X.values})
            #detrended_std_xarr= xarray.where(detrended_std_xarr == 0, 0.001, detrended_std_xarr)
            clim_xarr[input_var]= (clim_xarr[input_var] - clim_xarr[input_var + '_trend'])/detrended_std_xarr 
            
        elif scaling_flag == 'normalized':
            clim_xarr[input_var + '_mean']= xarray.DataArray(np.tile(clim_xarr[input_var][:trend_mons, :, :].mean(dim= 'time'), (tot_months, 1, 1)), dims=('time', 'Y', 'X'), \
                                                                                        coords={'time': clim_xarr.time.values, 'Y': clim_xarr.Y.values, 'X': clim_xarr.X.values})
            clim_xarr[input_var + '_std']= xarray.DataArray(np.tile(clim_xarr[input_var][:trend_mons, :, :].std(dim= 'time'), (tot_months, 1, 1)), dims=('time', 'Y', 'X'), \
                                                                                        coords={'time': clim_xarr.time.values, 'Y': clim_xarr.Y.values, 'X': clim_xarr.X.values})
            clim_xarr[input_var]= (clim_xarr[input_var] - clim_xarr[input_var][:trend_mons, :, :].mean(dim= 'time'))/clim_xarr[input_var][:trend_mons, :, :].std(dim= 'time')
        elif scaling_flag == 'minmax':
            clim_xarr[input_var + '_min']= xarray.DataArray(np.tile(clim_xarr[input_var].min(axis= 0), (tot_months, 1, 1)), dims=('time', 'Y', 'X'), \
                                                                                        coords={'time': clim_xarr.time.values, 'Y': clim_xarr.Y.values, 'X': clim_xarr.X.values})
            clim_xarr[input_var + '_max']= xarray.DataArray(np.tile(clim_xarr[input_var].max(axis= 0), (tot_months, 1, 1)), dims=('time', 'Y', 'X'), \
                                                                                        coords={'time': clim_xarr.time.values, 'Y': clim_xarr.Y.values, 'X': clim_xarr.X.values})
            clim_xarr[input_var]= (clim_xarr[input_var] - clim_xarr[input_var].min(axis= 0))/(clim_xarr[input_var].max(axis= 0) - clim_xarr[input_var].min(axis= 0))
        elif scaling_flag == 'hybrid':
            if input_var == 'Prec':
                clim_xarr[input_var + '_min']= xarray.DataArray(np.tile(clim_xarr[input_var].min(axis= 0), (tot_months, 1, 1)), dims=('time', 'Y', 'X'), \
                                                                                        coords={'time': clim_xarr.time.values, 'Y': clim_xarr.Y.values, 'X': clim_xarr.X.values})
                clim_xarr[input_var + '_max']= xarray.DataArray(np.tile(clim_xarr[input_var].max(axis= 0), (tot_months, 1, 1)), dims=('time', 'Y', 'X'), \
                                                                                        coords={'time': clim_xarr.time.values, 'Y': clim_xarr.Y.values, 'X': clim_xarr.X.values})
                clim_xarr[input_var]= (clim_xarr[input_var] - clim_xarr[input_var].min(axis= 0))/(clim_xarr[input_var].max(axis= 0) - clim_xarr[input_var].min(axis= 0))
            else:
                clim_xarr[input_var + '_mean']= xarray.DataArray(np.tile(clim_xarr[input_var][:trend_mons, :, :].mean(dim= 'time'), (tot_months, 1, 1)), dims=('time', 'Y', 'X'), \
                                                                                        coords={'time': clim_xarr.time.values, 'Y': clim_xarr.Y.values, 'X': clim_xarr.X.values})
                clim_xarr[input_var + '_std']= xarray.DataArray(np.tile(clim_xarr[input_var][:trend_mons, :, :].std(dim= 'time'), (tot_months, 1, 1)), dims=('time', 'Y', 'X'), \
                                                                                        coords={'time': clim_xarr.time.values, 'Y': clim_xarr.Y.values, 'X': clim_xarr.X.values})
                clim_xarr[input_var]= (clim_xarr[input_var] - clim_xarr[input_var][:trend_mons, :, :].mean(dim= 'time'))/clim_xarr[input_var][:trend_mons, :, :].std(dim= 'time')
    
    return clim_xarr


def ds_latlon_subset(ds,area,latname='latitude',lonname='longitude'):
    """
    Function to subset a dataset based on a lat/lon bounding box ("borrowed" from C3S tutorial for seasonal forecasting)
    """
    lon1 = area[1] % 360
    lon2 = area[3] % 360
    if lon2 >= lon1:
        masklon = ( (ds[lonname]<=lon2) & (ds[lonname]>=lon1) ) 
    else:
        masklon = ( (ds[lonname]<=lon2) | (ds[lonname]>=lon1) ) 
        
    mask = ((ds[latname]<=area[0]) & (ds[latname]>=area[2])) * masklon
    dsout = ds.where(mask,drop=True)
    
    if lon2 < lon1:
        dsout[lonname] = (dsout[lonname] + 180) % 360 - 180
        dsout = dsout.sortby(dsout[lonname])        
    
    return dsout

def vapor_pressure(temp):
    return 6.0178*np.exp((17.629*(temp - 273.15)/(237.3 + (temp - 273.15)))) #actual vapor pressure in hPa (1 mb = 1 hPa)

def seas5_monthly_anomaly_func(pred_var, system= None, fyear= 2021, init_month= 5, area_flag= True, subarea= None, regrid_flag= True, regrid_scheme= 'bilinear', dsout= None, anom_type= 'raw'):
    
    """
    Function to calculate monthly anomaly for a given predictor variable
    
    pred_var: predictor variable
    system: hindcast system (None or 51)
    fyear: forecast year
    init_month: initialisation month
    area_flag: whether to use subset of global data (True or False)
    regrid_flag: whether to regrid data to coarser/finer resolution (True or False)
    regrid_scheme: regridding scheme (bilinear or conservative)
    dsout: output grid for regridding
    anom_type: type of anomaly (raw or standardized)
    """
    
    if pred_var == 'VPD':
        if system == None:
            ds_hindcast_d2m= xr.open_dataset(f'{DATADIR}/ecmwf_seas5_1993-2016_amj_hindcast_monthly_Tdew.grib', engine='cfgrib', backend_kwargs=dict(time_dims=('forecastMonth', 'time')))
            ds_hindcast_tmax= xr.open_dataset(f'{DATADIR}/ecmwf_seas5_1993-2016_amj_hindcast_monthly_Tmax.grib', engine='cfgrib', backend_kwargs=dict(time_dims=('forecastMonth', 'time')))
            ds_hindcast_tmin= xr.open_dataset(f'{DATADIR}/ecmwf_seas5_1993-2016_amj_hindcast_monthly_Tmin.grib', engine='cfgrib', backend_kwargs=dict(time_dims=('forecastMonth', 'time')))
            seas5_forecast_d2m= xr.open_dataset(f'{DATADIR}/ecmwf_seas5_%s'%fyear + '_amj_forecast_monthly_Tdew.grib', engine='cfgrib', 
                                        backend_kwargs=dict(time_dims=('forecastMonth', 'time')))
            seas5_forecast_tmax= xr.open_dataset(f'{DATADIR}/ecmwf_seas5_%s'%fyear + '_amj_forecast_monthly_Tmax.grib', engine='cfgrib', 
                                        backend_kwargs=dict(time_dims=('forecastMonth', 'time')))
            seas5_forecast_tmin= xr.open_dataset(f'{DATADIR}/ecmwf_seas5_%s'%fyear + '_amj_forecast_monthly_Tmin.grib', engine='cfgrib', 
                                        backend_kwargs=dict(time_dims=('forecastMonth', 'time')))
        else:
            ds_hindcast_d2m= xr.open_dataset(f'{DATADIR}/ecmwf_seas5_51_1993-2016_amj_hindcast_monthly_Tdew.grib', engine='cfgrib', backend_kwargs=dict(time_dims=('forecastMonth', 'time')))
            ds_hindcast_tmax= xr.open_dataset(f'{DATADIR}/ecmwf_seas5_51_1993-2016_amj_hindcast_monthly_Tmax.grib', engine='cfgrib', backend_kwargs=dict(time_dims=('forecastMonth', 'time')))
            ds_hindcast_tmin= xr.open_dataset(f'{DATADIR}/ecmwf_seas5_51_1993-2016_amj_hindcast_monthly_Tmin.grib', engine='cfgrib', backend_kwargs=dict(time_dims=('forecastMonth', 'time')))
            seas5_forecast_d2m= xr.open_dataset(f'{DATADIR}/ecmwf_seas5_51_%s'%fyear + '_amj_forecast_monthly_Tdew.grib', engine='cfgrib', 
                                        backend_kwargs=dict(time_dims=('forecastMonth', 'time')))
            seas5_forecast_tmax= xr.open_dataset(f'{DATADIR}/ecmwf_seas5_51_%s'%fyear + '_amj_forecast_monthly_Tmax.grib', engine='cfgrib', 
                                        backend_kwargs=dict(time_dims=('forecastMonth', 'time')))
            seas5_forecast_tmin= xr.open_dataset(f'{DATADIR}/ecmwf_seas5_51_%s'%fyear + '_amj_forecast_monthly_Tmin.grib', engine='cfgrib', 
                                        backend_kwargs=dict(time_dims=('forecastMonth', 'time')))

        
        es_hindcast= vapor_pressure((ds_hindcast_tmax['mx2t24'] + ds_hindcast_tmin['mn2t24'])/2)
        ea_hindcast= vapor_pressure(ds_hindcast_d2m['d2m'])
        vpd_hindcast= es_hindcast - ea_hindcast
        vpd_hindcast_mean= vpd_hindcast.groupby('time.month')[init_month].mean(['number', 'time'])
        vpd_hindcast_std= vpd_hindcast.groupby('time.month')[init_month].std(['number', 'time'])

        es_forecast= vapor_pressure((seas5_forecast_tmax['mx2t24'] + seas5_forecast_tmin['mn2t24'])/2)
        ea_forecast= vapor_pressure(seas5_forecast_d2m['d2m'])
        vpd_forecast= es_forecast - ea_forecast

        if anom_type == 'raw':
            seas5_anomalies= vpd_forecast.groupby('time.month')[init_month] - vpd_hindcast_mean
        elif anom_type == 'standardized':
            seas5_anomalies= (vpd_forecast.groupby('time.month')[init_month] - vpd_hindcast_mean)/vpd_hindcast_std
        valid_time= [pd.to_datetime(seas5_anomalies.time.values[0]) + relativedelta(months=fcmonth-1) for fcmonth in seas5_anomalies.forecastMonth]
        seas5_anomalies= seas5_anomalies.assign_coords(valid_time=('forecastMonth',valid_time))
        numdays= [monthrange(dd.year,dd.month)[1] for dd in valid_time]
        seas5_anomalies= seas5_anomalies.assign_coords(numdays=('forecastMonth',numdays))
        seas5_anomalies_vpd= seas5_anomalies
        seas5_anomalies_vpd.attrs['units']= 'hPa'
        seas5_anomalies_vpd.attrs['long_name']= 'Vapor pressure deficit anomaly'
        if area_flag:
            if regrid_flag:
                seas5_anomalies_vpd_conus= ds_latlon_subset(seas5_anomalies_vpd.sel(time= '%s'%fyear + '-0%s'%init_month + '-01'), subarea)
                regridder_vpd= xe.Regridder(seas5_anomalies_vpd_conus, dsout, regrid_scheme)
                return regridder_vpd(seas5_anomalies_vpd_conus, keep_attrs=True)
            else:
                return ds_latlon_subset(seas5_anomalies_vpd.sel(time= '%s'%fyear + '-0%s'%init_month + '-01'), subarea)
        else:
            return seas5_anomalies_vpd

    else:
        if pred_var == 'Prec': 
            pred_var_name= 'tprate'
        elif pred_var == 'Tmax':
            pred_var_name= 'mx2t24'
        elif pred_var == 'Tmin':
            pred_var_name= 'mn2t24' #convert into dictionary
        if system == None:
            ds_hindcast= xr.open_dataset(f'{DATADIR}/ecmwf_seas5_1993-2016_amj_hindcast_monthly_%s.grib'%pred_var, engine='cfgrib', backend_kwargs=dict(time_dims=('forecastMonth', 'time')))
            seas5_forecast = xr.open_dataset(f'{DATADIR}/ecmwf_seas5_%s'%fyear + '_amj_forecast_monthly_%s.grib'%pred_var, engine='cfgrib', 
                                        backend_kwargs=dict(time_dims=('forecastMonth', 'time')))
        else:
            ds_hindcast= xr.open_dataset(f'{DATADIR}/ecmwf_seas5_51_1993-2016_amj_hindcast_monthly_%s.grib'%pred_var, engine='cfgrib', backend_kwargs=dict(time_dims=('forecastMonth', 'time')))
            seas5_forecast = xr.open_dataset(f'{DATADIR}/ecmwf_seas5_51_%s'%fyear + '_amj_forecast_monthly_%s.grib'%pred_var, engine='cfgrib', 
                                        backend_kwargs=dict(time_dims=('forecastMonth', 'time')))
        pred_var_hindcast= ds_hindcast[pred_var_name]
        pred_var_hindcast_mean= pred_var_hindcast.groupby('time.month')[init_month].mean(['number', 'time'])
        pred_var_hindcast_std= pred_var_hindcast.groupby('time.month')[init_month].std(['number', 'time'])

        if anom_type == 'raw':
            seas5_anomalies= (seas5_forecast.groupby('time.month')[init_month][pred_var_name] - pred_var_hindcast_mean)
        elif anom_type == 'standardized':
            seas5_anomalies= (seas5_forecast.groupby('time.month')[init_month][pred_var_name] - pred_var_hindcast_mean)/pred_var_hindcast_std 
        valid_time = [pd.to_datetime(seas5_anomalies.time.values[0]) + relativedelta(months=fcmonth-1) for fcmonth in seas5_anomalies.forecastMonth]
        seas5_anomalies= seas5_anomalies.assign_coords(valid_time=('forecastMonth',valid_time))
        numdays = [monthrange(dd.year,dd.month)[1] for dd in valid_time]
        seas5_anomalies = seas5_anomalies.assign_coords(numdays=('forecastMonth',numdays))
        if pred_var == 'Prec':
            seas5_anomalies_tp= seas5_anomalies * seas5_anomalies.numdays * 24 * 60 * 60 * 1000
            seas5_anomalies_tp.attrs['units']= 'mm'
            seas5_anomalies_tp.attrs['long_name']= 'Total precipitation anomaly' 

            if area_flag:
                if regrid_flag:
                    seas5_anomalies_tp_conus= ds_latlon_subset(seas5_anomalies_tp.sel(time= '%s'%fyear + '-0%s'%init_month + '-01'), subarea)
                    regridder_tp= xe.Regridder(seas5_anomalies_tp_conus, dsout, regrid_scheme)
                    return regridder_tp(seas5_anomalies_tp_conus, keep_attrs=True)
                else:
                    return ds_latlon_subset(seas5_anomalies_tp.sel(time= '%s'%fyear + '-0%s'%init_month + '-01'), subarea)
            else:
                return seas5_anomalies_tp

        elif pred_var == 'Tmax':
            seas5_anomalies_tmax= seas5_anomalies
            seas5_anomalies_tmax.attrs['units']= 'degC'
            seas5_anomalies_tmax.attrs['long_name']= 'Maximum 2m temperature anomaly' 

            if area_flag:
                if regrid_flag:
                    seas5_anomalies_tmax_conus= ds_latlon_subset(seas5_anomalies_tmax.sel(time= '%s'%fyear + '-0%s'%init_month + '-01'), subarea)
                    regridder_tmax= xe.Regridder(seas5_anomalies_tmax_conus, dsout, regrid_scheme)
                    return regridder_tmax(seas5_anomalies_tmax_conus, keep_attrs=True)
                else:
                    return ds_latlon_subset(seas5_anomalies_tmax.sel(time= '%s'%fyear + '-0%s'%init_month + '-01'), subarea)
            else:
                return seas5_anomalies_tmax
        
        elif pred_var == 'Tmin':
            seas5_anomalies_tmin= seas5_anomalies
            seas5_anomalies_tmin.attrs['units']= 'degC'
            seas5_anomalies_tmin.attrs['long_name']= 'Minimum 2m temperature anomaly' 

            if area_flag:
                if regrid_flag:
                    seas5_anomalies_tmin_conus= ds_latlon_subset(seas5_anomalies_tmin.sel(time= '%s'%fyear + '-0%s'%init_month + '-01'), subarea)
                    regridder_tmin= xe.Regridder(seas5_anomalies_tmin_conus, dsout, regrid_scheme)
                    return regridder_tmin(seas5_anomalies_tmin_conus, keep_attrs=True)
                else:
                    return ds_latlon_subset(seas5_anomalies_tmin.sel(time= '%s'%fyear + '-0%s'%init_month + '-01'), subarea)
            else:
                return seas5_anomalies_tmin

def fire_pred_df_func(clim_df, target_yr, pred_mon_arr, pred_var_arr, firemon_pred_flag, sys_no= None, ens_no= None, pred_drop_cols= ['SWE_mean', 'SWE_max', 'AvgSWE_3mo'], rescale_flag= True, freq_flag= 'prediction'):

    '''
    Function to create a dataframe of fire predictors 

    clim_df: dataframe with climate, vegetation, and human predictors
    target_yr: year for which fire predictions are to be made
    pred_mon_arr: array of months for which fire predictions are to be made
    pred_var_arr: array of fire month climate variables 
    firemon_pred_flag: statistical/dynamical forecasts or observations
    ens_no: ensemble member number for dynamical forecasts
    pred_drop_cols: columns to be dropped from the dataframe of fire predictors
    rescale_flag: flag for rescaling fire predictors
    freq_flag: flag for returning dataframe of fire predictors for single prediction or ensemble plots
    '''
    
    tmax_xr= xarray.open_dataarray('../data/12km/climate/2023/primary/tmax.nc')

    if firemon_pred_flag == 'statistical_forecasts':
        if target_yr != 2023:
            fire_freq_df= clim_df[clim_df.month.isin(pred_mon_arr)]['fire_freq'].reset_index(drop= True)
        else:
            fire_freq_df= pd.DataFrame({'fire_freq': np.zeros(len(tmax_xr[0].values.flatten())*len(pred_mon_arr), dtype= np.int64)})
        tmpdf= clim_df[clim_df.month.isin([pred_mon_arr[0]])].drop(columns= pred_var_arr).drop(columns= 'fire_freq').reset_index(drop= True)
        obsdf= pd.concat([tmpdf.replace(pred_mon_arr[0], m) for m in pred_mon_arr], ignore_index= True) 
        
        climdf= pd.read_hdf('../data/clim_12km_1952_2023_data.h5')
        input_var_arr= ['Tmax', 'Solar', 'VPD', 'Tmin', 'Prec', 'RH', 'SM_0_100cm', 'PDSI', 'FFWI_max7', 'CAPE']
        scaling_flag= 'trend_w_seasonal_cycle' # 'normalized', 'trend' , 'trend_w_seasonal_cycle'
        trend_mons= 700 #700 --> 2010; 792 --> 2018
        tstart_mon= 0  # 336 --> 1980; 468 --> 1991
        tot_months= len(climdf.month.unique())
        if scaling_flag == 'trend_w_seasonal_cycle':
            start_mon= 2
            end_mon= 8

        clim_xarr= clim_xarr_init(climdf, input_var_arr, scaling_flag, tstart_mon, trend_mons, start_mon, end_mon)

        input_var_arr= np.append([i for i in clim_xarr.data_vars], ['Elev', 'time', 'X', 'Y', 'mei', 'nino34', 'rmm1', 'rmm2']) #'Southness', 
        if scaling_flag == 'trend_w_seasonal_cycle':
            totmonarr= np.sort(np.hstack([np.arange(m, 857, 12) for m in range(start_mon - 1, end_mon)]))
            climdf= pd.concat([clim_xarr.to_dataframe().reset_index(), climdf[climdf.month.isin(totmonarr)][['Elev', 'mei', 'nino34', 'rmm1', 'rmm2']].reset_index(drop= True)], axis= 1) #'Southness', 
        else:
            climdf= pd.concat([clim_xarr.to_dataframe().reset_index(), climdf[['Elev', 'mei', 'nino34', 'rmm1', 'rmm2']]], axis= 1) #'Southness',
        climdf.time= (climdf.time.dt.year*12 + climdf.time.dt.month) - (1952*12 + 1)

        seas_anomalies_df= pd.DataFrame([])
        pred_var_tot_df= pd.DataFrame([])
        for pred_var in tqdm(pred_var_arr):
            stat_fcast_anomalies_conus= ens_stat_forecast_func(climdf= climdf, pred_var= pred_var, target_yr= target_yr, samp_flag= True)
            pred_var_df= stat_fcast_anomalies_conus.to_dataframe(pred_var).reset_index()[pred_var]
            pred_var_df/= pred_var_df.std() # standardizing the dynamical forecasts
            pred_var_tot_df= pd.concat([pred_var_tot_df, pred_var_df], axis=1, ignore_index= True)
        seas_anomalies_df= pd.concat([seas_anomalies_df, pred_var_tot_df], axis= 0, ignore_index= True)
        seas_anomalies_df.rename(columns= {i: pred_var_arr[i] for i in range(len(pred_var_arr))}, inplace= True)
        obsdf= pd.concat([obsdf, fire_freq_df], axis= 1)
        
        X_pred_ur_df= pd.concat([seas_anomalies_df, obsdf], axis= 1)
        X_pred_ur_df.drop(columns= pred_drop_cols, inplace= True)
        X_pred_df= X_pred_ur_df[X_pred_ur_df.columns].dropna()
        
    elif firemon_pred_flag == 'dynamical_forecasts':
        # Downscaling, regridding, and interpolating dynamical forecasts to match APW's 12km grid
        
        sub = (51.6, -128, 26.5, -101) # North/West/South/East
        ds_out = xr.Dataset(
            {
                "lat": (["lat"], np.arange(26.5, 51.6, 0.125), {"units": "degrees_north"}),
                "lon": (["lon"], np.arange(-128, -101, 0.125), {"units": "degrees_west"}),
            }
            )
        x_fire_grid= xr.DataArray(coord_transform(tmax_xr.X.values, tmax_xr.Y.values, "epsg:5070", "epsg:4326")[0], dims=('Y','X'))
        y_fire_grid= xr.DataArray(coord_transform(tmax_xr.X.values, tmax_xr.Y.values, "epsg:5070", "epsg:4326")[1], dims=('Y','X'))

        if target_yr != 2023:
            fire_freq_df= clim_df[clim_df.month.isin(pred_mon_arr)]['fire_freq'].reset_index(drop= True)
        else:
            fire_freq_df= pd.DataFrame({'fire_freq': np.zeros(len(tmax_xr[0].values.flatten())*len(pred_mon_arr), dtype= np.int64)})
        seas_anomalies_df= pd.DataFrame([])
        obsdf= pd.DataFrame([])
        init_month_arr= [4, 6]
        for indx in tqdm(range(len(init_month_arr))):
            # repeats the dynamical forecasts for the first 3/last 2 months of the fire season depending on the initialization month
            tmpdf= clim_df[clim_df.month.isin([pred_mon_arr[indx] - 1])].drop(columns= pred_var_arr).drop(columns= 'fire_freq').reset_index(drop= True)
            if indx== 0:
                tmppredarr= pred_mon_arr[0:3]
            else:
                tmppredarr= pred_mon_arr[3:5]
            tmpdf= pd.concat([tmpdf.replace(pred_mon_arr[indx] - 1, m) for m in tmppredarr], ignore_index= True) 
            obsdf= pd.concat([obsdf, tmpdf], axis= 0, ignore_index= True)
            
            pred_var_tot_df= pd.DataFrame([])
            for pred_var in pred_var_arr:
                seas5_anomalies_conus_regridded= seas5_monthly_anomaly_func(pred_var= pred_var, system= sys_no, fyear= target_yr, init_month= init_month_arr[indx], subarea= sub, regrid_scheme= 'bilinear', \
                                                                                                                                                        dsout= ds_out, anom_type= 'raw')
                seas5_anomalies_conus_regridded= seas5_anomalies_conus_regridded.interp({'lat':y_fire_grid, 'lon':x_fire_grid}, method='linear').load()
                seas5_anomalies_conus_regridded= seas5_anomalies_conus_regridded.assign_coords({'X': (('X'), tmax_xr.X.data, {"units": "meters"}), \
                                                                                                            'Y': (('Y'), tmax_xr.Y.data, {"units": "meters"})}).drop_vars(['lat','lon'])
                pred_var_df= seas5_anomalies_conus_regridded[ens_no][indx+1:4].where(~np.isnan(tmax_xr[0].drop('time'))).to_dataframe(pred_var).reset_index()[pred_var]
                pred_var_df/= pred_var_df.std() # standardizing the dynamical forecasts
                pred_var_tot_df= pd.concat([pred_var_tot_df, pred_var_df], axis=1, ignore_index= True)
            seas_anomalies_df= pd.concat([seas_anomalies_df, pred_var_tot_df], axis= 0, ignore_index= True)
        seas_anomalies_df.rename(columns= {i: pred_var_arr[i] for i in range(len(pred_var_arr))}, inplace= True)
        obsdf= pd.concat([obsdf, fire_freq_df], axis= 1)
        
        X_pred_ur_df= pd.concat([seas_anomalies_df, obsdf], axis= 1)
        X_pred_ur_df.drop(columns= pred_drop_cols, inplace= True)
        X_pred_df= X_pred_ur_df[X_pred_ur_df.columns].dropna()

    elif firemon_pred_flag == 'observations':
        X_pred_ur_df= clim_df[clim_df.month.isin(pred_mon_arr)]
        X_pred_ur_df.drop(columns= pred_drop_cols, inplace= True)
        X_pred_ur_df.reset_index(drop= True, inplace= True)
        X_pred_df= X_pred_ur_df[X_pred_ur_df.columns].dropna() #.reset_index(drop= True)

    ### Scaling predictors for forcing the trained fire model

    n_features= 43 - len(pred_drop_cols) # 39/46 for df with rescaled/unscaled 2021-2020 data; 35/43 for df with rescaled/unscaled 2022 data
    if not rescale_flag:
        start_month= 444
        tot_test_months= 36
        end_month= start_month + tot_test_months
        rseed= np.random.randint(1000)
        clim_df= clim_df.dropna().reset_index().drop(columns=['index'])
        fire_freq_test_ur_df= clim_df[(clim_df.month >= start_month) & (clim_df.month < end_month)]
        fire_freq_train_ur_df= clim_df.drop(fire_freq_test_ur_df.index)

        tmp_freq_df= X_pred_df[X_pred_df.iloc[:, 0:n_features].columns] 
        X_pred_test_df= pd.DataFrame({})
        scaler= StandardScaler().fit(fire_freq_train_ur_df.iloc[:, 0:n_features]) # scaling parameters derived from climatological data
        X_pred_test_df[tmp_freq_df.columns]= scaler.transform(tmp_freq_df)

        X_pred_test_df.loc[:, 'reg_indx']= X_pred_df.reg_indx
        X_pred_test_df.loc[:, 'month']= X_pred_df.month
        X_pred_test_df= X_pred_test_df.drop(columns=['Solar', 'Ant_Tmax', 'RH', 'Ant_RH', 'RH_min3', 'FFWI_max7', 'Avgprec_4mo', 'Avgprec_2mo', 'AvgVPD_4mo', 'AvgVPD_2mo', \
                                                            'Tmax_max7', 'VPD_max7', 'Tmin_max7', 'Elev', 'Delta_T', 'CAPE', 'Southness'])
    else:
        X_pred_test_df= X_pred_df[X_pred_df.iloc[:, 0:n_features].columns]
        X_pred_test_df.loc[:, 'reg_indx']= X_pred_df.reg_indx
        X_pred_test_df.loc[:, 'month']= X_pred_df.month
        X_pred_test_df= X_pred_test_df.drop(columns=['Solar', 'Ant_Tmax', 'RH', 'Ant_RH', 'RH_min3', 'FFWI_max7', 'Avgprec_4mo', 'Avgprec_2mo', 'AvgVPD_4mo', 'AvgVPD_2mo', \
                                                            'Tmax_max7', 'VPD_max7', 'Tmin_max7', 'Elev', 'Delta_T', 'CAPE', 'Southness'])
        
    if freq_flag == 'ensemble':
        return X_pred_ur_df[['Tmax', 'reg_indx', 'month', 'fire_freq']], X_pred_test_df[['reg_indx', 'month']]
    elif freq_flag == 'prediction':
        return X_pred_ur_df, X_pred_test_df
        
def fire_prob_pred_func(freq_id= None, seed= None, X_tot_df= None, X_test_df= None, pred_mon_arr= None, sav_flag= False, target_year= None, firemon_pred_flag= 'observations', ens_no= None):
    
    """ 
    Function to predict fire probability for a given month
    
    freq_id, seed: id and seed of trained ML freq model
    X_tot_df: dataframe containing all predictor variables including nan
    X_test_df: dataframe containing non-nan predictor variables
    pred_mon_arr: array containing months to predict
    sav_flag: flag to save predictions
    target_year: year for which predictions are made
    firemon_pred_flag: flag to indicate whether predictions are made for observations, dynamical forecasts or statistical forecasts
    ens_no: ensemble member number for dynamical or statistical forecasts
    """

    if seed == None:
        mdn_zipd= tf.keras.models.load_model('../sav_files/fire_freq_mods/mdn_ds_%s'%freq_id, custom_objects= {'zipd_loss': zipd_loss, 'zipd_accuracy': zipd_accuracy})
    else:
        mdn_zipd= tf.keras.models.load_model('../sav_files/fire_freq_mods/mdn_ds_rs_%s'%freq_id + '_%s'%seed, custom_objects= {'zipd_loss': zipd_loss, 'zipd_accuracy': zipd_accuracy})

    param_vec= []
    for m in pred_mon_arr:
        X_arr= np.array(X_test_df.groupby('month').get_group(m).drop(columns= ['reg_indx', 'month']), dtype= np.float32)
        param_vec.append(mdn_zipd.predict(x= tf.constant(X_arr)))
    param_vec= np.array(param_vec).reshape(len(pred_mon_arr)*23903, 3) #np.array(param_vec).reshape(3*23903, 3)

    # script to convert predicted fire probability in non-nan grid cells to a xarray with nan grid cells
    X_tot_df['pred_fire_prob']= np.zeros_like(X_tot_df['fire_freq'], dtype= np.float32)
    X_tot_df.loc[~X_tot_df['Elev'].isna(), 'pred_fire_prob']= param_vec[:, 1] # 1 --> rate parameter as mean fire probability
    X_tot_df.loc[X_tot_df['Elev'].isna(), 'pred_fire_prob']= np.nan # choose grid cells where Elev is not NaN because it has the fewest NaNs

    pred_prob_xarr= xarray.DataArray(data= X_tot_df['pred_fire_prob'].to_numpy().reshape(len(pred_mon_arr), 208, 155),
        dims=["month", "Y", "X"],
        coords=dict(
            X=(["X"], np.linspace(-2349250, -501250, 155)),
            Y=(["Y"], np.linspace(3166500, 682500, 208)),
            time= (["month"], pred_mon_arr),),) #np.linspace(0, len(pred_mon_arr) - 1, len(pred_mon_arr), dtype= np.int64)
    
    if sav_flag:
        if firemon_pred_flag == 'observations':
            pred_prob_xarr.to_netcdf('../sav_files/ssf_pred_files/pred_prob_xarr_%s'%freq_id + '_%d_'%seed + 'obs_%s.nc'%target_year)
        elif firemon_pred_flag == 'dynamical_forecasts':
            pred_prob_xarr.to_netcdf('../sav_files/ssf_pred_files/dynamical_forecasts/%s'%target_year + '/pred_prob_xarr_%s'%freq_id + '_%d_'%seed + 'df_%d'%ens_no + '_%s.nc'%target_year)
        else:
            pred_prob_xarr.to_netcdf('../sav_files/ssf_pred_files/statistical_forecasts/pred_prob_xarr_%s'%freq_id + '_%d_'%seed + 'sf_%d'%ens_no + '_%s.nc'%target_year)
        return print('Saved fire probability xarray for %s'%target_year)
    else:
        return pred_prob_xarr
    
def mon_fire_prob_pred(freq_id= '08_07_23', seed= 654, plot_yr= 2019, fmon= 5, fire_df= None, firemon_pred_flag= 'observations', ens_no= None, pred_fire_df= None):
    """
    Function to rescale predicted fire probability with climatological baseline fire probability and observed number of fires

    freq_id, seed: id and seed of trained ML freq model
    plot_yr: year for which predictions are made
    fmon: month for which predictions are made;  January --> 0, February --> 1, ..., June --> 5 etc.
    fire_df: dataframe containing observed number of fires
    firemon_pred_flag: flag to indicate whether predictions are made for observations, dynamical forecasts or statistical forecasts
    ens_no: ensemble member number for dynamical or statistical forecasts
    pred_fire_df: dataframe containing predicted fire probability
    """
    
    if firemon_pred_flag == 'observations':
        pred_prob_xarr= xarray.open_dataarray('../sav_files/ssf_pred_files/pred_prob_xarr_%s'%freq_id + '_%d_'%seed + 'obs_%s.nc'%plot_yr)
        n_fires_yr= len(fire_df[fire_df['fire_month'] == (plot_yr - 1984)*12 + fmon])
    elif firemon_pred_flag == 'dynamical_forecasts':
        pred_prob_xarr= xarray.open_dataarray('../sav_files/ssf_pred_files/dynamical_forecasts/pred_prob_xarr_%s'%freq_id + '_%d_'%seed + 'df_%d'%ens_no + '_%s.nc'%plot_yr)
        n_fires_yr= pred_fire_df[pred_fire_df.month == (plot_yr - 1984)*12 + fmon]['pred_mean_freq'].sum()
    else:
        pred_prob_xarr= xarray.open_dataarray('../sav_files/ssf_pred_files/statistical_forecasts/pred_prob_xarr_%s'%freq_id + '_%d_'%seed + 'sf_%d'%ens_no + '_%s.nc'%plot_yr)
        n_fires_yr= pred_fire_df[pred_fire_df.month == (plot_yr - 1984)*12 + fmon]['pred_mean_freq'].sum()

    baseline_arr= np.arange(209, 426, 12)
    n_fires_baseline= len(fire_df[fire_df['fire_month'].isin(baseline_arr + (fmon - 5))])/20

    pred_prob_baseline= xarray.open_dataarray('../sav_files/ssf_pred_files/pred_prob_xarr_%s'%freq_id + '_%d_'%seed + 'obs_baseline.nc')
    pred_prob_xarr_baseline= pred_prob_baseline[pred_prob_baseline.time.isin(baseline_mon_arr_func(start_yr= 2001, end_yr= 2019, mindx= (fmon + 1)).values), :, :].mean(dim= 'month')
    #fmon + 1 because months start from 1 in baseline formulation    
    return 10**(np.log10(pred_prob_xarr[(fmon - 5), :, :]) - np.log10(pred_prob_xarr_baseline) - np.log10(n_fires_baseline/n_fires_yr))

def ens_mon_fire_prob_pred(freq_id= '08_07_23', seed= 654, plot_yr= 2019, smon= 4, fmon= 5, fire_df= None, firemon_pred_flag= 'dynamical_forecasts', statistic= 'mean'):
    """
    Function to rescale predicted fire probability with climatological baseline fire probability and observed number of fires

    freq_id, seed: id and seed of trained ML freq model
    plot_yr: year for which predictions are made
    smon: start month of baseline array;  January --> 0, February --> 1, ..., May --> 4 etc.
    fmon: month for which predictions are made;  January --> 0, February --> 1, ..., June --> 5 etc.
    fire_df: dataframe containing observed number of fires
    firemon_pred_flag: flag to indicate whether predictions are made for observations, dynamical forecasts or statistical forecasts
    """
    
    if firemon_pred_flag == 'dynamical_forecasts':
        ens_pred_prob_xarr= xarray.concat([xarray.open_dataarray('../sav_files/ssf_pred_files/dynamical_forecasts/%s'%plot_yr + '/pred_prob_xarr_%s'%freq_id + '_%d_'%seed + \
                                                                                            'df_%d'%ens_no + '_%s.nc'%plot_yr) for ens_no in range(51)], dim= 'ens_no')
        mdn_tot_df= pd.DataFrame([])
        for ens_no in range(51):
            mdn_tmp_df= pd.read_hdf('../sav_files/fire_freq_pred_dfs/dynamical_forecasts/%s'%plot_yr + '/mdn_ssf_%s'%freq_id + '_%d'%seed +  '_fire_freq_%d'%ens_no + '_%d.h5'%plot_yr)
            mdn_tot_df= pd.concat([mdn_tot_df, mdn_tmp_df], axis= 0)
        pred_fire_df= mdn_tot_df.groupby(mdn_tot_df.index).mean().round().astype(int)
        if statistic == 'mean':
            pred_prob_xarr= ens_pred_prob_xarr.mean(dim= 'ens_no')
        elif statistic == 'std':
            pred_prob_xarr= ens_pred_prob_xarr.std(dim= 'ens_no')[(fmon - smon):(fmon - smon)+2].mean(dim= 'month')
        if fmon == 9:
            n_fires_yr= 11
        else:
            n_fires_yr= pred_fire_df[pred_fire_df.month == (plot_yr - 1984)*12 + fmon]['pred_mean_freq'].sum()
    else:
        ens_pred_prob_xarr= xarray.concat([xarray.open_dataarray('../sav_files/ssf_pred_files/statistical_forecasts/pred_prob_xarr_%s'%freq_id + '_%d_'%seed + \
                                                                                            'df_%d'%ens_no + '_%s.nc'%plot_yr) for ens_no in range(51)], dim= 'ens_no')
        pred_prob_xarr= xarray.open_dataarray('../sav_files/ssf_pred_files/statistical_forecasts/pred_prob_xarr_%s'%freq_id + '_%d_'%seed + 'sf_%d'%ens_no + '_%s.nc'%plot_yr)
        n_fires_yr= pred_fire_df[pred_fire_df.month == (plot_yr - 1984)*12 + fmon]['pred_mean_freq'].sum()

    baseline_arr= baseline_mon_arr_func(start_yr= 2001, end_yr= 2019, mindx= smon).values
    n_fires_baseline= len(fire_df[fire_df['fire_month'].isin(baseline_arr + (fmon - smon))])/20

    pred_prob_baseline= xarray.open_dataarray('../sav_files/ssf_pred_files/pred_prob_xarr_%s'%freq_id + '_%d_'%seed + 'obs_baseline.nc')
    pred_prob_xarr_baseline= pred_prob_baseline[pred_prob_baseline.time.isin(baseline_mon_arr_func(start_yr= 2001, end_yr= 2019, mindx= fmon).values), :, :].mean(dim= 'month')
    #fmon + 1 because months start from 1 in baseline formulation

    if statistic == 'mean':   
        return 10**(np.log10(pred_prob_xarr[(fmon - smon), :, :]) - np.log10(pred_prob_xarr_baseline) - np.log10(n_fires_baseline/n_fires_yr))
        #return np.log10(pred_prob_xarr[(fmon - smon), :, :]), np.log10(pred_prob_xarr_baseline), np.log10(n_fires_baseline/n_fires_yr)
    elif statistic == 'std':
        return 10**(np.log10(pred_prob_xarr) - np.log10(pred_prob_xarr_baseline) - np.log10(n_fires_baseline/n_fires_yr))
    
def grid_ssf_freq_predict(X_test_dat, freq_test_df= None, n_regs= 18, ml_model= None, func_flag= 'zipd', pred_mon_flag= True, pred_mons= None, seas_flag= None,\
                                                                                    base_yr= 1984, fcast_yr= None, loc_flag= False, rseed= 99):
    
    '''
    Predicts the fire frequency for each L3 ecoregion at seasonal and S2S timescales
    # todo: design a separate function that generates seasonal frequencies using this function and initialized dataframe for calibrating SSF using statistical/dynamical climate forecasts
    
    X_test_dat: dataframe containing predictor variables for each grid cell
    freq_test_df: dataframe containing observed fire frequency for each grid cell
    n_regs: number of L3 ecoregions
    ml_model: trained ML model
    func_flag: flag to choose between zipd and logistic regression
    pred_mon_flag: flag to choose between predicting for pred_mons or for all months in the seas_flag period
    pred_mons: array containing months to predict
    seas_flag: season flag
    base_yr: base year for the study period
    fcast_yr: start year of the forecast period
    rseed: random seed
    '''
    if pred_mon_flag:
        pred_mon_arr= pred_mons
    else:
        if seas_flag == 'JJA':
            start_month= 5 + (fcast_yr - base_yr)*12
        elif seas_flag == 'SON':
            start_month= 8 + (fcast_yr - base_yr)*12
        final_month= start_month + 3
        pred_mon_arr= np.sort(np.append(np.append(np.arange(start_month, final_month, 12), np.arange(start_month + 1, final_month, 12)), np.arange(start_month + 2, final_month, 12)))
    
    ml_freq_df= pd.DataFrame([])
    #tot_rfac_arr= []
    X_test_reg_groups= X_test_dat.groupby('reg_indx')
    freq_test_reg_groups= freq_test_df.groupby('reg_indx')
    for r in tqdm(range(n_regs)):  
        pred_freq= []
        pred_freq_sig= []
        freq_arr= []
        X_test_mon_groups= X_test_reg_groups.get_group(r+1).groupby('month')
        freq_test_mon_groups= freq_test_reg_groups.get_group(r+1).groupby('month')
        for m in pred_mon_arr:
            X_arr= np.array(X_test_mon_groups.get_group(m).dropna().drop(columns= ['reg_indx', 'month']), dtype= np.float32)
            if func_flag == 'zipd':
                param_vec= ml_model.predict(x= tf.constant(X_arr))
                freq_samp= zipd_model(param_vec).sample(1000, seed= rseed)
                pred_freq.append(tf.reduce_sum(tf.cast(tf.reduce_mean(freq_samp, axis= 0), tf.int64)).numpy())
                pred_freq_sig.append(np.sqrt(tf.reduce_sum(tf.pow(tf.cast(tf.math.reduce_std(freq_samp, axis= 0), tf.int64), 2)).numpy()).astype(np.int64))

                freq_arr.append(tf.cast(tf.reduce_mean(freq_samp, axis= 0), tf.int64).numpy())

            elif func_flag == 'logistic':
                reg_predictions= ml_model.predict(x= tf.constant(X_arr)).flatten()
                freq_arr.append([1 if p > 0.5 else 0 for p in reg_predictions])
                pred_freq.append(np.sum([1 if p > 0.5 else 0 for p in reg_predictions]))

        pred_freq_arr= [np.sum(freq_arr[m - pred_mon_arr[0]]) for m in pred_mon_arr]
        reg_indx_arr= np.ones(len(pred_mon_arr), dtype= np.int64)*(r+1)  
        #tot_rfac_arr.append((np.std(obs_freqs)/np.std(pred_freq)))
        
        if func_flag == 'zipd':
            pred_high_2sig= np.ceil((np.array(pred_freq) + 2*np.array(pred_freq_sig)))
            pred_low= np.array(pred_freq) - 2*np.array(pred_freq_sig)
            pred_low[pred_low < 0]= 0
            pred_low_2sig= np.floor(pred_low)

            if pred_mon_flag:
                if loc_flag:
                    ml_freq_df= pd.concat([ml_freq_df, pd.DataFrame({'freq_loc_arr': freq_arr, 'month': pred_mon_arr, 'reg_indx': reg_indx_arr})], axis= 0, ignore_index= True)
                else:
                    ml_freq_df= pd.concat([ml_freq_df, pd.DataFrame({'pred_mean_freq': pred_freq_arr, 'pred_high_2sig': pred_high_2sig, 'pred_low_2sig': pred_low_2sig, \
                                                                                               'month': pred_mon_arr, 'reg_indx': reg_indx_arr})], axis= 0, ignore_index= True)
            else:
                if loc_flag:
                    ml_freq_df= ml_freq_df.append(pd.DataFrame({'freq_loc_arr': freq_arr, 'month': pred_mon_arr, 'reg_indx': reg_indx_arr}))
                else:
                    ml_freq_df= ml_freq_df.append(pd.DataFrame({'obs_freq': obs_freqs, 'pred_mean_freq': pred_freq_arr, 'pred_high_2sig': pred_high_2sig, 'pred_low_2sig': pred_low_2sig, \
                                                                                                                                        'reg_indx': reg_indx_arr}))

        elif func_flag == 'logistic':
            obs_freqs= [np.sum(freq_test_mon_groups.get_group(m).fire_freq) for m in pred_mon_arr]
            ml_freq_df= ml_freq_df.append(pd.DataFrame({'obs_freq': obs_freqs, 'pred_mean_freq': pred_freq_arr, 'reg_indx': reg_indx_arr}))
                
    return ml_freq_df.reset_index(drop= True) #, tot_rfac_arr

def calib_ssf_freq_predict(freq_train_df, freq_test_df, n_regs, n_train_years, n_pred_mons, input_type= 'std', pred_type= 'std', regtype= 'polynomial'):
    
    """
    Derives a calibration factor by using a linear model to predict the annual std/mean of monthly observed frequencies with annual std/mean of monthly predicted frequencies

    freq_train_df: dataframe containing observed and predicted frequencies for training period
    freq_test_df: dataframe containing predicted frequencies for testing period
    n_regs: number of L3 ecoregions
    n_train_years: number of years in the training period
    n_pred_mons: number of months in the training and test seasons
    input_type: whether to use std or mean of monthly predicted frequencies
    pred_type: whether to use std or mean of monthly observed frequencies
    regtype: type of regression model to use
    """
    freq_train_groups= freq_train_df.groupby('reg_indx')
    freq_test_groups= freq_test_df.groupby('reg_indx')
    seas_train_arr= np.linspace(0, n_train_years*n_pred_mons, n_train_years + 1, dtype=int)
    seas_test_arr= np.arange(0, n_pred_mons, 1, dtype=int)
    pred_norm_df= pd.DataFrame({})

    for r in tqdm(range(n_regs)):
        pred_freqs_train= np.array(freq_train_groups.get_group(r+1)['pred_mean_freq'])
        obs_freqs_train= np.array(freq_train_groups.get_group(r+1)['obs_freq'])
        pred_freqs_test= np.array(freq_test_groups.get_group(r+1)['pred_mean_freq'])
        
        if input_type == 'std':
            X_mat_train= np.array([np.std(pred_freqs_train[seas_train_arr[t]:seas_train_arr[t+1]]) \
                            for t in range(len(seas_train_arr) - 1)])
            X_mat= np.std(pred_freqs_test[seas_test_arr])
        else:
            X_mat_train= np.array([np.mean(pred_freqs_train[seas_train_arr[t]:seas_train_arr[t+1]]) \
                            for t in range(len(seas_train_arr) - 1)])
            X_mat= np.mean(pred_freqs_test[seas_test_arr])
        
        if pred_type == 'std':
            Y_arr_train= np.array([np.std(obs_freqs_train[seas_train_arr[t]:seas_train_arr[t+1]]) \
                            for t in range(len(seas_train_arr) - 1)])
            #Y_arr= np.array([np.std(ml_freq_groups.get_group(regindx)['obs_freq'][ann_arr[t]:ann_arr[t+1]]) \
            #                for t in range(len(ann_arr) - 1)])
        else:
            Y_arr_train= np.array([np.mean(obs_freqs_train[seas_train_arr[t]:seas_train_arr[t+1]]) \
                            for t in range(len(seas_train_arr) - 1)])
            
        if regtype == 'linear':
            reg_pred= LinearRegression().fit(X_mat_train.reshape(-1, 1), Y_arr_train)
            pred_norm= reg_pred.predict(X_mat.reshape(-1, 1))
            #r_pred= reg_pred.score(X_mat.reshape(-1, 1), Y_arr)
        elif regtype == 'polynomial':
            poly_feat= PolynomialFeatures(3)
            ann_poly_x= poly_feat.fit_transform(X_mat.reshape(-1, 1))
        elif regtype == 'spline':
            spline_feat= SplineTransformer(degree= 3, n_knots= 5)
            scaler= spline_feat.fit(X_mat_train.reshape(-1, 1))
            ann_poly_x_train= scaler.transform(X_mat_train.reshape(-1, 1))
            ann_poly_x= scaler.transform(X_mat.reshape(-1, 1))
            
            model= make_pipeline(SplineTransformer(n_knots= 5, degree= 3), LinearRegression())
            reg_pred= model.fit(X_mat_train.reshape(-1, 1), Y_arr_train)
            pred_norm= reg_pred.predict(X_mat.reshape(-1, 1))
            #r_pred= reg_pred.score(X_mat.reshape(-1, 1), Y_arr)

        pred_norm_df= pd.concat([pred_norm_df, pd.DataFrame({'month': freq_test_groups.get_group(r+1).month, 'reg_indx': np.ones(len(seas_test_arr), dtype= np.int64)*(r+1), \
                                'pred_freq': np.repeat(X_mat, len(seas_test_arr)), 'pred_obs_freq': np.repeat(pred_norm, len(seas_test_arr))})], axis= 0, ignore_index= True)    
        # pred_freq --> mean/std of predicted frequencies; pred_obs_freq --> mean/std of observed frequencies as predicted by the calibration model
    
    return pred_norm_df #, r_pred

def fire_loc_ssf_func(loc_df, ml_freq_df, X_test_dat, regindx, pred_mon_flag= True, pred_mons= None, seas_flag= None, base_yr= 1984, fcast_yr= None, freqlabel= 'pred_mean_freq'):
    
    '''
    Predicts the grid scale location of fire frequencies/probabilites for each L3 ecoregion

    loc_df: dataframe containing simulated or observed fire locations
    ml_freq_df: dataframe containing median and +/- 2 sigma predicted fire frequencies
    X_test_dat: dataframe containing predictor variables for each grid cell
    regindx: L3 ecoregion index
    pred_mon_flag: flag to choose between predicting for pred_mons or for all months in the seas_flag period
    pred_mons: array containing months to predict
    seas_flag: season flag
    base_yr: base year for the study period
    fcast_yr: start year of the forecast period
    freqlabel: 'pred_mean_freq' or 'pred_high_2sig' or 'pred_low_2sig'
    '''
    
    if pred_mon_flag:
        pred_mon_arr= pred_mons
    else:
        if seas_flag == 'JJA':
            start_month= 5 + (fcast_yr - base_yr)*12
        elif seas_flag == 'SON':
            start_month= 8 + (fcast_yr - base_yr)*12
        final_month= start_month + 3
        pred_mon_arr= np.sort(np.append(np.append(np.arange(start_month, final_month, 12), np.arange(start_month + 1, final_month, 12)), np.arange(start_month + 2, final_month, 12)))

    ml_freq_groups= ml_freq_df.groupby('reg_indx')
    X_test_groups= X_test_dat.groupby('reg_indx').get_group(regindx).groupby('month')
    loc_df_groups= loc_df.groupby('reg_indx').get_group(regindx)
    fire_loc_arr= []
    
    for m in pred_mon_arr:
        mindx= m - pred_mon_arr[0]
        if np.count_nonzero(loc_df_groups['freq_loc_arr'].iloc[[mindx]].to_numpy()[0]) in [0, 1]:
            fire_loc_arr.append(np.array([0]))
        else:
            indarr= np.random.choice(np.nonzero(loc_df.groupby('reg_indx').get_group(regindx)['freq_loc_arr'].iloc[[mindx]].to_numpy()[0])[0], \
                                     ml_freq_groups.get_group(regindx)[freqlabel].iloc[[mindx]].to_numpy()[0].astype(int)) #np.nonzero() returns the indices of all non-zero elements
            fire_loc_arr.append(X_test_groups.get_group(m).index.to_numpy()[indarr])
    
    return fire_loc_arr

def loc_ind_ssf_func(loc_df, ml_freq_df, X_test_dat, n_regs= 18, pred_mon_flag= True, pred_mons= None, seas_flag= None, base_yr= 1984, fcast_yr= None, freqlabel= 'pred_mean_freq'):
    
    '''
    Returns a dataframe of grid indices of all predicted fires over the whole study region

    # same arguments as fire_loc_ssf_func()
    '''

    pred_loc_arr= []
    for r in tqdm(range(n_regs)): #range(n_regs)
        tmplocarr= fire_loc_ssf_func(loc_df, ml_freq_df, X_test_dat, regindx= r+1, pred_mon_flag= pred_mon_flag, pred_mons= pred_mons, seas_flag= seas_flag, \
                                                                                                                base_yr= base_yr, fcast_yr= fcast_yr, freqlabel= freqlabel)
        tmplocarr= np.hstack(tmplocarr)
        pred_loc_arr.append(tmplocarr[tmplocarr != 0])

    return pred_loc_arr


def fire_freq_pred_func(target_yr, firemon_pred_flag= 'dynamical_forecasts', ens_no= None, freq_id= None, seed= None, freqlabel= None):

    """
    Function to predict fire frequency for a given year using a trained ML model

    target_yr: year for which predictions are made
    firemon_pred_flag: flag to indicate whether predictions are made for observations, dynamical forecasts or statistical forecasts
    ens_no: ensemble member number for dynamical or statistical forecasts
    freq_id: id of trained ML freq model
    seed: seed of trained ML freq model
    freqlabel: 'pred_mean_freq' or 'pred_high_2sig' or 'pred_low_2sig'
    """

    if target_yr != 2023:
        clim_df= pd.read_hdf('../data/clim_fire_freq_12km_w2022_rescaled_data.h5')
        sys_no= None
    else:
        clim_df= pd.read_hdf('../data/clim_fire_freq_12km_w%s_rescaled_data.h5'%target_yr)
        sys_no= 51
    clim_df.loc[clim_df[clim_df.fire_freq > 1].index, 'fire_freq']= np.ones(len(clim_df[clim_df.fire_freq > 1].index), dtype= np.int64)

    pred_var_arr= ['Tmax', 'Prec', 'VPD', 'Tmin'] #'VPD', 'FFWI',
    if target_yr == 'baseline':
        pred_mon_arr= baseline_mon_arr_func(start_yr= 2001, end_yr= 2019, mindx= [5, 6, 7, 8, 9]).values # np.sort(np.append(np.append(np.arange(209, 426, 12), np.arange(210, 427, 12)), np.arange(211, 428, 12))) #2001-2020 
    else:
        pred_mon_arr=  np.array([460, 461, 462, 463, 464]) - (2022 - target_yr)*12  #464

    X_pred_ur_df, X_pred_test_df= fire_pred_df_func(clim_df, target_yr, pred_mon_arr, pred_var_arr, firemon_pred_flag, sys_no= sys_no, ens_no= ens_no, freq_flag= 'prediction') #freq_flag= 'ensemble' or 'prediction'

    start_year= 1984
    end_year= 2019
    tot_years= end_year - start_year + 1
    start_month= 5 # index of forecast start month with January = 1 
    tot_months= len(pred_mon_arr)
    seas_mon_arr= baseline_mon_arr_func(start_yr= start_year, end_yr= end_year, mindx= np.arange(0, tot_months, 1) + start_month)

    # load predicted fire frequency to calculate rescaling factor
    mdn_freq_train_ur_df= pd.read_hdf('../sav_files/fire_freq_pred_dfs/mdn_ds_%s'%freq_id + '_mon_fire_freq_%s.h5'%seed) 
    mdn_freq_train_ur_df= mdn_freq_train_ur_df.reset_index().rename(columns= {'index': 'month'})
    mdn_freq_train_df= mdn_freq_train_ur_df[mdn_freq_train_ur_df.month.isin(seas_mon_arr)].reset_index(drop= True)

    if seed == None:
        mdn_zipd= tf.keras.models.load_model('../sav_files/fire_freq_mods/mdn_ds_%s'%freq_id, custom_objects= {'zipd_loss': zipd_loss, 'zipd_accuracy': zipd_accuracy})
    else:
        mdn_zipd= tf.keras.models.load_model('../sav_files/fire_freq_mods/mdn_ds_rs_%s'%freq_id + '_%s'%seed, custom_objects= {'zipd_loss': zipd_loss, 'zipd_accuracy': zipd_accuracy})
    mdn_freq_test_df= grid_ssf_freq_predict(X_test_dat= X_pred_test_df, freq_test_df= X_pred_ur_df.dropna(), n_regs= 18, ml_model= mdn_zipd, func_flag= 'zipd', pred_mon_flag= True, \
                                                                                                                                        pred_mons= pred_mon_arr, loc_flag= False, rseed= 87)
    rescale_fac_df= calib_ssf_freq_predict(freq_train_df= mdn_freq_train_df, freq_test_df= mdn_freq_test_df, n_regs= 18, n_train_years= tot_years, n_pred_mons= tot_months, \
                                                                                                                        input_type= 'mean', pred_type= 'mean', regtype= 'linear')
    scale_fac= np.array(rescale_fac_df['pred_obs_freq']/rescale_fac_df['pred_freq'])
    scale_fac[np.isinf(scale_fac)]= 0
    scale_fac[scale_fac < 0]= 0

    mdn_freq_test_df['pred_mean_freq']= np.floor(mdn_freq_test_df['pred_mean_freq']*scale_fac)
    mdn_freq_test_df['pred_high_2sig']= np.floor(mdn_freq_test_df['pred_high_2sig']*scale_fac)
    mdn_freq_test_df['pred_low_2sig']= np.floor(mdn_freq_test_df['pred_low_2sig']*scale_fac)

    freq_loc_df= grid_ssf_freq_predict(X_test_dat= X_pred_test_df, freq_test_df= X_pred_ur_df.dropna(), n_regs= 18, ml_model= mdn_zipd, func_flag= 'zipd', pred_mon_flag= True, \
                                                                                                            pred_mons= pred_mon_arr, loc_flag= True, rseed= 87)
    
    pred_loc_arr= loc_ind_ssf_func(loc_df= freq_loc_df, ml_freq_df= mdn_freq_test_df, X_test_dat= X_pred_test_df, pred_mon_flag= True, pred_mons= pred_mon_arr, freqlabel= freqlabel)
    X_pred_ur_df['pred_fire_freq']= np.zeros_like(X_pred_ur_df['fire_freq'])
    for r in tqdm(range(18)):
        X_pred_ur_df.loc[X_pred_ur_df.groupby('reg_indx').get_group(r+1).index, 'pred_fire_freq']= 0

    for ind in tqdm(np.hstack(pred_loc_arr)):
        X_pred_ur_df.loc[ind, 'pred_fire_freq']+=1 

    nan_ind_arr= X_pred_ur_df['Tmax'].isna()
    X_pred_ur_df.loc[nan_ind_arr, 'pred_fire_freq']= np.nan

    return mdn_freq_test_df, freq_loc_df, X_pred_ur_df

def grid_ssf_size_func(mdn_model, stat_model, max_size_arr, sum_size_arr, pred_mon_flag= True, pred_mons= None, seas_flag= None, base_yr= 1984, fcast_yr= None, \
                        nsamps= 1000, loc_df= None, ml_freq_df= None, X_test_dat= None, seed= None):
    
    '''
    Given a NN model and fire locations, the function returns the monthly burned area time series for all L3 regions
    # todo: include effect of frequency uncertainty
    '''
    
    #tf.random.set_seed(seed)
    if seed == None:
        seed= np.random.randint(1000)
    if pred_mon_flag:
        pred_mon_arr= pred_mons
    else:
        if seas_flag == 'JJA':
            start_month= 5 + (fcast_yr - base_yr)*12
        elif seas_flag == 'SON':
            start_month= 8 + (fcast_yr - base_yr)*12
        final_month= start_month + 3
        pred_mon_arr= np.sort(np.append(np.append(np.arange(start_month, final_month, 12), np.arange(start_month + 1, final_month, 12)), np.arange(start_month + 2, final_month, 12)))
    tot_months= len(pred_mon_arr)

    n_regions= 18
    reg_size_df= pd.DataFrame({'mean_size': pd.Series(dtype= 'int'), 'low_1sig_size': pd.Series(dtype= 'int'), 'high_1sig_size': pd.Series(dtype= 'int'), \
                                                                                           'reg_indx': pd.Series(dtype= 'int')})
    X_test_dat_reg_groups= X_test_dat[X_test_dat['pred_fire_freq'] > 0].groupby('reg_indx')
    
    for r in tqdm(range(n_regions)): #tqdm --> removed for hyperparameter runs
        mean_burnarea_tot= np.zeros(tot_months)
        high_1sig_burnarea_tot= np.zeros(tot_months)
        low_1sig_burnarea_tot= np.zeros(tot_months)

        try:
            X_test_dat_mon_groups= X_test_dat_reg_groups.get_group(r+1).groupby('month')

            for m in pred_mon_arr:
                mindx= m - pred_mon_arr[0]
                samp_arr= tf.zeros([nsamps, 0])

                try: 
                    fire_loc_arr= X_test_dat_mon_groups.get_group(m).index.values
                    n_fires= X_test_dat_mon_groups.get_group(m).pred_fire_freq.values
                except KeyError:
                    fire_loc_arr= [0]

                # for sampling from frequency distribution, create additional function from here
                if np.nonzero(fire_loc_arr)[0].size == 0: #if mean freqs from distribution is zero, then set burned area to be zero
                    mean_burnarea_tot[mindx]= 0
                    high_1sig_burnarea_tot[mindx]= 0
                    low_1sig_burnarea_tot[mindx]= 0
                        
                else:
                    ml_param_vec= mdn_model.predict(x= np.array(X_test_dat_mon_groups.get_group(m).drop(columns=['reg_indx', 'month', 'pred_fire_freq']), dtype= np.float32)) #note: different indexing than the fire_size_test df
                    samp_arr= tf.concat([samp_arr, tf.reshape(stat_model(ml_param_vec).sample(nsamps, seed= seed), (nsamps, ml_param_vec.shape[0]))], axis= 1)
                    size_samp_arr= tf.reduce_mean(samp_arr, axis= 0).numpy()
                    std_size_arr= tf.math.reduce_std(samp_arr, axis= 0).numpy()
                                
                    high_1sig_err= deepcopy(std_size_arr) #tfp.stats.percentile(tf.reduce_sum(samps, axis= 1), 95, axis= 0)
                    tot_l1sig_arr= np.sqrt(np.sum(std_size_arr**2))
                            
                    size_samp_arr[size_samp_arr > max_size_arr[r]]= max_size_arr[r]
                    high_1sig_err[high_1sig_err > max_size_arr[r]]= max_size_arr[r] 
                    tot_h1sig_arr= np.sqrt(np.sum(high_1sig_err**2))

                    if np.sum(size_samp_arr) > 3*sum_size_arr[r][m]:
                        mean_burnarea_tot[mindx]= sum_size_arr[r][m]
                    else:
                        mean_burnarea_tot[mindx]= np.sum(size_samp_arr)

                    high_1sig_burnarea_tot[mindx]= mean_burnarea_tot[mindx] + tot_h1sig_arr
                    low_1sig_burnarea_tot[mindx]= mean_burnarea_tot[mindx] - tot_l1sig_arr
                    if (mean_burnarea_tot[mindx] - tot_l1sig_arr) < 0: 
                        low_1sig_burnarea_tot[mindx]= 0
        except KeyError:
            n_fires= 0

        reg_indx_arr= (r+1)*np.ones(tot_months, dtype= np.int64)
        reg_size_df= pd.concat([reg_size_df, pd.DataFrame({'mean_size': mean_burnarea_tot, 'low_1sig_size': low_1sig_burnarea_tot, 'high_1sig_size': high_1sig_burnarea_tot, \
                                                                    'month': pred_mon_arr, 'reg_indx': reg_indx_arr})], axis= 0, ignore_index=True)
        
    return reg_size_df.astype({'month': 'int64'})

def baseline_mon_arr_func(start_yr= 2000, end_yr= 2020, mindx= 6, base_yr= 1984):

    """
    Function to calculate the baseline months for a given month or list of months

    start_yr: start year for baseline period
    end_yr: end year for baseline period
    mindx: month index or list of month indices
    """

    baselinemons= pd.date_range(start='%s-01-01'%start_yr, end='%s-12-01'%end_yr, freq='MS')
    if isinstance(mindx, (list, tuple, np.ndarray)):
        junoctmons= baselinemons[baselinemons.month.isin(mindx)]
    else:
        junoctmons= baselinemons[baselinemons.month.isin([mindx])]
    baseline_mon_arr= junoctmons.year*12 + junoctmons.month - (base_yr*12 + 1)

    return baseline_mon_arr

def obs_fire_freq_climatology(fire_freq_df, pred_mons, regindx, start_yr= 2000, end_yr= 2020):
    
    """
    Function for calculating the observed fire frequency climatology for a given region and months

    fire_freq_df: dataframe containing fire frequency data
    pred_mons: list of months to be predicted (e.g. [6, 7, 8, 9, 10] for Jun-Oct)
    regindx: region index
    start_yr: start year for baseline period
    end_yr: end year for baseline period
    """

    oba_clim_arr= []
    fire_freq_reg_df= fire_freq_df.groupby('reg_indx').get_group(regindx)
    for m in pred_mons:
        baseline_mon_arr= baseline_mon_arr_func(start_yr= start_yr, end_yr= end_yr, mindx= m)
        oba_clim_arr.append(np.floor(np.mean(fire_freq_reg_df[fire_freq_reg_df.month.isin(baseline_mon_arr)].groupby('month')['fire_freq'].sum())))
    
    return np.nan_to_num(oba_clim_arr)

def obs_burned_area_climatology(fire_size_df, pred_mons, regindx, start_yr= 2000, end_yr= 2020):

    """
    Function for calculating the observed burned area climatology for a given region and months

    fire_size_df: dataframe containing fire size data
    pred_mons: list of months to be predicted (e.g. [6, 7, 8, 9, 10] for Jun-Oct)
    regindx: region index
    start_yr: start year for baseline period
    end_yr: end year for baseline period
    """

    oba_clim_arr= []
    fire_size_reg_df= fire_size_df.groupby('reg_indx').get_group(regindx)
    for m in pred_mons:
        baseline_mon_arr= baseline_mon_arr_func(start_yr= start_yr, end_yr= end_yr, mindx= m)
        oba_clim_arr.append(np.mean(fire_size_reg_df[fire_size_reg_df.fire_month.isin(baseline_mon_arr)].groupby('fire_month')['fire_size'].sum()/1e6))
    
    return np.nan_to_num(oba_clim_arr)

def obs_burned_area_ts(fire_size_df, pred_mons, regindx):

    """
    Function for calculating the time series of observed burned area for hindcasts

    fire_size_df: dataframe containing fire size data
    pred_mons: list of months to be predicted (e.g. [461, 462, 463] for Jun-Aug 2022)
    regindx: region index
    """

    oba_pred_arr= []
    fire_size_reg_df= fire_size_df.groupby('reg_indx').get_group(regindx)
    for m in pred_mons:
        if len(fire_size_reg_df[fire_size_reg_df.fire_month == m]) == 0:
            oba_pred_arr.append(0)
        else:
            oba_pred_arr.append(fire_size_reg_df[fire_size_reg_df.fire_month == m].fire_size.sum()/1e6)
    
    return np.array(oba_pred_arr)


def ens_stat_forecast_func(climdf, pred_var, target_yr, samp_flag= True, debug= False):
    
    """
    Function to generate ensemble members from a statistical forecast for a given prediction variable, target year, and prediction months

    climdf: dataframe containing the climate data
    pred_var: variable to be predicted
    target_yr: target year for which the forecast is being made
    pred_mons: prediction months
    samp_flag: if True, return ensemble members; else, return mean and standard deviation of the ensemble members

    to-do: include separate models for m, jja, and s
    """
    pred_mons= np.array([856, 857, 858, 859]) - (2023 - target_yr)*12
    run_id_arr= ['normalized_lead1mo_seas_detrended', 'normalized_lead2mo_seas_detrended', 'normalized_lead3mo_seas_detrended', 'normalized_lead3mo_seas_detrended']
    mb_frac= 0.1
    n_lead_months_arr= [1, 2, 3, 3]

    lead1mons_fcast_arr= pred_mons - 1
    lead2mons_fcast_arr= pred_mons - 2
    lead3mons_fcast_arr= pred_mons - 3

    lead1mons_fcast_df= climdf[climdf['time'].isin(lead1mons_fcast_arr)].dropna().reset_index().drop(columns= ['index'])
    lead2mons_fcast_df= climdf[climdf['time'].isin(lead2mons_fcast_arr)].dropna().reset_index().drop(columns= ['index'])
    lead3mons_fcast_df= climdf[climdf['time'].isin(lead3mons_fcast_arr)].dropna().reset_index().drop(columns= ['index'])
    pred_arr_dynamic= {'Tmax': ['Tmax', 'RH', 'SM_0_100cm', 'PDSI'], 'Prec': ['Prec', 'Tmax', 'RH', 'SM_0_100cm', 'PDSI', 'CAPE'], \
                   'VPD': ['VPD', 'Tmax', 'Prec', 'RH', 'SM_0_100cm', 'PDSI'], 'FFWI': ['FFWI_max7', 'Tmax', 'Prec', 'RH', 'SM_0_100cm', 'PDSI'], \
                    'Tmin': ['Tmin', 'Tmax', 'Prec', 'RH', 'SM_0_100cm', 'PDSI']}

    tmax_xr= xarray.open_dataarray('../data/12km/climate/primary/tmax.nc') # for getting the lat/lon coordinates
    ind_nan= climdf.groupby('time').get_group(list(climdf.groupby('time').groups.keys())[0]).dropna().index
    len_xcoord= len(tmax_xr.X)
    len_ycoord= len(tmax_xr.Y)

    for i in range(len(pred_mons)):
        run_id= run_id_arr[i]
        n_lead_month= n_lead_months_arr[i]
        if 'lead1mo' in run_id.split('_'):
            leadmonsdf= lead1mons_fcast_df
            leadsmons_fcast_arr= np.sort(lead1mons_fcast_arr)
        elif 'lead2mo' in run_id.split('_'):
            leadmonsdf= lead2mons_fcast_df
            leadsmons_fcast_arr= np.sort(lead2mons_fcast_arr)
        elif 'lead3mo' in run_id.split('_'):
            leadmonsdf= lead3mons_fcast_df
            leadsmons_fcast_arr= np.sort(lead3mons_fcast_arr)

        ngb_path= 'ngb_%s'%pred_var + '_%s'%run_id + '_mb_%.2f.p'%(mb_frac)
        file_path = Path.home()/'Desktop/seasonal_fire_pred/sav_files/ngb_mods'/ngb_path
        with file_path.open("rb") as f:
            ngb= pickle.load(f)

        Y_dist_loc= []
        Y_dist_scale= []
        
        m= leadsmons_fcast_arr[i]
        #if debug:
        #    return leadmonsdf
        leadmonsdf_groups= leadmonsdf.groupby(['time'], as_index= False)
        X_test= leadmonsdf_groups.get_group(m)
        Y_test_std= climdf.dropna().groupby('time').get_group(m - (target_yr == 2023)*12 + n_lead_month)[pred_var + '_std'] # consider std and trend values of the same month in the previous year # - (12 - n_lead_months)
        Y_test_mean= climdf.dropna().groupby('time').get_group(m - (target_yr == 2023)*12 + n_lead_month)[pred_var + '_trend']# ensure that Y is always the target variable and not a predictor

        X_test_scaled= pd.concat([X_test[pred_arr_dynamic[pred_var]], X_test[['Elev', 'nino34', 'mei', 'rmm1', 'rmm2']]], axis= 1).replace(np.nan, 0)
        poly = PolynomialFeatures(1)
        X_test_poly= poly.fit_transform(X_test_scaled)
            
        tmplocarr= np.ones(len_xcoord*len_ycoord)*np.nan
        tmpsigarr= np.ones(len_xcoord*len_ycoord)*np.nan
        tmplocarr[ind_nan]= ngb.pred_dist(X_test_poly).params['loc']*Y_test_std  #+ Y_test_mean
        tmpsigarr[ind_nan]= ngb.pred_dist(X_test_poly).params['scale']*Y_test_std
        Y_dist_loc.append(tmplocarr)
        Y_dist_scale.append(tmpsigarr)

        if samp_flag:
            if i== 0:
                xarr_pred_samp_tyear= xarray.DataArray(np.random.normal(Y_dist_loc, Y_dist_scale).reshape(len_ycoord, len_xcoord), \
                                                        coords= {'Y': tmax_xr.Y, 'X': tmax_xr.X}, dims= ['Y', 'X'])
            else:
                xarr_pred_samp_tyear= xarray.concat([xarr_pred_samp_tyear, xarray.DataArray(np.random.normal(Y_dist_loc, Y_dist_scale).reshape(len_ycoord, len_xcoord), \
                                                        coords= {'Y': tmax_xr.Y, 'X': tmax_xr.X}, dims= ['Y', 'X'])], dim= 'time')
        else:
            if i== 0:
                xarr_pred_loc_tyear= xarray.DataArray(np.array(Y_dist_loc).reshape(len_ycoord, len_xcoord), \
                                                        coords= {'Y': tmax_xr.Y, 'X': tmax_xr.X}, dims= ['Y', 'X'])
                xarr_pred_scale_tyear= xarray.DataArray(np.array(Y_dist_scale).reshape(len_ycoord, len_xcoord), \
                                                        coords= {'Y': tmax_xr.Y, 'X': tmax_xr.X}, dims= ['Y', 'X'])
            else:
                xarr_pred_loc_tyear= xarray.concat([xarr_pred_loc_tyear, xarray.DataArray(np.array(Y_dist_loc).reshape(len_ycoord, len_xcoord), \
                                                        coords= {'Y': tmax_xr.Y, 'X': tmax_xr.X}, dims= ['Y', 'X'])], dim= 'time')
                xarr_pred_scale_tyear= xarray.concat([xarr_pred_scale_tyear, xarray.DataArray(np.array(Y_dist_scale).reshape(len_ycoord, len_xcoord), \
                                                        coords= {'Y': tmax_xr.Y, 'X': tmax_xr.X}, dims= ['Y', 'X'])], dim= 'time')
    
    if samp_flag:
        return xarr_pred_samp_tyear
    else:
        return xarr_pred_loc_tyear, xarr_pred_scale_tyear


def fire_activity_ensemble_ssf(tot_ens_mems= 51, target_yr= 2022, firemon_pred_flag= 'dynamical_forecasts', fcast_type= 'dual_init', \
                                                    pred_var_arr= ['Tmax', 'Prec', 'VPD', 'Tmin'], freq_id= '08_07_23', seed= 654, size_id= '08_21_23', debug= False):
    """
    Function to generate SSF fire frequency and size forecasts for all members of a dynamical or statistical forecast ensemble

    tot_ens_mems: total number of ensemble members
    target_yr: target year for which forecasts are generated
    firemon_pred_flag: flag to indicate whether the forecasts are generated from dynamical forecasts ('dynamical_forecasts') or statistical forecasts ('statistical_forecasts')
    fcast_type: flag to indicate whether the forecasts are generated from a single initialization ('single_init') or dual initialization ('dual_init') forecast ensemble
    pred_var_arr: list of climate predictors used to generate the forecasts
    freq_id: identifier for the fire frequency model
    seed: seed for the fire frequency model
    size_id: identifier for the fire size model
    """
    
    ## Loading observed and forecast climate predictors
    
    if target_yr < 2023:
        clim_df= pd.read_hdf('../data/clim_fire_freq_12km_w2022_rescaled_data.h5')
        sys_no= None
    else:
        clim_df= pd.read_hdf('../data/clim_fire_freq_12km_w%s_rescaled_data.h5'%target_yr)
        sys_no= 51
    clim_df.loc[clim_df[clim_df.fire_freq > 1].index, 'fire_freq']= np.ones(len(clim_df[clim_df.fire_freq > 1].index), dtype= np.int64)
    pred_drop_cols= ['SWE_mean', 'SWE_max', 'AvgSWE_3mo']
    n_features= 43 - len(pred_drop_cols)

    if fcast_type == 'dual_init':
        pred_mon_arr=  np.array([460, 461, 462, 463, 464]) - (2022 - target_yr)*12
        start_month= 5 # index of start month with January = 1
        init_month_arr= [4, 6]
    else:
        pred_mon_arr=  np.array([461, 462, 463]) - (2022 - target_yr)*12
        start_month= 6 # index of start month with January = 1
        init_month_arr= [5]
    
    start_year= 1984
    end_year= 2019
    tot_years= end_year - start_year + 1
    tot_months= len(pred_mon_arr)
    seas_mon_arr= baseline_mon_arr_func(start_yr= start_year, end_yr= end_year, mindx= np.arange(0, tot_months, 1) + start_month)

    mdn_freq_train_ur_df= pd.read_hdf('../sav_files/fire_freq_pred_dfs/mdn_ds_%s'%freq_id + '_mon_fire_freq_%s.h5'%seed)
    mdn_freq_train_ur_df= mdn_freq_train_ur_df.reset_index().rename(columns= {'index': 'month'})
    mdn_freq_train_df= mdn_freq_train_ur_df[mdn_freq_train_ur_df.month.isin(seas_mon_arr)].reset_index(drop= True)
    if seed == None:
        mdn_zipd= tf.keras.models.load_model('../sav_files/fire_freq_mods/mdn_ds_%s'%freq_id, custom_objects= {'zipd_loss': zipd_loss, 'zipd_accuracy': zipd_accuracy})
    else:
        mdn_zipd= tf.keras.models.load_model('../sav_files/fire_freq_mods/mdn_ds_rs_%s'%freq_id + '_%s'%seed, custom_objects= {'zipd_loss': zipd_loss, 'zipd_accuracy': zipd_accuracy})
    
    nregions= 18
    X_sizes_train, X_sizes_val, y_sizes_train, y_sizes_val, fire_size_train, fire_size_test, X_sizes_test, y_sizes_test= fire_size_data(res= '12km', \
                                dropcols= drop_col_func(mod_type= 'normal', add_var_flag= True, add_var_list= ['SWE_max', 'SWE_mean', 'AvgSWE_3mo', 'Delta_T']), \
                                start_month= 444, tot_test_months= 24, threshold= 4, scaled= True, tflag= True, final_year= 2022) #tflag= True; scaled= True, rh_flag= True
    X_sizes_train_df= pd.concat([X_sizes_train, X_sizes_val], sort= False).reset_index().drop(columns=['index'])
    X_sizes_tot= pd.concat([X_sizes_train_df, X_sizes_test], sort= False).reset_index().drop(columns=['index'])
    fire_size_tot= pd.concat([fire_size_train, fire_size_test], sort= False).reset_index().drop(columns=['index'])

    max_fire_train_arr= []
    sum_fire_train_arr= []
    for r in range(nregions):
        max_fire_train_arr.append(np.max(np.concatenate([fire_size_train.groupby('reg_indx').get_group(r+1).groupby('fire_month').get_group(k).fire_size.to_numpy()/1e6 \
                                        for k in fire_size_train.groupby('reg_indx').get_group(r+1).groupby('fire_month').groups.keys()])))
        #sum_fire_train_arr.append(np.max([np.sum(fire_size_train.groupby('reg_indx').get_group(r+1).groupby('fire_month').get_group(k).fire_size.to_numpy()/1e6) \
        #                                for k in fire_size_train.groupby('reg_indx').get_group(r+1).groupby('fire_month').groups.keys()]))
        
    max_fire_train_arr= np.asarray(max_fire_train_arr)
    sum_fire_train_arr= max_fire_size_sum_func(fire_size_df= fire_size_tot, final_month= (target_yr + 1 - 1984)*12) #update final month for 2023!

    if firemon_pred_flag == 'statistical_forecasts':
        climdf= pd.read_hdf('../data/clim_12km_1952_2023_data.h5')
        input_var_arr= ['Tmax', 'Solar', 'VPD', 'Tmin', 'Prec', 'RH', 'SM_0_100cm', 'PDSI', 'FFWI_max7', 'CAPE']
        scaling_flag= 'trend_w_seasonal_cycle' # 'normalized', 'trend' , 'trend_w_seasonal_cycle'
        trend_mons= 700 #700 --> 2010; 792 --> 2018
        tstart_mon= 0  # 336 --> 1980; 468 --> 1991
        tot_months= len(climdf.month.unique())
        if scaling_flag == 'trend_w_seasonal_cycle':
            start_mon= 2
            end_mon= 8

        clim_xarr= clim_xarr_init(climdf, input_var_arr, scaling_flag, tstart_mon, trend_mons, start_mon, end_mon)

        input_var_arr= np.append([i for i in clim_xarr.data_vars], ['Elev', 'time', 'X', 'Y', 'mei', 'nino34', 'rmm1', 'rmm2']) #'Southness', 
        if scaling_flag == 'trend_w_seasonal_cycle':
            totmonarr= np.sort(np.hstack([np.arange(m, 857, 12) for m in range(start_mon - 1, end_mon)]))
            climdf= pd.concat([clim_xarr.to_dataframe().reset_index(), climdf[climdf.month.isin(totmonarr)][['Elev', 'mei', 'nino34', 'rmm1', 'rmm2']].reset_index(drop= True)], axis= 1) #'Southness', 
        else:
            climdf= pd.concat([clim_xarr.to_dataframe().reset_index(), climdf[['Elev', 'mei', 'nino34', 'rmm1', 'rmm2']]], axis= 1) #'Southness',
        climdf.time= (climdf.time.dt.year*12 + climdf.time.dt.month) - (1952*12 + 1)
    
    for ens_no in tqdm(range(tot_ens_mems)):
        if firemon_pred_flag == 'statistical_forecasts':
            if target_yr < 2023:
                fire_freq_df= clim_df[clim_df.month.isin(pred_mon_arr)]['fire_freq'].reset_index(drop= True)
            else:
                fire_freq_df= pd.DataFrame({'fire_freq': np.zeros(len(tmax_xr[0].values.flatten())*len(pred_mon_arr), dtype= np.int64)})
            tmpdf= clim_df[clim_df.month.isin([pred_mon_arr[0]])].drop(columns= pred_var_arr).drop(columns= 'fire_freq').reset_index(drop= True)
            obsdf= pd.concat([tmpdf.replace(pred_mon_arr[0], m) for m in pred_mon_arr], ignore_index= True) 

            seas_anomalies_df= pd.DataFrame([])
            pred_var_tot_df= pd.DataFrame([])
            for pred_var in tqdm(pred_var_arr):
                stat_fcast_anomalies_conus= ens_stat_forecast_func(climdf= climdf, pred_var= pred_var, target_yr= target_yr, samp_flag= True)
                pred_var_df= stat_fcast_anomalies_conus.to_dataframe(pred_var).reset_index()[pred_var]
                pred_var_df/= pred_var_df.std() # standardizing the dynamical forecasts
                pred_var_tot_df= pd.concat([pred_var_tot_df, pred_var_df], axis=1, ignore_index= True)
            seas_anomalies_df= pd.concat([seas_anomalies_df, pred_var_tot_df], axis= 0, ignore_index= True)
            seas_anomalies_df.rename(columns= {i: pred_var_arr[i] for i in range(len(pred_var_arr))}, inplace= True)
            obsdf= pd.concat([obsdf, fire_freq_df], axis= 1)
            
            X_pred_ur_df= pd.concat([seas_anomalies_df, obsdf], axis= 1)
            X_pred_ur_df.drop(columns= pred_drop_cols, inplace= True)
            X_pred_df= X_pred_ur_df[X_pred_ur_df.columns].dropna()
        elif firemon_pred_flag == 'dynamical_forecasts':
            # Downscaling, regridding, and interpolating dynamical forecasts to match APW's 12km grid
            
            sub = (51.6, -128, 26.5, -101) # North/West/South/East
            ds_out = xr.Dataset(
                {
                    "lat": (["lat"], np.arange(26.5, 51.6, 0.125), {"units": "degrees_north"}),
                    "lon": (["lon"], np.arange(-128, -101, 0.125), {"units": "degrees_west"}),
                }
                )
            tmax_xr= xarray.open_dataarray('../data/12km/climate/%s/primary/tmax.nc'%target_yr)
            x_fire_grid= xr.DataArray(coord_transform(tmax_xr.X.values, tmax_xr.Y.values, "epsg:5070", "epsg:4326")[0], dims=('Y','X'))
            y_fire_grid= xr.DataArray(coord_transform(tmax_xr.X.values, tmax_xr.Y.values, "epsg:5070", "epsg:4326")[1], dims=('Y','X'))

            if target_yr < 2023:
                fire_freq_df= clim_df[clim_df.month.isin(pred_mon_arr)]['fire_freq'].reset_index(drop= True)
            else:
                fire_freq_df= pd.DataFrame({'fire_freq': np.zeros(len(tmax_xr[0].values.flatten())*len(pred_mon_arr), dtype= np.int64)})
            seas_anomalies_df= pd.DataFrame([])
            obsdf= pd.DataFrame([])
            for indx in range(len(init_month_arr)):
                # repeats the dynamical forecasts for the first 3/last 2 months of the fire season depending on the initialization month
                tmpdf= clim_df[clim_df.month.isin([pred_mon_arr[indx] - 1])].drop(columns= pred_var_arr).drop(columns= 'fire_freq').reset_index(drop= True)
                if indx== 0:
                    tmppredarr= pred_mon_arr[0:3]
                else:
                    tmppredarr= pred_mon_arr[3:5]
                tmpdf= pd.concat([tmpdf.replace(pred_mon_arr[indx] - 1, m) for m in tmppredarr], ignore_index= True) 
                obsdf= pd.concat([obsdf, tmpdf], axis= 0, ignore_index= True)
            
                pred_var_tot_df= pd.DataFrame([])
                for pred_var in pred_var_arr:
                    seas5_anomalies_conus_regridded= seas5_monthly_anomaly_func(pred_var= pred_var, system= sys_no, fyear= target_yr, init_month= init_month_arr[indx], subarea= sub, regrid_scheme= 'bilinear', \
                                                                                                                                                            dsout= ds_out, anom_type= 'raw')
                    seas5_anomalies_conus_regridded= seas5_anomalies_conus_regridded.interp({'lat':y_fire_grid, 'lon':x_fire_grid}, method='linear').load()
                    seas5_anomalies_conus_regridded= seas5_anomalies_conus_regridded.assign_coords({'X': (('X'), tmax_xr.X.data, {"units": "meters"}), \
                                                                                                                'Y': (('Y'), tmax_xr.Y.data, {"units": "meters"})}).drop_vars(['lat','lon'])
                    pred_var_df= seas5_anomalies_conus_regridded[ens_no][indx+1:4].where(~np.isnan(tmax_xr[0].drop('time'))).to_dataframe(pred_var).reset_index()[pred_var]
                    pred_var_df/= pred_var_df.std() # standardizing the dynamical forecasts
                    pred_var_tot_df= pd.concat([pred_var_tot_df, pred_var_df], axis=1, ignore_index= True)
                seas_anomalies_df= pd.concat([seas_anomalies_df, pred_var_tot_df], axis= 0, ignore_index= True)
            seas_anomalies_df.rename(columns= {i: pred_var_arr[i] for i in range(len(pred_var_arr))}, inplace= True)
            obsdf= pd.concat([obsdf, fire_freq_df], axis= 1)
            
            X_pred_ur_df= pd.concat([seas_anomalies_df, obsdf], axis= 1)
            X_pred_ur_df.drop(columns= pred_drop_cols, inplace= True)
            X_pred_df= X_pred_ur_df[X_pred_ur_df.columns].dropna()

            X_pred_test_df= X_pred_df[X_pred_df.iloc[:, 0:n_features].columns]
            X_pred_test_df.loc[:, 'reg_indx']= X_pred_df.reg_indx
            X_pred_test_df.loc[:, 'month']= X_pred_df.month
            X_pred_test_df= X_pred_test_df.drop(columns=['Solar', 'Ant_Tmax', 'RH', 'Ant_RH', 'RH_min3', 'FFWI_max7', 'Avgprec_4mo', 'Avgprec_2mo', 'AvgVPD_4mo', 'AvgVPD_2mo', \
                                                                'Tmax_max7', 'VPD_max7', 'Tmin_max7', 'Elev', 'Delta_T', 'CAPE', 'Southness'])
        
        ## Function to generate a spatial map of fire probability forecasts from a saved MDN model and save the resultant xarray for post-processing

        fire_prob_pred_func(freq_id= freq_id, seed= seed, X_tot_df= X_pred_ur_df, X_test_df= X_pred_test_df, pred_mon_arr= pred_mon_arr, sav_flag= True, target_year= target_yr, \
                                                                                                                            firemon_pred_flag= firemon_pred_flag, ens_no= (ens_no))
        
        ## Fire frequency trends and locations

        mdn_freq_test_df= grid_ssf_freq_predict(X_test_dat= X_pred_test_df, freq_test_df= X_pred_df, n_regs= nregions, ml_model= mdn_zipd, func_flag= 'zipd', \
                                                                                        pred_mon_flag= True, pred_mons= pred_mon_arr, loc_flag= False, rseed= 87)
        rescale_fac_df= calib_ssf_freq_predict(freq_train_df= mdn_freq_train_df, freq_test_df= mdn_freq_test_df, n_regs= nregions, n_train_years= tot_years, \
                                                                                        n_pred_mons= tot_months, input_type= 'mean', pred_type= 'mean', regtype= 'linear')
        scale_fac= np.array(rescale_fac_df['pred_obs_freq']/rescale_fac_df['pred_freq'])
        scale_fac[np.isinf(scale_fac)]= 0
        scale_fac[scale_fac < 0]= 0

        mdn_freq_test_df['pred_mean_freq']= np.floor(mdn_freq_test_df['pred_mean_freq']*scale_fac)
        mdn_freq_test_df['pred_high_2sig']= np.floor(mdn_freq_test_df['pred_high_2sig']*scale_fac)
        mdn_freq_test_df['pred_low_2sig']= np.floor(mdn_freq_test_df['pred_low_2sig']*scale_fac)

        freq_loc_df= grid_ssf_freq_predict(X_test_dat= X_pred_test_df, freq_test_df= X_pred_df, n_regs= 18, ml_model= mdn_zipd, func_flag= 'zipd', pred_mon_flag= True, \
                                                                                                                pred_mons= pred_mon_arr, loc_flag= True, rseed= 87) # df of predicted frequency locations averaged over 1000 samples

        if firemon_pred_flag == 'observations':
            mdn_freq_test_df.to_hdf('../sav_files/fire_freq_pred_dfs/mdn_ssf_%s'%freq_id + '_%d'%seed +  '_fire_freq_%d'%target_yr + '_%s.h5'%firemon_pred_flag, key= 'df', mode= 'w')
            freq_loc_df.to_hdf('../sav_files/fire_freq_pred_dfs/freq_loc_ssf_%s'%freq_id + '_%d'%seed +  '_fire_freq_%d'%target_yr + '_%s.h5'%firemon_pred_flag, key= 'df', mode= 'w')
        else:
            mdn_freq_test_df.to_hdf('../sav_files/fire_freq_pred_dfs/' + '%s/'%firemon_pred_flag + '%s/'%target_yr + 'mdn_ssf_%s'%freq_id + '_%d'%seed +  '_fire_freq_%d'%(ens_no) + '_%d.h5'%target_yr, key= 'df', mode= 'w')
            freq_loc_df.to_hdf('../sav_files/fire_freq_pred_dfs/' + '%s/'%firemon_pred_flag + '%s/'%target_yr + 'freq_loc_ssf_%s'%freq_id + '_%d'%seed +  '_fire_freq_%d'%(ens_no) + '_%d.h5'%target_yr, key= 'df', mode= 'w')
        
        for freqlabel in ['pred_mean_freq', 'pred_high_2sig']: # 'pred_mean_freq', 'pred_high_2sig', 'pred_low_2sig'
            pred_loc_arr= loc_ind_ssf_func(loc_df= freq_loc_df, ml_freq_df= mdn_freq_test_df, X_test_dat= X_pred_test_df, pred_mon_flag= True, pred_mons= pred_mon_arr, freqlabel= freqlabel)
            X_pred_ur_df['pred_fire_freq']= np.zeros_like(X_pred_ur_df['fire_freq'])
            for r in range(18):
                X_pred_ur_df.loc[X_pred_ur_df.groupby('reg_indx').get_group(r+1).index, 'pred_fire_freq']= 0

            for ind in np.hstack(pred_loc_arr):
                X_pred_ur_df.loc[ind, 'pred_fire_freq']+=1 

            nan_ind_arr= X_pred_ur_df['Tmax'].isna()
            X_pred_ur_df.loc[nan_ind_arr, 'pred_fire_freq']= np.nan
            if firemon_pred_flag == 'observations':
                X_pred_ur_df.to_hdf('../sav_files/ssf_pred_files/%s_dataframe'%freqlabel + '_%s'%freq_id + '_%d'%seed +  '_obs_%d.h5'%target_yr, key= 'df', mode= 'w')
            else:
                X_pred_ur_df.to_hdf('../sav_files/ssf_pred_files/' + '%s'%firemon_pred_flag + '/%s_dataframe'%freqlabel + '_%s'%freq_id + '_%d'%seed +  '_%d'%(ens_no) +\
                                                                                                                                             '_%d.h5'%target_yr, key= 'df', mode= 'w')
                        
            ## Fire size trends and locations

            mdn_gpd_mod= tf.keras.models.load_model('../sav_files/fire_size_mods/mdn_gpd_size_model_%s'%size_id, custom_objects= {'gpd_loss': gpd_loss, 'gpd_accuracy': gpd_accuracy})
            mdn_gpd_ext_mod= tf.keras.models.load_model('../sav_files/fire_size_mods/mdn_gpd_ext_size_model_%s'%size_id, custom_objects= {'gpd_loss': gpd_loss, 'gpd_accuracy': gpd_accuracy})
            X_sizes_test_df= X_pred_ur_df.drop(columns= ['Solar', 'Ant_Tmax', 'RH', 'Ant_RH', 'RH_min3', 'FFWI_max7', 'Avgprec_4mo',  'Avgprec_2mo', 'AvgVPD_4mo', 'AvgVPD_2mo', 'Tmax_max7', \
                                                                                    'VPD_max7', 'Tmin_max7', 'Elev', 'Delta_T', 'CAPE', 'Southness', 'X', 'Y', 'fire_freq', 'pred_fire_prob'])
            
            for sizelabel in ['gpd', 'gpd_ext']:
                if sizelabel == 'gpd_ext':
                    reg_gpd_ml_pred_size_df= grid_ssf_size_func(mdn_model= mdn_gpd_ext_mod, stat_model= gpd_model, max_size_arr= max_fire_train_arr, sum_size_arr= sum_fire_train_arr, pred_mon_flag= True, pred_mons= pred_mon_arr, \
                                                        nsamps= 1000, loc_df= freq_loc_df, ml_freq_df= mdn_freq_test_df, X_test_dat= X_sizes_test_df)
                elif sizelabel == 'gpd':
                    reg_gpd_ml_pred_size_df= grid_ssf_size_func(mdn_model= mdn_gpd_mod, stat_model= gpd_model, max_size_arr= max_fire_train_arr, sum_size_arr= sum_fire_train_arr, pred_mon_flag= True, pred_mons= pred_mon_arr, \
                                                        nsamps= 1000, loc_df= freq_loc_df, ml_freq_df= mdn_freq_test_df, X_test_dat= X_sizes_test_df)
                if firemon_pred_flag == 'observations':
                    reg_gpd_ml_pred_size_df.to_hdf('../sav_files/fire_size_pred_dfs/pred_size_df_ml_gpd_%s'%size_id + '_%s'%('_'.join(freqlabel.split('_')[1:])) \
                                                                                                                    + '_%s'%sizelabel + '_%s.h5'%target_yr, key= 'df', mode= 'w')
                else:
                    reg_gpd_ml_pred_size_df.to_hdf('../sav_files/fire_size_pred_dfs/' + '%s/'%firemon_pred_flag + '%s/'%target_yr + 'pred_size_df_ml_gpd_%s'%size_id + '_%s'%('_'.join(freqlabel.split('_')[1:])) \
                                                                                                                + '_%s'%sizelabel + '_%d'%(ens_no) + '_%s.h5'%target_yr, key= 'df', mode= 'w')
