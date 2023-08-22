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
# import xesmf as xe # for regridding

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
    tot_months= len(pd.date_range(start='1952-01-01', end= xarr_end_date, freq='MS'))
    
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

def seas5_monthly_anomaly_func(pred_var, system= None, fyear= 2021, init_month= 5, area_flag= True, subarea= None, regrid_flag= True, regrid_scheme= 'bilinear', dsout= None):
    
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
    """
    
    if pred_var == 'tp':
        pred_var_name= 'tprate'
    elif pred_var == 'tmax':
        pred_var_name= 'mx2t24'
    if system == None:
        ds_hindcast= xr.open_dataset(f'{DATADIR}/ecmwf_seas5_1993-2016_amj_hindcast_monthly_%s.grib'%pred_var, engine='cfgrib', backend_kwargs=dict(time_dims=('forecastMonth', 'time')))
        seas5_forecast = xr.open_dataset(f'{DATADIR}/ecmwf_seas5_%s'%fyear + '_amj_forecast_monthly_%s.grib'%pred_var, engine='cfgrib', 
                                    backend_kwargs=dict(time_dims=('forecastMonth', 'time')))
    else:
        ds_hindcast= xr.open_dataset(f'{DATADIR}/ecmwf_seas5_51_1993-2016_amj_hindcast_monthly_%s.grib'%pred_var, engine='cfgrib', backend_kwargs=dict(time_dims=('forecastMonth', 'time')))
        seas5_forecast = xr.open_dataset(f'{DATADIR}/ecmwf_seas5_51_%s'%fyear + '_amj_forecast_monthly_%s.grib'%pred_var, engine='cfgrib', 
                                    backend_kwargs=dict(time_dims=('forecastMonth', 'time')))
    pred_var_hindcast = ds_hindcast[pred_var_name]
    pred_var_hindcast = pred_var_hindcast.groupby('time.month')[init_month].mean(['number', 'time'])

    seas5_anomalies= seas5_forecast.groupby('time.month')[init_month][pred_var_name] - pred_var_hindcast
    valid_time = [pd.to_datetime(seas5_anomalies.time.values[0]) + relativedelta(months=fcmonth-1) for fcmonth in seas5_anomalies.forecastMonth]
    seas5_anomalies= seas5_anomalies.assign_coords(valid_time=('forecastMonth',valid_time))
    numdays = [monthrange(dd.year,dd.month)[1] for dd in valid_time]
    seas5_anomalies = seas5_anomalies.assign_coords(numdays=('forecastMonth',numdays))
    if pred_var == 'tp':
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

    elif pred_var == 'tmax':
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
        
def fire_prob_pred_func(freq_id= None, seed= None, X_tot_df= None, X_test_df= None, pred_mon_arr= None, sav_flag= False, target_year= None):
    
    """ 
    Function to predict fire probability for a given month
    
    freq_id, seed: id and seed of trained ML freq model
    X_tot_df: dataframe containing all predictor variables including nan
    X_test_df: dataframe containing non-nan predictor variables
    pred_mon_arr: array containing months to predict
    sav_flag: flag to save predictions
    target_year: year for which predictions are made
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

    X_tot_df['pred_fire_prob']= np.zeros_like(X_tot_df['fire_freq'], dtype= np.float32)
    X_tot_df.loc[~X_tot_df['Elev'].isna(), 'pred_fire_prob']= param_vec[:, 1] # choose grid cells where Elev is not NaN because it has the fewest NaNs
    X_tot_df.loc[X_tot_df['Elev'].isna(), 'pred_fire_prob']= np.nan

    pred_prob_xarr= xarray.DataArray(data= X_tot_df['pred_fire_prob'].to_numpy().reshape(len(pred_mon_arr), 208, 155),
        dims=["month", "Y", "X"],
        coords=dict(
            X=(["X"], np.linspace(-2349250, -501250, 155)),
            Y=(["Y"], np.linspace(3166500, 682500, 208)),
            time= (["month"], np.linspace(0, len(pred_mon_arr) - 1, len(pred_mon_arr), dtype= np.int64)),),)
    
    if sav_flag:
        pred_prob_xarr.to_netcdf('../sav_files/ssf_pred_files/pred_prob_xarr_rs_%s'%freq_id + '_%d_'%seed + 'obs_%s.nc'%target_year)
        return print('Saved fire probability xarray for %s'%target_year)
    else:
        return pred_prob_xarr
    
def mon_fire_prob_pred(freq_id= '08_07_23', seed= 654, plot_yr= 2019, fmon= 5, fire_df= None):
    """
    Function to rescale predicted fire probability with climatological baseline fire probability and observed number of fires

    freq_id, seed: id and seed of trained ML freq model
    plot_yr: year for which predictions are made
    fmon: month for which predictions are made
    fire_df: dataframe containing observed number of fires
    """
    
    pred_prob_xarr= xarray.open_dataarray('../sav_files/ssf_pred_files/pred_prob_xarr_rs_%s'%freq_id + '_%d_'%seed + 'obs_%s.nc'%plot_yr)
    n_fires_yr= len(fire_df[fire_df['fire_month'] == (plot_yr - 1984)*12 + fmon])
    baseline_arr= np.arange(209, 426, 12)
    n_fires_baseline= len(fire_df[fire_df['fire_month'].isin(baseline_arr + (fmon - 5))])/20

    pred_prob_baseline= xarray.open_dataarray('../sav_files/ssf_pred_files/pred_prob_xarr_rs_%s'%freq_id + '_%d_'%seed + 'obs_baseline.nc')
    pred_prob_xarr_baseline= pred_prob_baseline[(fmon - 5)::3, :, :].mean(dim= 'month')
        
    return 10**(np.log10(pred_prob_xarr[(fmon - 5), :, :]) - np.log10(pred_prob_xarr_baseline) - np.log10(n_fires_baseline/n_fires_yr))

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
                    ml_freq_df= ml_freq_df.append(pd.DataFrame({'freq_loc_arr': freq_arr, 'month': pred_mon_arr, 'reg_indx': reg_indx_arr}))
                else:
                    ml_freq_df= ml_freq_df.append(pd.DataFrame({'pred_mean_freq': pred_freq_arr, 'pred_high_2sig': pred_high_2sig, 'pred_low_2sig': pred_low_2sig, \
                                                                                                                                    'month': pred_mon_arr, 'reg_indx': reg_indx_arr}))
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

        pred_norm_df= pred_norm_df.append(pd.DataFrame({'month': freq_test_groups.get_group(r+1).month, 'reg_indx': np.ones(len(seas_test_arr), dtype= np.int64)*(r+1), \
                                                                    'pred_freq': np.repeat(X_mat, len(seas_test_arr)), 'pred_obs_freq': np.repeat(pred_norm, len(seas_test_arr))}))    
    
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

def grid_size_ssf_func(mdn_model, stat_model, max_size_arr, sum_size_arr, pred_mon_flag= True, pred_mons= None, seas_flag= None, base_yr= 1984, fcast_yr= None, \
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

        reg_indx_arr= (r+1)*np.ones(tot_months, dtype= np.int64)
        reg_size_df= reg_size_df.append(pd.DataFrame({'mean_size': mean_burnarea_tot, 'low_1sig_size': low_1sig_burnarea_tot, 'high_1sig_size': high_1sig_burnarea_tot, \
                                                                                    'month': pred_mon_arr, 'reg_indx': reg_indx_arr}), ignore_index=True)
        
    return reg_size_df.astype({'month': 'int64'})

def baseline_mon_arr_func(start_yr= 2000, end_yr= 2020, mindx= 6):

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
    baseline_mon_arr= junoctmons.year*12 + junoctmons.month - (1984*12 + 1)

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