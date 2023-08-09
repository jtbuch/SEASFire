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