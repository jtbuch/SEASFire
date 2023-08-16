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

def grid_ssf_freq_predict(X_test_dat, freq_test_df= None, n_regs= 18, ml_model= None, func_flag= 'zipd', pred_mon_flag= True, pred_mons= None, seas_flag= None, final_month= None, rseed= 99):
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
    final_month: final month of the forecast period
    rseed: random seed
    '''
    if pred_mon_flag:
        pred_mon_arr= pred_mons
    else:
        if seas_flag == 'JJA':
            start_month= 5
        elif seas_flag == 'SON':
            start_month= 8
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
                ml_freq_df= ml_freq_df.append(pd.DataFrame({'pred_mean_freq': pred_freq_arr, 'pred_high_2sig': pred_high_2sig, 'pred_low_2sig': pred_low_2sig, \
                                                                                                                                    'month': pred_mon_arr, 'reg_indx': reg_indx_arr}))
            else:
                obs_freqs= [np.sum(freq_test_mon_groups.get_group(m).fire_freq) for m in pred_mon_arr]
                ml_freq_df= ml_freq_df.append(pd.DataFrame({'obs_freq': obs_freqs, 'pred_mean_freq': pred_freq_arr, 'pred_high_2sig': pred_high_2sig, 'pred_low_2sig': pred_low_2sig, \
                                                                                                                                        'reg_indx': reg_indx_arr}))

        elif func_flag == 'logistic':
            obs_freqs= [np.sum(freq_test_mon_groups.get_group(m).fire_freq) for m in pred_mon_arr]
            ml_freq_df= ml_freq_df.append(pd.DataFrame({'obs_freq': obs_freqs, 'pred_mean_freq': pred_freq_arr, 'reg_indx': reg_indx_arr}))
                
    return ml_freq_df.reset_index(drop= True) #, tot_rfac_arr

def calib_ssf_freq_predict(freq_train_df, freq_test_df, n_regs, n_train_years, n_pred_mons, input_type= 'std', pred_type= 'std', regtype= 'polynomial'):
    """
    Derives a calibration factor by using a linear model to predict the annual std/mean of monthly observed frequencies with annual std/mean of monthly predicted frequencies
    
    """
    freq_train_groups= freq_train_df.groupby('reg_indx')
    freq_test_groups= freq_test_df.groupby('reg_indx')
    seas_train_arr= np.linspace(0, n_train_years*3, n_train_years + 1, dtype=int)
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

        pred_norm_df= pred_norm_df.append(pd.DataFrame({'month':freq_test_groups.get_group(r+1).month, 'reg_indx': np.ones(len(seas_test_arr), dtype= np.int64)*(r+1), \
                                                                    'pred_freq': np.repeat(X_mat, len(seas_test_arr)), 'pred_obs_freq': np.repeat(pred_norm, len(seas_test_arr))}))    
    
    return pred_norm_df #, r_pred