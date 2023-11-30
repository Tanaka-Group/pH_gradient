# -*- coding: utf-8 -*-
"""
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import leastsq
import seaborn as sns
from numba import jit

sigmoid_a   = 200

# x: height. 0 < x < 1
x_res = 0.001 # resolution of x
x = np.arange(-0.1, 1.01, x_res)

# basic functions
@jit('f8[:](f8[:],f8,f8)', nopython=True)
def sigmoid(x, a, b):
    return 1 / (1 + np.exp(-a * (x - b))) #a: slope, b: inflection point

def exp_fit(x,y):
    def exp_residual(param, x1, y1):
            residual = np.log(y1) - (np.log(param[0]) + param[1] * x1)
            return residual
    r1    =  leastsq(exp_residual, [1,1], args=(x, y))
    return r1[0][0], r1[0][1]

def poly6_fit(x,y):
    def poly6_residual(param, x1, y1):
            residual = y1 - (param[0]*x1**6 + param[1]*x1**5 + param[2]*x1**4 + param[3]*x1**3 + param[4]*x1**2 + param[5]*x1**1 + param[6])
            return residual
    r1    =  leastsq(poly6_residual, [0,0,0,0,0,0,0], args=(x, y))
    return r1[0][0], r1[0][1], r1[0][2], r1[0][3], r1[0][4], r1[0][5], r1[0][6]

def poly3_fit(x,y):
    def poly3_residual(param, x1, y1):
            residual = y1 - (param[0]*x1**3 + param[1]*x1**2 + param[2]*x1**1 + param[3])
            return residual
    r1    =  leastsq(poly3_residual, [0,0,0,0], args=(x, y))
    return r1[0][0], r1[0][1], r1[0][2], r1[0][3]

# pH, KD, pHact functions
@jit('f8[:](f8[:])', nopython=True)
def pH7_5(x):       #conventional
    a = 7 - 1.6*x 
    a = np.where(a > 7, 7, a)
    return a

@jit('f8[:](f8[:])', nopython=True)
def pH5(x):       # assume Tmem79 KO mice
    return 6.2 - 0.8*np.sign(x)

@jit('f8[:](f8[:])', nopython=True)
def pH7(x):         # assume Claudin-1 KO mice
    return 7 -x*0

@jit('f8[:](f8[:])', nopython=True)
def pH5_7step(x):
    return 7 - sigmoid(x, sigmoid_a, 0)*1 - sigmoid(x, sigmoid_a, 0.3)*0.6 + sigmoid(x, sigmoid_a, 0.8)*1.3

@jit('f8[:](f8[:])', nopython=True)
def pH5_7step1(x):
    return 7 - sigmoid(x, sigmoid_a, 0)*1.6 + sigmoid(x, sigmoid_a, 0.8)*1.3

@jit('f8[:](f8[:])', nopython=True)
def pH757v(x):      
    a = 5.4 + 3.2*np.abs(x-0.5) 
    a = np.where(a > 7, 7, a)
    a = np.where(x > 0.5, a - 0.6*(x-0.5), a)
    return a

def ka_KLK5(pH):
    x      = np.array([4.5,	    5.5,	6.5,	7.5])
    y      = np.array([9.91E+03,	3.06E+04,	3.44E+04,	5.71E+04])
    a, b    = exp_fit(x,y)
    ka      = a*np.exp(b*pH)
    return ka

def kd_KLK5(pH):
    x      = np.array([4.5,	    5.5,	6.5,	7.5])
    y      = np.array([8.04E-04,	3.38E-05,	1.83E-06,	8.70E-08])
    a, b    = exp_fit(x,y)
    kd      = a*np.exp(b*pH)
    return kd

def pHact_KLK5(pH):
    x          = np.array([4.1, 4.4, 4.7, 5.0, 5.3, 5.6, 5.8, 6.0, 6.2, 6.4, 6.6, 6.8, 7.0, 7.2, 7.4, 7.6, 7.8, 8.0])
    y          = np.array([0.068, 0.056,0.082,0.121,0.153,0.215,0.255,0.336,0.368,0.423,0.513,0.604,0.687,0.802,0.851,0.946,1.000,0.918])
    a, b, c, d, e, f, g    = poly6_fit(x, y)
    pHact   = a*pH**6 + b*pH**5 + c*pH**4 + d*pH**3 + e*pH**2 + f*pH**1 + g
    return pHact

def mesotrpsin_act(pH):
    x          = np.array([4.1, 4.4 ,4.7 ,5.0 ,5.3 ,5.6 ,5.8 ,6.0 ,6.2 ,6.4 ,6.6 ,6.8 ,7.0 ,7.2 ,7.4 ,7.6 ,7.8 ,8.0])
    y          = np.array([0.283,0.330,0.412,0.532,0.622,0.711,0.701,0.742,0.852,0.828,0.859,0.903,0.883,0.948,0.948,1.000,1.000,0.963])
    a, b, c, d    = poly3_fit(x,y)
    meso_act   = a*pH**3 + b*pH**2 + c*pH**1 + d
    return meso_act

# assumed pH profiles
def pH_plot():
    df = pd.DataFrame({'x'            : x,
                       'pH7_5'        : pH7_5(x),
                       'pH5'          : pH5(x),
                       'pH7'          : pH7(x),
                       'pH7_5_7step'  : pH5_7step(x)})

    fig, ax = plt.subplots(figsize = (3,3), dpi=300)
    sns.lineplot(data = df, x='x', y='pH7_5', label = 'pH7_5 (gradient)', color = 'black')
    sns.lineplot(data = df, x='x', y='pH5',   label = 'pH5 (Tmem79 KO)'  , color = 'blue')
    sns.lineplot(data = df, x='x', y='pH7',   label = 'pH7 (Claudin-1 KO)'  , color = 'purple')
    sns.lineplot(data = df, x='x', y='pH7_5_7step', label = 'pH7_5_7 (stepwise)', color = 'red')
    ax.set(xlabel ='Height_x (0:SG/SC, 1:SC surface)',ylabel='pH',\
           xlim=(-0.1, 1), ylim=(5.3,7.1))
    ax.set_xticks(np.linspace(-0, 1, 6))
    ax.set_yticks(np.linspace(5.4, 7, 9))
    ax.lines[0].set_linestyle('dashed')
    ax.lines[1].set_linestyle('dashdot')
    ax.lines[2].set_linestyle('dashdot')
    ax.lines[3].set_linestyle('solid')
    ax.legend(loc='best', bbox_to_anchor=(1, 1))
pH_plot()

# assumed pH profiles
def pH_plot():
    df = pd.DataFrame({'x'          : x,
                       'pH75'       : pH7_5(x),
                       'pH757v'     : pH757v(x),
                       'pH757s'  : pH5_7step(x)})

    fig, ax = plt.subplots(figsize = (3,3), dpi=300)
    sns.lineplot(data = df, x='x', y='pH75', label = 'pH7_5 (gradient)', color = 'black')
    sns.lineplot(data = df, x='x', y='pH757v', label = 'pH7_5_7 (gradient)' , color = 'green')
    sns.lineplot(data = df, x='x', y='pH757s', label = 'pH7_5_7 (stepwise)', color = 'red')
    ax.set(xlabel ='Height_x (0:SG/SC, 1:SC surface)',ylabel='pH',\
           xlim=(-0.1, 1), ylim=(5.3,7.1))
    ax.set_xticks(np.linspace(-0, 1, 6))
    ax.set_yticks(np.linspace(5.4, 7, 9))
    ax.lines[0].set_linestyle('dashed')
    ax.lines[1].set_linestyle('dashdot')
    ax.lines[2].set_linestyle('solid')
    ax.legend(loc='best', bbox_to_anchor=(1.5, 1))
pH_plot()

# assumed pH profiles
def pH_plot():
    df = pd.DataFrame({'x'            : x,
                       'pH5'          : pH5(x),
                       'pH7'          : pH7(x),
                       'pH7_5_7step'  : pH5_7step(x)})

    fig, ax = plt.subplots(figsize = (3,3), dpi=300)
    sns.lineplot(data = df, x='x', y='pH5',   label = 'pH5 (Tmem79 KO)'  , color = 'blue')
    sns.lineplot(data = df, x='x', y='pH7',   label = 'pH7 (Claudin-1 KO)'  , color = 'purple')
    sns.lineplot(data = df, x='x', y='pH7_5_7step', label = 'pH7_5_7 (stepwise)', color = 'red')
    ax.set(xlabel ='Height_x (0:SG/SC, 1:SC surface)',ylabel='pH',\
           xlim=(-0.1, 1), ylim=(5.3,7.1))
    ax.set_xticks(np.linspace(-0, 1, 6))
    ax.set_yticks(np.linspace(5.4, 7, 9))
    ax.lines[0].set_linestyle('dashdot')
    ax.lines[1].set_linestyle('dashdot')
    ax.lines[2].set_linestyle('solid')
    ax.legend(loc='best', bbox_to_anchor=(1.2, 1))
pH_plot()

# assumed pH profiles
def pH_plot():
    df = pd.DataFrame({'x'            : x,
                       'pH757step1' : pH5_7step1(x),
                       'pH757step'  : pH5_7step(x)})
    fig, ax = plt.subplots(figsize = (3,3), dpi=300)
    sns.lineplot(data = df, x='x', y='pH757step1', label = 'pH7_5_7 w/o pH6 step'  , color = 'gray')
    sns.lineplot(data = df, x='x', y='pH757step', label = 'pH7_5_7 (stepwise)', color = 'red')
    ax.set(xlabel ='Height_x (0:SG/SC, 1:SC surface)',ylabel='pH',\
           xlim=(-0.1, 1), ylim=(5.3,7.1))
    ax.set_xticks(np.linspace(-0, 1, 6))
    ax.set_yticks(np.linspace(5.4, 7, 9))
    ax.lines[0].set_linestyle('dashdot')
    ax.lines[1].set_linestyle('solid')
    ax.legend(loc='best', bbox_to_anchor=(1.2, 1))
pH_plot()


# plot KD profiles
def kad_plot():
    pH     = np.arange(4, 8, 0.01)

    #raw data for KLK5
    x_5    = np.array([4.5,	    5.5,	6.5,	7.5])
    y_5    = np.array([9.91E+03,	3.06E+04,	3.44E+04,	5.71E+04])
    ka_5   = ka_KLK5(pH)
    df_5 = pd.DataFrame({'pH'           : pH,
                         'curve-fitted' : ka_5})
    df_5ref = pd.DataFrame({'pH_ref'    : x_5,
                            'ka_ref'    : y_5})

    y_5d    = np.array([8.04E-04,	3.38E-05,	1.83E-06,	8.70E-08])
    kd_5   = kd_KLK5(pH)
    df_5d = pd.DataFrame({'pH'           : pH,
                         'curve-fitted' : kd_5})
    df_5refd = pd.DataFrame({'pH_ref'    : x_5,
                            'kd_ref'    : y_5d})
        
    fig, ax1 = plt.subplots(figsize = (3,3), dpi=300)
    sns.lineplot(data = df_5, x='pH', y='curve-fitted', label = 'ka (curve-fitted)', color = 'blue')
    sns.regplot(data = df_5ref, x='pH_ref', y='ka_ref', label = 'ka (reference)'  , color = 'blue', fit_reg=False)
    ax1.set(xlabel ='pH',ylabel='ka (M-1 s-1)',\
            xlim=(4,8), ylim=(1E+3,1e+5))
    ax1.set_xticks(np.linspace(4, 8, 5))
    
    ax2 = ax1.twinx()
    sns.lineplot(data = df_5d, x='pH', y='curve-fitted', label = 'kd (curve-fitted)', ax=ax2, color = 'black', legend=False)
    sns.regplot(data = df_5refd, x='pH_ref', y='kd_ref', label = 'kd (reference)', ax=ax2 , color = 'black', fit_reg=False, marker="D")

    ax2.set(xlabel ='pH',ylabel='kd (s-1)', ylim=(1E-8, 1E-3))
    ax2.lines[0].set_linestyle("dotted")
    ax1.set_yscale("log")
    ax2.set_yscale("log")    
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h2+h1, l2+l1, loc='best', bbox_to_anchor=(2.1, 1))
#kad_plot()

# plot pHact profiles KLK5
def pHact_plot():
    pH    = np.arange(4, 8, 0.01)
    x_5   = np.array([4.1, 4.4, 4.7, 5.0, 5.3, 5.6, 5.8, 6.0, 6.2, 6.4, 6.6, 6.8, 7.0, 7.2, 7.4, 7.6, 7.8, 8.0])
    y_5   = np.array([0.068, 0.056,0.082,0.121,0.153,0.215,0.255,0.336,0.368,0.423,0.513,0.604,0.687,0.802,0.851,0.946,1.000,0.918])
    yerrd = np.array([0.022,0.019,0.026,0.037,0.033,0.028,0.037,0.036,0.042,0.034,0.021,0.028,0.024,0.024,0.022,0.018,0.035,0.020])

    pHact_5   = pHact_KLK5(pH)
    df_5 = pd.DataFrame({'pH'           : pH,
                         'curve-fitted' : pHact_5})
    df_5ref = pd.DataFrame({'pH_ref'    : x_5,
                            'pHact_ref' : y_5})

    
    fig, ax = plt.subplots(figsize = (3,3), dpi=300)
    sns.lineplot(data = df_5, x='pH', y='curve-fitted', label = 'curve-fitted', color = 'black')
    sns.regplot(data = df_5ref, x='pH_ref', y='pHact_ref', label = 'reference'  , color = 'blue', fit_reg=False)
    ax.errorbar(x_5, y_5, yerr=yerrd, fmt='none', capsize=5, zorder=1, color='blue')
    ax.set(xlabel ='pH',ylabel='Relative activity (ratio to activity at pH7.8)',  xlim=(4,8), ylim=(0,1.1))
    ax.legend(loc='best', bbox_to_anchor=(1, 1))
pHact_plot()

# plot mesotrypsin profiles
def mesot_plot():
    pH     = np.arange(4, 8, 0.01)

    #raw data
    x     = np.array([4.1, 4.4 ,4.7 ,5.0 ,5.3 ,5.6 ,5.8 ,6.0 ,6.2 ,6.4 ,6.6 ,6.8 ,7.0 ,7.2 ,7.4 ,7.6 ,7.8 ,8.0])
    y     = np.array([0.283,0.330,0.412,0.532,0.622,0.711,0.701,0.742,0.852,0.828,0.859,0.903,0.883,0.948,0.948,1.000,1.000,0.963])
    yerrd = np.array([0.010,0.010,0.010,0.003,0.010,0.011,0.015,0.017,0.010,0.010,0.013,0.013,0.016,0.012,0.016,0.018,0.013,0.009])

    mesot   = pd.DataFrame(list(map(mesotrpsin_act, pH)))
    mesot.columns = ['curve-fitted']
    pH_df = pd.DataFrame(pH, columns=['pH'])
    df = pd.concat([mesot, pH_df], axis=1)
    dfref = pd.DataFrame({'pH_ref'    : x,
                          'mesot_ref' : y})
    fig, ax = plt.subplots(figsize = (3,3), dpi=300)
    sns.lineplot(data = df, x='pH', y='curve-fitted', label = 'curve-fitted', color = 'black')
    sns.regplot(data = dfref, x='pH_ref', y='mesot_ref', label = 'reference'  , color = 'blue', fit_reg=False)
    ax.errorbar(x, y, yerr=yerrd, fmt='none', capsize=5, zorder=1, color='blue')
    ax.set(xlabel ='pH',ylabel='Relative activity (ratio to activity at pH7.8)',  xlim=(4,8), ylim=(0,1.1))
    ax.set_xticks(np.linspace(4, 8, 5))
    ax.legend(loc='best', bbox_to_anchor=(1, 1))
mesot_plot()

def KLKdepth_plot():
    #raw data
    a_x           = np.array([1.00, 0.64, 0.50, 0.32, 0.00])
    KLK5a_y       = np.array([1.00, 0.83, 0.59, 0.51, 0.45])
    KLK7a_y       = np.array([1.00, 0.76, 0.50, 0.40, 0.43])
    df = pd.DataFrame({'a_x'   : a_x,
                       'KLK5a' : KLK5a_y,
                       'KLK7a' : KLK7a_y})
    fig, ax = plt.subplots(figsize = (3,3), dpi=300)
    sns.regplot(data = df, x='a_x', y='KLK5a', label = 'KLK5' , color = 'blue', fit_reg=False)
    sns.regplot(data = df, x='a_x', y='KLK7a', label = 'KLK7' , color = 'black', fit_reg=False)
    ax.set(xlabel ='       Height in SC (0: last tape strip, 1: first tape strip)',\
           ylabel='Relative activity (ratio to activity at x=1)',  xlim=(-0.1,1.1), ylim=(0,1.1))
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.legend(loc='best', bbox_to_anchor=(1, 1))
KLKdepth_plot()
