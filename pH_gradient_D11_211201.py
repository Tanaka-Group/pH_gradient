# -*- coding: utf-8 -*-
"""
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#from scipy.optimize import leastsq
import seaborn as sns
#import copy
from scipy.integrate import odeint
from numba import jit
from matplotlib.colors import LinearSegmentedColormap

# Parameters
k_deg       = 28.0 # 20
k_auto      = 22.0  # 50
k_meso      = 1.0 # 10

C_pKLK0     = 1.0E-10
C_KLK0      = 0
C_Ltotal    = 1.0E-10
C_meso      = 1.0E-10
V           = 2.3E-6 #human: 5.8E-7, mouse: 2.3E-6
sigmoid_a   = 200
dpi_set     = 100 

def generate_cmap(colors):
    values = range(len(colors))
    vmax = np.ceil(np.max(values))
    color_list = []
    for v, c in zip(values, colors):
        color_list.append( ( v/ vmax, c) )
    return LinearSegmentedColormap.from_list('custom_cmap', color_list)

class ODE(object):
    def __init__(self, diff_eq, init_con):
        self.diff_eq  = diff_eq
        self.init_con = init_con
        
    def cal_equation(self, x_end, k_deg, k_auto, k_meso):
        dx = 0.001 # resolution of x
        N = round(x_end/dx) + 1 #total steps of x
        x = np.linspace(0, x_end, N) # prepare range of x
        v = odeint(self.diff_eq, self.init_con, x, rtol=1e-13, atol=1e-19, args=(k_deg, k_auto, k_meso))
        return v

x_res = 0.001 # resolution of x
x = np.arange(-0.1, 1.01, x_res) # x: height. 0 < x < 1

# basic functions
@jit('f8[:](f8[:],f8,f8)', nopython=True)
def sigmoid(x, a, b):
    return 1 / (1 + np.exp(-a * (x - b))) #a: slope, b: inflection point
@jit('f8(f8,f8,f8)', nopython=True)
def sigmoid_p(x, a, b):
    return 1 / (1 + np.exp(-a * (x - b))) #a: slope, b: inflection point

#def exp_fit(x,y):
#    def exp_residual(param, x1, y1):
#            residual = np.log(y1) - (np.log(param[0]) + param[1] * x1)
#            return residual
#    r1    =  leastsq(exp_residual, [1,1], args=(x, y))
#    return r1[0][0], r1[0][1]
#
#def poly6_fit(x,y):
#    def poly6_residual(param, x1, y1):
#            residual = y1 - (param[0]*x1**6 + param[1]*x1**5 + param[2]*x1**4 + param[3]*x1**3 + param[4]*x1**2 + param[5]*x1**1 + param[6])
#            return residual
#    r1    =  leastsq(poly6_residual, [0,0,0,0,0,0,0], args=(x, y))
#    return r1[0][0], r1[0][1], r1[0][2], r1[0][3], r1[0][4], r1[0][5], r1[0][6]
#
#def poly3_fit(x,y):
#    def poly3_residual(param, x1, y1):
#            residual = y1 - (param[0]*x1**3 + param[1]*x1**2 + param[2]*x1**1 + param[3])
#            return residual
#    r1    =  leastsq(poly3_residual, [0,0,0,0,0], args=(x, y))
#    return r1[0][0], r1[0][1], r1[0][2], r1[0][3], r1[0][4]

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

@jit('f8(f8)', nopython=True)
def pH7_5_p(x):       #conventional
    a = 7 - 1.6*x 
    if a>7:
        a=a
    else:
        a=7
    return a
@jit('f8(f8)', nopython=True)
def pH5_p(x):       # assume Tmem79 KO mice
    return 6.2 - 0.8*np.sign(x)
@jit('f8(f8)', nopython=True)
def pH7_p(x):         # assume Claudin-1 KO mice
    return 7 -x*0
@jit('f8(f8)', nopython=True)
def pH5_7step_p(x):
    return 7 - sigmoid_p(x, sigmoid_a, 0)*1 - sigmoid_p(x, sigmoid_a, 0.3)*0.6 + sigmoid_p(x, sigmoid_a, 0.8)*1.3
@jit('f8(f8)', nopython=True)
def pH5_7step1_p(x):
    return 7 - sigmoid_p(x, sigmoid_a, 0)*1.6 + sigmoid_p(x, sigmoid_a, 0.8)*1.3
@jit('f8(f8)', nopython=True)
def pH757v_p(x):
    a = 5.4 + 3.2*np.abs(x-0.5) 
    if a > 7:
        a = 7
    if x > 0.5:
        a = a - 0.6*(x-0.5)
    return a

@jit('f8(f8)', nopython=True)
def ka_KLK5(pH):
#    x      = np.array([4.5,	    5.5,	6.5,	7.5])
#    y      = np.array([9.91E+03,	3.06E+04,	3.44E+04,	5.71E+04])
#    a, b    = exp_fit(x,y)
    a   = 1107.2170512153116
    b   = 0.5370835922533095
    ka      = a*np.exp(b*pH)
    return ka

@jit('f8(f8)', nopython=True)
def kd_KLK5(pH):
#    x      = np.array([4.5,	    5.5,	6.5,	7.5])
#    y      = np.array([8.04E-04,	3.38E-05,	1.83E-06,	8.70E-08])
#    a, b    = exp_fit(x,y)
    a   = 641.5660425153252
    b   = -3.0310484112381295
    kd  = a*np.exp(b*pH)
    return kd

@jit('f8(f8)', nopython=True)
def pHact_KLK5(pH):
#    x  = np.array([4.1, 4.4, 4.7, 5.0, 5.3, 5.6, 5.8, 6.0, 6.2, 6.4, 6.6, 6.8, 7.0, 7.2, 7.4, 7.6, 7.8, 8.0])
#    y  = np.array([0.068, 0.056,0.082,0.121,0.153,0.215,0.255,0.336,0.368,0.423,0.513,0.604,0.687,0.802,0.851,0.946,1.000,0.918])
#    a, b, c, d, e, f, g = poly6_fit(x, y)
    a = -0.004746254429466693
    b = 0.1589424238008589
    c = -2.194742504812747
    d = 15.993391066635198
    e = -64.7859033497406
    f = 138.16196094242454
    g = -121.04562652656601
    pHact   = a*pH**6 + b*pH**5 + c*pH**4 + d*pH**3 + e*pH**2 + f*pH**1 + g
    return pHact

@jit('f8[:](f8[:])', nopython=True)
def pHact_KLK5_l(pH):
#    x  = np.array([4.1, 4.4, 4.7, 5.0, 5.3, 5.6, 5.8, 6.0, 6.2, 6.4, 6.6, 6.8, 7.0, 7.2, 7.4, 7.6, 7.8, 8.0])
#    y  = np.array([0.068, 0.056,0.082,0.121,0.153,0.215,0.255,0.336,0.368,0.423,0.513,0.604,0.687,0.802,0.851,0.946,1.000,0.918])
#    a, b, c, d, e, f, g = poly6_fit(x, y)
    a = -0.004746254429466693
    b = 0.1589424238008589
    c = -2.194742504812747
    d = 15.993391066635198
    e = -64.7859033497406
    f = 138.16196094242454
    g = -121.04562652656601
    pHact   = a*pH**6 + b*pH**5 + c*pH**4 + d*pH**3 + e*pH**2 + f*pH**1 + g
    return pHact

@jit('f8(f8)', nopython=True)
def mesotrypsin_act(pH):
#    x          = np.array([4.1, 4.4 ,4.7 ,5.0 ,5.3 ,5.6 ,5.8 ,6.0 ,6.2 ,6.4 ,6.6 ,6.8 ,7.0 ,7.2 ,7.4 ,7.6 ,7.8 ,8.0])
#    y          = np.array([0.283,0.330,0.412,0.532,0.622,0.711,0.701,0.742,0.852,0.828,0.859,0.903,0.883,0.948,0.948,1.000,1.000,0.963])
#    a, b, c, d    = poly3_fit(x,y)
#    meso_act   = a*pH**3 + b*pH**2 + c*pH**1 + d
    a   = -0.0031336145910445218
    b   = 0.01660872174079904
    c   = 0.3405034981494503
    d   = -1.199922472107838
    meso_act   = a*pH**3 + b*pH**2 + c*pH**1 + d
    return meso_act

@jit('f8[:](f8[:],f8,f8,f8,f8,)', nopython=True)
def diff_pH5_7step(c, x, k_deg, k_auto, k_meso):  # c: conc. of [pKLK, KLK, LEKTI, K-L compelx], x: height
    pH          = pH5_7step_p(x)
    ka          = ka_KLK5(pH)
    kd          = kd_KLK5(pH)
    a_KLK       = pHact_KLK5(pH)
    a_meso      = mesotrypsin_act(pH)
    dpKLKdt     = -c[0]/V*(k_auto*a_KLK*c[1] + k_meso*a_meso*C_meso)
    dKLKdt      = 1/V* (c[0]*(k_auto*a_KLK*c[1]  + k_meso*a_meso*C_meso)\
                        - ka*c[1]*c[2] + kd*c[3] + k_deg*a_meso*C_meso*c[3])
    dLEKTIdt    = 1/V*( - ka*c[1]*c[2] + kd*c[3] - k_deg*a_meso*C_meso*c[2]) 
    dComplexdt  = 1/V*(   ka*c[1]*c[2] - kd*c[3] - k_deg*a_meso*C_meso*c[3])
    results_all = np.array([dpKLKdt, dKLKdt, dLEKTIdt, dComplexdt])
    return results_all

@jit('f8[:](f8[:],f8,f8,f8,f8,)', nopython=True)
def diff_pH7_5(c, x, k_deg, k_auto, k_meso):  # c: conc. of [pKLK, KLK, LEKTI, K-L compelx], x: height
    pH          = pH7_5_p(x)
    ka          = ka_KLK5(pH)
    kd          = kd_KLK5(pH)
    a_KLK       = pHact_KLK5(pH)
    a_meso      = mesotrypsin_act(pH)
    dpKLKdt     = -c[0]/V*(k_auto*a_KLK*c[1] + k_meso*a_meso*C_meso)
    dKLKdt      = 1/V* (c[0]*(k_auto*a_KLK*c[1]  + k_meso*a_meso*C_meso)\
                        - ka*c[1]*c[2] + kd*c[3] + k_deg*a_meso*C_meso*c[3])
    dLEKTIdt    = 1/V*( - ka*c[1]*c[2] + kd*c[3] - k_deg*a_meso*C_meso*c[2]) 
    dComplexdt  = 1/V*(   ka*c[1]*c[2] - kd*c[3] - k_deg*a_meso*C_meso*c[3])
    results_all = np.array([dpKLKdt, dKLKdt, dLEKTIdt, dComplexdt])
    return results_all

@jit('f8[:](f8[:],f8,f8,f8,f8,)', nopython=True)
def diff_pH5_7step1(c, x, k_deg, k_auto, k_meso):  # c: conc. of [pKLK, KLK, LEKTI, K-L compelx], x: height
    pH          = pH5_7step1_p(x)
    ka          = ka_KLK5(pH)
    kd          = kd_KLK5(pH)
    a_KLK       = pHact_KLK5(pH)
    a_meso      = mesotrypsin_act(pH)
    dpKLKdt     = -c[0]/V*(k_auto*a_KLK*c[1] + k_meso*a_meso*C_meso)
    dKLKdt      = 1/V* (c[0]*(k_auto*a_KLK*c[1]  + k_meso*a_meso*C_meso)\
                        - ka*c[1]*c[2] + kd*c[3] + k_deg*a_meso*C_meso*c[3])
    dLEKTIdt    = 1/V*( - ka*c[1]*c[2] + kd*c[3] - k_deg*a_meso*C_meso*c[2]) 
    dComplexdt  = 1/V*(   ka*c[1]*c[2] - kd*c[3] - k_deg*a_meso*C_meso*c[3])
    results_all = np.array([dpKLKdt, dKLKdt, dLEKTIdt, dComplexdt])
    return results_all

@jit('f8[:](f8[:],f8,f8,f8,f8,)', nopython=True)
def diff_pH5(c, x, k_deg, k_auto, k_meso):  # c: conc. of [pKLK, KLK, LEKTI, K-L compelx], x: height
    pH          = pH5_p(x)
    ka          = ka_KLK5(pH)
    kd          = kd_KLK5(pH)
    a_KLK       = pHact_KLK5(pH)
    a_meso      = mesotrypsin_act(pH)
    dpKLKdt     = -c[0]/V*(k_auto*a_KLK*c[1] + k_meso*a_meso*C_meso)
    dKLKdt      = 1/V* (c[0]*(k_auto*a_KLK*c[1]  + k_meso*a_meso*C_meso)\
                        - ka*c[1]*c[2] + kd*c[3] + k_deg*a_meso*C_meso*c[3])
    dLEKTIdt    = 1/V*( - ka*c[1]*c[2] + kd*c[3] - k_deg*a_meso*C_meso*c[2]) 
    dComplexdt  = 1/V*(   ka*c[1]*c[2] - kd*c[3] - k_deg*a_meso*C_meso*c[3])
    results_all = np.array([dpKLKdt, dKLKdt, dLEKTIdt, dComplexdt])
    return results_all

@jit('f8[:](f8[:],f8,f8,f8,f8,)', nopython=True)
def diff_pH7(c, x, k_deg, k_auto, k_meso):  # c: conc. of [pKLK, KLK, LEKTI, K-L compelx], x: height
    pH          = pH7_p(x)
    ka          = ka_KLK5(pH)
    kd          = kd_KLK5(pH)
    a_KLK       = pHact_KLK5(pH)
    a_meso      = mesotrypsin_act(pH)
    dpKLKdt     = -c[0]/V*(k_auto*a_KLK*c[1] + k_meso*a_meso*C_meso)
    dKLKdt      = 1/V* (c[0]*(k_auto*a_KLK*c[1]  + k_meso*a_meso*C_meso)\
                        - ka*c[1]*c[2] + kd*c[3] + k_deg*a_meso*C_meso*c[3])
    dLEKTIdt    = 1/V*( - ka*c[1]*c[2] + kd*c[3] - k_deg*a_meso*C_meso*c[2]) 
    dComplexdt  = 1/V*(   ka*c[1]*c[2] - kd*c[3] - k_deg*a_meso*C_meso*c[3])
    results_all = np.array([dpKLKdt, dKLKdt, dLEKTIdt, dComplexdt])
    return results_all

@jit('f8[:](f8[:],f8,f8,f8,f8,)', nopython=True)
def diff_pH757v(c, x, k_deg, k_auto, k_meso):  # c: conc. of [pKLK, KLK, LEKTI, K-L compelx], x: height
    pH          = pH757v_p(x)
    ka          = ka_KLK5(pH)
    kd          = kd_KLK5(pH)
    a_KLK       = pHact_KLK5(pH)
    a_meso      = mesotrypsin_act(pH)
    dpKLKdt     = -c[0]/V*(k_auto*a_KLK*c[1] + k_meso*a_meso*C_meso)
    dKLKdt      = 1/V* (c[0]*(k_auto*a_KLK*c[1]  + k_meso*a_meso*C_meso)\
                        - ka*c[1]*c[2] + kd*c[3] + k_deg*a_meso*C_meso*c[3])
    dLEKTIdt    = 1/V*( - ka*c[1]*c[2] + kd*c[3] - k_deg*a_meso*C_meso*c[2]) 
    dComplexdt  = 1/V*(   ka*c[1]*c[2] - kd*c[3] - k_deg*a_meso*C_meso*c[3])
    results_all = np.array([dpKLKdt, dKLKdt, dLEKTIdt, dComplexdt])
    return results_all


# --- Range of kdeg to reproduce qualitative distribution of LEKTI ---
def fR_Ltotal_pH757(k_deg):
    ode      = ODE(diff_pH5_7step, [C_pKLK0, C_KLK0, C_Ltotal, 0])
    sim      = ode.cal_equation(1, k_deg, k_auto, k_meso) # x, k_deg, k_auto, k_meso
    R_Ltotal = (sim[1000, 2] + sim[1000, 3])/sim[0, 2]
    return R_Ltotal

def fR_Ltotal_pH75(k_deg):
    ode      = ODE(diff_pH7_5, [C_pKLK0, C_KLK0, C_Ltotal, 0])
    sim      = ode.cal_equation(1, k_deg, k_auto, k_meso) # x, k_deg, k_auto, k_meso
    R_Ltotal = (sim[1000, 2] + sim[1000, 3])/sim[0, 2]
    return R_Ltotal

def fR_Ltotal_pH757v(k_deg):
    ode      = ODE(diff_pH757v, [C_pKLK0, C_KLK0, C_Ltotal, 0])
    sim      = ode.cal_equation(1, k_deg, k_auto, k_meso) # x, k_deg, k_auto, k_meso
    R_Ltotal = (sim[1000, 2] + sim[1000, 3])/sim[0, 2]
    return R_Ltotal

def C_total_kdeg_plot(R_Ltotal, Title, kdeg_min, kdeg_max, kdeg_range):
    df = pd.DataFrame({'k_deg'    : kdeg_range,
                       'R_Ltotal' : R_Ltotal})
    fig, ax = plt.subplots(figsize = (3,3), dpi=300)
    sns.lineplot(data = df, x='k_deg', y='R_Ltotal')
#    ax.set(xlabel ='k_deg (s-1)',ylabel='C_Ltotal(1)/C_Ltotal(0) (-)',\
    ax.set(xlabel ='k_deg (s-1)',ylabel='C_Ltotal(1)/C_Ltotal(0) (-)',\
           xlim=(np.min(kdeg_range), np.max(kdeg_range)), ylim=(0.000001,1))
    plt.vlines(kdeg_min, 0, 0.5, "gray", linestyles='dashed') 
    plt.hlines(0.5, 0, kdeg_min, "gray", linestyles='dashed')
    plt.vlines(kdeg_max, 0, 0.0001, "gray", linestyles='dotted') 
    plt.hlines(0.0001, 0, kdeg_max, "gray", linestyles='dotted')
#    ax.set_xscale("log") 
    ax.set_yscale("log")  
#    ax.set_xticks(np.logspace(0,6,7))
    ax.set_title(Title) 

kdeg_range = np.linspace(0,1e6,1000)
R_Ltotal_pH757 = np.array(list(map(fR_Ltotal_pH757, kdeg_range)))
R_Ltotal_pH757_max = kdeg_range[np.sum(np.where(R_Ltotal_pH757 > 0.0001, 1, 0))]
R_Ltotal_pH757_min = kdeg_range[np.sum(np.where(R_Ltotal_pH757 > 0.5, 1, 0))]

R_Ltotal_pH75 = np.array(list(map(fR_Ltotal_pH75, kdeg_range)))
R_Ltotal_pH75_max = kdeg_range[np.sum(np.where(R_Ltotal_pH75 > 0.0001, 1, 0))]
R_Ltotal_pH75_min = kdeg_range[np.sum(np.where(R_Ltotal_pH75 > 0.5, 1, 0))]

R_Ltotal_pH757v = np.array(list(map(fR_Ltotal_pH757v, kdeg_range)))
R_Ltotal_pH757v_max = kdeg_range[np.sum(np.where(R_Ltotal_pH757v > 0.0001, 1, 0))]
R_Ltotal_pH757v_min = kdeg_range[np.sum(np.where(R_Ltotal_pH757v > 0.5, 1, 0))]

C_total_kdeg_plot(R_Ltotal_pH757, "", R_Ltotal_pH757_min, R_Ltotal_pH757_max, kdeg_range)
C_total_kdeg_plot(R_Ltotal_pH75, "", R_Ltotal_pH75_min, R_Ltotal_pH75_max, kdeg_range)
C_total_kdeg_plot(R_Ltotal_pH757v, "", R_Ltotal_pH757v_min, R_Ltotal_pH757v_max, kdeg_range)


# plot C_Ltotal with different k_deg
def sim_Ltotal(diff_func, k_deg):
    ode    = ODE(diff_func, [C_pKLK0, C_KLK0, C_Ltotal, 0])
    sim    = ode.cal_equation(1, k_deg, k_auto, k_meso)
    Ltotal = sim[:,2] + sim[:,3]
    Ltotal = np.concatenate([[C_Ltotal] * 109, Ltotal]).reshape((1110,1))
    return Ltotal

def C_Ltotal_plot(L1, Title, pHfunc, kmin, kmax):
    np.set_printoptions(suppress=True, precision=2)
    pH  = pHfunc(x).reshape((1110,1))
    res = pd.DataFrame(np.concatenate([pH.T,L1.T])).T
    res.columns =     ['pH', 'L1', 'L2', 'L3']
    x_df = pd.DataFrame(x, columns=['x'])
    res = pd.concat([res, x_df], axis=1)
    
    fig, ax1 = plt.subplots(figsize = (3,3), dpi=300)
    label1 = 'k_deg = {:.0e}'.format(0)
    sns.lineplot(data = res[109:], x='x', y='L1', \
                 label = label1, \
                 color = sns.color_palette("deep", 6)[0])
    label2 = 'k_deg = {:.0e}'.format(kmin)
    sns.lineplot(data = res[109:], x='x', y='L2', \
                 label = label2, \
                 color = sns.color_palette("deep", 6)[0])
    label3 = 'k_deg = {:.0e}'.format(kmax)
    sns.lineplot(data = res[109:], x='x', y='L3', \
                 label = label3, \
                 color = sns.color_palette("deep", 6)[0])
    ax1.set(xlabel ='Height_x (0:SG/SC, 1:SC surface)',ylabel='C_Ltotal (M)',\
            xlim=(0,1), ylim=(-1E-12,1.42e-10))
    ax1.lines[0].set_linestyle("solid")
    ax1.lines[1].set_linestyle("dashed")
    ax1.lines[2].set_linestyle("dotted")
    ax1.set_xticks(np.linspace(0, 1, 6))
    ax2 = ax1.twinx()
    sns.lineplot(data = res, x='x', y='pH', label = 'pH', ax=ax2,\
                 legend=False, color = 'gray')
    ax2.set(xlim=(-0.1, 1), ylim=(-0.1,7.1))
    ax2.lines[0].set_linestyle("--")
#    ax2.set_yticks(np.linspace(0, 7, 8))
    ax2.set_ylabel('')
    ax2.set_yticks([])
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h2+h1, l2+l1, loc='best', bbox_to_anchor=(1.9, 1))
    ax1.set_title(Title) 

L1 = np.concatenate([sim_Ltotal(diff_pH5_7step,0),
                     sim_Ltotal(diff_pH5_7step,R_Ltotal_pH757_min),
                     sim_Ltotal(diff_pH5_7step,R_Ltotal_pH757_max)], 1)
L2 = np.concatenate([sim_Ltotal(diff_pH7_5,0),
                     sim_Ltotal(diff_pH7_5,R_Ltotal_pH75_min),
                     sim_Ltotal(diff_pH7_5,R_Ltotal_pH75_max)], 1)
L3 = np.concatenate([sim_Ltotal(diff_pH757v,0),
                     sim_Ltotal(diff_pH757v,R_Ltotal_pH757v_min),
                     sim_Ltotal(diff_pH757v,R_Ltotal_pH757v_max)], 1)
    
    
C_Ltotal_plot(L1,'', pH5_7step, R_Ltotal_pH757_min, R_Ltotal_pH757_max)
C_Ltotal_plot(L2,'', pH7_5, R_Ltotal_pH75_min, R_Ltotal_pH75_max)
C_Ltotal_plot(L3,'', pH757v, R_Ltotal_pH757v_min, R_Ltotal_pH757v_max)

# --- Range of kauto and kmeso to reproduce qualitative distribution of KLK ---
def f_P1_KLK_pH757(k_auto,k_meso):
    ode     = ODE(diff_pH5_7step, [C_pKLK0, C_KLK0, C_Ltotal, 0])
    sim     = ode.cal_equation(1, k_deg, k_auto, k_meso) # x, k_deg, k_auto, k_meso
    Ktot  = sim[:,1] + sim[:,3]
    P_KLK   = (Ktot[1000] - Ktot[500])/Ktot[500]
    return P_KLK

def f_P2_KLK_pH757(k_auto,k_meso):
    ode     = ODE(diff_pH5_7step, [C_pKLK0, C_KLK0, C_Ltotal, 0])
    sim     = ode.cal_equation(1, k_deg, k_auto, k_meso) # x, k_deg, k_auto, k_meso
    Ktot  = sim[:,1] + sim[:,3]
    P_KLK   = Ktot[1000]/C_pKLK0
    return P_KLK

def f_P1_KLK_pH75(k_auto,k_meso):
    ode     = ODE(diff_pH7_5, [C_pKLK0, C_KLK0, C_Ltotal, 0])
    sim     = ode.cal_equation(1, k_deg, k_auto, k_meso) # x, k_deg, k_auto, k_meso
    Ktot  = sim[:,1] + sim[:,3]
    P_KLK   = (Ktot[1000] - Ktot[500])/Ktot[500]
    return P_KLK

def f_P2_KLK_pH75(k_auto,k_meso):
    ode     = ODE(diff_pH7_5, [C_pKLK0, C_KLK0, C_Ltotal, 0])
    sim     = ode.cal_equation(1, k_deg, k_auto, k_meso) # x, k_deg, k_auto, k_meso
    Ktot  = sim[:,1] + sim[:,3]
    P_KLK   = Ktot[1000]/C_pKLK0
    return P_KLK

def f_P1_KLK_pH757v(k_auto,k_meso):
    ode     = ODE(diff_pH757v, [C_pKLK0, C_KLK0, C_Ltotal, 0])
    sim     = ode.cal_equation(1, k_deg, k_auto, k_meso) # x, k_deg, k_auto, k_meso
    Ktot  = sim[:,1] + sim[:,3]
    P_KLK   = (Ktot[1000] - Ktot[500])/Ktot[500]
    return P_KLK

def f_P2_KLK_pH757v(k_auto,k_meso):
    ode     = ODE(diff_pH757v, [C_pKLK0, C_KLK0, C_Ltotal, 0])
    sim     = ode.cal_equation(1, k_deg, k_auto, k_meso) # x, k_deg, k_auto, k_meso
    Ktot  = sim[:,1] + sim[:,3]
    P_KLK   = Ktot[1000]/C_pKLK0
    return P_KLK

def sim_profiles(func):
    List_kauto = np.sort(np.array(list(np.logspace(2,7,201))*201))
    List_kmeso = np.array(list(np.logspace(2,7,201))*201)
    res = np.array(list(map(func, List_kauto, List_kmeso)))
    res = res.reshape((201,201))
    res = pd.DataFrame(res)
    res.index   = list(np.array(np.logspace(2,7,201), np.int32))
    res.columns = list(np.array(np.logspace(2,7,201), np.int32))
    return res
    
def P_KLK_plot(P_KLK, Title, label1, cmap1, vmin1, vmax1, tick1):
    fig, ax = plt.subplots(figsize = (5,4), dpi=100)
    sns.heatmap(P_KLK, vmin=vmin1, vmax=vmax1, \
                xticklabels=40, yticklabels=40, \
                cmap=cmap1, cbar_kws={'label': label1, 'ticks': tick1})
    ax.set(xlabel ='k_meso (M-1 s-1)',ylabel='k_auto (M-1 s-1)')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    ax.set_title(Title) 

#k_deg = 0
#k_deg = R_Ltotal_pH757_min
k_deg = R_Ltotal_pH757_max

P1_KLK_pH757 = sim_profiles(f_P1_KLK_pH757)    
P2_KLK_pH757 = sim_profiles(f_P2_KLK_pH757)
P1a_KLK_pH757 = np.where((0.5 < P1_KLK_pH757) &\
                        (P1_KLK_pH757 < 1.5), 1, 0)
P2a_KLK_pH757 = np.where((P2_KLK_pH757>0.5), 1, 0)
P3_KLK_pH757 = np.where((0.5 < P1_KLK_pH757) &\
                        (P1_KLK_pH757 < 1.5) & \
                        (P2_KLK_pH757>0.5), 1, 0)
P_KLK_plot(P1_KLK_pH757, "pH757 (stepwise)",'[C_Ktotal(1)-C_Ktotal(0.5)]/[C_Ktotal(0.5)-C_Ktotal(0)]',\
           generate_cmap(['#3C4EC2',\
                          '#E0E1E2',\
                          '#B40426',\
                          '#E0E1E2',\
                          sns.color_palette("deep", 10)[5]]\
                          ),0, 2, [0, 0.5, 1, 1.5, 2])
#P_KLK_plot(P2_KLK_pH757, "pH757 (stepwise)",'','coolwarm',0, 1, [0, 0.5, 1])
P_KLK_plot(P1a_KLK_pH757, "pH757 (stepwise)",'','coolwarm',0, 1, [0, 1])
P_KLK_plot(P2a_KLK_pH757, "pH757 (stepwise)",'','coolwarm',0, 1, [0, 1])
P_KLK_plot(P3_KLK_pH757, "pH757 (stepwise)",'','coolwarm',0, 1, [0, 1])

P_KLK_plot(P2_KLK_pH757, "",'C_Ktotal(1)/C_pKLK_SG/SC','coolwarm',0, 1, [0, 0.5, 1])

P3= pd.DataFrame(P3_KLK_pH757)
P3.index = P1_KLK_pH757.index
P3.columns = P1_KLK_pH757.columns

#k_deg = 0
#k_deg = R_Ltotal_pH75_min
k_deg = R_Ltotal_pH75_max

P1_KLK_pH75 = sim_profiles(f_P1_KLK_pH75)
P2_KLK_pH75 = sim_profiles(f_P2_KLK_pH75)
P1a_KLK_pH75 = np.where((0.5 < P1_KLK_pH75) &\
                        (P1_KLK_pH75 < 1.5), 1, 0)
P2a_KLK_pH75 = np.where((P2_KLK_pH75>0.5), 1, 0)
P3_KLK_pH75 = np.where((0.5 < P1_KLK_pH75) &\
                        (P1_KLK_pH75 < 1.5) & \
                        (P2_KLK_pH75>0.5), 1, 0)

P_KLK_plot(P1_KLK_pH75, "pH75 (gradient)",'[C_Ktotal(1)-C_Ktotal(0.5)]/[C_Ktotal(0.5)-C_Ktotal(0)]',\
           generate_cmap(['#3C4EC2',\
                          '#E0E1E2',\
                          '#B40426',\
                          '#E0E1E2',\
                          sns.color_palette("deep", 10)[5]]\
                          ),0, 2, [0, 0.5, 1, 1.5, 2])
#P_KLK_plot(P2_KLK_pH75, "pH75 (gradient)",'C_Ktotal(1)/C_pKLK(0)','coolwarm',0, 1, [0, 0.5, 1])

P_KLK_plot(P1a_KLK_pH75, "pH75 (gradient)",'C_Ktotal(1)/C_pKLK(0)','coolwarm',0, 1, [0, 1])
P_KLK_plot(P2a_KLK_pH75, "pH75 (gradient)",'C_Ktotal(1)/C_pKLK(0)','coolwarm',0, 1, [0, 1])
P_KLK_plot(P3_KLK_pH75, "pH75 (gradient)",'C_Ktotal(1)/C_pKLK(0)','coolwarm',0, 1, [0, 1])

P_KLK_plot(P2_KLK_pH75, "",'C_Ktotal(1)/C_pKLK(0)','coolwarm',0, 1, [0, 0.5, 1])

P3= pd.DataFrame(P3_KLK_pH75)
P3.index = P1_KLK_pH75.index
P3.columns = P1_KLK_pH75.columns

#k_deg = 0
#k_deg = R_Ltotal_pH757v_min
k_deg = R_Ltotal_pH757v_max

P1_KLK_pH757v = sim_profiles(f_P1_KLK_pH757v)    
P2_KLK_pH757v = sim_profiles(f_P2_KLK_pH757v)
P1a_KLK_pH757v = np.where((0.5 < P1_KLK_pH757v) &\
                        (P1_KLK_pH757v < 1.5), 1, 0)
P2a_KLK_pH757v = np.where((P2_KLK_pH757v>0.5), 1, 0)
P3_KLK_pH757v = np.where((0.5 < P1_KLK_pH757v) &\
                        (P1_KLK_pH757v < 1.5) & \
                        (P2_KLK_pH757v>0.5), 1, 0)
P_KLK_plot(P1_KLK_pH757v, "pH757 (gradient)",'[C_Ktotal(1)-C_Ktotal(0.5)]/[C_Ktotal(0.5)-C_Ktotal(0)]',\
           generate_cmap(['#3C4EC2',\
                          '#E0E1E2',\
                          '#B40426',\
                          '#E0E1E2',\
                          sns.color_palette("deep", 10)[5]]\
                          ),0, 2, [0, 0.5, 1, 1.5, 2])
#P_KLK_plot(P2_KLK_pH757v, "pH757 (gradient)",'','coolwarm',0, 1, [0, 0.5, 1])
P_KLK_plot(P1a_KLK_pH757v, "pH757 (gradient)",'','coolwarm',0, 1, [0, 1])
P_KLK_plot(P2a_KLK_pH757v, "pH757 (gradient)",'','coolwarm',0, 1, [0, 1])
P_KLK_plot(P3_KLK_pH757v, "pH757 (gradient)",'','coolwarm',0, 1, [0, 1])

P_KLK_plot(P2_KLK_pH757v, "",'C_Ktotal(1)/C_pKLK_SG/SC','coolwarm',0, 1, [0, 0.5, 1])

P3= pd.DataFrame(P3_KLK_pH757v)
P3.index = P1_KLK_pH757v.index
P3.columns = P1_KLK_pH757v.columns


# plot E_KLK with different k_auto/meso
def sim_Ktotal(diff_func, pH_func, k_auto, k_meso):
    ode    = ODE(diff_func, [C_pKLK0, C_KLK0, C_Ltotal, 0])
    sim    = ode.cal_equation(1, k_deg, k_auto, k_meso)
    Ktotal = sim[:,1] + sim[:,3]
    Ktotal = np.concatenate([[C_pKLK0] * 109, Ktotal]).reshape((1110,1))
    return Ktotal

# p:pH757stepwise, q:pH75 gradient, r: pH757v
p1 = [1e+2,   2.3e+4] #100, all: 22387-44668
p2 = [1e+2,   4.4e+4] #100, all: 22387-44668
p3 = [1e+5,   3e+4] #1e5, kdeg=0: 13335-44668, min/max: 12589-44668, kdeg_max: 
p4 = [1.6e+6, 1e+2] #,100 kged0/min: 1584893-1678804, max: 1496235-1678804
p5 = [1e+3,   1e+3]
p6 = [1e+6,   1e+6] 
 
q1 = [1e+2, 2e+4] # 100  all:17782-33496
q2 = [1e+2, 3e+4] # 100  all:17782-33496
q3 = [1e+5, 1.6e+4] # 1e5, k0:7079-25118, min7079-26607, max7943-25118
q4 = [5.95662e+5, 1e+2] #,100  k0:595662-668343, min:595662-630957, max 562341-595662
q5 = [1e+3, 1e+3] #
q6 = [1e+6, 1e+6] #

r1 = [1e+2, 2.2e+4] # 100, all 21134-37583,
r2 = [1e+2, 3.7e+4] # 100, all 21134-37583,
r3 = [1e+5, 2.3e+4] # 1e5, k0 11885-35481,min 11220-37583, max 10592-35481
r4 = [1.1e+6, 1e+2] # ,100 k0min:1059253-1188502,max 1000000-1122018
r5 = [1e+3, 1e+3] #
r6 = [1e+6, 1e+6] #

def evaluate_series(sim_func, diff_func, step_func, p1, p2, p3, p4, p5, p6):
    res = np.concatenate([sim_func(diff_func, step_func, p1[0], p1[1]),
                          sim_func(diff_func, step_func, p2[0], p2[1]),
                          sim_func(diff_func, step_func, p3[0], p3[1]),
                          sim_func(diff_func, step_func, p4[0], p4[1]),
                          sim_func(diff_func, step_func, p5[0], p5[1]),
                          sim_func(diff_func, step_func, p6[0], p6[1])], 1)
    return res

def Ktotal_plot(K0, K1, K2, Title, Y_label, Y_max, pHfunc, p1, p2, p3, p4, p5, p6, degmin, degmax, Anchor):
    pH  = pHfunc(x).reshape((1110,1))
    res = pd.DataFrame(np.concatenate([pH.T,K1.T])).T
    res.columns =     ['pH', 'K1', 'K2', 'K3', 'K4', 'K5', 'K6']
    x_df = pd.DataFrame(x, columns=['x'])
    res = pd.concat([res, x_df], axis=1)

    res2 = pd.DataFrame(np.concatenate([pH.T,K2.T])).T
    res2.columns =     ['pH', 'K1', 'K2', 'K3', 'K4', 'K5', 'K6']
    x_df2 = pd.DataFrame(x, columns=['x'])
    res2 = pd.concat([res2, x_df2], axis=1)

    res0 = pd.DataFrame(np.concatenate([pH.T,K0.T])).T
    res0.columns =     ['pH', 'K1', 'K2', 'K3', 'K4', 'K5', 'K6']
    x_df0 = pd.DataFrame(x, columns=['x'])
    res0 = pd.concat([res0, x_df0], axis=1)
    
    fig, ax1 = plt.subplots(figsize = (3,3), dpi=300)
    label01 = '[k_deg, k_auto,k_meso] = [{0:.0e}, {1:.0e}, {2:.0e}] '.format(0, p1[0], p1[1])
    sns.lineplot(data = res0[109:], x='x', y='K1', \
                 label = label01,\
                 color = sns.color_palette("bright", 8)[1])
    label02 = '[k_deg, k_auto,k_meso] = [{0:.0e}, {1:.0e}, {2:.0e}] '.format(0, p2[0], p2[1])
    sns.lineplot(data = res0[109:], x='x', y='K2', \
                 label = label02,\
                 color = sns.color_palette("bright", 8)[2])
    label03 = '[k_deg, k_auto,k_meso] = [{0:.0e}, {1:.0e}, {2:.0e}] '.format(0, p3[0], p3[1])
    sns.lineplot(data = res0[109:], x='x', y='K3', \
                 label = label03,\
                 color = sns.color_palette("bright", 8)[4])
    label04 = '[k_deg, k_auto,k_meso] = [{0:.0e}, {1:.0e}, {2:.0e}] '.format(0, p4[0], p4[1])
    sns.lineplot(data = res0[109:], x='x', y='K4', \
                 label = label04,\
                 color = sns.color_palette("bright", 8)[5])
    label05 = '[k_deg, k_auto,k_meso] = [{0:.0e}, {1:.0e}, {2:.0e}] '.format(0, p5[0], p5[1])
    sns.lineplot(data = res0[109:], x='x', y='K5', \
                 label = label05,\
                 color = sns.color_palette("bright", 8)[0])
    label06 = '[k_deg, k_auto,k_meso] = [{0:.0e}, {1:.0e}, {2:.0e}] '.format(0, p6[0], p6[1])
    sns.lineplot(data = res0[109:], x='x', y='K6', \
                 label = label06,\
                 color = sns.color_palette("bright", 8)[3])    
    label1 = '[k_deg, k_auto,k_meso] = [{0:.0e}, {1:.0e}, {2:.0e}] '.format(degmin, p1[0], p1[1])
    sns.lineplot(data = res[109:], x='x', y='K1', \
                 label = label1,\
                 color = sns.color_palette("pastel", 8)[1])
    label2 = '[k_deg, k_auto,k_meso] = [{0:.0e}, {1:.0e}, {2:.0e}] '.format(degmin, p2[0], p2[1])
    sns.lineplot(data = res[109:], x='x', y='K2', \
                 label = label2,\
                 color = sns.color_palette("pastel", 8)[2])
    label3 = '[k_deg, k_auto,k_meso] = [{0:.0e}, {1:.0e}, {2:.0e}] '.format(degmin, p3[0], p3[1])
    sns.lineplot(data = res[109:], x='x', y='K3', \
                 label = label3,\
                 color = sns.color_palette("pastel", 8)[4])
    label4 = '[k_deg, k_auto,k_meso] = [{0:.0e}, {1:.0e}, {2:.0e}] '.format(degmin, p4[0], p4[1])
    sns.lineplot(data = res[109:], x='x', y='K4', \
                 label = label4,\
                 color = sns.color_palette("pastel", 8)[5])
    label5 = '[k_deg, k_auto,k_meso] = [{0:.0e}, {1:.0e}, {2:.0e}] '.format(degmin, p5[0], p5[1])
    sns.lineplot(data = res[109:], x='x', y='K5', \
                 label = label5,\
                 color = sns.color_palette("pastel", 8)[0])
    label6 = '[k_deg, k_auto,k_meso] = [{0:.0e}, {1:.0e}, {2:.0e}] '.format(degmin, p6[0], p6[1])
    sns.lineplot(data = res[109:], x='x', y='K6', \
                 label = label6,\
                 color = sns.color_palette("pastel", 8)[3])
    label7 = '[k_deg, k_auto,k_meso] = [{0:.0e}, {1:.0e}, {2:.0e}] '.format(degmax, p1[0], p1[1])
    sns.lineplot(data = res2[109:], x='x', y='K1', \
                 label = label7,\
                 color = sns.color_palette("dark", 8)[1])
    label8 = '[k_deg, k_auto,k_meso] = [{0:.0e}, {1:.0e}, {2:.0e}] '.format(degmax, p2[0], p2[1])
    sns.lineplot(data = res2[109:], x='x', y='K2', \
                 label = label8,\
                 color = sns.color_palette("dark", 8)[2])
    label9 = '[k_deg, k_auto,k_meso] = [{0:.0e}, {1:.0e}, {2:.0e}] '.format(degmax, p3[0], p3[1])
    sns.lineplot(data = res2[109:], x='x', y='K3', \
                 label = label9,\
                 color = sns.color_palette("dark", 8)[4])
    label10 = '[k_deg, k_auto,k_meso] = [{0:.0e}, {1:.0e}, {2:.0e}] '.format(degmax, p4[0], p4[1])
    sns.lineplot(data = res2[109:], x='x', y='K4', \
                 label = label10,\
                 color = sns.color_palette("dark", 8)[5])
    label11 = '[k_deg, k_auto,k_meso] = [{0:.0e}, {1:.0e}, {2:.0e}] '.format(degmax, p5[0], p5[1])
    sns.lineplot(data = res2[109:], x='x', y='K5', \
                 label = label11,\
                 color = sns.color_palette("dark", 8)[0])
    label12 = '[k_deg, k_auto,k_meso] = [{0:.0e}, {1:.0e}, {2:.0e}] '.format(degmax, p6[0], p6[1])
    sns.lineplot(data = res2[109:], x='x', y='K6', \
                 label = label12,\
                 color = sns.color_palette("dark", 8)[3])    


    ax1.set(xlabel ='Height_x (0:SG/SC, 1:SC surface)',ylabel = Y_label,\
            xlim=(0,1), ylim=(-1E-12,1.42E-10))
    ax1.lines[0].set_linestyle("solid")
    ax1.lines[1].set_linestyle("solid")
    ax1.lines[2].set_linestyle("solid")
    ax1.lines[3].set_linestyle("solid")
    ax1.lines[4].set_linestyle("solid")
    ax1.lines[5].set_linestyle("solid")

    ax1.lines[6].set_linestyle("dashed")
    ax1.lines[7].set_linestyle("dashed")
    ax1.lines[8].set_linestyle("dashed")
    ax1.lines[9].set_linestyle("dashed")
    ax1.lines[10].set_linestyle("dashed")
    ax1.lines[11].set_linestyle("dashed")

    ax1.lines[12].set_linestyle("dotted")
    ax1.lines[13].set_linestyle("dotted")
    ax1.lines[14].set_linestyle("dotted")
    ax1.lines[15].set_linestyle("dotted")
    ax1.lines[16].set_linestyle("dotted")
    ax1.lines[17].set_linestyle("dotted")

    ax1.set_xticks(np.linspace(0, 1, 6))
    ax2 = ax1.twinx()
    sns.lineplot(data = res, x='x', y='pH', label = 'pH', ax=ax2,\
                 legend=False, color = 'gray')
    ax2.set(xlim=(-0.1, 1), ylim=(-0.1,7.1))
    ax2.lines[0].set_linestyle("--")
    ax2.set_ylabel('')
    ax2.set_yticks([])
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h2+h1, l2+l1, loc='best', bbox_to_anchor=(Anchor, 1))
    ax1.set_title(Title) 

k_deg = 0
K0_757 = evaluate_series(sim_Ktotal, diff_pH5_7step, pH5_7step, p1, p2, p3, p4, p5, p6)
k_deg = R_Ltotal_pH757_min
K1_757 = evaluate_series(sim_Ktotal, diff_pH5_7step, pH5_7step, p1, p2, p3, p4, p5, p6)
k_deg = R_Ltotal_pH757_max
K2_757 = evaluate_series(sim_Ktotal, diff_pH5_7step, pH5_7step, p1, p2, p3, p4, p5, p6)

k_deg = 0
K0_75 = evaluate_series(sim_Ktotal, diff_pH7_5, pH7_5, q1, q2, q3, q4, q5, q6)
k_deg = R_Ltotal_pH75_min
K1_75 = evaluate_series(sim_Ktotal, diff_pH7_5, pH7_5, q1, q2, q3, q4, q5, q6)
k_deg = R_Ltotal_pH75_max
K2_75 = evaluate_series(sim_Ktotal, diff_pH7_5, pH7_5, q1, q2, q3, q4, q5, q6)

k_deg = 0
K0_757v = evaluate_series(sim_Ktotal, diff_pH757v, pH757v, r1, r2, r3, r4, r5, r6)
k_deg = R_Ltotal_pH75_min
K1_757v = evaluate_series(sim_Ktotal, diff_pH757v, pH757v, r1, r2, r3, r4, r5, r6)
k_deg = R_Ltotal_pH75_max
K2_757v = evaluate_series(sim_Ktotal, diff_pH757v, pH757v, r1, r2, r3, r4, r5, r6)
  
  
Ktotal_plot(K0_757, K1_757, K2_757,'      pH7_5_7 (stepwise)','C_Ktotal (M)', 1.5e-10, pH5_7step, p1, p2, p3, p4, p5, p6, R_Ltotal_pH757_min, R_Ltotal_pH757_max, 1.2)
Ktotal_plot(K0_75, K1_75,  K2_75, '      pH7_5 (gradient)', 'C_Ktotal (M)', 1.5e-10, pH7_5, q1, q2, q3, q4, q5, q6, R_Ltotal_pH75_min, R_Ltotal_pH75_max, 1.2)
Ktotal_plot(K0_757v, K1_757v, K2_757v,'      pH757 (gradient)','C_Ktotal (M)', 1.5e-10, pH757v, p1, p2, p3, p4, p5, p6, R_Ltotal_pH757_min, R_Ltotal_pH757_max, 1.2)


# --- 3.2.1 Comparison of stepwise pH profiles with gradient pH profiles ---

def Ktotal_plot2(K0, K1, K2, Title, Y_label, Y_max, pHfunc, p1, p2, p3, p4, degmin, degmax, Anchor):
    pH  = pHfunc(x).reshape((1110,1))
    res = pd.DataFrame(np.concatenate([pH.T,K1.T])).T
    res.columns =     ['pH', 'K1', 'K2', 'K3', 'K4', 'K5', 'K6']
    x_df = pd.DataFrame(x, columns=['x'])
    res = pd.concat([res, x_df], axis=1)

    res2 = pd.DataFrame(np.concatenate([pH.T,K2.T])).T
    res2.columns =     ['pH', 'K1', 'K2', 'K3', 'K4', 'K5', 'K6']
    x_df2 = pd.DataFrame(x, columns=['x'])
    res2 = pd.concat([res2, x_df2], axis=1)

    res0 = pd.DataFrame(np.concatenate([pH.T,K0.T])).T
    res0.columns =     ['pH', 'K1', 'K2', 'K3', 'K4', 'K5', 'K6']
    x_df0 = pd.DataFrame(x, columns=['x'])
    res0 = pd.concat([res0, x_df0], axis=1)
    
    fig, ax1 = plt.subplots(figsize = (3,3), dpi=300)

    label01 = '[k_deg, k_auto,k_meso] = [{0:.0e}, {1:.0e}, {2:.0e}] '.format(0, p1[0], p1[1])
    sns.lineplot(data = res0[109:], x='x', y='K1', \
                 label = label01,\
                 color = sns.color_palette("bright", 8)[1], legend=False)
    label02 = '[k_deg, k_auto,k_meso] = [{0:.0e}, {1:.0e}, {2:.0e}] '.format(0, p2[0], p2[1])
    sns.lineplot(data = res0[109:], x='x', y='K2', \
                 label = label02,\
                 color = sns.color_palette("bright", 8)[2], legend=False)
    label03 = '[k_deg, k_auto,k_meso] = [{0:.0e}, {1:.0e}, {2:.0e}] '.format(0, p3[0], p3[1])
    sns.lineplot(data = res0[109:], x='x', y='K3', \
                 label = label03,\
                 color = sns.color_palette("bright", 8)[4], legend=False)
    label04 = '[k_deg, k_auto,k_meso] = [{0:.0e}, {1:.0e}, {2:.0e}] '.format(0, p4[0], p4[1])
    sns.lineplot(data = res0[109:], x='x', y='K4', \
                 label = label04,\
                 color = sns.color_palette("bright", 8)[5], legend=False)
    label1 = '[k_deg, k_auto,k_meso] = [{0:.0e}, {1:.0e}, {2:.0e}] '.format(degmin, p1[0], p1[1])
    sns.lineplot(data = res[109:], x='x', y='K1', \
                 label = label1,\
                 color = sns.color_palette("pastel", 8)[1], legend=False)
    label2 = '[k_deg, k_auto,k_meso] = [{0:.0e}, {1:.0e}, {2:.0e}] '.format(degmin, p2[0], p2[1])
    sns.lineplot(data = res[109:], x='x', y='K2', \
                 label = label2,\
                 color = sns.color_palette("pastel", 8)[2], legend=False)
    label3 = '[k_deg, k_auto,k_meso] = [{0:.0e}, {1:.0e}, {2:.0e}] '.format(degmin, p3[0], p3[1])
    sns.lineplot(data = res[109:], x='x', y='K3', \
                 label = label3,\
                 color = sns.color_palette("pastel", 8)[4], legend=False)
    label4 = '[k_deg, k_auto,k_meso] = [{0:.0e}, {1:.0e}, {2:.0e}] '.format(degmin, p4[0], p4[1])
    sns.lineplot(data = res[109:], x='x', y='K4', \
                 label = label4,\
                 color = sns.color_palette("pastel", 8)[5], legend=False)
    label5 = '[k_deg, k_auto,k_meso] = [{0:.0e}, {1:.0e}, {2:.0e}] '.format(degmax, p1[0], p1[1])
    sns.lineplot(data = res2[109:], x='x', y='K1', \
                 label = label5,\
                 color = sns.color_palette("dark", 8)[1], legend=False)
    label6 = '[k_deg, k_auto,k_meso] = [{0:.0e}, {1:.0e}, {2:.0e}] '.format(degmax, p2[0], p2[1])
    sns.lineplot(data = res2[109:], x='x', y='K2', \
                 label = label6,\
                 color = sns.color_palette("dark", 8)[2], legend=False)
    label7 = '[k_deg, k_auto,k_meso] = [{0:.0e}, {1:.0e}, {2:.0e}] '.format(degmax, p3[0], p3[1])
    sns.lineplot(data = res2[109:], x='x', y='K3', \
                 label = label7,\
                 color = sns.color_palette("dark", 8)[4], legend=False)
    label8 = '[k_deg, k_auto,k_meso] = [{0:.0e}, {1:.0e}, {2:.0e}] '.format(degmax, p4[0], p4[1])
    sns.lineplot(data = res2[109:], x='x', y='K4', \
                 label = label8,\
                 color = sns.color_palette("dark", 8)[5], legend=False)
    ax1.set(xlabel ='Height_x (0:SG/SC, 1:SC surface)',ylabel = Y_label,\
            xlim=(0,1), ylim=(-1E-12,Y_max))
    ax1.lines[0].set_linestyle("solid")
    ax1.lines[1].set_linestyle("solid")
    ax1.lines[2].set_linestyle("solid")
    ax1.lines[3].set_linestyle("solid")

    ax1.lines[4].set_linestyle("dashed")
    ax1.lines[5].set_linestyle("dashed")
    ax1.lines[6].set_linestyle("dashed")
    ax1.lines[7].set_linestyle("dashed")

    ax1.lines[8].set_linestyle("dotted")
    ax1.lines[9].set_linestyle("dotted")
    ax1.lines[10].set_linestyle("dotted")
    ax1.lines[11].set_linestyle("dotted")
    ax1.set_xticks(np.linspace(0, 1, 6))
    ax2 = ax1.twinx()
    sns.lineplot(data = res, x='x', y='pH', label = 'pH', ax=ax2,\
                 legend=False, color = 'gray')
    ax2.set(xlim=(-0.1, 1), ylim=(-0.1,7.1))
    ax2.lines[0].set_linestyle("--")
    ax2.set_ylabel('')
    ax2.set_yticks([])
#    ax2.set_yticks(np.linspace(0, 7, 8))
#    h1, l1 = ax1.get_legend_handles_labels()
#    h2, l2 = ax2.get_legend_handles_labels()
#    ax1.legend(h2+h1, l2+l1, loc='best', bbox_to_anchor=(Anchor, 1))
    ax1.set_title(Title) 

def sim_IKLK(k_auto,k_meso):
    pH      = pH5_7step(x)
    ode     = ODE(diff_pH5_7step, [C_pKLK0, C_KLK0, C_Ltotal, 0])
    sim     = ode.cal_equation(1, k_deg, k_auto, k_meso) # x, k_deg, k_auto, k_meso = 1
    E_KLK   = pHact_KLK5_l(pH)[109:]*sim[:,1]
    I_KLK   = E_KLK[1000] - E_KLK[700]
    return I_KLK

def sim_IKLK2(k_auto,k_meso):
    pH      = pH7_5(x)
    ode     = ODE(diff_pH7_5, [C_pKLK0, C_KLK0, C_Ltotal, 0])
    sim     = ode.cal_equation(1, k_deg, k_auto, k_meso) # x, k_deg, k_auto, k_meso = 1
    E_KLK   = pHact_KLK5_l(pH)[109:]*sim[:,1]
    I_KLK   = E_KLK[1000] - E_KLK[700]
    return I_KLK

def sim_IKLK6(k_auto,k_meso):
    pH      = pH5_7step1(x)
    ode     = ODE(diff_pH5_7step1, [C_pKLK0, C_KLK0, C_Ltotal, 0])
    sim     = ode.cal_equation(1, k_deg, k_auto, k_meso) # x, k_deg, k_auto, k_meso
    E_KLK   = pHact_KLK5_l(pH)[109:]*sim[:,1]
    I_KLK   = E_KLK[1000] - E_KLK[700]
    return I_KLK

def sim_IKLKv(k_auto,k_meso):
    pH      = pH757v(x)
    ode     = ODE(diff_pH757v, [C_pKLK0, C_KLK0, C_Ltotal, 0])
    sim     = ode.cal_equation(1, k_deg, k_auto, k_meso) # x, k_deg, k_auto, k_meso
    E_KLK   = pHact_KLK5_l(pH)[109:]*sim[:,1]
    I_KLK   = E_KLK[1000] - E_KLK[700]
    return I_KLK

def plot_profiles(data, Title, cmap1, vmin1, vmax1, tick1, label1):
    fig, ax = plt.subplots(figsize = (5,4), dpi=100)
    sns.heatmap(data, cmap=cmap1, vmin=vmin1, vmax=vmax1,
                xticklabels=40, yticklabels=40, cbar_kws={'label': label1, 'ticks': tick1})
    ax.set(xlabel ='k_meso (M-1 s-1)',ylabel='k_auto (M-1 s-1)')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    ax.set_title(Title)

def plot_profiles2(data, Title, cmap1, vmin1, vmax1, tick1, label1):
    fig, ax = plt.subplots(figsize = (4,4), dpi=100)
    sns.heatmap(data, cmap=cmap1, vmin=vmin1, vmax=vmax1,
                xticklabels=40, yticklabels=40, cbar=False)
    ax.set(xlabel ='k_meso (M-1 s-1)',ylabel='k_auto (M-1 s-1)')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    ax.set_title(Title)

def plot_profiles3(data, Title, cmap1, vmin1, vmax1, tick1, label1):
    fig, ax = plt.subplots(figsize = (5,4), dpi=100)
    sns.heatmap(data, cmap=cmap1, vmin=vmin1, vmax=vmax1,
                xticklabels=40, yticklabels=40, \
                cbar_kws={'label': label1, 'ticks': tick1,\
                          'use_gridspec':False, 'location':"top"})
    ax.set(xlabel ='k_meso (M-1 s-1)',ylabel='k_auto (M-1 s-1)')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    ax.set_title(Title)

def sim_EKLK(diff_func, pH_func, k_auto, k_meso):
    pH    = pH_func(x)
    ode   = ODE(diff_func, [C_pKLK0, C_KLK0, C_Ltotal, 0])
    sim   = ode.cal_equation(1, k_deg, k_auto, k_meso)
    E_KLK = pHact_KLK5_l(pH)[109:]*sim[:,1]
    E_KLK = np.concatenate([[0] * 109, E_KLK]).reshape((1110,1))
    return E_KLK

k_deg = 0
Iklk1 = sim_profiles(sim_IKLK)
Iklk6 = sim_profiles(sim_IKLK6)
Iklk2 = sim_profiles(sim_IKLK2)
Iklkv = sim_profiles(sim_IKLKv)
plot_profiles2(Iklk1, "pH757 (stepwise)", 'coolwarm',\
              -1e-12, 1e-12, [-8e-13, -4e-13, 0, 4e-13, 8e-13], 'E_KLK(1.0)-E_KLK(0.7)')
plot_profiles(Iklk2, "pH75 (gradient)", 'coolwarm',\
              -1e-12, 1e-12, [-8e-13, -4e-13, 0, 4e-13, 8e-13], 'E_KLK(1.0)-E_KLK(0.7)')
plot_profiles(Iklk6, "pH757 (w/o pH6)", 'coolwarm',\
              -1e-12, 1e-12, [-8e-13, -4e-13, 0, 4e-13, 8e-13], 'E_KLK(1.0)-E_KLK(0.7)')
plot_profiles(Iklkv, "pH757v (gradient)", 'coolwarm',\
              -1e-12, 1e-12, [-8e-13, -4e-13, 0, 4e-13, 8e-13], 'E_KLK(1.0)-E_KLK(0.7)')


k_deg = R_Ltotal_pH757_min
Iklk1 = sim_profiles(sim_IKLK)
Iklk6 = sim_profiles(sim_IKLK6)
k_deg = R_Ltotal_pH75_min
Iklk2 = sim_profiles(sim_IKLK2)
k_deg = R_Ltotal_pH757v_min
Iklkv = sim_profiles(sim_IKLKv)
plot_profiles2(Iklk1, "pH757 (stepwise)", 'coolwarm',\
              -1e-12, 1e-12, [-8e-13, -4e-13, 0, 4e-13, 8e-13], 'E_KLK(1.0)-E_KLK(0.7)')
plot_profiles(Iklk2, "pH75 (gradient)", 'coolwarm',\
              -1e-12, 1e-12, [-8e-13, -4e-13, 0, 4e-13, 8e-13], 'E_KLK(1.0)-E_KLK(0.7)')
plot_profiles(Iklk6, "pH757 (w/o pH6)", 'coolwarm',\
              -1e-12, 1e-12, [-8e-13, -4e-13, 0, 4e-13, 8e-13], 'E_KLK(1.0)-E_KLK(0.7)')
plot_profiles(Iklkv, "pH757v (gradient)", 'coolwarm',\
              -1e-12, 1e-12, [-8e-13, -4e-13, 0, 4e-13, 8e-13], 'E_KLK(1.0)-E_KLK(0.7)')

k_deg = R_Ltotal_pH757_max
Iklk1 = sim_profiles(sim_IKLK)
Iklk6 = sim_profiles(sim_IKLK6)
k_deg = R_Ltotal_pH75_max
Iklk2 = sim_profiles(sim_IKLK2)
k_deg = R_Ltotal_pH757v_max
Iklkv = sim_profiles(sim_IKLKv)
plot_profiles2(Iklk1, "pH757 (stepwise)", 'coolwarm',\
              -1e-12, 1e-12, [-8e-13, -4e-13, 0, 4e-13, 8e-13], 'E_KLK(1.0)-E_KLK(0.7)')
plot_profiles(Iklk2, "pH75 (gradient)", 'coolwarm',\
              -1e-12, 1e-12, [-8e-13, -4e-13, 0, 4e-13, 8e-13], 'E_KLK(1.0)-E_KLK(0.7)')
plot_profiles(Iklk6, "pH757 (w/o pH6)", 'coolwarm',\
              -1e-12, 1e-12, [-8e-13, -4e-13, 0, 4e-13, 8e-13], 'E_KLK(1.0)-E_KLK(0.7)')
plot_profiles(Iklkv, "pH757v (gradient)", 'coolwarm',\
              -1e-12, 1e-12, [-8e-13, -4e-13, 0, 4e-13, 8e-13], 'E_KLK(1.0)-E_KLK(0.7)')


k_deg = 0
E0_757 = evaluate_series(sim_EKLK, diff_pH5_7step, pH5_7step, p1, p2, p3, p4, p5, p6)
E0_5 = evaluate_series(sim_EKLK, diff_pH5, pH5, p1, p2, p3, p4, p5, p6)
E0_7 = evaluate_series(sim_EKLK, diff_pH7, pH7, p1, p2, p3, p4, p5, p6)
E0_7576 = evaluate_series(sim_EKLK, diff_pH5_7step1, pH5_7step1, p1, p2, p3, p4, p5, p6)

k_deg = R_Ltotal_pH757_min
E1_757 = evaluate_series(sim_EKLK, diff_pH5_7step, pH5_7step, p1, p2, p3, p4, p5, p6)
E1_5 = evaluate_series(sim_EKLK, diff_pH5, pH5, p1, p2, p3, p4, p5, p6)
E1_7 = evaluate_series(sim_EKLK, diff_pH7, pH7, p1, p2, p3, p4, p5, p6)
E1_7576 = evaluate_series(sim_EKLK, diff_pH5_7step1, pH5_7step1, p1, p2, p3, p4, p5, p6)

k_deg = R_Ltotal_pH757_max
E2_757 = evaluate_series(sim_EKLK, diff_pH5_7step, pH5_7step, p1, p2, p3, p4, p5, p6)
E2_5 = evaluate_series(sim_EKLK, diff_pH5, pH5, p1, p2, p3, p4, p5, p6)
E2_7 = evaluate_series(sim_EKLK, diff_pH7, pH7, p1, p2, p3, p4, p5, p6)
E2_7576 = evaluate_series(sim_EKLK, diff_pH5_7step1, pH5_7step1, p1, p2, p3, p4, p5, p6)

k_deg = 0
E0_75 = evaluate_series(sim_EKLK, diff_pH7_5, pH7_5, q1, q2, q3, q4, q5, q6)
k_deg = R_Ltotal_pH75_min
E1_75 = evaluate_series(sim_EKLK, diff_pH7_5, pH7_5, q1, q2, q3, q4, q5, q6)
k_deg = R_Ltotal_pH75_max
E2_75 = evaluate_series(sim_EKLK, diff_pH7_5, pH7_5, q1, q2, q3, q4, q5, q6)

k_deg = 0
E0_757v = evaluate_series(sim_EKLK, diff_pH757v, pH757v, r1, r2, r3, r4, r5, r6)
k_deg = R_Ltotal_pH757v_min
E1_757v = evaluate_series(sim_EKLK, diff_pH757v, pH757v, r1, r2, r3, r4, r5, r6)
k_deg = R_Ltotal_pH757v_max
E2_757v = evaluate_series(sim_EKLK, diff_pH757v, pH757v, r1, r2, r3, r4, r5, r6)


Ktotal_plot2(E0_757, E1_757, E2_757, '      pH757 (stepwise)','E_KLK (M)', \
            0.71e-10, pH5_7step, p1, p2, p3, p4, R_Ltotal_pH757_min, R_Ltotal_pH757_max, 1.2)
Ktotal_plot2(E0_75, E1_75, E2_75,   '      pH75 (gradient)', 'E_KLK (M)', \
            0.71e-10, pH7_5, q1, q2, q3, q4, R_Ltotal_pH75_min, R_Ltotal_pH75_max, 3)
Ktotal_plot2(E0_5, E1_5, E2_5,'      pH5 (Tmem79 KO)'   ,'E_KLK (M)',\
            0.71e-10, pH5, p1, p2, p3, p4, R_Ltotal_pH757_min, R_Ltotal_pH757_max, 3)
Ktotal_plot2(E0_7, E1_7, E2_7,'      pH7 (Claudin-1 KO)', 'E_KLK (M)',\
            0.71e-10, pH7, q1, q2, q3, q4, R_Ltotal_pH757_min, R_Ltotal_pH757_max, 1.2)
Ktotal_plot2(E0_7576, E1_7576, E2_7576,'           pH757 (w/o pH6 step)', 'E_KLK (M)',\
            0.71e-10, pH5_7step1, p1, p2, p3, p4, R_Ltotal_pH757_min, R_Ltotal_pH757_max, 1.2)
Ktotal_plot2(E0_757v, E1_757v, E2_757v,'           pH757 (gradient)', 'E_KLK (M)',\
            0.71e-10, pH757v, r1, r2, r3, r4, R_Ltotal_pH757_min, R_Ltotal_pH757_max, 1.2)


# --- 3.2.2 pH profiles of Claudin-1 KO and Tmem79 KO mice (concs of LEKTI and KLK)
def sim_Ltotal2(diff_func, pH_func, k_auto, k_meso):
    ode    = ODE(diff_func, [C_pKLK0, C_KLK0, C_Ltotal, 0])
    sim    = ode.cal_equation(1, k_deg, k_auto, k_meso)
    Ltotal = sim[:,2] + sim[:,3]
    Ltotal = np.concatenate([[C_Ltotal] * 109, Ltotal]).reshape((1110,1))
    return Ltotal

def sim_Ktotal2(diff_func, pH_func, k_auto, k_meso):
    ode    = ODE(diff_func, [C_pKLK0, C_KLK0, C_Ltotal, 0])
    sim    = ode.cal_equation(1, k_deg, k_auto, k_meso)
    Ktotal = sim[:,1] + sim[:,3]
    Ktotal = np.concatenate([[C_KLK0] * 109, Ktotal]).reshape((1110,1))
    return Ktotal

k_deg = 0
Ls0_757     = evaluate_series(sim_Ltotal2, diff_pH5_7step, pH5_7step, p1, p2, p3, p4, p5, p6)
Ls0_5       = evaluate_series(sim_Ltotal2, diff_pH5, pH5, p1, p2, p3, p4, p5, p6)
Ls0_7       = evaluate_series(sim_Ltotal2, diff_pH7, pH7, p1, p2, p3, p4, p5, p6)
Ls0_757wo   = evaluate_series(sim_Ltotal2, diff_pH5_7step1, pH5_7step1, p1, p2, p3, p4, p5, p6)
Ls0_757v    = evaluate_series(sim_Ltotal2, diff_pH757v, pH757v, p1, p2, p3, p4, p5, p6)
Ks0_757     = evaluate_series(sim_Ktotal2, diff_pH5_7step, pH5_7step, p1, p2, p3, p4, p5, p6)
Ks0_5       = evaluate_series(sim_Ktotal2, diff_pH5, pH5, p1, p2, p3, p4, p5, p6)
Ks0_7       = evaluate_series(sim_Ktotal2, diff_pH7, pH7, p1, p2, p3, p4, p5, p6)
Ks0_757wo   = evaluate_series(sim_Ktotal2, diff_pH5_7step1, pH5_7step1, p1, p2, p3, p4, p5, p6)
Ks0_757v    = evaluate_series(sim_Ktotal2, diff_pH757v, pH757v, p1, p2, p3, p4, p5, p6)

k_deg = R_Ltotal_pH757_min
Ls1_757     = evaluate_series(sim_Ltotal2, diff_pH5_7step, pH5_7step, p1, p2, p3, p4, p5, p6)
Ls1_5       = evaluate_series(sim_Ltotal2, diff_pH5, pH5, p1, p2, p3, p4, p5, p6)
Ls1_7       = evaluate_series(sim_Ltotal2, diff_pH7, pH7, p1, p2, p3, p4, p5, p6)
Ls1_757wo   = evaluate_series(sim_Ltotal2, diff_pH5_7step1, pH5_7step1, p1, p2, p3, p4, p5, p6)
Ls1_757v    = evaluate_series(sim_Ltotal2, diff_pH757v, pH757v, p1, p2, p3, p4, p5, p6)
Ks1_757     = evaluate_series(sim_Ktotal2, diff_pH5_7step, pH5_7step, p1, p2, p3, p4, p5, p6)
Ks1_5       = evaluate_series(sim_Ktotal2, diff_pH5, pH5, p1, p2, p3, p4, p5, p6)
Ks1_7       = evaluate_series(sim_Ktotal2, diff_pH7, pH7, p1, p2, p3, p4, p5, p6)
Ks1_757wo   = evaluate_series(sim_Ktotal2, diff_pH5_7step1, pH5_7step1, p1, p2, p3, p4, p5, p6)
Ks1_757v    = evaluate_series(sim_Ktotal2, diff_pH757v, pH757v, p1, p2, p3, p4, p5, p6)

k_deg = R_Ltotal_pH757_max
Ls2_757     = evaluate_series(sim_Ltotal2, diff_pH5_7step, pH5_7step, p1, p2, p3, p4, p5, p6)
Ls2_5       = evaluate_series(sim_Ltotal2, diff_pH5, pH5, p1, p2, p3, p4, p5, p6)
Ls2_7       = evaluate_series(sim_Ltotal2, diff_pH7, pH7, p1, p2, p3, p4, p5, p6)
Ls2_757wo   = evaluate_series(sim_Ltotal2, diff_pH5_7step1, pH5_7step1, p1, p2, p3, p4, p5, p6)
Ls2_757v    = evaluate_series(sim_Ltotal2, diff_pH757v, pH757v, p1, p2, p3, p4, p5, p6)
Ks2_757     = evaluate_series(sim_Ktotal2, diff_pH5_7step, pH5_7step, p1, p2, p3, p4, p5, p6)
Ks2_5       = evaluate_series(sim_Ktotal2, diff_pH5, pH5, p1, p2, p3, p4, p5, p6)
Ks2_7       = evaluate_series(sim_Ktotal2, diff_pH7, pH7, p1, p2, p3, p4, p5, p6)
Ks2_757wo   = evaluate_series(sim_Ktotal2, diff_pH5_7step1, pH5_7step1, p1, p2, p3, p4, p5, p6)
Ks2_757v    = evaluate_series(sim_Ktotal2, diff_pH757v, pH757v, p1, p2, p3, p4, p5, p6)

Ktotal_plot2(Ls0_757, Ls1_757, Ls2_757,'      pH757 (stepwise)', 'C_Ltotal (M)', 1.42e-10, pH5_7step, p1, p2, p3, p4, R_Ltotal_pH757_min, R_Ltotal_pH757_max, 1.2)
Ktotal_plot2(Ls0_5, Ls1_5, Ls2_5,'      pH5 (Tmem79 KO)', 'C_Ltotal (M)', 1.42e-10, pH5, p1, p2, p3, p4, R_Ltotal_pH757_min, R_Ltotal_pH757_max, 1.2)
Ktotal_plot2(Ls0_7, Ls1_7, Ls2_7,'      pH7 (Claudin-1 KO)', 'C_Ltotal (M)', 1.42e-10, pH7, p1, p2, p3, p4,R_Ltotal_pH757_min, R_Ltotal_pH757_max,  1.2)
Ktotal_plot2(Ls0_757wo, Ls1_757wo, Ls2_757wo,'      pH757 (w/o pH6)', 'C_Ltotal (M)', 1.42e-10, pH5_7step1, p1, p2, p3, p4, R_Ltotal_pH757_min, R_Ltotal_pH757_max, 1.2)

Ktotal_plot2(Ks0_757, Ks1_757, Ks2_757,'      pH757 (stepwise)', 'C_Ktotal (M)', 1.42e-10, pH5_7step, p1, p2, p3, p4,R_Ltotal_pH757_min, R_Ltotal_pH757_max,  1.2)
Ktotal_plot2(Ks0_5, Ks1_5, Ks2_5,'      pH5 (Tmem79 KO)', 'C_Ktotal (M)', 1.42e-10, pH5, p1, p2, p3, p4, R_Ltotal_pH757_min, R_Ltotal_pH757_max,  1.2)
Ktotal_plot2(Ks0_7, Ks1_7, Ks2_7,'      pH7 (Claudin-1 KO)', 'C_Ktotal (M)', 1.42e-10, pH7, p1, p2, p3, p4, R_Ltotal_pH757_min, R_Ltotal_pH757_max,  1.2)
Ktotal_plot2(Ks0_757wo, Ks1_757wo, Ks2_757wo,'      pH757 (w/o pH6)', 'C_Ktotal (M)', 1.42e-10, pH5_7step1, p1, p2, p3, p4,R_Ltotal_pH757_min, R_Ltotal_pH757_max,  1.2)

Ktotal_plot2(Ls0_757v, Ls1_757v, Ls2_757v,'      pH757 (V-shape)', 'C_Ltotal (M)', 1.42e-10, pH757v, p1, p2, p3, p4, R_Ltotal_pH757_min, R_Ltotal_pH757_max, 1.2)
Ktotal_plot2(Ks0_757v, Ks1_757v, Ks2_757v,'      pH757 (V-shape)', 'C_Ktotal (M)', 1.42e-10, pH757v, p1, p2, p3, p4, R_Ltotal_pH757_min, R_Ltotal_pH757_max,  1.2)


# --- 3.2.3 Contribution of pH6 step in the stepwise pH profile

def sim_Xrise(k_auto, k_meso):
    pH     = pH5_7step(x)
    ode    = ODE(diff_pH5_7step, [C_pKLK0, C_KLK0, C_Ltotal, 0])
    sim    = ode.cal_equation(1, k_deg, k_auto, k_meso)
    E_KLK  = pHact_KLK5_l(pH)[109:]*sim[:,1]    # 0<x<1
    a = np.where(E_KLK < E_KLK[1000]*0.9, 1, 0)
    X_rise = (np.sum(a) - 800)/1000
    return X_rise

def sim_Xrise2(k_auto, k_meso):
    pH     = pH5_7step1(x)
    ode    = ODE(diff_pH5_7step1, [C_pKLK0, C_KLK0, C_Ltotal, 0])
    sim    = ode.cal_equation(1, k_deg, k_auto, k_meso)
    E_KLK  = pHact_KLK5_l(pH)[109:]*sim[:,1]    # 0<x<1
    a = np.where(E_KLK < E_KLK[1000]*0.9, 1, 0)
    X_rise = (np.sum(a) - 800)/1000
    return X_rise

k_deg = 0
X_rise1 = sim_profiles(sim_Xrise)
X_rise2 = sim_profiles(sim_Xrise2)
X_rise3 = X_rise2 - X_rise1
plot_profiles(X_rise1, "pH757 (stepwise)", 'Reds', 0, 0.17, [0, 0.05, 0.1,0.15],'X_rise')
plot_profiles(X_rise2, "pH757 (w/o pH6)", 'Reds', 0, 0.17, [0, 0.05, 0.1,0.15],'X_rise')
plot_profiles(X_rise3, "pH757 (w/o pH6 step - w/ pH6 step)", 'coolwarm', -0.05, 0.05, [-0.05, 0, 0.05],'X_rise')

k_deg = R_Ltotal_pH757_min
X_rise1 = sim_profiles(sim_Xrise)
X_rise2 = sim_profiles(sim_Xrise2)
X_rise3 = X_rise2 - X_rise1
plot_profiles(X_rise1, "pH757 (stepwise)", 'Reds', 0, 0.17, [0, 0.05, 0.1,0.15],'X_rise')
plot_profiles(X_rise2, "pH757 (w/o pH6)", 'Reds', 0, 0.17, [0, 0.05, 0.1,0.15],'X_rise')
plot_profiles(X_rise3, "pH757 (w/o pH6 step - w/ pH6 step)", 'coolwarm', -0.05, 0.05, [-0.05, 0, 0.05],'X_rise')

k_deg = R_Ltotal_pH757_max
X_rise1 = sim_profiles(sim_Xrise)
X_rise2 = sim_profiles(sim_Xrise2)
X_rise3 = X_rise2 - X_rise1
plot_profiles(X_rise1, "", 'Reds', 0, 0.17, [0, 0.05, 0.1,0.15],'X_rise')
plot_profiles(X_rise2, "", 'Reds', 0, 0.17, [0, 0.05, 0.1,0.15],'X_rise')
plot_profiles(X_rise3, "", 'coolwarm', -0.05, 0.05, [-0.05, 0, 0.05],'X_rise')

# horizontal cbar
plot_profiles3(X_rise1, "", 'Reds', 0, 0.17, [0, 0.05, 0.1,0.15],'X_rise')
plot_profiles3(X_rise2, "", 'Reds', 0, 0.17, [0, 0.05, 0.1,0.15],'X_rise')
plot_profiles3(X_rise3, "", 'coolwarm', -0.05, 0.05, [-0.05, 0, 0.05],'X_rise')

