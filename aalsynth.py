# TBD List:
# "clean up" means to standardize the coding format (e.g., function definitions),
#                                 parameter names (e.g., with underscores),
#                  to remove commented and unused code
# "test imports" means to determine if each of the imported packages needs to be there
# General  -- AALSynth documentation and user guide
#          -- Github archive -- doing this now!
#          -- RL NV COS paper
# aalsynth -- Clean up, test imports
#          -- Read in parameters, directory locations, program flow from ASCII file
# AbsCloud -- Clean up, test imports
#          -- Consider temperature profile of disk (currently hydrostatic, but check Cloudy)
#          -- Improve placement of new clouds
# atomic   -- Clean up, test imports
# cloudy   -- Clean up, test imports
#          -- Test for existence of Cloudy_runs/ subdirectory and creation if not there
# doppler  -- Clean up, test imports
# hstqso   -- Clean up, test imports
# mcgv     -- Get steady-state velocity field
#          -- Speed up force multiplier calculation --> use quasar._calculate_absorbed_flux_gaussleg
#          -- Cloudy simulations for emissivity/source function
# ntdisk   -- Clean up, test imports
#          -- Slim disk, ADAF models                          #FUTURE
#          -- MAD models                                      #FUTURE
# quasar   -- Clean up, test imports
#          -- Incorporate emission lines into integration of absorbed flux
#          -- Fitting routine parameter uncertainties
#                             potential for fixed parameters  #FUTURE

import copy
import matplotlib
matplotlib.use("TkAgg")

import os
import platform
import subprocess
import sys

import matplotlib.pyplot  as plt
import matplotlib.patches as pth
import numpy              as np
import time               as tm

from astropy                 import constants as const
from astropy                 import units as u
from astropy.convolution     import convolve, Gaussian1DKernel
from astropy.io              import fits, ascii
from astropy.modeling.models import BlackBody
from astropy.table           import Table
from astropy.visualization   import astropy_mpl_style, quantity_support
from functools               import partial
from multiprocessing         import Pool
from numpy.polynomial        import chebyshev
from scipy                   import special
from scipy                   import interpolate
from scipy.integrate         import solve_ivp,odeint
from scipy.interpolate       import griddata, RegularGridInterpolator, CubicSpline
from scipy.special           import gamma
from time                    import sleep
from tqdm                    import tqdm

from atomic                  import atomic
from doppler                 import calcvel,calcwave
from ntdisk                  import ntdisk
from AbsCloud                import AbsCloud
from mcgv                    import mcgv
from quasar                  import Quasar
from cloudy                  import cloudy

#######################################################################################
breakstr = "#" * 50
fu = (u.erg / (u.s * u.cm * u.cm * u.Hz))
######################################################

print(breakstr)
print(breakstr)
print("Welcome to AALSynth. Your one-stop-shop for modelling associated absorbers.")
print(breakstr)

######################################################
print("Defining initial parameters")
if os.name == 'posix':
  datapath    = "/content/drive/MyDrive/AALSynth_data/"
  cloudypath  = datapath
  if os.uname().nodename == 'astronomy':
    datapath = "/mnt/data/AALSynth_data/"
    cloudypath  = "/mnt/data/c23.01/source/"
if os.name == 'nt':
  datapath    = "D:\\ganguly\\AALSynth2\\AALSynth_data\\"
  cloudypath  = datapath

print(f"\tData path: {datapath}\tCloudy Path: {cloudypath}")

###################################################################
# Black hole parameters
sbh         = np.array([  0.01,   0.01,   0.99,   0.01,   0.99])
mbh         = np.array([1.2e+8, 1.0e+8, 1.0e+8, 1.0e+8, 1.0e+8]) # Solar masses
# Accretion parameters
nr          = np.array([   300,    300,    300,    300,    300])
rlo         = np.array([   0.1,    0.1,    0.1,    0.1,    0.1]) # rms
rhi         = np.array([1.0e+5, 1.0e+5, 1.0e+5, 1.0e+5, 1.0e+5]) # rms
mdot        = np.array([  0.72,    1.8,    0.5,    0.5,    1.8]) # solar mass per year
alpha       = np.array([   0.1,    0.1,    0.1,    0.1,    0.1])
ctype       =          [   "k",    "k",    "b",    "g",    "r"]
# Observer parameters
inclination = np.array([ 20.0,         30.0,   30.0,   30.0,   30.0]) * u.degree
robs        = np.array([  0.084,        1.0,    1.0,    1.0,    1.0]) * 1.0e+15 # rg
thetaobs    = np.array([  0.0,          0.0,    0.0,    0.0,    0.0])
zqso        = np.array([  0.114272     ])
raqso       = np.array([175.19958333333]) * u.deg
decqso      = np.array([ 46.36805555556]) * u.deg
##################################################################
# Wind parameters
winddxlo    = np.array([    70,     70,     70,     70,     70], dtype=np.int16)
winddxhi    = np.array([   250,    250,    250,    250,    250], dtype=np.int16)
# Absorber parameters
xclp        = np.array([])#1.0])#,         0.5]) # rg
yclp        = np.array([])#0.0])#,         0.5]) # rg
zcl         = np.array([])#2.0e+6])#,   1.9e+6]) # rg
rhoindex    = np.array([])#2.0])#,         2.0]) 
logrhoscale = np.array([])#15.0])#,       14.5]) # log(cm)
logrho0     = np.array([])#3.5])#,         2.5]) # log(cm**-3)
vcl         = np.array([])#-1575.9]) * (u.km / u.s) #, -1800.0]) 
# Spectral synthesis
anum        = np.array([1,     7,  8])
ion         = np.array([1,     5,  6])
trandx      = np.array([0,     0,  1])
plot_code   =          ["g", "r", "m"]
vlo         = -2000.0 * (u.km/u.s)
vhi         = -1000.0 * (u.km/u.s)
vres        = 3.0 * (u.km/u.s)
# Display parameters
zoom = 5000
verbose = True
veryverbose = True
# Optimization parameters
maxchi      = 1.5
#nproc = np.int16(np.max([os.cpu_count()-20,1]))
nproc = 30
if os.name == 'nt':
    nproc = 1

######################################################

# Program flow
calcwind  = False # Calculate the wind/outflow(s)?
calcabscl = True  # Calculate absorbing cloud(s)?

######################################################
if calcabscl: # Do just the first quasar
  myquasar = Quasar(verbose,datapath,cloudypath,zqso[0],                  # Program flow
                    inclination[0],robs[0],raqso[0],decqso[0],            # Observer parms
                    mbh[0], sbh[0],                                       # Black hole parms
                    nr[0], rlo[0], rhi[0], mdot[0], alpha[0],             # Disk parms
                    xclp, yclp, zcl, rhoindex, logrhoscale, logrho0, vcl, # Absorbing clouds
                    anum,ion,trandx,                                      # Atomic physics parameters
                    plot_code,vlo,vhi,vres,nproc,                         # Spectral synthesis parameters
                    wind = False, abscloud = True,                        # Optional program flow
                    minlox=9.9,                                           # Optional atomic physics parameters
                    dtheta_fac = 0.5                                      # Optional accretion disk parameters
                    )
  print(breakstr)

  if True:
    # Remove the last cloud
    #if len(myquasar.clouds) > 2:
    #myquasar.write_clouds([myquasar.clouds[0]])
    #input("Removed clouds")
    #myquasar.clouds = myquasar.read_clouds()
    myquasar.readspec("WISEA J114047", "WISEA_J114047", redospline=False, vlo=vlo, vhi=vhi)
    #myquasar.readspec("7C 1138", "7C-1138", redospline=False)
    myquasar._build_modwave(wres = vres * np.average(myquasar.mydata.new_w) / const.c.to(u.km/u.s))
    print(breakstr)
    clouds = myquasar.fitabs(maxchi,
                             first_time = os.path.exists(myquasar.cloud_filename),
                             nr = 150, ntheta = 60, maxiter = 100, mcmin = True
                             )

elif calcwind:
  myquasar = Quasar(verbose,datapath,cloudypath,zqso[0],                  # Program flow
                    inclination[1],robs[1],raqso[0],decqso[0],            # Observer parms
                    mbh[1], sbh[1],                                       # Black hole parms
                    nr[1], rlo[1], rhi[1], mdot[1], alpha[1],             # Disk parms
                    xclp, yclp, zcl, rhoindex, logrhoscale, logrho0, vcl, # Absorbing clouds
                    anum,ion,tran,                                        # Atomic physics parameters
                    plot_code,vlo,vhi,vres,nproc,                         # Spectral synthesis parameters
                    wind = True, abscloud = False                         # Optional program flow
                   )
  
