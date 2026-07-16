# TBD List:
# "clean up" means to standardize the coding format (e.g., function definitions),
#                                 parameter names (e.g., with underscores),
#                  to remove commented and unused code
# "test imports" means to determine if each of the imported packages needs to be there
# General  -- AALSynth documentation and user guide
#          -- RL NV COS paper
# aalsynth -- Add wind modelling
# AbsCloud -- Clean up, test imports
#          -- Revise criterion for running new Cloudy model (e.g., in between IPs that have been simlated previously)
#          -- Improve placement of new clouds
# atomic   -- Clean up, test imports
# cloudy   -- Clean up, test imports
#          -- Test for existence of Cloudy_runs/ subdirectory and creation if not there
#          -- Effect of allowing metalicity to deviate from solar
# doppler  -- Clean up, test imports
# hstqso   -- Clean up, test imports
#          -- Fix datapath
# mcgv     -- Get steady-state velocity field
#          -- Speed up force multiplier calculation --> use quasar._calculate_absorbed_flux_gaussleg
#          -- Cloudy simulations for emissivity/source function
# ntdisk   -- Clean up, test imports
#          -- Slim disk, ADAF models                          #FUTURE
#          -- MAD models                                      #FUTURE
# quasar   -- Clean up, test imports
#          -- Incorporate emission lines into integration of absorbed flux
#          -- Fitting routine -- parameter uncertainties
#                             -- potential for fixed parameters  #FUTURE
# readpars -- Add wind parameters?

import os
import sys

import matplotlib.pyplot  as plt
import numpy              as np

from astropy  import constants as const
from astropy  import units     as u
from quasar   import Quasar
from readpars import readpars

#######################################################################################
breakstr = "#" * 50

if len(sys.argv) > 1:
  inputfile = sys.argv[1]
else:
  inputfile = "AALSynth_input.txt"

inputfile = "AALSynth_input.txt"

print(breakstr)
print(breakstr)
print("Welcome to AALSynth. Your one-stop-shop for modelling associated absorbers.")
print(breakstr)

######################################################
mypars = readpars(inputfile)
#input("Please review input parameters")
######################################################
myquasar = Quasar(mypars)
######################################################
if mypars.calcabscl:
  print(breakstr)

  myquasar.readspec()
  myquasar._build_modwave(wres = mypars.vres * np.average(myquasar.mydata.new_w) / const.c.to(u.km/u.s))
  print(breakstr)
  myquasar.clouds = myquasar.fitabs()
  
  myquasar.printpars()
  myquasar._abs_write_clouds(myquasar.clouds)
