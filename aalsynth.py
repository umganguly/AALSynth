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
import copy

import matplotlib.pyplot  as plt
import numpy              as np

from astropy  import constants as const
from astropy  import units     as u
from quasar   import Quasar
from readpars import readpars

#######################################################################################
def print_abs_menu(clouds,
                   xclp,
                   yclp,
                   cloud_index = 0):
  print("AALSynth Absorber Cloud menu:")
  print("-----------------------------")
  if clouds is not None:
    print(f"Current cloud: {cloud_index}")
    print("-----------------------------")
    print(f"i. change number density profile Index ({clouds[cloud_index].rhoindex})")
    print(f"m. change Metallicity of cloud ({clouds[cloud_index].logZ})")
    print(f"n. change Number density at clouds surface (log n = {clouds[cloud_index].logrho0})")
    print(f"s. change Size of cloud (log rhoscale = {clouds[cloud_index].logrhoscale})")
    print(f"v. change Velocity ({clouds[cloud_index].vlos})")
    print(f"x. change Xclp ({xclp[cloud_index]})")
    print(f"y. change Yclp ({yclp[cloud_index]})")
    print(f"z. change Zcl ({clouds[cloud_index].zcl})")
  else:
    print("No clouds chosen.")

  print("-----------------------------")
  print("A. Add a cloud (copies the last one)")
  print("D. Delete a cloud from the cloud list")
  print("R. Refesh all clouds")
  print("S. Switch chosen cloud")
  print('W. Write clouds to file')
  print("-----------------------------")
  print("Q. Enough! I quit!")

#######################################################################################
def interactive_absorption(myquasar):
  if myquasar.clouds is not None:
    clouds = copy.deepcopy(myquasar.clouds)
  else:
    print("There are no clouds! Fine, I'll make one.")
    clouds = myquasar.makeclouds(np.array([0.0]),
                                 np.array([0.0]), 
                                 np.array([1.0e+7]),
                                 np.array([1.0]), 
                                 np.array([16.0]), 
                                 np.array([3.5]), 
                                 np.array([0.3]),
                                 np.array([0.0])
                                 )

  cloud_index = 0
  user_choice = 'R'
  done = False
  while not done:
    match user_choice:
      case 'i':
        new_rhoindex = input("Enter new rhoindex (or None for no change): ")
        if new_rhoindex != "None":
          clouds[cloud_index].rhoindex = float(new_rhoindex)
      case 'n':
        new_logrho0 = input("Enter new log n0 (or None for no change): ")
        if new_logrho0 != "None":
          clouds[cloud_index].logrho0 = float(new_logrho0)
      case 's':
        new_logrhoscale = input("Enter new log n scale (or None for no change): ")
        if new_logrhoscale != "None":
          clouds[cloud_index].logrhoscale = float(new_logrhoscale)
      case 'v':
        new_velocity = input("Enter new velocity (or None for no change): ")
        if new_velocity != "None":
          clouds[cloud_index].vlos = float(new_velocity)
      case 'x':
        new_xclp = input("Enter new projected x coordinate (or None for no change): ")
        if new_xclp != "None":
          (xclp, yclp) =  myquasar._abs_project_clouds(clouds[cloud_index].rcl, 
                                                       clouds[cloud_index].zcl, 
                                                       clouds[cloud_index].thetacl
                                                       )
          print(f"Changing projected coordinates ({xclp}, {yclp}) --> ({new_xclp}, {yclp})")
          new_xcl, new_ycl = myquasar._abs_deproject_clouds(float(new_xclp), yclp,
                                                            clouds[cloud_index].zcl
                                                            )
          new_rcl = np.sqrt(new_xcl*new_xcl + new_ycl*new_ycl)
          new_thetacl = np.atan2(new_ycl, new_xcl).to(u.deg)
          xcl = clouds[cloud_index].rcl * np.cos(clouds[cloud_index].thetacl)
          ycl = clouds[cloud_index].rcl * np.sin(clouds[cloud_index].thetacl)
          print(f"Changing actual coordinates ({xcl}, {ycl}) --> ({new_xcl}, {new_ycl})  [Cartesian]")
          print(f"Changing actual coordinates ({clouds[cloud_index].rcl}, {clouds[cloud_index].thetacl}) --> ({new_rcl}, {new_thetacl})  [Polar]")
          clouds[cloud_index].rcl = new_rcl
          clouds[cloud_index].thetacl = new_thetacl
      case 'y':
        new_yclp = input("Enter new projected y coordinate (or None for no change): ")
        if new_yclp != "None":
          (xclp, yclp) =  myquasar._abs_project_clouds(clouds[cloud_index].rcl, 
                                                       clouds[cloud_index].zcl, 
                                                       clouds[cloud_index].thetacl
                                                       )
          print(f"Changing projected coordinates ({xclp}, {yclp}) --> ({xclp}, {new_yclp})")
          new_xcl, new_ycl = myquasar._abs_deproject_clouds(xclp, float(new_yclp),
                                                            clouds[cloud_index].zcl
                                                            )
          new_rcl = np.sqrt(new_xcl*new_xcl + new_ycl*new_ycl)
          new_thetacl = np.atan2(new_ycl, new_xcl).to(u.deg)
          xcl = clouds[cloud_index].rcl * np.cos(clouds[cloud_index].thetacl)
          ycl = clouds[cloud_index].rcl * np.sin(clouds[cloud_index].thetacl)
          print(f"Changing actual coordinates ({xcl}, {ycl}) --> ({new_xcl}, {new_ycl})  [Cartesian]")
          print(f"Changing actual coordinates ({clouds[cloud_index].rcl}, {clouds[cloud_index].thetacl}) --> ({new_rcl}, {new_thetacl})  [Polar]")
          clouds[cloud_index].rcl = new_rcl
          clouds[cloud_index].thetacl = new_thetacl
      case 'z':
        new_zcl = input("Enter new z coordinate (or None for no change): ")
        if new_zcl != "None":
          clouds[cloud_index].zcl = float(new_zcl)
      ###########################################################################
      case 'A':
        print("Adding copy of last cloud to cloud list")
        clouds.append(clouds[-1])
      case 'D':
        del_cloud_index = np.int16(input("Which cloud do you want to delete? "))
        try:
          clouds.pop(del_cloud_index)
          if del_cloud_index < cloud_index and cloud_index > 0:
            cloud_index -= 1
        except IndexError:
          print("Nope.")
      case 'Q':
        print("Bye!")
        done = True
      case 'R':
        clouds = myquasar.refresh_clouds(clouds)
        (totflux, unabsflux) = myquasar._calculate_absorbed_flux_gaussleg(clouds)
        myquasar._abs_plot(totflux, unabsflux,
                           vcl = myquasar.grab_cloud_pars(clouds)[-1],
                           clouds = clouds
                           )
      case 'S':
        myquasar.print_clouds(clouds)
        cloud_index = np.int16(input("Which cloud do you want? "))
      case 'W':
        myquasar._abs_write_clouds(clouds)

    if user_choice != 'Q':
      myquasar.print_clouds(clouds)
      rcl, zcl, thetacl, logrhoscale, rhoindex, logrho0, logZ, vcl = myquasar.grab_cloud_pars(clouds)
      (xclp, yclp) =  myquasar._abs_project_clouds(rcl, 
                                                   zcl, 
                                                   thetacl
                                                   ) 
      print_abs_menu(clouds,
                     xclp,
                     yclp,
                     cloud_index = cloud_index)
      user_choice = input("Choose, but choose wisely: ")
    else:
      pass

  return clouds

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
######################################################
myquasar = Quasar(mypars)
print(breakstr)
myquasar.readspec()
myquasar._build_modwave(wres = mypars.vres * np.average(myquasar.mydata.new_w) / const.c.to(u.km/u.s))
######################################################
if mypars.interactive:
  dummy_show_geometry = mypars.showgeometry
  mypars.showgeometry = True
  myquasar.clouds = interactive_absorption(myquasar)

elif mypars.calcabscl:
  print(breakstr)

  myquasar.clouds = myquasar.fitabs()
  
  myquasar.printpars()
  myquasar._abs_write_clouds(myquasar.clouds)
