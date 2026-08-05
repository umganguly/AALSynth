# TBD List:
# "clean up" means to standardize the coding format (e.g., function definitions),
#                                 parameter names (e.g., with underscores),
#                  to remove commented and unused code
# "test imports" means to determine if each of the imported packages needs to be there
# General  -- AALSynth documentation and user guide
#          -- RL NV COS paper
# aalsynth -- Add wind modelling
# AbsCloud -- Clean up, test imports
#          -- Improve placement of new clouds
# atomic   -- Clean up, test imports
# cloudy   -- Clean up, test imports
#          -- Test for existence of Cloudy_runs/ subdirectory and creation if not there
#          -- Effect of allowing metalicity to deviate from solar
# corona   -- Placement of corona tuned to UV/X-ray photometric data?
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
import re
import sys
import copy

import matplotlib.pyplot  as plt
import numpy              as np

from AbsCloud            import AbsCloud
from astropy             import constants as const
from astropy             import units     as u
from astropy.io          import ascii
from quasar              import Quasar
from readpars            import readpars

#######################################################################################
def print_main_menu():
  print("AALSynth Main menu:")
  print("-----------------------------")
  print("A. Absorption cloud menu")
  print("D. accretion Disk menu")
  print("W. Emission lines menu")
  print("-----------------------------")
  print("Q. Enough! I quit!")

def interactive_main(myquasar):
  print_main_menu()
  done = False
  while not done:
    print("-----------------------------")
    user_choice = input("Choose, but choose wisely: ")
    match user_choice:
      case 'A':
        myquasar.clouds = interactive_absorption(myquasar)
      case 'D':
        myquasar = interactive_disk(myquasar)
        print("Nope")
      case 'W':
        print("Nope")
      case 'Q':
        print("Bye!")
        done = True

    if user_choice != 'Q':
      print_main_menu()

  return myquasar

#######################################################################################
# Accretion disk
def print_disk_menu(myquasar):
  print("AALSynth Accretion disk menu:")
  print("-----------------------------")
  print("Black hole parameters:")
  print("-----------------------------")
  print(f"M. change black hole Mass ({myquasar.mypars.mbh})")
  print(f"S. change black hole Spin ({myquasar.mypars.sbh})")
  print("-----------------------------")
  print("Disk parameters:")
  print("-----------------------------")
  print(f"A. change Accretion rate ({myquasar.mypars.mdot})")
  print(f"D. change Dtheta_fac ({myquasar.mypars.dtheta_fac})")
  print(f"H. change rHi (outer radius - {myquasar.mypars.rhi})")
  print(f"L. change rLo (outer radius - {myquasar.mypars.rlo})")
  print(f"N. change Nr (number of annuli - {myquasar.mypars.nr})")
  print(f"V. change Shakura-Sunyaev Viscosity parameter alpha ({myquasar.mypars.viscosity_alpha})")
  print("-----------------------------")
  print("Q. Enough! I quit!")
  print("R. Remake the disk")

def interactive_disk(myquasar):
  myquasar.mydisk.pltdisk()
  print_disk_menu(myquasar)
  done = False
  while not done:
    print("-----------------------------")
    user_choice = input("Choose, but choose wisely: ")
    match user_choice:
      case 'A':
        new_mdot = input("Enter new accretion rate (or None for no change): ")
        if new_mdot != "None":
          myquasar.mypars.mdot = float(new_mdot) * u.M_sun / u.yr
      case 'D':
        new_dtheta_fac = input("Enter new dtheta_fac (or None for no change): ")
        if new_dtheta_fac != "None":
          myquasar.mypars.dtheta_fac = float(new_dtheta_fac)
      case 'H':
        new_rhi = input("Enter new rhi (or None for no change): ")
        if new_rhi != "None":
          myquasar.mypars.rhi = float(new_rhi)
      case 'L':
        new_rlo = input("Enter new rlo (or None for no change): ")
        if new_rlo != "None":
          myquasar.mypars.rlo = float(new_rlo)
      case 'M':
        new_mass = input("Enter new black hole mass (or None for no change): ")
        if new_mass != "None":
          myquasar.mypars.sbh = float(new_mass) * u.M_sun
      case 'N':
        new_nr = input("Enter new nr (or None for no change): ")
        if new_nr != "None":
          myquasar.mypars.nr = int(new_nr)
      case 'Q':
        print("Bye!")
        done = True
      case 'R':
        myquasar.mydisk.mypars = myquasar.mypars
        print("\tRecalculating disk")
        myquasar.mydisk.makedisk()
        print("\tRedetermining disk photosphere")
        myquasar.mydisk.photosphere(overwrite=True)
        print("Reinitializing corona")
        myquasar.mycorona.__init__(myquasar.mypars, myquasar.mydisk)
        myquasar.mycorona.activate_lamppost()
      case 'S':
        new_spin = input("Enter new black hole spin (or None for no change): ")
        if new_spin != "None":
          myquasar.mypars.sbh = float(new_spin)
      case 'V':
        new_alpha = input("Enter new viscosity parameter alpha (or None for no change): ")
        if new_alpha != "None":
          myquasar.mypars.viscosity_alpha = float(new_alpha)

    if user_choice != 'Q':
      myquasar.mydisk.pltdisk()
      print_disk_menu(myquasar)

  return myquasar

#######################################################################################
# Absorbing clouds
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
  print("F. Minimize Chi^2 using current clouds")
  print("M. Toggle minimization method")
  print("R. Refesh all clouds")
  print("S. Switch chosen cloud")
  print("V. change Velocity limits")
  print('W. Write clouds to file')
  print("-----------------------------")
  print("Q. Enough! I quit!")

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
                                 np.array([0.0]) * (u.km/u.s)
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
      case 'm':
        new_logZ = input("Enter new log Z (or None for no change): ")
        if new_logZ != "None":
          clouds[cloud_index].logZ = float(new_logZ)
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
          clouds[cloud_index].vlos = float(new_velocity) * (u.km/u.s)
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
        rcl, zcl, thetacl, logrhoscale, rhoindex, logrho0, logZ, vcl = myquasar.grab_cloud_pars(clouds)
        (xclp, yclp) = myquasar._abs_project_clouds(rcl, zcl, thetacl)
        zcl = np.append(zcl, 100000.0)
        xclp = np.append(xclp, 0.0)
        yclp = np.append(yclp, 0.0)
        print(f"Adding a new cloud at xclp=0, yclp=0, zclp={zcl[-1]} to cloud list")
        new_xcl, new_ycl = myquasar._abs_deproject_clouds(xclp, yclp, zcl)
        rcl = np.sqrt(new_xcl*new_xcl + new_ycl*new_ycl)
        thetacl = np.atan2(new_ycl, new_xcl)
        print(f"\t--> xcl = {new_xcl[-1]}, ycl = {new_ycl[-1]} --> rcl = {rcl[-1]}, theta_cl = {thetacl[-1].to(u.deg)}")
        clouds = myquasar.makeclouds(xclp, 
                                     yclp, 
                                     zcl, 
                                     np.append(rhoindex, 2.0), 
                                     np.append(logrhoscale, 16.0), 
                                     np.append(logrho0, 3.5), 
                                     np.append(logZ, 0.5), 
                                     np.append(vcl.to(u.km/u.s).value, 0.0) * (u.km/u.s), 
                                     ntabs = 1)
      case 'D':
        del_cloud_index = np.int16(input("Which cloud do you want to delete? "))
        try:
          clouds.pop(del_cloud_index)
          if del_cloud_index < cloud_index and cloud_index > 0:
            cloud_index -= 1
        except IndexError:
          print("Nope.")
      case 'F':
        myquasar.clouds = copy.deepcopy(clouds)
        dummy_add_clouds = myquasar.mypars.add_clouds
        myquasar.mypars.add_clouds = False
        clouds = myquasar.fitabs()
        myquasar.mypars.add_clouds = dummy_add_clouds
      case 'M':
        myquasar.mypars.mcmin = not myquasar.mypars.mcmin
        prtstr = 'Minimization method set to '
        if myquasar.mypars.mcmin:
          prtstr += 'mcmin'
        else:
          prtstr += 'scipy least squares'
        print(prtstr)
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
      case 'V':
        new_vlo = input("Enter new lower velocity limit (or None for no change): ")
        if new_vlo != "None":
          myquasar.mypars.vlo = float(new_vlo) * (u.km/u.s)
        new_vhi = input("Enter new upper velocity limit (or None for no change): ")
        if new_vhi != "None":
          myquasar.mypars.vhi = float(new_vhi) * (u.km/u.s)
        nvel = np.int16((myquasar.mypars.vhi-myquasar.mypars.vlo)/myquasar.mypars.vres)
        myquasar.velocity   = np.linspace(start = myquasar.mypars.vlo,
                                          stop  = myquasar.mypars.vhi,
                                          num   = nvel
                                          )
      case 'W':
        myquasar._abs_write_clouds(clouds)

    if user_choice != 'Q':
      myquasar.print_clouds(clouds)
      rcl, zcl, thetacl, logrhoscale, rhoindex, logrho0, logZ, vcl = myquasar.grab_cloud_pars(clouds)
      myquasar._abs_plot(totflux, unabsflux,
                         vcl = vcl,
                         clouds = clouds
                         )
      (xclp, yclp) =  myquasar._abs_project_clouds(rcl, 
                                                   zcl, 
                                                   thetacl
                                                   ) 
      print_abs_menu(clouds,
                     xclp,
                     yclp,
                     cloud_index = cloud_index)
      print("-----------------------------")
      user_choice = input("Choose, but choose wisely: ")

  return clouds

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

inputfile = "AALSynth_input.txt"

if len(sys.argv) > 1:
  inputfile = sys.argv[1]


print(breakstr)
print(breakstr)
print("Welcome to AALSynth. Your one-stop-shop for modelling associated absorbers.")
print(breakstr)

######################################################
mypars = readpars(inputfile)
######################################################
print(breakstr)
myquasar = Quasar(mypars)
print(breakstr)
myquasar.readspec()
myquasar._build_modwave(wres = mypars.vres * np.average(myquasar.mydata.new_w) / const.c.to(u.km/u.s))
######################################################
print(breakstr)
myquasar.readspec()
myquasar._build_modwave(wres = mypars.vres * np.average(myquasar.mydata.new_w) / const.c.to(u.km/u.s))
######################################################
print(breakstr)
if mypars.interactive:
  dummy_show_geometry = mypars.showgeometry
  mypars.showgeometry = True
  myquasar = interactive_main(myquasar)
  mypars.showgeometry = dummy_show_geometry
elif mypars.calcabscl:
  print(breakstr)

  myquasar.clouds = myquasar.fitabs()
  
  myquasar.printpars()
  myquasar._abs_write_clouds(myquasar.clouds)
