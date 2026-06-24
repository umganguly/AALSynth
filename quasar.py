import copy
import os
import matplotlib

import matplotlib.patches as pth
import matplotlib.pyplot  as plt
import numpy              as np
import time               as tm

from astropy                 import constants as const
from astropy                 import units as u
from astropy.units           import Quantity
from astropy.convolution     import convolve, Gaussian1DKernel
from astropy.coordinates     import SkyCoord
from astropy.cosmology       import LambdaCDM
from astropy.io              import ascii
from astropy.modeling.models import BlackBody
from astropy.modeling.functional_models import Voigt1D
from astropy.table           import Table
from astropy.visualization   import astropy_mpl_style, quantity_support
from functools               import partial
from multiprocessing         import Pool
from numpy.polynomial        import chebyshev
from scipy.interpolate       import CubicSpline, interp1d
from scipy.optimize          import minimize, least_squares
from scipy.stats             import f as Ftest

from AbsCloud                import AbsCloud
from atomic                  import atomic
from corona                  import corona
from doppler                 import calcvel,calcwave
from hstqso                  import hstqso
from mcgv                    import mcgv
from ntdisk                  import ntdisk
from readpars                import readpars
from tqdm                    import tqdm

class Quasar:
  #def __init__(self,verbose,datapath,cloudypath,zqso,                # Program flow
  #             inclination,robs,ra,dec,                              # Observer parameters
  #             mbh, sbh,                                             # Black Hole parameters
  #             nr, rlo, rhi, mdot, alpha,                            # Accretion disk parameters
  #             xclp, yclp, zcl, rhoindex, logrhoscale, logrho0, vcl, # Absorber parameters
  #             anum,ion,trandx,                                      # Atomic physics parameters
  #             plot_code,vlo,vhi,vres,nproc,                         # Spectral synthesis parameters
  #             wind = True, abscloud = True,                         # Optional program flow
  #             minlox=8.5,                                           # Optional atomic physics parameters
  #             dtheta_fac = 2.0,                                     # Optional accretion disk parameters
  #             absfile = "Clouds.fits"                               # Optional absorber parameters
  #             ):
  def __init__(self,mypars):
    self.verbose     = mypars.verbose
    self.datapath    = mypars.datapath
    self.cloudypath  = mypars.cloudypath
    self.zqso        = mypars.zqso
    self.inclination = mypars.inclination
    self.robs        = mypars.robs
    self.ra          = mypars.ra
    self.dec         = mypars.dec
    self.mbh         = mypars.mbh
    self.sbh         = mypars.sbh
    self.nr          = mypars.nr
    self.rlo         = mypars.rlo
    self.rhi         = mypars.rhi
    self.mdot        = mypars.mdot
    self.alpha       = mypars.alpha
    self.xclp        = mypars.xclp
    self.yclp        = mypars.yclp
    self.zcl         = mypars.zcl
    self.rhoindex    = mypars.rhoindex
    self.logrhoscale = mypars.logrhoscale
    self.logrho0     = mypars.logrho0
    self.vcl         = mypars.vcl

    ###############################################################################
    self.pltcount = 0
    self.nproc    = mypars.nproc

    ###############################################################################
    print("#" * 50)
    print("Grabbing atomic data")
    self.myatoms = atomic(mypars.datapath,900 * u.Angstrom,3000 * u.Angstrom,minlox=mypars.minlox)

    self.datapath    = mypars.datapath
    self.cloudypath  = mypars.cloudypath
    self.anum        = mypars.anum
    self.ion         = mypars.ion
    self.trandx      = mypars.trandx
    self.plot_code   = mypars.plot_code

    self.vres       = mypars.vres
    self.velocity   = np.linspace(mypars.vlo,mypars.vhi,num=np.int16((mypars.vhi-mypars.vlo)/mypars.vres))

    fu = (u.erg / (u.s * u.cm * u.cm * u.Hz))
    ###############################################################################
    print("Setting observer")
    self.zqso        = mypars.zqso
    self.inclination = mypars.inclination
    self.skycoord    = SkyCoord(ra=mypars.ra,dec=mypars.dec)

    ###############################################################################
    print("Initializing disk")
    self.mydisk = ntdisk(mypars.sbh, mypars.mbh,
                         mypars.mdot, mypars.alpha,
                         mypars.inclination, mypars.robs,
                         mypars.nr, mypars.rlo, mypars.rhi,
                         self.datapath,
                         dtheta_fac = mypars.dtheta_fac)
    comove_dist = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7).comoving_distance(self.zqso)
    self.robs =  (comove_dist * np.sin(self.inclination) / self.mydisk.rg).decompose()
    self.zobs = self.robs / np.tan(self.inclination)
    
    print("\tCalculating disk")
    self.mydisk.makedisk()
    print("\tDetermining disk photosphere")
    self.mydisk.photosphere()

    ###############################################################################
    print("Initializing corona")
    self.mycorona = corona(self.mydisk)
    self.mycorona.activate_lamppost()

    ###############################################################################
    if wind:
      print("Initializing wind...")
      self.mywind = mcgv(self.mydisk, self.mycorona, self.myatoms, 90, self.datapath)
      forcemultfile  = self.datapath+f"Sbh{self.mydisk.sbh}-MBH{np.log10(self.mydisk.mbh / const.M_sun):.2f}"
      forcemultfile += f"-Mdot{(self.mydisk.mdot/(const.M_sun/u.year)).decompose()}-alpha{self.mydisk.alpha}"
      forcemultfile += f"-fmultgrid_{self.mywind.nr}x{self.mywind.ntheta}.fits"
      print(f"\tLooking for {forcemultfile}")
      if os.path.exists(forcemultfile):
          print("\t\t\tReading "+forcemultfile)
          data = Table.read(forcemultfile, format="fits")
          self.mywind.Mrgrid     = np.array(data['Mrgrid'])
          self.mywind.Mthetagrid = np.array(data['Mthetagrid'])

      # IN DEVELOPMENT
      which_cells = self.mywind.boundary_mask & ((self.mywind.Mrgrid == 0) | (self.mywind.Mthetagrid == 0))

      if np.sum(which_cells) > 0:
        plt.ion()
        plt.figure()
        if os.name == 'nt':
          plt.get_current_fig_manager().window.state("zoomed")

        t1 = tm.time() * u.s
        for rdx in range(self.mywind.nr):
          which_cells = self.mywind.boundary_mask & (self.mywind.RR == self.mywind.r[rdx]) & \
            ((self.mywind.Mrgrid == 0) | (self.mywind.Mthetagrid == 0))
          if np.sum(which_cells) > 0:
            for (pnum,title,colarr) in [(1, r'$\log M_\mathrm{r}$',
                                         np.log10(    self.mywind.Mrgrid[:-1,:-1] + 0.1*np.min(np.extract(    self.mywind.Mrgrid > 0.0, self.mywind.Mrgrid)))),
                                        (2, r'$\log M_\mathrm{\theta}$',
                                         np.log10(self.mywind.Mthetagrid[:-1,:-1] + 0.1*np.min(np.extract(self.mywind.Mthetagrid > 0.0, self.mywind.Mthetagrid))))
                                        ]:
              plt.subplot(2,1,pnum)
              plt.cla()
              plt.title(title)
              plt.pcolormesh(self.mywind.RR_cyl, self.mywind.ZZ_cyl, colarr,  shading='flat')
              plt.xlabel(r'r ($r_g$)')
              plt.ylabel(r'z ($r_g$)')
              plt.plot(self.mydisk.rstar, self.mydisk.diskheight)
              plt.plot(self.mydisk.rstar, self.mydisk.zt1)
              for pltr in np.logspace(np.log10(self.mydisk.rstar[0]),np.log10(self.mydisk.rstar[-1]),num=10):
                  plt.plot(pltr * np.cos(self.mywind.theta), pltr * np.sin(self.mywind.theta), 'k:', alpha=0.1)
              for pltth in (np.linspace(0.0, np.pi/2.0, num=90) * u.rad):
                  plt.plot(self.mydisk.rstar * np.cos(pltth), self.mydisk.rstar * np.sin(pltth), 'k:', alpha=0.1)
              plt.xlim(left = self.mydisk.rstar[0]) #, right = 3.0e+3)
              plt.ylim(bottom = 0.3) #, top = 3.0e+3)
              plt.xscale("log")
              plt.yscale("log")
              plt.colorbar()
              plt.tight_layout()
            plt.pause(0.1)

            prtstr  = f"\t\tr[{rdx}] = {self.mywind.r[rdx] / self.mydisk.rg} rg "
            prtstr += f"--> {np.sum(which_cells)} cells {self.mywind._mcgv_timer(t1)} / {(self.mywind._mcgv_timer(t1)) * (self.mywind.nr-rdx)/rdx}"
            print(prtstr)
            mrg,mrt = self.mywind.fmultfunc(which_cells, verbose=False)
            print(f"\t\t\tMrg = [{np.min(mrg)},{np.max(mrg)}]  Mrt = [{np.min(mrt)},{np.max(mrt)}]")
            self.mywind.Mrgrid[    which_cells] = mrg
            self.mywind.Mthetagrid[which_cells] = mrt

            data = Table(data=[self.mywind.Mrgrid,self.mywind.Mthetagrid], names=["Mrgrid","Mthetagrid"])
            data.write(forcemultfile, format="fits", overwrite=True)

      self.mywind.calcstreamline_relativistic(dtime = 10.0 * u.s,
                                              mindt = 1.0 * u.s,
                                              vres = 0.5 * const.c.to(u.km/u.s),
                                              minr_rg = 10.0,
                                              plotstream=True,
                                              mupdate = False
                                              )
      #  .
      #  .
      #  .
    else:
      self.mywind = None

    ###############################################################################
    if abscloud:
      print("Initializing absorbing clouds")
      self.reset_observer()
      self.cloud_filename = self.datapath+mypars.abscloudfile
      print(f"\tLooking for {self.cloud_filename}")
      if os.path.exists(self.cloud_filename):
        print(f"\t\tFound it!")
        self.clouds = self.read_clouds()
      else:
        self.clouds = None
      self.bestfit = None
    else:
      self.clouds = None

  #######################################################################################
  def _abs_bounds(self,
                  clouds
                  ):
    abs_lower_bounds = []
    abs_upper_bounds = []
    for cld in clouds:
      abs_lower_bounds.append(                               -self.mydisk.rstar[-1] ) # xcl
      abs_lower_bounds.append(                               -self.mydisk.rstar[-1] ) # ycl
      abs_lower_bounds.append(10.0**(1.5 + cld.logrhoscale) * u.cm / self.mydisk.rg ) # zcl
      abs_lower_bounds.append(                                                  0.0 ) # rhoindex
      abs_lower_bounds.append(                                cld.logrhoscale - 3.0 ) # logrhoscale
      abs_lower_bounds.append(                                    cld.logrho0 - 4.0 ) # logrho0
      abs_lower_bounds.append(   (cld.vlos - 100.0 * (u.km/u.s)).to(u.km/u.s).value ) # vlos

      abs_upper_bounds.append(                             self.mydisk.rstar[-1] ) # xcl
      abs_upper_bounds.append(                             self.mydisk.rstar[-1] ) # ycl
      abs_upper_bounds.append(        (1.0 * u.kpc / self.mydisk.rg).decompose() ) # zcl
      abs_upper_bounds.append(                                               5.0 ) # rhoindex
      abs_upper_bounds.append(                             cld.logrhoscale + 1.5 ) # logrhoscale
      abs_upper_bounds.append(                                 cld.logrho0 + 4.0 ) # logrho0
      abs_upper_bounds.append((cld.vlos + 100.0 * (u.km/u.s)).to(u.km/u.s).value ) # vlos

    return (abs_lower_bounds,abs_upper_bounds)
    
  #######################################################################################
  # Callback routine for the scipy.optimize.minimize fitter
  def _abs_callback(self,
                    intermediate_result
                    ):
    clouds = self._abs_unpack(intermediate_result.x)
    # Observer coordinates (reset here for sanity)
    self.reset_observer()

    (totflux, unabsflux) = self._calculate_absorbed_flux_gaussleg(clouds, lograd = True)
    chisq = np.sum(self._abs_chisq(totflux, unabsflux))
    self.bestfit = totflux/unabsflux
    self.print_clouds(clouds, ntabs=2)
    try:
      print(f"\t\tChi^2 = {chisq}  ({tm.time() * u.s - self._abscall_t0:e} since last callback)")
      self._abs_plot(totflux, unabsflux, 0, 0, self._abscall_t0)
    except NameError:
      print(f"\t\tChi^2 = {chisq}")
      self._abs_plot(totflux, unabsflux, 0, 0, 0 * u.s)

    self._abscall_t0 = tm.time() * u.s

    # Write out clouds to a file so that we can pick up where we left off...
    self.write_clouds(clouds)

  #######################################################################################
  # Compute the chisq, summing across all max(f-lambda) species that are covered in the velocity range specified
  def _abs_chisq(self,
                 totflux, unabsflux
                 ):
    try:
      chisq_spec = np.zeros((self.anum.size, self.obswave.size))
      cosflux    = self.mydata._lsf_convolve(self.wavelength, totflux.value)/unabsflux.value
      for i in range(self.anum.size):
        myatoms_index = self.myatoms.getspecies(self.anum[i],
                                                self.ion[i]
                                                )[self.trandx[i]]
        velocity_mask = (self.obsvel[myatoms_index,:] > self.velocity[0]) & (self.obsvel[myatoms_index,:] < self.velocity[-1])

        chisq_anum = np.square((self.normobsflux[velocity_mask] - np.interp(self.obsvel[myatoms_index,velocity_mask],
                                                                            np.squeeze(calcvel(self.wavelength,
                                                                                               np.array(self.myatoms.wave[myatoms_index])
                                                                                               )
                                                                                       ),
                                                                            cosflux
                                                                            )
                                )/self.normobsferr[velocity_mask]
                               )
        chisq_spec[i,velocity_mask] = chisq_anum

    except AttributeError:
      chisq_spec = np.zeros((self.anum.size, self.wavelength.size))

    self.chisq_spec = chisq_spec
    return chisq_spec

  #######################################################################################
  # Wrapper for minimization purposes - the function that is to be minimized
  def _abs_chisqfunc(self,
                     x,
                     nr = 300, ntheta = 300,
                     verbose = False
                     ):
    clouds = self._abs_unpack(x, verbose = verbose)

    # Observer coordinates (reset here for sanity)
    self.reset_observer()

    (totflux, unabsflux) = self._calculate_absorbed_flux_gaussleg(clouds, nr = nr, ntheta = ntheta, verbose = verbose)
    chisq_spec = self._abs_chisq(totflux, unabsflux)

    return chisq_spec.flatten()

  #######################################################################################
  def _abs_deproject_clouds(self,
                            xclp, yclp,
                            zcl
                            ):
    self.reset_observer()
    a       = -1 - self.zobs / (zcl - self.zobs)
    xcl     = (xclp - self.robs * np.cos(self.mydisk.thetaobs)) / (1 + a) + self.robs * np.cos(self.mydisk.thetaobs)
    ycl     = yclp / (1 + a)

    return xcl, ycl
  
  #######################################################################################
  def _abs_mcminimize(self,
                      clouds,
                      nr = 300, ntheta = 300,
                      maxiter = 1000,
                      step_size = 1.0,
                      minstep = 1.0e-5,
                      verbose = False
                      ):

    ncl = len(clouds)
    self.reset_observer()
    (totflux, unabsflux) = self._calculate_absorbed_flux_gaussleg(clouds,
                                                                  nr = nr,
                                                                  ntheta = ntheta,
                                                                  verbose = verbose
                                                                  )
    chisq = np.sum(self._abs_chisq(totflux,
                                   unabsflux
                                   )
                   )
    self.bestfit = totflux/unabsflux
    x = self._abs_pack(clouds)

    better_clouds = copy.deepcopy(clouds)
    niter = maxiter * x.size
    rng = np.random.default_rng()
    good_direction = False
    check_negative_direction = False
    while niter > 0 and step_size > minstep:
      self.reset_observer()
      (rcl, zcl, thetacl, logrhoscale, rhoindex, logrho0, vcl) = self.grab_cloud_pars(better_clouds)
      (xclp, yclp) = self._abs_project_clouds(rcl,
                                              zcl,
                                              thetacl
                                              )

      if not good_direction:
        if not check_negative_direction:
          dx = 2 * rng.uniform(size=x.shape) - 1.0
          dx /= np.sqrt(np.sum(dx*dx))
          check_negative_direction = True
        else:
          dx = -dx
          check_negative_direction = False

      # Change the parameters
      print("\t"+"-"*20)
      print(f"\tProposing changes to clouds (step size = {step_size}, good_direction = {good_direction}, check_negative_direction = {check_negative_direction})...")
      maxrad = self.mydisk.rstar[-1] + 10.0**logrhoscale * u.cm / self.mydisk.rg

      pos_step = 1.0
      while np.any(np.fabs(xclp + pos_step * step_size * dx[0:ncl]) > maxrad):
        pos_step *= 0.95
      xclp += pos_step * step_size * dx[0:ncl]

      pos_step = 1.0
      while np.any(np.fabs(yclp + pos_step * step_size * dx[0:ncl]) > maxrad):
        pos_step *= 0.95
      yclp += pos_step * step_size * dx[ncl:2*ncl]

      logrhoscale += step_size * dx[3*ncl:4*ncl]

      zmin = np.interp(rcl, self.mydisk.rstar, self.mydisk.zt1) + 10.0**(1+logrhoscale) * u.cm / self.mydisk.rg
      zcl += step_size * dx[2*ncl:3*ncl]
      zmask = zcl < zmin
      if np.any(zmask):
        print(f"\t\tResetting heights for clouds {np.arange(ncl)[zmask]}: {zcl[zmask]} < {zmin[zmask]}")
        zcl[zmask] = zmin[zmask]

      rhoindex    += step_size * dx[4*ncl:5*ncl]
      logrho0     += step_size * dx[4*ncl:5*ncl]

      vcl += 5.0 * step_size * dx[6*ncl:7*ncl] * (u.km/u.s)

      # Make the potential new clouds
      potential_clouds = self.makeclouds(xclp, yclp, zcl,
                                         rhoindex, logrhoscale, logrho0,
                                         vcl
                                         )
      self.reset_observer()

      print("\tTrying the following changes:")
      self.print_diff_clouds(better_clouds,
                             potential_clouds,
                             ntabs=2)

      # Determine chisq
      (newtotflux, newunabsflux) = self._calculate_absorbed_flux_gaussleg(potential_clouds,
                                                                          nr = nr,
                                                                          ntheta = ntheta,
                                                                          verbose = verbose
                                                                          )
      newchisq = np.sum(self._abs_chisq(newtotflux,
                                        newunabsflux
                                        )
                        )

      # If the chisq is better, then keep it and continue.
      # If it is not better, increment the iteration number and try again until the iteration number is bunk
      if newchisq < chisq:
        print(f"\n\tIMPROVED FIT! iterations remaining: {niter}    chisq = {chisq} - {chisq - newchisq}\n")
        x             = self._abs_pack(potential_clouds)
        better_clouds = self._abs_unpack(x)
        chisq         = newchisq
        totflux       = newtotflux
        unabsflux     = newunabsflux
        self.bestfit  = totflux/unabsflux

        if newchisq < chisq-1.0:
          niter = maxiter * x.size

        good_direction = True
        step_size *= 1.05

        self.write_clouds(better_clouds)
        self._abscall_t0 = tm.time() * u.s
      else:
        good_direction = False
        niter -= 1
        step_size *= 0.95

        print(f"\n\tKeeping old fit! iterations remaining: {niter}    chisq = {chisq} + {newchisq-chisq}")

      self.print_clouds(better_clouds, ntabs=2)

    return better_clouds, chisq
                     
  #######################################################################################
  # Wrapper to take a cloud class and pack the parameters into a 1D array to feed into scipy.optimize.minimize
  def _abs_pack(self,
                clouds
                ):
    x = np.array([])
    for cld in clouds:
      xclp, yclp = self._abs_project_clouds(cld.rcl, cld.zcl, cld.thetacl)
      x = np.append(x, [xclp, yclp, cld.zcl, cld.rhoindex, cld.logrhoscale, cld.logrho0, cld.vlos.value])

    return x

  #######################################################################################
  # Make a plot of the normalized absorption spectra (observed and predicted)
  def _abs_plot(self,
                totflux, unabsflux,
                ncur, ntot,
                t0,
                vcl = np.array([])
                ):
    plt.clf()

    plt.plot(self.velocity, np.zeros(self.velocity.size), "k--")
    for tdx in range(self.anum.size):
      myatoms_index = self.myatoms.getspecies(self.anum[tdx],
                                              self.ion[tdx]
                                              )[self.trandx[tdx]]
      plt.plot(self.velocity, np.ones(self.velocity.size) + tdx, "k--")
      #----------------------------------------------------------
      try:
        plt.step(self.obsvel[myatoms_index,:],
                 self.normobsflux + tdx,
                 c=self.plot_code[tdx],
                 label=f"{self.myatoms.specstr[myatoms_index]}")
      except AttributeError:
        print("Oops.. no data read in yet...")
      #----------------------------------------------------------
      try:
        plt.plot(self.species_velocity[:,myatoms_index],
                 totflux/unabsflux + tdx,
                 self.plot_code[tdx])
      except IndexError:
        print("Oops... totflux don't have 'nuff dimensions")
      #----------------------------------------------------------
      try:
        if self.bestfit is not None:
          plt.plot(self.species_velocity[:,myatoms_index],
                   self.bestfit + tdx,
                   f"{self.plot_code[tdx]}--")
      except AttributeError:
        print("Oops... no bestfit yet")

    if vcl.size > 0:
      for v in vcl:
        v_nounits = v.to(u.km/u.s).value
        plt.plot(np.array([v_nounits,v_nounits]),
                 np.array([-0.2, 1.2*self.anum.size]),
                 "k--")

    plt.legend()
    
    if ntot > 0:
      title_str  = f"sightline = {ncur}/{ntot} time: {tm.time()-t0:.3f} "
    else:
      title_str = ""

    title_str += r"$\chi^2 = $"
    chisq_spec = self._abs_chisq(totflux, unabsflux)
    chisq = np.sum(chisq_spec)
    title_str += f"{chisq:.3f}"
    
    if ntot > 0:
      title_str += f"  est: {(tm.time()-t0) * (ntot/(ncur+1.0e-30) - 1):.3f} / {(tm.time()-t0) * (ntot/(ncur+1.0e-30)):.3f}"
    try:
      title_str += f" {tm.time() * u.s - self._abscall_t0:.3f}"
    except:
      title_str += f" {tm.time():.3f}"

    plt.title(title_str)
    plt.xlim([self.velocity[0].to(u.km/u.s).value, self.velocity[-1].to(u.km/u.s).value])
    plt.ylim([-0.2, 1.2*self.anum.size])
    plt.show(block=False)
    plt.pause(0.001)

  #######################################################################################
  def _abs_project_clouds(self,
                          rcl, zcl, thetacl
                          ):
    self.reset_observer()
    xobs = self.robs * np.cos(self.mydisk.thetaobs)
    yobs = self.robs * np.sin(self.mydisk.thetaobs)

    a    = - zcl / (zcl - self.zobs) # --> 0 if zobs >>> zcl
    xclp = -a * xobs + (1+a) * rcl * np.cos(thetacl)
    yclp = -a * yobs + (1+a) * rcl * np.sin(thetacl)

    return xclp, yclp
  
  #######################################################################################
  # Wrapper to take the parameters fed into/from scipy.optimize.minimize and unpack it into a cloud class
  def _abs_unpack(self,
                  x,
                  verbose = False
                  ):
    if len(x)//7 > 0:
      xclp        = np.zeros(len(x)//7)
      yclp        = np.zeros(len(x)//7)
      zcl         = np.zeros(len(x)//7)
      rhoindex    = np.zeros(len(x)//7)
      logrhoscale = np.zeros(len(x)//7)
      logrho0     = np.zeros(len(x)//7)
      vcl         = np.zeros(len(x)//7) * (u.km/u.s)
      for i in range(len(x)//7):
        xclp[i]        = x[7*i]
        yclp[i]        = x[7*i+1]
        zcl[i]         = x[7*i+2]
        rhoindex[i]    = x[7*i+3]
        logrhoscale[i] = x[7*i+4]
        logrho0[i]     = x[7*i+5]
        vcl[i]         = x[7*i+6] * (u.km/u.s)

      clouds = self.makeclouds(xclp, yclp, zcl, rhoindex, logrhoscale, logrho0, vcl, verbose = False)
      if verbose:
        self.print_clouds(clouds, ntabs=1)
    else:
      clouds = None

    return clouds

  #######################################################################################
  def _add_to_cheb(self,
                   cloud
                   ):
    print("\t\tExpanding Chebyshev fits with new Cloudy run...")
    (lognuFnu, log_ion_parm) = self._calc_ion_parm(cloud)

    self.cheb_log_ion_parm_list = np.append(self.cheb_log_ion_parm_list, log_ion_parm)

    # Iterate through ions and the ion fraction...
    ionfrac = np.zeros(cloud.iondensity.shape)
    for i in range(self.myatoms.idx.size):
      ionfrac[:,i] = cloud.iondensity[:,i] / (cloud.density * np.power(10.0, self.myatoms.abund[self.myatoms.idx[i]]-12))
    if self.cheb_ionfrac_list.size > 0:
      self.cheb_ionfrac_list = np.append(self.cheb_ionfrac_list, ionfrac, axis=0)
    else:
      self.cheb_ionfrac_list = np.copy(ionfrac)
      self.cheb_degree = np.ones(self.myatoms.idx.size, dtype=np.int16)
      self.cheb_coeff_list = np.zeros((self.myatoms.idx.size,1))

    lnmin = np.min(self.cheb_log_ion_parm_list)
    lnmax = np.max(self.cheb_log_ion_parm_list)
    xnorm = 2 * (self.cheb_log_ion_parm_list - lnmin)/(lnmax-lnmin) - 1
    sdx = np.argsort(xnorm)
    for i in range(self.myatoms.idx.size):
      done = False
      oldchisq = 9.99e+99
      olddeg = self.cheb_degree[i]
      ion_mask = self.cheb_ionfrac_list[:,i] > 0
      if not np.any(ion_mask):
        self.cheb_degree[i] = 1
        coeff = np.array([0.0])
      else:
        while not done:
          coeff,res = chebyshev.chebfit(xnorm[ion_mask],np.log10(self.cheb_ionfrac_list[ion_mask,i]), self.cheb_degree[i], full=True)
          if res[0].size > 0:
            fstat = (oldchisq / olddeg )  / (np.squeeze(res[0]) / self.cheb_degree[i])
            p_value = Ftest.sf(fstat, olddeg, self.cheb_degree[i])
            if p_value < 0.48:
              self.cheb_degree[i] += 1
              done = False
            else:
              self.cheb_degree[i] = np.max(np.array([1, self.cheb_degree[i]-1]))
              coeff,res = chebyshev.chebfit(xnorm[ion_mask],np.log10(self.cheb_ionfrac_list[ion_mask,i]), self.cheb_degree[i], full=True)
              done = True
          else:
            self.cheb_degree[i] = np.max(np.array([1, self.cheb_degree[i]-1]))
            coeff,res = chebyshev.chebfit(xnorm[ion_mask],np.log10(self.cheb_ionfrac_list[ion_mask,i]), self.cheb_degree[i], full=True)
            done = True
          olddeg = self.cheb_degree[i]
          oldchisq = np.squeeze(res[0])

          if coeff.size > self.cheb_coeff_list.shape[1]:
            dum = np.copy(self.cheb_coeff_list)
            self.cheb_coeff_list = np.zeros((self.myatoms.idx.size, coeff.size  ))
            self.cheb_coeff_list[:,:dum.shape[1]] = np.copy(dum)
            self.cheb_coeff_list[i,:] = 0.0
          self.cheb_coeff_list[i,:coeff.size] = np.copy(coeff)

    chebfile  = self.datapath+f"/Cloudy_runs/Sbh{self.mydisk.sbh}-MBH{np.log10(self.mydisk.mbh / const.M_sun):.2f}"
    chebfile += f"-Mdot{(self.mydisk.mdot/(const.M_sun/u.year)).decompose()}-alpha{self.mydisk.alpha}_ionfracs_lox{self.myatoms.minlox}.fits"
    datatab = Table(data=[self.cheb_log_ion_parm_list, self.cheb_ionfrac_list], names=["log_ion_parm_list", "ionfrac_list"])
    datatab.write(chebfile, format="fits", overwrite=True)
    datatab2 = Table(data=[self.cheb_degree,self.cheb_coeff_list], names=["DEGREE","COEFFS"])
    datatab2.write(chebfile, format="fits", append=True)

  #######################################################################################
  def _build_modwave(self,
                     wres = 0.01 * u.Angstrom):
    nwave = np.int64(((np.max(self.obswave) - np.min(self.restwave)) / wres).decompose())
    print(f"Building model wavelength ranges from {np.min(self.restwave)} to {np.max(self.obswave)} in {nwave} {wres}-bins")
    self.wavelength = np.linspace(np.min(self.restwave.to(u.Angstrom).value),
                                  np.max(self.obswave.to(u.Angstrom).value),
                                  nwave
                                  ) * u.Angstrom

    print("Calculating velocity ranges for posterity")
    self.species_velocity = calcvel(self.wavelength,self.myatoms.wave).to(u.km/u.s)

  #######################################################################################
  def _calc_ion_parm(self,
                     cloud
                     ):
    fu = (u.erg / (u.s * u.cm * u.cm * u.Hz))
    lognuFnu = np.interp((const.Ryd).to(u.Hz, equivalencies=u.spectral()),
                         cloud.ionspecfreq,
                         np.log10((cloud.ionspecfreq * cloud.ionspecflux).value))
    log_ion_parm = lognuFnu - np.log10(cloud.density.to(u.cm**-3) * \
                                       ( (const.h * const.c *const.Ryd).to(u.erg) * \
                                         (const.c.to(u.cm/u.s)) ) / (fu * u.Hz))

    return lognuFnu, log_ion_parm

  #######################################################################################
  def _calculate_absorbed_flux_gaussleg(self,
                                        clouds,
                                        nr         = 300,
                                        ntheta     = 300,
                                        nproc      = 20,
                                        robs       = None,
                                        thetaobs   = None,
                                        zobs       = None,
                                        wavelength = None,
                                        debug      = False,
                                        lograd     = False,
                                        verbose    = False,
                                        noplot     = False
                                        ):
    fu = (u.erg / (u.s * u.cm * u.cm * u.Hz))
    if verbose:
      print("\tSetting up integration for spectral synthesis")
    t0 = tm.time()
    if wavelength is None:
      wavelength = self.wavelength
    totflux    = np.zeros(wavelength.shape) * fu
    unabsflux  = np.zeros(wavelength.shape) * fu

    if robs is None:
      robs     = self.robs
      thetaobs = 0.0
      zobs     = self.zobs

    self.reset_observer(robs     =    robs,
                        thetaobs = thetaobs,
                        zobs      =    zobs
                        )
    robs_vec = np.array([robs * np.cos(thetaobs),
                         robs * np.sin(thetaobs),
                         zobs
                         ])
    if debug:
      print(f"robs_vec = {robs_vec}")

    done = False
    scale = 1
    while not done:
      done = True
      # Set up Gauss-Legendre grid for integration
      if verbose:
        print(f"\t\tDetermining Gauss-Legendre positions and weights for a {scale*nr} x {scale*ntheta} grid ({tm.time()-t0})")
      gaussleg_y_r,     gaussleg_w_r     = np.polynomial.legendre.leggauss(scale*nr)     # Cylindrical radius (normalized)
      gaussleg_y_theta, gaussleg_w_theta = np.polynomial.legendre.leggauss(scale*ntheta) # Azimuhtal angle (normalized)

      plt.ion()
      if os.name == 'nt':
        plt.get_current_fig_manager().window.state("zoomed")

      pool_tuple_input = []
      for rdx in range(nr):
        if lograd:
          rdisk = self.mydisk.rstar[0] * np.power(self.mydisk.rstar[-1]/self.mydisk.rstar[0], (gaussleg_y_r[rdx] + 1)/2) # Units rg (log)
        else:
          rdisk = self.mydisk.rstar[0] + (self.mydisk.rstar[-1] - self.mydisk.rstar[0]) * (gaussleg_y_r[rdx] + 1) / 2 # Units rg (linear)

        thetadisk = np.pi * (gaussleg_y_theta + 1.) # Azimuthal angle
      
        pool_tuple_input.append((rdisk,
                                 thetadisk,
                                 clouds,
                                 robs_vec,
                                 gaussleg_y_r[rdx],
                                 gaussleg_w_r[rdx],
                                 gaussleg_w_theta,
                                 wavelength,
                                 lograd
                                 )
                                )

      if verbose:
        print(f"\t\tIntegrating across disk with {nproc} processors...({tm.time()-t0})")

      min_impact_parameters_all_sightlines = None
      if nproc > 1:
        with Pool(nproc) as pool:
          istart = 0
          iend = nproc
          while iend < len(pool_tuple_input):
            flux_output_tuple = pool.starmap(self._flux_sightline, pool_tuple_input[istart:iend])
            for (fluxrtnu, optical_depth, min_impact_parameter) in flux_output_tuple:  # fluxrtnu.shape = optical_depth.shape = (wavelength.size, thetadisk.size)
              totflux += np.sum(fluxrtnu * np.exp(-optical_depth), axis=-1) # Sum over sightlines
              unabsflux += np.sum(fluxrtnu, axis=-1) # Sum over sightlines
              if min_impact_parameter is not None:
                try:
                  min_impact_parameters_all_sightlines = np.append(min_impact_parameters_all_sightlines, [min_impact_parameter], axis=0)
                except ValueError:
                  min_impact_parameters_all_sightlines = np.array([min_impact_parameter])
            if not noplot:
              self._abs_plot(totflux, unabsflux,
                             iend, len(pool_tuple_input), t0,
                             self.grab_cloud_pars(clouds)[-1]
                             )
            istart = iend
            iend = np.min([istart+nproc,len(pool_tuple_input)])
      else:
        iend = 0
        while iend < len(pool_tuple_input):
          (fluxrtnu, optical_depth, min_impact_parameter) = self._flux_sightline(*pool_tuple_input[iend]) # fluxrtnu.shape = optical_depth.shape = (wavelength.size, thetadisk.size)
          totflux += np.sum(fluxrtnu * np.exp(-optical_depth), axis=-1) # Sum over sightlines
          unabsflux += np.sum(fluxrtnu, axis=-1) # Sum over sightlines
          if min_impact_parameter is not None:
            try:
              min_impact_parameters_all_sightlines = np.append(min_impact_parameters_all_sightlines, [min_impact_parameter], axis=0)
            except ValueError:
              min_impact_parameters_all_sightlines = np.array([min_impact_parameter])
          if iend % 10 == 90 and not noplot:
            self._abs_plot(totflux, unabsflux,
                           iend, len(pool_tuple_input), t0,
                           self.grab_cloud_pars(clouds)[-1]
                           )
          iend += 1
      if verbose:
        print(f"\t\tIntegration complete...({tm.time()-t0})\n")

      if clouds is not None:
        min_impact_parameters_all_sightlines = np.min(min_impact_parameters_all_sightlines, axis=0)
        ip_mask = min_impact_parameters_all_sightlines > 1
        if np.any(ip_mask):
          scale *= np.int16(np.max(np.array([np.sqrt(np.max(min_impact_parameters_all_sightlines[ip_mask])), 2])))
          done = False
          prtstr  = f"\t\tThe following cloud"
          if np.sum(ip_mask) > 1:
            prtstr += f"s were "
          else:
            prtstr += f" was "
          prtstr += f"never intercepted: {np.arange(len(clouds))[ip_mask]}."
          prtstr += f" The minimum impact parameter"
          if np.sum(ip_mask) > 1:
            prtstr += f"s were "
          else:
            prtstr += f" was "
          prtstr += f"{min_impact_parameters_all_sightlines[ip_mask]} cloud radii"
          print(prtstr)
          print(f"\t\tIncreasing scale to {scale}")
          if np.any(scale*np.array([nr,ntheta]) > 1000):
            print(f"\t\t\tThis is would be too expensive - bailing")
            done = True

    return totflux,unabsflux # shape = wavelength.shape

  #######################################################################################
  def _cld_all_optical_depth(self,
                             clouds,           # List of AbsClouds
                             rdisk_vec, R_vec, # 2D nd.arrays with shapes (3,thetadisk.size)
                             wavelength        # 1d nd.array
                             ):

    optical_depth = np.zeros((wavelength.size, rdisk_vec[0,:].size)) # shape = (wavelength.size, thetadisk.size)

    if clouds is not None:
      rcl, zcl, thetacl, logrhoscale, rhoindex, logrho0, vcl  = self.grab_cloud_pars(clouds)
      ncl = len(clouds)

      xcl = rcl * np.cos(thetacl)
      ycl = rcl * np.sin(thetacl)

      Rmag_squared = np.sum(R_vec*R_vec, axis=0)

      min_impact_parameter = 10.0 * self.mydisk.rstar[-1] * self.mydisk.rg / (np.array([np.max(clouds[cdx].radius.to(u.cm).value) for cdx in range(ncl) ]) * u.cm)

      for cdx in range(ncl):
        rcl_vec = np.broadcast_to(np.array([xcl[cdx],
                                            ycl[cdx],
                                            zcl[cdx]
                                            ]),
                                  rdisk_vec.T.shape
                                  ).T
        R_dot_rclmrdisk = np.sum(R_vec * (rcl_vec - rdisk_vec), axis=0)

        impact_parameter_vec = (R_dot_rclmrdisk/Rmag_squared) * R_vec - rcl_vec + rdisk_vec

        impact_parameter  = np.sqrt(np.sum(impact_parameter_vec * impact_parameter_vec, axis=0)) * self.mydisk.rg

        min_impact_parameter[cdx] = np.min([min_impact_parameter[cdx],
                                            np.min(impact_parameter)/np.max(clouds[cdx].radius)
                                            ]
                                           )

        impact_parameter_mask = impact_parameter < np.max(clouds[cdx].radius)
        if np.any(impact_parameter_mask):
          cld_optical_depth = self._cld_optical_depth(impact_parameter, # 1D nd.array with rdisk_vec[0,:].size
                                                      clouds[cdx],      # AbsCloud
                                                      wavelength        # 1D nd.array
                                                      )                 # shape (wavelength.shape + impact_parameter.shape + cloud.radius.shape)
          optical_depth += cld_optical_depth.sum(axis=-1) # This sums over all radial shells pierced by the sightlines
    else:
      min_impact_parameter = None

    return optical_depth, min_impact_parameter

  #######################################################################################
  def _cld_optical_depth(self,
                         impact_parameter, # 1D nd.array
                         cloud,            # AbsCloud
                         wavelength        # 1D nd.array
                         ): 
    optical_depth = np.zeros(wavelength.shape + impact_parameter.shape + cloud.radius.shape)

    tau_const = (2 * np.sqrt(np.pi) * const.e.esu * const.e.esu / (const.m_e * const.c) ).decompose()

    if np.any(impact_parameter < np.max(cloud.radius)):

      ip_ratio = np.outer(impact_parameter,
                          1 / (cloud.radius + 1 * u.cm)
                          ) # shape (impact_parameter.size, cloud.radius.size)
      ip_ratio[ip_ratio > 1] = 1
      dxarray = np.broadcast_to(cloud.dr.to(u.cm), ip_ratio.shape) * np.sqrt(1 - np.square(ip_ratio ) ) * u.cm  # shape (impact_parameter.size, cloud.radius.size)

      if wavelength.shape == self.wavelength.shape:
        velocity = self.species_velocity
      else:
        velocity = calcvel(wavelength,
                           self.myatoms.wave
                           )  # shape (wavelength.size, self.myatoms.wave.size)

      for myatoms_index in range(self.myatoms.anum.size):
        bvalue = np.sqrt(2 * const.k_B * np.broadcast_to(cloud.temperature.to(u.K),
                                                         dxarray.shape) * u.K / self.myatoms.amass[myatoms_index]) # shape (impact_parameter.size, cloud.radius.size)
        dvel = np.outer((velocity[:,myatoms_index] - cloud.vlos),
                        1 / bvalue
                        ).decompose().reshape((wavelength.shape)+(bvalue.shape)) # shape (wavelength.shape + impact_parameter.shape + cloud.radius.shape)

        dvel_mask = np.abs(velocity[:,myatoms_index] - cloud.vlos)/np.average(bvalue) < 300.0
        if np.any(dvel_mask):
          which_ion_index = (self.myatoms.ions == 100*self.myatoms.anum[myatoms_index]+self.myatoms.ion[myatoms_index])
        
          column_density = dxarray * np.broadcast_to(np.squeeze(cloud.iondensity[:,which_ion_index]),
                                                     dxarray.shape) * (u.cm**-3) # shape (impact_parameter.size, cloud.radius.size)
          tau0 = (tau_const * self.myatoms.flam[myatoms_index] * column_density / bvalue ).decompose() # shape (impact_parameter.size, cloud.radius.size)

          column_density = np.broadcast_to(column_density.to(u.cm**-2), dvel[dvel_mask,:,:].shape) * (u.cm**-2)
          bvalue = np.broadcast_to(bvalue.to(u.km/u.s),                 dvel[dvel_mask,:,:].shape) * (u.km/u.s)
          tau0   = np.broadcast_to(tau0,                                dvel[dvel_mask,:,:].shape)

          if self.myatoms.gamma[myatoms_index].value > 0:
            a = (self.myatoms.gamma[myatoms_index] * self.myatoms.wave[myatoms_index] / bvalue).decompose()
            V1   = Voigt1D(x_0         = 0,
                           amplitude_L = 1/(np.pi * a),
                           fwhm_L      = 2 * a,
                           fwhm_G      = 2 * Voigt1D.sqrt_ln2
                           )
            optical_depth_species = tau0 * V1(dvel[dvel_mask,:,:])
          else:
            optical_depth_species = tau0 * np.exp(-np.square(dvel[dvel_mask,:,:]))

          optical_depth[dvel_mask,:,:] += optical_depth_species

    return optical_depth # shape (wavelength.shape + impact_parameter.shape + cloud.radius.shape)

  #######################################################################################
  def _flux_sightline(self,
                      rdisk, thetadisk, # Disk parameters. rdisk is scalar, thetadisk is 1D nd.array
                      clouds,           # Clouds parameters (list of AbsClouds)
                      robs_vec,         # Observer parameters (nd.array with shape (3,) )
                      gaussleg_y_r, gaussleg_w_r, gaussleg_w_theta, # Integration parameters [y = radial/log-radial locations, w = weights]
                      wavelength,       # Spectral parameters (i.e., wavelengths in a 1D nd.array)
                      lograd            # boolean for quadrature method
                      ):
    fu = (u.erg / (u.s * u.cm * u.cm * u.Hz))

    zt1cs = CubicSpline(self.mydisk.rstar, self.mydisk.zt1)
    Tt1cs = CubicSpline(self.mydisk.rstar, self.mydisk.tempt1.to(u.K).value)

    frequency = wavelength.to(u.Hz, equivalencies=u.spectral())

    rdisk_vec = np.array([rdisk * np.cos(thetadisk),
                          rdisk * np.sin(thetadisk),
                          np.broadcast_to(zt1cs(rdisk), (thetadisk.size,))
                          ])

    dzt1dr = zt1cs(rdisk, nu=1)
    Tt1    = Tt1cs(rdisk) * u.K

    gradz_vec = np.array([-dzt1dr * np.cos(thetadisk),
                          -dzt1dr * np.sin(thetadisk),
                          np.ones(thetadisk.size)
                          ])
    gradzmag = np.sqrt(np.sum(gradz_vec*gradz_vec, axis=0))

    R_vec = np.broadcast_to(robs_vec, rdisk_vec.T.shape).T - rdisk_vec
    Rmag = np.sqrt(np.sum(R_vec*R_vec, axis=0))
    if(np.any(Rmag == 0)):
      input("Rmag is zero")

    cosbeta = np.array([np.dot(R_vec[:,i], gradz_vec[:,i]) for i in range(thetadisk.size)]) / (gradzmag * Rmag)

    # With the doppler shift, we have to figure out what frequencies are being emitted by the patch that are observed at "frequency"
    # So, we have to unshift the frequency array for each patch
    disk_velpar_vec = np.array([-np.sin(thetadisk) / rdisk,
                                np.cos(thetadisk) / rdisk,
                                np.zeros(thetadisk.size)
                                ]) # shape (3, thetadisk.size)
    disk_lorentz_fac = 1./np.sqrt(np.sum(disk_velpar_vec*disk_velpar_vec, axis=0))  # shape (thetadisk.size,)
    disk_velpar_dot_Rhat = np.sum(disk_velpar_vec*R_vec, axis=0) / Rmag             # shape (thetadisk.size,)
    doppler_beam_fac = 1. / (disk_lorentz_fac * (1.0 - disk_velpar_dot_Rhat))       # shape (thetadisk.size,)
    doppler_unshift_freq = np.outer(frequency,
                                    np.sqrt((1 + disk_velpar_dot_Rhat) / (1 + disk_velpar_dot_Rhat))
                                    ) # shape = (wavelength.size, thetadisk.size)

    Bnu = np.zeros(doppler_unshift_freq.shape) * fu / u.sr
    Bnu_mask = const.h * doppler_unshift_freq / (const.k_B * Tt1) < 670.74 # This is to prevent underflows... 670.74 was just determined through trial and error
    Bnu[Bnu_mask] = BlackBody().evaluate(doppler_unshift_freq[Bnu_mask],Tt1,1) # Emitted at doppler_unshift_freq, observed at gaussleg_freq

    if lograd:
      fluxrtnu  = np.pi * ((self.mydisk.rstar[0])**2) * (np.log(self.mydisk.rstar[-1]/self.mydisk.rstar[0])) * np.power(self.mydisk.rstar[-1]/self.mydisk.rstar[0],
                                                                                                                        gaussleg_y_r + 1)
    else:
      fluxrtnu = np.pi * (self.mydisk.rstar[-1] - self.mydisk.rstar[0]) * rdisk / 2.0

    fluxrtnu *=  (Bnu * u.sr) * gaussleg_w_r * gaussleg_w_theta * (doppler_beam_fac**3) * cosbeta / Rmag**2

    optical_depth, min_impact_parameter = self._cld_all_optical_depth(clouds,           # List of AbsClouds
                                                                      rdisk_vec, R_vec, # 2D nd.arrays with shapes (3,thetadisk.size)
                                                                      wavelength        # 1d nd.array
                                                                      ) # shape = (wavelength.size, thetadisk.size)

    return fluxrtnu, optical_depth, min_impact_parameter

  #######################################################################################
  def _read_cheb_files(self):
    chebfile  = self.datapath+f"/Cloudy_runs/Sbh{self.mydisk.sbh}-MBH{np.log10(self.mydisk.mbh / const.M_sun):.2f}"
    chebfile += f"-Mdot{(self.mydisk.mdot/(const.M_sun/u.year)).decompose()}-alpha{self.mydisk.alpha}_ionfracs_lox{self.myatoms.minlox}.fits"
    if os.path.exists(chebfile):
      datatab1 = Table.read(chebfile, hdu=1)
      datatab2 = Table.read(chebfile, hdu=2)

      self.cheb_log_ion_parm_list = datatab1["log_ion_parm_list"]
      self.cheb_ionfrac_list      = datatab1["ionfrac_list"]
      self.cheb_coeff_list        = datatab2["COEFFS"]
      self.cheb_degree            = datatab2["DEGREE"]
    else:
      self.cheb_log_ion_parm_list = np.array([])
      self.cheb_ionfrac_list      = np.array([])
      self.cheb_coeff_list        = np.array([])
      self.cheb_degree            = np.array([])
      

  #######################################################################################
  def fitabs(self,
             maxchi,
             first_time = True,
             nproc = 1,
             nr = 300, ntheta = 300,
             maxiter = 100,
             mcmin = False
             ):
    plt.close('all')
    plt.ion()
    if os.name == 'nt':
      plt.get_current_fig_manager().window.state("zoomed")

    fu = (u.erg / (u.s * u.cm * u.cm * u.Hz))
    print(f"Calculating spectrum from initial parameters")
    self.printpars()
    # Reset observer location...
    self.reset_observer()
    # Initial Chisq...
    (totflux,unabsflux) = self._calculate_absorbed_flux_gaussleg(self.clouds,
                                                                 nr = nr, ntheta = ntheta,
                                                                 nproc = 20, verbose = True)
    self.bestfit = totflux/unabsflux
    self._abs_plot(totflux, unabsflux, 0, 0, 0)
    chisq = np.sum(self._abs_chisq(totflux, unabsflux))
    print(f"Initial chisq = {chisq}")

    if self.clouds is not None:
      (rcl, zcl, thetacl, logrhoscale, rhoindex, logrho0, vcl) = self.grab_cloud_pars(self.clouds)
      xclp,yclp = self._abs_project_clouds(rcl, zcl, thetacl)
    else:
      first_time = False

    list_of_bad_velocities = np.array([]) * (u.km/u.s)
    res = None
    oldres = None
    done = False
    while not done:
      print("Determining what velocity to put a new component...")
      tot_chisq_spec = np.zeros(self.velocity.size)
      for i in range(self.anum.size):
        velocity_mask = (self.obsvel[i,-1] > self.velocity) & (self.obsvel[i,0] < self.velocity) & \
          (np.interp(self.velocity, self.obsvel[i,:], self.bigew, left=0, right=0) > np.interp(self.velocity, self.obsvel[i,:], self.bigsew, left=0, right=0))
        for bad_velocity in list_of_bad_velocities:
          velocity_mask = velocity_mask & ((self.velocity < bad_velocity - self.vres) | (self.velocity > bad_velocity + self.vres))
        tot_chisq_spec[velocity_mask] += np.interp(self.velocity[velocity_mask], self.obsvel[i,:], self.chisq_spec[i,:])

      potential_bad_vel = np.extract(tot_chisq_spec == np.max(tot_chisq_spec), self.velocity)[0]

      if not first_time:
        print(f"\t... and adding it at {potential_bad_vel} with badness {np.max(tot_chisq_spec)}")
        try:
          xclp        = np.append(                  xclp,                                  0.0)
          yclp        = np.append(                  yclp,                                  0.0)
          rhoindex    = np.append(              rhoindex,                                  2.0)
          logrhoscale = np.append(           logrhoscale,                                 15.5)
          logrho0     = np.append(               logrho0,                                  2.5)
          vcl         = np.append(vcl.to(u.km/u.s).value, potential_bad_vel.to(u.km/u.s).value) * (u.km/u.s)

          zcl_prop = (np.random.rand(1) * u.kpc / self.mydisk.rg).decompose()
          while (zcl_prop < 10.0**(1.5 + logrhoscale[-1]) * u.cm / self.mydisk.rg) or (zcl_prop > (1.0 * u.kpc / self.mydisk.rg).decompose()):
            zcl_prop = (np.random.rand(1) *  u.kpc / self.mydisk.rg).decompose()
          zcl         = np.append(                   zcl, zcl_prop)
        except:
          xclp        = np.array([ 0.0])
          yclp        = np.array([ 0.0]) 
          rhoindex    = np.array([ 2.0])
          logrhoscale = np.array([15.5])
          logrho0     = np.array([ 3.5])
          vcl         = np.array([potential_bad_vel.to(u.km/u.s).value]) * (u.km/u.s)

          zcl = (np.random.rand(1) * u.kpc / self.mydisk.rg).decompose()
          while (zcl < 10.0**(1.5 + logrhoscale[-1]) * u.cm / self.mydisk.rg) or (zcl > (1.0 * u.kpc / self.mydisk.rg).decompose()):
            zcl = (np.random.rand(1) * u.kpc / self.mydisk.rg).decompose()

        clouds = self.makeclouds(xclp, yclp, zcl,
                                 rhoindex, logrhoscale, logrho0,
                                 vcl,
                                 verbose = True,
                                 nr = nr, ntheta = ntheta
                                 )
        self.write_clouds(clouds)
      else:
        print(f"\t... but not actually adding it...sigh... (would have been {potential_bad_vel})")
        first_time = False
        clouds = self.clouds

      self.reset_observer()

      print("#" * 50)
      print(f"Beginning optimization with {len(clouds)} clouds...")
      self._abscall_t0 = tm.time() * u.s
      oldchisq = chisq
      t0 = tm.time()
      if mcmin:
        mcminimize_clouds, chisq = self._abs_mcminimize(clouds,
                                                        nr = nr, ntheta = ntheta,
                                                        maxiter = maxiter,
                                                        verbose = True
                                                        )
        x = self._abs_pack(mcminimize_clouds)
      else:
        x = self._abs_pack(clouds)
        if not res is None:
          oldres = res.copy()
        res = least_squares(self._abs_chisqfunc,
                            x,
                            bounds=self._abs_bounds(clouds),
                            jac="2-point",
                            callback=self._abs_callback,
                            kwargs = {'nr': nr, 'ntheta': ntheta, 'verbose': True}
                            )
        x = res.x
        
      print(f"Optimized (in {tm.time()-t0} seconds)!  Cleaning up...") 
      if res.success:
        print("\tSupposedly, the least-squares fit was successful")
        chisq = np.sum(self._abs_chisqfunc(res.x, nr = nr, ntheta = ntheta))
        self.clouds = self._abs_unpack(res.x)
      elif mcmin:
        self.clouds = mcminimize_clouds
      else:
        print("\tSomething barfed")
      print(f"   Chi^2 = {chisq}")
      # Was adding this cloud a statistically significant improvement in the fit?
      # We need to run F-test
      dof = 0
      for i in range(self.anum.size):
        velocity_mask = (self.obsvel[i,:] > self.velocity[0]) & (self.obsvel[i,:] < self.velocity[-1])
        dof += np.sum(velocity_mask)
      dof -= 7 * len(self.clouds)
      old_dof = dof - 7
      if mcmin:
        F_stat = (oldchisq / old_dof) / (chisq / dof)
      else:
        F_stat = (oldchisq / old_dof) / (res.cost / dof)

      p_value = Ftest.sf(F_stat, dof, old_dof)
      print(f"\tF-stat = {F_stat} --> probablility that the new and old fits are statistically consistent {p_value}")
      if p_value < 0.05 or first_time:
        print("\t\tKEEPING NEW FIT!")
        clouds = self._abs_unpack(x)
        self.write_clouds(clouds)
        self.reset_observer()
        if not mcmin:
          chisq = np.sum(self._abs_chisq(*self._calculate_absorbed_flux_gaussleg(self.clouds,
                                                                                 nr = nr, ntheta = ntheta,
                                                                                 nproc = 20, verbose = True
                                                                                 )
                                         )
                         )
        first_time = False
      else:
        print("\t\tRESETTING BACK TO OLD FIT AND FLAGGING VELOCITY!")
        list_of_bad_velocities = np.append(list_of_bad_velocities.to(u.km/u.s).value,
                                           potential_bad_vel.to(u.km/u.s).value) * (u.km/u.s)
        xclp = xclp[:-1]
        yclp = yclp[:-1]
        zcl  = zcl[:-1]
        rhoindex = rhoindex[:-1]
        logrhoscale = logrhoscale[:-1]
        logrho0 = logrho0[:-1]
        vcl = vcl[:-1]
        # reset res to the old fit
        if oldres is not None:
          res = oldres.copy()
        else:
          res = None
        chisq = oldchisq
  
      # Do we need another cloud?
      if np.max(self.chisq_spec) < maxchi:
        done = True

    return clouds

  #######################################################################################
  def grab_cloud_pars(self,
                      clouds
                      ):
    if clouds is not None:
      rcl         = np.array([cld.rcl                     for cld in clouds])
      zcl         = np.array([cld.zcl                     for cld in clouds])
      thetacl     = np.array([cld.thetacl.to(u.rad).value for cld in clouds])
      logrhoscale = np.array([cld.logrhoscale             for cld in clouds])
      rhoindex    = np.array([cld.rhoindex                for cld in clouds])
      logrho0     = np.array([cld.logrho0                 for cld in clouds])
      vcl         = np.array([cld.vlos.to(u.km/u.s).value for cld in clouds]) * (u.km/u.s)
    else:
      rcl         = np.array([])
      zcl         = np.array([])
      thetacl     = np.array([])
      logrhoscale = np.array([])
      rhoindex    = np.array([])
      logrho0     = np.array([])
      vcl         = np.array([]) * (u.km/u.s)

    return rcl, zcl, thetacl, logrhoscale, rhoindex, logrho0, vcl

  #######################################################################################
  def makeclouds(self,
                 xclp, yclp, zcl,
                 rhoindex, logrhoscale, logrho0,
                 vcl,
                 verbose = False,
                 nr = 300, ntheta = 300
                 ):
    fu = (u.erg / (u.s * u.cm * u.cm * u.Hz))

    if xclp.size > 0:
      if verbose:
        print(f"Computing cloud positions (apparent positions given as ({xclp}, {yclp}, {zcl})....")
        print(f"\t\twith observer located at r,z = {self.robs}, {self.zobs}")
      xcl, ycl = self._abs_deproject_clouds(xclp, yclp, zcl)
      rcl     = np.sqrt(xcl * xcl + ycl * ycl)
      thetacl = np.arctan2(ycl, xcl)
      if not isinstance(thetacl, Quantity):
        thetacl *= u.rad
      thetacl = thetacl.to(u.deg)

      if verbose: print("Making clouds!")
      clouds = []
      ncl    = rcl.size
      for i in range(ncl):
        t0 = tm.time()
        if verbose:
          print("#" * 50)
          print(f"\t{i} Making a cloud with the following parameters: xclp = {xclp[i]} yclp = {yclp[i]} --> ")
          print("\t"*7 + f"xcl = {xcl[i]:.3f}   ycl = {ycl[i]:.3f} zcl={zcl[i]:.3f}  --> r = {rcl[i]:.3f}  theta = {thetacl[i]:.3f}")
          print("\t"*7 + f"rhoindex = {rhoindex[i]}  logrhoscale = {logrhoscale[i]} logrho0 = {logrho0[i]}")
          print("\t"*7 + f"vcl_los = {vcl[i]}")
        clouds.append(AbsCloud(self.datapath, self.mydisk, self.mycorona, self.myatoms,
                               rcl[i], zcl[i], thetacl[i], rhoindex=rhoindex[i], logrhoscale=logrhoscale[i], logrho0=logrho0[i], vcl_los=vcl[i]))
        if verbose:
          print(f"\t{i} Determining ionizing spectrum")
        cloudy_rootname = f"ABS-rho0{logrho0[i]}-index{rhoindex[i]}-scale{logrhoscale[i]}-zcl{zcl[i]}"
        clouds[i].calcionspec(cloudyfileroot = cloudy_rootname, structure_only = True)
        totflux = self._calculate_absorbed_flux_gaussleg(None,
                                                         nr = nr, ntheta = ntheta,
                                                         robs = rcl[i], thetaobs = thetacl[i], zobs = zcl[i],
                                                         wavelength = clouds[i].ionspecfreq.to(u.Angstrom, equivalencies=u.spectral()),
                                                         lograd = True, noplot = True
                                                         )[0]
        try:
          dum = self._spectrum_scale
        except AttributeError:
          print("\tSetting spectral scale 'cause I'm stupid and can't integrate...")
          clouds[i].calcionspec(cloudyfileroot = cloudy_rootname, structure_only = False)
          self._spectrum_scale = np.max(clouds[i].ionspecfreq*clouds[i].ionspecflux) / np.max(clouds[i].ionspecfreq*totflux)
          while np.any(self._spectrum_scale * totflux > clouds[i].ionspecflux):
            self._spectrum_scale *= 0.999

        corona_flux = np.squeeze(self.mycorona.fnu_lamppost(clouds[i].ionspecfreq,rcl[i],zcl[i]))
        totflux = self._spectrum_scale * totflux + corona_flux
        clouds[i].ionspecflux = np.where(totflux < 1.0e-100 * fu, 1.0e-100 * fu, totflux)

        if verbose:
          print(f"\t{i} Getting cloudy with it {self.cloudypath}")
        # This is for creating clouds[i].radius, clouds[i].density, clouds[i].temperature, clouds[i].iondensity
        clouds[i].getcloudy(self.cloudypath, verbose = verbose, runcloudy = False)
        (lognuFnu, log_ion_parm) = self._calc_ion_parm(clouds[i])
        sdx = np.argsort(log_ion_parm)

        # Need to fill the clouds[i].iondensity array
        try:
          dum = self.cheb_log_ion_parm_list
        except AttributeError:
          print("\t\tOOPS - FORGOT TO READ CHEBYSHEV TABLES...")
          self._read_cheb_files()

        if self.cheb_log_ion_parm_list.size > 0:
          lnmin = np.min(self.cheb_log_ion_parm_list)
          lnmax = np.max(self.cheb_log_ion_parm_list)
        else:
          print("\t\tOOPS - CHEBYSHEV TABLES DON'T EXIST...")
          lnmin = np.min(log_ion_parm) + 10.0
          lnmax = np.max(log_ion_parm) - 10.0

        # If we have ionization parameters outside of the Chebyshev range, run Cloudy and expand the range
        if np.any(log_ion_parm > lnmax) or np.any(log_ion_parm < lnmin):
          prtstr = f"\t\tNeed ionization parameter range ({np.min(log_ion_parm)}, {np.max(log_ion_parm)})"
          prtstr += f" which is outside the range ({lnmin},{lnmax})"
          print(prtstr)
          clouds[i].getcloudy(self.cloudypath, verbose = verbose, runcloudy = True)
          (lognuFnu, log_ion_parm) = self._calc_ion_parm(clouds[i])
          sdx = np.argsort(log_ion_parm)
          self._add_to_cheb(clouds[i])
          lnmin = np.min(self.cheb_log_ion_parm_list)
          lnmax = np.max(self.cheb_log_ion_parm_list)
        
        xnorm = 2 * (log_ion_parm - lnmin)/(lnmax-lnmin) - 1
        for iondx in range(self.myatoms.nion):
          ion_parm_mask1 = self.cheb_ionfrac_list[:,iondx] > 0
          if np.any(ion_parm_mask1):
            ion_parm_mask2 = (log_ion_parm < np.max(self.cheb_log_ion_parm_list[ion_parm_mask1])) & (xnorm >= -1.0) & (xnorm <= 1.0)

            logionfrac  = np.zeros(log_ion_parm.size) + self.myatoms.abund[self.myatoms.idx[iondx]] - 12.0
            if np.any(ion_parm_mask2):
              logionfrac[ion_parm_mask2] += chebyshev.chebval(xnorm[ion_parm_mask2],
                                                              self.cheb_coeff_list[iondx,:self.cheb_degree[iondx]+1]
                                                              )
              logionfrac[~ion_parm_mask2] += chebyshev.chebval(-1.0,
                                                               self.cheb_coeff_list[iondx,:self.cheb_degree[iondx]+1]
                                                               )

            logionfrac = np.where(logionfrac > 0.0, 0.0, logionfrac)

            clouds[i].iondensity[sdx,iondx] = clouds[i].density[sdx] * np.power(10.0,logionfrac)

          if not np.all(np.isfinite(clouds[i].iondensity)):
            print("ERROR IN GETTING ION DENSITIES:")
            print(f"logionfrac = {logionfrac}")
            print(f"cloud density = {clouds[i].density}")
            print(f"cloud ion density = {clouds[i].iondensity}")
            input("Stopped in quasar.makeclouds")

        if verbose:
          print(f"\t{i} Cloud took {tm.time()-t0} s")

    else:
      clouds = None

    return clouds

  #######################################################################################
  def printpars(self):
    print("-" * 70)
    print(f"Observer: zqso = {self.zqso}, inclination = {self.inclination}, coordinates = {self.skycoord.to_string('hmsdms')}")

    print(f"Black hole: mass {self.mydisk.mbh.to(u.Msun):e}, spin {self.mydisk.sbh}")
    print(f"            Rg = {self.mydisk.rg:e} = {self.mydisk.rg.to(u.AU)}")

    print(f"Accretion disk: accretion rate {self.mydisk.mdot.to(u.Msun/u.yr)}, viscosity parameter {self.mydisk.alpha}")
    if not self.mywind is None:
      print(f"                Eddington ratio {self.mywind.Eddington_ratio}")
    print(f"                Zone 1/2 boundary (pressure) {self.mydisk.x1} rg  Zone 2/3 (opacity) boundary {self.mydisk.x2} rg")
    print(f"                Inner radius {self.mydisk.rstar[0]} rg, Outer radius {self.mydisk.rstar[-1]} rg, nr = {self.mydisk.nr}")

    print(f"Lamp post: location {self.mycorona.lamp_r_cyl}, {self.mycorona.lamp_z}")
    prtstr  = f"           spectrum L_nu_2keV = {self.mycorona.lamp_L_nu_2keV} @ nu_xo_2keV = {self.mycorona.lamp_nu_2keV}, "
    prtstr += f" alpha_x = {self.mycorona.lamp_alpha_x}, E_c = {(const.h * self.mycorona.cutoff_freq).to(u.keV)}"
    print(prtstr)

    if not self.mywind is None:
      print(f"Wind: grid (nr,ntheta) = ({self.mywind.nr},{self.mywind.ntheta})")

    if not self.clouds is None:
      self.print_clouds(self.clouds)
    print("-" * 70)

    return

  #######################################################################################
  def print_clouds(self,
                   clouds,
                   ntabs = 0):
    print("\t" * ntabs + "Absorbing clouds:")
    print("\t" * ntabs + "        num rcl          zcl          thetacl           rhoindex     rhoscale/rg  logrho0      vlos")
    for i in range(len(clouds)):
      prtstr  = "\t" * ntabs + f"\t {i:2d} {clouds[i].rcl:e} {clouds[i].zcl:e} {clouds[i].thetacl:e} {clouds[i].rhoindex:e} "
      prtstr += f"{(10.0**clouds[i].logrhoscale * u.cm)/self.mydisk.rg:e} {clouds[i].logrho0:e} {clouds[i].vlos:e}"
      print(prtstr)

    return

  #######################################################################################
  def print_diff_clouds(self,
                        clouds1,
                        clouds2,
                        ntabs = 0):
    print("\t" * ntabs + "Differences (cloud2 - cloud1):")
    print("\t" * ntabs + "        num  drcl          dzcl          dthetacl         drhoindex     drhoscale/rg  dlogrho0      dvlos")
    for i in range(len(clouds1)):
      prtstr  = "\t" * ntabs + f"\t {i:2d} {clouds2[i].rcl-clouds1[i].rcl:13.6e} {clouds2[i].zcl-clouds1[i].zcl:13.6e} {clouds2[i].thetacl-clouds1[i].thetacl:13.6e} "
      prtstr += f"{clouds2[i].rhoindex-clouds1[i].rhoindex:13.6e} {((10.0**clouds2[i].logrhoscale-10.0**clouds1[i].logrhoscale) * u.cm)/self.mydisk.rg:13.6e} "
      prtstr += f"{clouds2[i].logrho0-clouds1[i].logrho0:13.6e} {clouds2[i].vlos-clouds1[i].vlos:13.6e}"
      print(prtstr)

    return

  #######################################################################################
  def readspec(self,
               qname, qfileroot,
               redospline=False,
               vlo=0.0 * (u.km/u.s),
               vhi=0.0 * (u.km/u.s),
               nsigma = 1.1
               ):

    self.mydata = hstqso(self.datapath,
                         self.skycoord,
                         qfileroot,
                         self.zqso,
                         redospline=redospline)

    self.mydata.combspec(verbose=True)

    self.mydata.contfit()

    self.obswave, self.obsflux, self.obsivar = self.mydata.bin_spec(self.mydata.totwave, self.mydata.totflux, self.mydata.totivar)

    self.obswave *= u.Angstrom
    self.obsflux *= u.erg/u.s/u.cm**2/u.Hz
    
    self.restwave = self.obswave/(1 + self.zqso)
    self.obsferr  = 1.0/np.sqrt(self.obsivar)
    self.obsferr *= u.erg/u.s/u.cm**2/u.Hz

    print("Normalizing continuum")
    self.normobsflux = self.obsflux.value/self.mydata.continuum(self.obswave)
    self.normobsferr = self.obsferr.value/self.mydata.continuum(self.obswave)

    self.lilew, self.lilsew, self.bigew, self.bigsew = self.mydata.ew_spec(self.restwave,
                                                                           self.obsflux,
                                                                           self.obsferr
                                                                           )
    self.bigew_mask = self.bigew > nsigma * self.bigsew

    possible_codes = ["g", "r", "m", "b", "k"]
    self.plot_code = []
    self.wavelength = None
    for i in range(self.myatoms.wave.size):
      self.plot_code.append(possible_codes[np.mod(i,
                                                  len(possible_codes),
                                                  dtype=np.int32
                                                  )
                                           ]
                            )

    #self.wavelength = np.empty((self.anum.size,self.velocity.size)) * u.Angstrom
    #for i in range(self.anum.size):
    #  self.wavelength[i,:] = np.squeeze(calcwave(self.velocity,
    #                                             self.myatoms.wave[self.myatoms.getspecies(self.anum[i],
    #                                                                                       self.ion[i]
    #                                                                                       )[self.trandx[i]]
    #                                                               ]
    #                                             )
    #                                    ) * u.Angstrom

    print("Computing velocities")
    self.obsvel = np.empty((self.myatoms.wave.size,
                            self.obswave.size
                            )
                           ) * (u.km/u.s)
    for i in range(self.myatoms.wave.size):
      self.obsvel[i,:] = np.squeeze(calcvel(self.restwave,
                                            self.myatoms.wave[i]
                                            )
                                    )

    #self.obsvel = np.empty((self.anum.size,self.obswave.size)) * (u.km/u.s)
    #for i in range(self.anum.size):
    #  self.obsvel[i,:] = np.squeeze(calcvel(self.restwave,
    #                                        self.myatoms.wave[ self.myatoms.getspecies(self.anum[i],self.ion[i])[self.trandx[i]] ] ) )

  #######################################################################################
  def reset_observer(self,
                     robs = None,
                     thetaobs = None,
                     zobs = None
                     ):
    if robs is None:
      self.mydisk.robs     = self.robs
      self.mydisk.thetaobs = 0.0
      self.mydisk.zobs     = self.zobs
    else:
      self.mydisk.robs     = robs
      self.mydisk.thetaobs = thetaobs
      self.mydisk.zobs     = zobs

  #######################################################################################
  def write_clouds(self,
                   clouds
                   ):
    (rcl, zcl, thetacl, logrhoscale, rhoindex, logrho0, vcl_los) = self.grab_cloud_pars(clouds)
    xclp, yclp = self._abs_project_clouds(rcl, zcl, thetacl)
    
    print(f"\tWriting clouds to {self.cloud_filename}")
    cloud_table = Table(data=[xclp, yclp, zcl, rhoindex, logrhoscale, logrho0, vcl_los],
                        names=["xclp","yclp","zcl","rhoindex","logrhoscale","logrho0","vcl"])
    cloud_table.pprint()
    cloud_table.write(self.cloud_filename, format="fits", overwrite=True)
    
    return
  
  def read_clouds(self):
    print(f"\tReading clouds from {self.cloud_filename}")
    cloud_table = Table.read(self.cloud_filename, format="fits")
    cloud_table.pprint()
    xclp        = np.array(cloud_table["xclp"])
    yclp        = np.array(cloud_table["yclp"])
    zcl         = np.array(cloud_table["zcl"])
    rhoindex    = np.array(cloud_table["rhoindex"])
    logrhoscale = np.array(cloud_table["logrhoscale"])
    logrho0     = np.array(cloud_table["logrho0"])
    vcl_los     = np.array(cloud_table["vcl"]) * (u.km/u.s)

    clouds = self.makeclouds(xclp, yclp, zcl, rhoindex, logrhoscale, logrho0, vcl_los, verbose = True)

    return clouds
  
  #######################################################################################
