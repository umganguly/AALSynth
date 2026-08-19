import copy
import os
import matplotlib
import matplotlib.pyplot  as plt
import numpy              as np
import time               as tm

from astropy                 import constants as const
from astropy                 import units     as u
from astropy.units           import Quantity
from astropy.coordinates     import SkyCoord
from astropy.cosmology       import LambdaCDM
from astropy.io              import fits
from astropy.modeling.models import BlackBody
from astropy.table           import Table
from astroquery.ipac.ned     import Ned
from multiprocessing         import Pool
from numpy.polynomial        import chebyshev
from scipy.interpolate       import CubicSpline
from scipy.optimize          import least_squares
from scipy.special           import wofz
from scipy.stats             import f         as Ftest
from tqdm                    import tqdm

from AbsCloud                import AbsCloud
from atomic                  import atomic
from cloudy                  import cloudy
from corona                  import corona
from doppler                 import calcvel,calcwave
from hstqso                  import hstqso
from mcgv                    import mcgv
from ntdisk                  import ntdisk
from readpars                import readpars

class Quasar:
  def __init__(self,mypars,
               ntabs = 0):
    self.mypars = mypars

    ###############################################################################
    print("\t" * ntabs + "#" * 50)
    print("\t" * ntabs + "Grabbing atomic data")
    self.myatoms = atomic(self.mypars.datapath,900 * u.Angstrom,3000 * u.Angstrom,
                          minlox = self.mypars.minlox,
                          ntabs = ntabs+1
                          )

    nvel = np.int16((self.mypars.vhi-self.mypars.vlo)/self.mypars.vres)
    self.velocity   = np.linspace(start = self.mypars.vlo,
                                  stop  = self.mypars.vhi,
                                  num   = nvel
                                  )

    fu = (u.erg / (u.s * u.cm * u.cm * u.Hz))
    ###############################################################################
    print("\t" * ntabs + "Setting observer")
    self.skycoord    = SkyCoord(ra  = self.mypars.raqso,
                                dec = self.mypars.decqso)
    #print("\t" * ntabs + "Querying NED")
    #result_table = Ned.query_region(self.skycoord, radius = 5 * u.arcsec)
    #redshift_mask = result_table['Redshift Points'] == np.max(result_table['Redshift Points'])
    #result_table[redshift_mask].pprint_all()
    #self.mypars.zqso = result_table[redshift_mask]['Redshift'][0]
    comove_dist = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7).comoving_distance(self.mypars.zqso)
    print("\t" * ntabs + f"Quasar redshift = {self.mypars.zqso} --> Comoving distance = {comove_dist}")
    #lum_dist = (LambdaCDM(H0=70, Om0=0.3, Ode0=0.7).luminosity_distance(self.mypars.zqso)).to(u.cm)
    #photometry_table = Ned.get_table(result_table[redshift_mask]['Object Name'])
    #photo_mask = photometry_table['Frequency Mode'] == 'Broad-band measurement'
    #self.ned_freq = photometry_table[photo_mask]['Frequency']
    #self.ned_flux = (photometry_table[photo_mask]['Flux Density']).to(fu)

    ###############################################################################
    print("\t" * ntabs + f"Initializing disk with {self.mypars.nr} annuli from {self.mypars.rlo} to {self.mypars.rhi} rg")
    self.mydisk = ntdisk(self.mypars,
                         ntabs = ntabs+1)
    self.robs =  (comove_dist * np.sin(self.mypars.inclination) / self.mydisk.rg).decompose()
    self.zobs = self.robs / np.tan(self.mypars.inclination)
    self.reset_observer()
    print("\t" * ntabs + f"\tObserver located at (r,z) = ({self.robs}, {self.zobs}), inclination = {self.mypars.inclination}")

    print("\t" * ntabs + "\tCalculating disk")
    self.mydisk.makedisk(ntabs = ntabs+2)
    print("\t" * ntabs + "\tDetermining disk photosphere")
    self.mydisk.photosphere(ntabs = ntabs+1)

    ###############################################################################
    print("\t" * ntabs + "Initializing corona")
    self.mycorona = corona(self.mypars, self.mydisk)
    self.mycorona.activate_lamppost(ntabs = ntabs+1)

    #rest_freq = self.ned_freq.to(u.Hz) / (1+self.mypars.zqso)
    #pred_flux = self._calculate_absorbed_flux_gaussleg(None,
    #                                                   wavelength = rest_freq.to(u.Angstrom, equivalencies=u.spectral() ) 
    #                                                   )[0]
    #print(pred_flux)
    #plt.clf()
    #plt.scatter(rest_freq, self.ned_flux) # * 4 * np.pi * lum_dist * lum_dist)
    #plt.xscale('log')
    #plt.yscale('log')
    #plt.xlim(plt.xlim())
    #plt.ylim(plt.ylim())
    #plt.plot(rest_freq, 
    #         pred_flux
    #         )
    #plt.show(block=True)

    ###############################################################################
    if mypars.calcwind:
      print("\t" * ntabs + "Initializing wind...")
      self.mywind = mcgv(self.mydisk, 
                         self.mycorona, 
                         self.myatoms, 
                         self.mypars,
                         ntabs = ntabs+1
                         )

      if self.mywind.spherical_coords:
        Mtot_mag_grid = np.sqrt(self.mywind.Mrgrid*self.mywind.Mrgrid + self.mywind.Mthetagrid*self.mywind.Mthetagrid)
        tau_es = self.mywind.number_density * const.sigma_T.cgs * self.mywind.DRR
        which_cells = self.mywind.boundary_mask & (Mtot_mag_grid == 0)  & (self.mywind.number_density.to(u.cm**-3).value >= 9.0e-6) & \
          (self.mywind.number_density.to(u.cm**-3).value < 1.0e+15) & (tau_es < 0.7)

        if np.sum(which_cells) > 0:
          t1 = tm.time() * u.s
          print("\t" * (ntabs+1) + f"Calculating force multipliers for {np.sum(which_cells)} cells")
          self._wnd_force_multiplier(which_cells,
                                     ntabs = ntabs+2)

        self._wnd_calcstreamline_relativistic(dtime = 10.0 * u.s,
                                              mindt = 1.0 * u.s,
                                              vres = 0.5 * const.c.to(u.km/u.s),
                                              minr_rg = 10.0,
                                              plotstream=True,
                                              mupdate = True,
                                              ntabs = ntabs+2
                                              )
      else:
        Mtot_mag_grid = np.sqrt(self.mywind.MRgrid*self.mywind.MRgrid + self.mywind.MZgrid*self.mywind.MZgrid)
        try:
          tau_es = (self.mywind.number_density * const.sigma_T.cgs * self.mywind.DRR).decompose()
          which_cells = self.mywind.boundary_mask & (Mtot_mag_grid == 0)  & (self.mywind.number_density > 1.0e-5 / u.cm**3) & \
            (self.mywind.number_density < 1.0e+15 / u.cm**3) & (tau_es < 0.7)
        except:
          print(f"tau_es = {self.mywind.number_density} * {const.sigma_T.cgs} * {self.mywind.DRR}")
          input("pause")

        if np.sum(which_cells) > 0:
          t1 = tm.time() * u.s
          print("\t" * (ntabs+1) + f"Calculating force multipliers for {np.sum(which_cells)} cells")
          self._wnd_force_multiplier_cylindrical(which_cells,
                                                 ntabs = ntabs+2)
        #else:
        #  print("\t" * ntabs + "Didn't find any cells to calculate force multipliers initially:")
        #  print("\t" * (ntabs+1) + f"n(boundary_mask) = {np.sum(self.mywind.boundary_mask)}")
        #  print("\t" * (ntabs+1) + f"n(|Mtot| == 0) = {np.sum(Mtot_mag_grid == 0)}")
        #  print("\t" * (ntabs+1) + f"n(number_density > {1.0e-5 / u.cm**3} = {np.sum(self.mywind.number_density > 1.0e-5 / u.cm**3)}")
        #  print("\t" * (ntabs+1) + f"n(number_density < {1.0e+15 / u.cm**3} = {np.sum(self.mywind.number_density < 1.0e+15 / u.cm**3)}")
        #  print("\t" * (ntabs+1) + f"n(tau_es < 0.7) = {np.sum(tau_es < 0.7)}")
        #  input("Why no force multiplier????")

        self._solve_euler_cylindrical(dtime = 10.0 * u.s,
                                      mindt = 1.0 * u.s,
                                      vres = 0.5 * const.c.to(u.km/u.s),
                                      minr_rg = 10.0,
                                      plotstream=True,
                                      mupdate = True,
                                      ntabs = ntabs+1
                                      )
      #  .
      #  .
      #  .
    else:
      self.mywind = None

    ###############################################################################
    if mypars.calcabscl:
      print("\t" * ntabs + "Initializing absorbing clouds")
      self.reset_observer()
      self.cloud_filename = self.mypars.datapath + self.mypars.abscloudfile
      print("\t" * ntabs + f"\tLooking for {self.cloud_filename}")
      if os.path.exists(self.cloud_filename):
        print("\t" * ntabs + f"\t\tFound it!")
        self.clouds = self._abs_read_clouds(mypars,
                                            ntabs = ntabs+1)
      else:
        self.clouds = None
      self.bestfit = None
    else:
      self.clouds = None

  #######################################################################################
  def _abs_all_optical_depth(self,
                             clouds,           # List of AbsClouds
                             rdisk_vec, R_vec, # 2D nd.arrays with shapes (3,thetadisk.size)
                             wavelength,        # 1d nd.array
                             ntabs = 0
                             ):

    if clouds is not None:
      t0 = tm.time()

      rcl, zcl, thetacl, logrhoscale, rhoindex, logrho0, logZ, vcl  = self.grab_cloud_pars(clouds, ntabs = ntabs+1)
      ncl = len(clouds)

      xcl = rcl * np.cos(thetacl)
      ycl = rcl * np.sin(thetacl)

      Rmag_squared = np.sum(R_vec*R_vec, axis=0)

      min_impact_parameter = 10.0 * self.mydisk.rstar[-1] * self.mydisk.rg / (np.array([np.max(clouds[cdx].radius.to(u.cm).value) for cdx in range(ncl) ]) * u.cm)

      cloud_optical_depth = np.zeros(self.cld_totflux.T.shape + Rmag_squared.shape)

      for cdx in range(ncl):
        t0 = tm.time()
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
        
        try:
          self.optical_depth_geometry_time += tm.time() - t0
        except AttributeError:
          self.optical_depth_geometry_time = tm.time() - t0

        impact_parameter_mask = impact_parameter < np.max(clouds[cdx].radius)
        if np.any(impact_parameter_mask):
          cdx_optical_depth = self._abs_optical_depth(impact_parameter, # 1D nd.array with rdisk_vec[0,:].size
                                                      clouds[cdx],      # AbsCloud
                                                      wavelength,        # 1D nd.array
                                                      ntabs = ntabs+1
                                                      )                 # shape (wavelength.shape + impact_parameter.shape + cloud.radius.shape)

          cloud_optical_depth[cdx,:,:] += cdx_optical_depth.sum(axis=-1)
      
      try:
        self.t_abs_all_optical_depth += tm.time() - t0
      except:
        self.t_abs_all_optical_depth = tm.time() - t0
    else:
      min_impact_parameter = None
      cloud_optical_depth = None

    return cloud_optical_depth, min_impact_parameter

  #######################################################################################
  def _abs_bounds(self,
                  clouds,
                  ntabs = 0
                  ):

    abs_lower_bounds = []
    abs_upper_bounds = []
    for cld in clouds:

      max_radius = self.mydisk.rstar[-1] + (10.0**cld.logrhoscale * u.cm / self.mydisk.rg).decompose()

      abs_lower_bounds.append(                                          -max_radius ) # xcl
      abs_lower_bounds.append(                                          -max_radius ) # ycl
      abs_lower_bounds.append( self.mydisk.zt1cs(cld.rcl) + 10.0**(cld.logrhoscale) * u.cm / self.mydisk.rg ) # zcl
      abs_lower_bounds.append(                                                  0.0 ) # rhoindex
      abs_lower_bounds.append(                        np.log10(self.mydisk.rg/u.cm) ) # logrhoscale
      abs_lower_bounds.append(                                                - 4.0 ) # logrho0
      abs_lower_bounds.append(                                                 -3.0 ) # log Z
      abs_lower_bounds.append(   (cld.vlos - 100.0 * (u.km/u.s)).to(u.km/u.s).value ) # vlos

      abs_upper_bounds.append(                                        max_radius ) # xcl
      abs_upper_bounds.append(                                        max_radius ) # ycl
      abs_upper_bounds.append(        (1.0 * u.kpc / self.mydisk.rg).decompose() ) # zcl
      abs_upper_bounds.append(                                               5.0 ) # rhoindex
      abs_upper_bounds.append(               np.log10(self.mydisk.rg/u.cm) + 5.0 ) # logrhoscale
      abs_upper_bounds.append(                                               5.0 ) # logrho0
      abs_upper_bounds.append(                                               4.0 ) # log Z
      abs_upper_bounds.append((cld.vlos + 100.0 * (u.km/u.s)).to(u.km/u.s).value ) # vlos

    return (abs_lower_bounds,abs_upper_bounds)
    
  #######################################################################################
  # Callback routine for the scipy.optimize.minimize fitter
  def _abs_callback(self,
                    intermediate_result,
                    ntabs = 0
                    ):
    clouds = self._abs_unpack(intermediate_result.x,
                              ntabs = ntabs+1)
    # Observer coordinates (reset here for sanity)
    self.reset_observer()

    (totflux, unabsflux) = self._calculate_absorbed_flux_gaussleg(clouds, lograd = True, ntabs = ntabs+1)
    chisq = np.sum(self._abs_chisq(totflux, unabsflux, ntabs = ntabs+1))
    self.bestfit = totflux/unabsflux

    print("\t" * ntabs + '\n')
    print("\t" * ntabs + '\t\t' + '#'*20)
    self.print_clouds(clouds, ntabs = ntabs+1)
    try:
      print("\t" * ntabs + f"\t\tChi^2 = {chisq}  ({tm.time() * u.s - self._abscall_t0:e} since last callback)")
      self._abs_plot(totflux, 
                     unabsflux, 
                     vcl = self.grab_cloud_pars(clouds)[-1], 
                     clouds = clouds,
                     ntabs = ntabs+1
                     )
    except NameError:
      print("\t" * ntabs + f"\t\tChi^2 = {chisq}")
      self._abs_plot(totflux, 
                     unabsflux, 
                     vcl = self.grab_cloud_pars(clouds)[-1], 
                     clouds = clouds,
                     ntabs = ntabs+1
                     )

    self._abscall_t0 = tm.time() * u.s

    # Write out clouds to a file so that we can pick up where we left off...
    self._abs_write_clouds(clouds,
                           ntabs = ntabs+1
                           )
    print("\t" * ntabs + '\t\t' + '#'*20)
    print("\t" * ntabs + '\n')
 
  #######################################################################################
  # Compute the chisq, summing across all max(f-lambda) species that are covered in the velocity range specified
  def _abs_chisq(self,
                 totflux, unabsflux,
                 ntabs = 0
                 ):
    try:
      chisq_spec = np.zeros((self.mypars.anum.size, self.obswave.size))
      try:
        cosflux    = self.mydata._lsf_convolve(self.wavelength, 
                                               totflux.value,
                                               ntabs = ntabs+1
                                               )/unabsflux.value
      except RuntimeWarning:
        print("\t" * ntabs + f"totflux = {totflux}")
        print("\t" * ntabs + f"unabsflux = {unabsflux}")
        input("Paused")
      for i in range(self.mypars.anum.size):
        myatoms_index = self.myatoms.getspecies(self.mypars.anum[i],
                                                self.mypars.ion[i],
                                                ntabs = ntabs+1
                                                )[self.mypars.trandx[i]]
        velocity_mask = (self.obsvel[myatoms_index,:] > self.velocity[0]) & (self.obsvel[myatoms_index,:] < self.velocity[-1])

        chisq_anum = np.square((self.normobsflux[velocity_mask] - np.interp(self.obsvel[myatoms_index,velocity_mask],
                                                                            np.squeeze(calcvel(self.wavelength,
                                                                                               np.array(self.myatoms.wave[myatoms_index]),
                                                                                               ntabs = ntabs+1
                                                                                               )
                                                                                       ),
                                                                            cosflux
                                                                            )
                                )/self.normobsferr[velocity_mask]
                               )
        chisq_spec[i,velocity_mask] = chisq_anum

    except AttributeError:
      chisq_spec = np.zeros((self.mypars.anum.size, self.wavelength.size))

    self.chisq_spec = chisq_spec
    return chisq_spec

  #######################################################################################
  # Wrapper for minimization purposes - the function that is to be minimized
  def _abs_chisqfunc(self,
                     x,
                     ntabs = 0
                     ):
    self._chisqfunc_clouds = self._abs_unpack(x,
                                              ntabs = ntabs+1
                                              )

    # Observer coordinates (reset here for sanity)
    self.reset_observer()

    (totflux, unabsflux) = self._calculate_absorbed_flux_gaussleg(self._chisqfunc_clouds, 
                                                                  noplot=True,
                                                                  ntabs = ntabs+1)
    chisq_spec = self._abs_chisq(totflux, 
                                 unabsflux,
                                 ntabs = ntabs+1)

    return chisq_spec.flatten()

  #######################################################################################
  def _abs_deproject_clouds(self,
                            xclp, yclp,
                            zcl,
                            ntabs = 0
                            ):
    self.reset_observer()
    a       = -1 - self.zobs / (zcl - self.zobs)
    xcl     = (xclp - self.robs * np.cos(self.mydisk.thetaobs)) / (1 + a) + self.robs * np.cos(self.mydisk.thetaobs)
    ycl     = yclp / (1 + a)

    return xcl, ycl
  
  #######################################################################################
  def _abs_mcminimize(self,
                      clouds,
                      totflux = None,
                      unabsflux = None,
                      ntabs = 0
                      ):

    ncl = len(clouds)
    self.reset_observer()
    if totflux is None:
      (totflux, unabsflux) = self._calculate_absorbed_flux_gaussleg(clouds,
                                                                    ntabs = ntabs+1
                                                                    )
    chisq = np.sum(self._abs_chisq(totflux,
                                   unabsflux,
                                   ntabs = ntabs+1
                                   )
                   )
    self.bestfit = totflux/unabsflux
    x = self._abs_pack(clouds,
                       ntabs = ntabs+1)

    better_clouds = copy.deepcopy(clouds)
    niter = self.mypars.maxiter * x.size
    rng = np.random.default_rng()
    good_direction = False
    check_negative_direction = False
    step_size = self.mypars.initstep
    while niter > 0 and step_size > self.mypars.minstep:
      self.reset_observer()
      (rcl, zcl, thetacl, logrhoscale, rhoindex, logrho0, logZ, vcl) = self.grab_cloud_pars(better_clouds,
                                                                                            ntabs = ntabs+1)
      (xclp, yclp) = self._abs_project_clouds(rcl,
                                              zcl,
                                              thetacl,
                                              ntabs = ntabs+1
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
      print("\t" * ntabs + "\t"+"-"*20)
      print("\t" * ntabs + f"\tProposing changes to clouds [step size = {step_size/self.mypars.minstep} x minstep (={self.mypars.minstep}), good_direction = {good_direction}, check_negative_direction = {check_negative_direction}]...")
      maxrad = self.mydisk.rstar[-1] + 10.0**logrhoscale * u.cm / self.mydisk.rg

      pos_step = 1.0
      while np.any(np.fabs(xclp + pos_step * step_size * dx[0:ncl]) > maxrad):
        pos_step *= 0.95
      xclp += pos_step * step_size * dx[0:ncl]

      pos_step = 1.0
      while np.any(np.fabs(yclp + pos_step * step_size * dx[ncl:2*ncl]) > maxrad):
        pos_step *= 0.95
      yclp += pos_step * step_size * dx[ncl:2*ncl]

      logrhoscale += step_size * dx[2*ncl:3*ncl]

      logZ += step_size * dx[3*ncl:4*ncl]

      zmin = np.interp(rcl, self.mydisk.rstar, self.mydisk.zt1) + 10.0**(1+logrhoscale) * u.cm / self.mydisk.rg
      zcl += step_size * dx[4*ncl:5*ncl]
      zmask = zcl < zmin
      if np.any(zmask):
        print("\t" * ntabs + f"\t\tResetting heights for clouds {np.arange(ncl)[zmask]}: {zcl[zmask]} < {zmin[zmask]}")
        zcl[zmask] = zmin[zmask]

      rhoindex    += step_size * dx[5*ncl:6*ncl]
      logrho0     += step_size * dx[6*ncl:7*ncl]

      for i in range (rhoindex.size):
        rhoindex[i] = np.max([rhoindex[i],0])
 
      vcl += 5.0 * step_size * dx[7*ncl:8*ncl] * (u.km/u.s)

      # Make the potential new clouds
      potential_clouds = self.makeclouds(xclp, yclp, zcl,
                                         rhoindex, logrhoscale, logrho0,
                                         logZ,
                                         vcl,
                                         ntabs = ntabs+1
                                         )
      self.reset_observer()

      print("\t" * ntabs + "\tTrying the following changes:")
      self.print_diff_clouds(better_clouds,
                             potential_clouds,
                             ntabs = ntabs+1)

      # Determine chisq
      (newtotflux, newunabsflux) = self._calculate_absorbed_flux_gaussleg(potential_clouds,
                                                                          ntabs = ntabs+1)
      newchisq = np.sum(self._abs_chisq(newtotflux,
                                        newunabsflux,
                                        ntabs = ntabs+1
                                        )
                        )

      # If the chisq is better, then keep it and continue.
      # If it is not better, increment the iteration number and try again until the iteration number is bunk
      if newchisq < chisq:
        print(f"\n" + "\t" * ntabs + f"\tIMPROVED FIT! iterations remaining: {niter}    chisq = {chisq} - {chisq - newchisq}\n")
        better_clouds = copy.deepcopy(potential_clouds)
        chisq         = np.copy(newchisq)
        totflux       = np.copy(newtotflux)
        unabsflux     = np.copy(newunabsflux)
        self.bestfit  = totflux/unabsflux

        if newchisq < chisq-1.0:
          niter = self.mypars.maxiter * x.size

        good_direction = True
        step_size *= 1.0 + self.mypars.dstep

        self._abs_write_clouds(better_clouds,
                               ntabs = ntabs+1)
        self._abscall_t0 = tm.time() * u.s
      else:
        good_direction = False
        step_size *= 1.0 - self.mypars.dstep
        if not check_negative_direction:
          niter -= 1

        print("\t" * ntabs + f"\n\tKeeping old fit! iterations remaining: {niter}    chisq = {chisq} + {newchisq-chisq}")

      self.print_clouds(better_clouds, 
                        ntabs = ntabs+1)

    return better_clouds, chisq

  #######################################################################################
  # Absorption optical depth from sightlines (at impact_parameter) piercing one cloud
  def _abs_optical_depth(self,
                         impact_parameter, # 1D nd.array
                         cloud,            # AbsCloud
                         wavelength,       # 1D nd.array
                         ntabs = 0
                         ): 
    optical_depth = np.zeros(wavelength.shape + impact_parameter.shape + cloud.radius.shape)

    if np.any(impact_parameter < np.max(cloud.radius)):

      t0 = tm.time()

      ip_ratio = np.outer(impact_parameter,
                          1 / (cloud.radius + 1 * u.cm)
                          ) # shape (impact_parameter.size, cloud.radius.size)
      ip_ratio[ip_ratio > 1] = 1 # This ensures that dx=0 when the impact parameter is outside the cloud
      dxarray = cloud.dr[None,:].to(u.cm) * np.sqrt(1 - np.square(ip_ratio ) ) # shape (impact_parameter.size, cloud.radius.size)

      if wavelength.shape == self.wavelength.shape:
        velocity = self.species_velocity
      else:
        velocity = calcvel(wavelength,
                           self.myatoms.wave,
                           ntabs = ntabs+1
                           )  # shape (wavelength.size, self.myatoms.wave.size)

      # Parallel processing all of the absorbing species
      t0 = tm.time()
      pool_tuple_input = []
      for myatoms_index in range(self.myatoms.anum.size):
        pool_tuple_input.append([cloud.temperature,dxarray,myatoms_index,cloud.vlos,velocity,cloud.iondensity])
      try:
        self.optical_depth_pool_fill_time += tm.time() - t0
      except AttributeError:
        self.optical_depth_pool_fill_time = tm.time() - t0

      for tuple_input in pool_tuple_input:
        (optical_depth_species, dvel_mask, optical_depth_bv_fill_time, optical_depth_Ntau0_fill_time, optical_depth_voigt_calc_time) = self._abs_optical_depth_single_species(*tuple_input,
                                                                                                                                                                              ntabs = ntabs+1)
        optical_depth[dvel_mask,:,:] += optical_depth_species

    return optical_depth # shape (wavelength.size, impact_parameter.size, cloud.radius.size)

  #######################################################################################
  # Absorption optical depth by a single species from one sightline (at impact_parameter --> dxarray) piercing one cloud (vlos, iondiensity, temperature)
  def _abs_optical_depth_single_species(self,
                                        temperature,
                                        dxarray,
                                        myatoms_index,
                                        vlos,
                                        velocity,
                                        iondensity,
                                        ntabs = 0
                                        ):
    tau_const = (2 * np.sqrt(np.pi) * const.e.esu * const.e.esu / (const.m_e * const.c) ).decompose()

    t0 = tm.time()
    bvalue = np.sqrt(2 * const.k_B * temperature.to(u.K) / self.myatoms.amass[myatoms_index]) # shape (cloud.radius.size,)

    dvel = np.outer((velocity[:,myatoms_index] - vlos), # shape (velocity[:,myatoms_index].size,)
                     1 / bvalue                         # shape (cloud.radius.size,)
                    ).decompose()                       # shape (velocity[:,myatoms_index].size, cloud.radius.size)

    dvel_mask = np.abs(velocity[:,myatoms_index] - vlos)/np.average(bvalue) < 300.0
    if np.any(dvel_mask):
      which_ion_index = (self.myatoms.ions == 100*self.myatoms.anum[myatoms_index]+self.myatoms.ion[myatoms_index])

      t0 = tm.time()
      column_density = dxarray * np.squeeze(iondensity[:,which_ion_index])[None,:]                         # shape (impact_parameter.size, cloud.radius.size)
      tau0 = (tau_const * self.myatoms.flam[myatoms_index] * column_density / bvalue[None,:] ).decompose() # shape (impact_parameter.size, cloud.radius.size)
      optical_depth_Ntau0_fill_time = tm.time() - t0

      t0 = tm.time()
      if self.myatoms.gamma[myatoms_index].value > 0: # self.myatoms.wave[myatoms_index] > 1215.66 * u.Angstrom and self.myatoms.wave[myatoms_index] < 1215.68 * u.Angstrom:
        a = (self.myatoms.gamma[myatoms_index] * self.myatoms.wave[myatoms_index] / (4.0 * np.pi * bvalue)).decompose()        # shape (cloud.radius.size,)
        optical_depth_species = tau0[None,:,:] * wofz(dvel[dvel_mask,None,:] + 1j * a[None,None,:]).real
      else:
        optical_depth_species = tau0[None,:,:] * np.exp(-np.square(dvel[dvel_mask,None,:]))
      optical_depth_voigt_calc_time = tm.time() - t0

    else:
      optical_depth_species = np.zeros( (np.int16(np.sum(dvel_mask)),) + dxarray.shape) #  dvel[dvel_mask,None,:]*dxarray[None,:,:]
      optical_depth_Ntau0_fill_time = 0
      optical_depth_voigt_calc_time = 0

    optical_depth_bv_fill_time = tm.time() - t0

    return optical_depth_species, dvel_mask, optical_depth_bv_fill_time, optical_depth_Ntau0_fill_time, optical_depth_voigt_calc_time

  #######################################################################################
  # Wrapper to take a cloud class and pack the parameters into a 1D array to feed into scipy.optimize.minimize
  def _abs_pack(self,
                clouds,
                ntabs = 0
                ):
    x = np.array([])
    for cld in clouds:
      xclp, yclp = self._abs_project_clouds(cld.rcl, 
                                            cld.zcl, 
                                            cld.thetacl,
                                            ntabs = ntabs+1
                                            )
      x = np.append(x, [xclp, yclp, cld.zcl, cld.rhoindex, cld.logrhoscale, cld.logrho0, cld.logZ, cld.vlos.value])

    return x

  #######################################################################################
  # Make a plot of the normalized absorption spectra (observed and predicted)
  def _abs_plot(self,
                totflux, unabsflux,
                vcl = np.array([]),
                clouds = None,
                ntabs = 0
                ):
    plt.ion()
    plt.clf()

    #####################################################################################
    if self.mypars.showgeometry and clouds is not None:
      plt.subplot(121)
    plt.plot(self.velocity, np.zeros(self.velocity.size), "k--")
    for tdx in range(self.mypars.anum.size):
      try:
        myatoms_index = self.myatoms.getspecies(self.mypars.anum[tdx],
                                                self.mypars.ion[tdx],
                                                ntabs = ntabs+1
                                                )[self.mypars.trandx[tdx]]
      except IndexError:
        print("\t" * ntabs + self.mypars.anum[tdx])
        print("\t" * ntabs + self.mypars.ion[tdx])
        print("\t" * ntabs + self.mypars.trandx[tdx])
        input("Pause due to IndexError in anum/ion/trandx")

      plt.plot(self.velocity, np.ones(self.velocity.size) + tdx, "k--")
      #----------------------------------------------------------
      try:
        yhi = self.normobsflux + self.normobsferr + tdx
        ylo = self.normobsflux - self.normobsferr + tdx
        ylo[ylo < tdx] = tdx
        vel_mask = (self.obsvel[myatoms_index,:] > self.velocity[0]) & (self.obsvel[myatoms_index,:] < self.velocity[-1])
        plt.fill_between(self.obsvel[myatoms_index,vel_mask].to(u.km/u.s).value, 
                         yhi[vel_mask], 
                         ylo[vel_mask], 
                         step = 'mid',
                         color = self.mypars.plot_code[tdx], 
                         alpha=0.3
                         )
        plt.scatter(self.obsvel[myatoms_index,vel_mask],
                    self.normobsflux[vel_mask] + tdx,
                     color=self.mypars.plot_code[tdx],
                    label=f"{self.myatoms.specstr[myatoms_index]} "+r"$\lambda$"+f"{self.myatoms.wave[myatoms_index].to(u.Angstrom).value:.3f}")
      except:
        pass
      #----------------------------------------------------------
      try:
        for clddx in range(self.cld_totflux.shape[1]):
          plt.plot(self.species_velocity[:,myatoms_index],
                   self.cld_totflux[:,clddx]/unabsflux + tdx,
                   self.mypars.plot_code[tdx]+":"
                  )
      except:
        pass
      #----------------------------------------------------------
      try:
        for clddx in range(self.cld_totflux.shape[1]):
          plt.plot(self.species_velocity[:,myatoms_index],
                   self.cld_totflux[:,clddx]/unabsflux + tdx,
                   self.mypars.plot_code[tdx]+":",
                   lw=2*plt.rcParams['lines.linewidth']
                  )
      except:
        pass
      #----------------------------------------------------------
      try:
        plt.plot(self.species_velocity[:,myatoms_index],
                 totflux/unabsflux + tdx,
                 self.mypars.plot_code[tdx], lw=2*plt.rcParams['lines.linewidth'])
      except:
        pass
      #----------------------------------------------------------
      try:
        if self.bestfit is not None:
          plt.plot(self.species_velocity[:,myatoms_index],
                   self.bestfit + tdx,
                   f"{self.mypars.plot_code[tdx]}--",
                   lw=2*plt.rcParams['lines.linewidth'])
      except:
        pass

    if vcl.size > 0:
      for v in vcl:
        v_nounits = v.to(u.km/u.s).value
        plt.plot(np.array([v_nounits,v_nounits]),
                 np.array([-0.2, 1.2*self.mypars.anum.size]),
                 "k--")

    plt.legend(loc='lower right')

    title_str  = self.mypars.qname + f" zqso = {self.mypars.zqso} "
    try:
      chisq_spec = self._abs_chisq(totflux, 
                                   unabsflux,
                                   ntabs = ntabs+1)
      chisq      = np.sum(chisq_spec)
      title_str += r"$\chi^2 = $" + f"{chisq:.3f}"
    except:
      pass

    plt.title(title_str)
    plt.xlim([self.velocity[0].to(u.km/u.s).value, self.velocity[-1].to(u.km/u.s).value])
    plt.ylim([-0.2, 1.2*self.mypars.anum.size])

    #####################################################################################
    # To have osberver be in the xz plane, and the tile about the y-axis, we need to 
    # display things so that +x_obs is -y_display and +y_obs is +x_display
    if self.mypars.showgeometry and clouds is not None:
      plt.subplot(122)
      # Accretion disk annuli
      ecc = np.cos(self.mypars.inclination)
      theta = np.linspace(0,np.pi) * u.rad
      for brdx in range(self.mydisk.rstar.size):
        rdx = self.mydisk.rstar.size - brdx - 1
        a_out = self.mydisk.rstar[rdx] + 0.5*self.mydisk.drstar[rdx]
        b_out = a_out * np.sqrt(1 - ecc*ecc)
        a_in = self.mydisk.rstar[rdx] - 0.5*self.mydisk.drstar[rdx]
        b_in = a_in * np.sqrt(1 - ecc*ecc)
        norm_temp = self.mydisk.tempt1[rdx].value / np.max(self.mydisk.tempt1.value)

        plt.fill_between(a_out * np.cos(theta), 
                         b_out * np.sin(theta), 
                         -b_out * np.sin(theta),
                         color = 'y', 
                         alpha = norm_temp
                         )
        plt.fill_between(a_in * np.cos(theta), 
                         b_in * np.sin(theta), 
                         -b_in * np.sin(theta),
                         color = 'w', 
                         alpha = 1
                         )

      # Black hole
      plt.fill_between(np.cos(theta), 
                       np.sin(theta), 
                       -np.sin(theta),
                       color = 'k', 
                       alpha = 1
                       )

      # Absorbing clouds
      xlimit = 0.0
      for cldx in range(len(clouds)):
        xclp, yclp = self._abs_project_clouds(clouds[cldx].rcl,
                                              clouds[cldx].zcl,
                                              clouds[cldx].thetacl,
                                              ntabs = ntabs+1
                                             )
        xlimit = np.max([xlimit,
                         np.sqrt(xclp*xclp+yclp*yclp) + np.max(clouds[cldx].radius)/self.mydisk.rg
                        ])
        for rdx in range(clouds[cldx].radius.size):
          norm_dens = np.log(clouds[cldx].density[rdx].value) / np.log(np.max(clouds[cldx].density.value))
          plt.fill_between(clouds[cldx].radius[rdx] * np.cos(theta) / self.mydisk.rg + yclp,
                           clouds[cldx].radius[rdx] * np.sin(theta) / self.mydisk.rg - xclp,
                           -clouds[cldx].radius[rdx] * np.sin(theta) / self.mydisk.rg - xclp,
                           color = self.mypars.plot_code[cldx % len(self.mypars.plot_code)],
                           alpha = 0.01 * norm_dens
                          )
      if 1.5*xlimit > self.mydisk.x1:
        plt.plot(self.mydisk.x1 * np.cos(theta),
                 self.mydisk.x1 * np.sin(theta),
                 "k:"
                 )
        plt.plot(self.mydisk.x1 * np.cos(theta),
                 -self.mydisk.x1 * np.sin(theta),
                 "k:"
                 )
      if 1.5*xlimit > self.mydisk.x2:
        plt.plot(self.mydisk.x2 * np.cos(theta),
                 self.mydisk.x2 * np.sin(theta),
                 "k:"
                 )
        plt.plot(self.mydisk.x2 * np.cos(theta),
                 -self.mydisk.x2 * np.sin(theta),
                 "k:"
                 )

      plt.xlim(-1.5*xlimit,
               1.5*xlimit
               )
      plt.ylim(-1.5*xlimit,
               1.5*xlimit
               )

    plt.show(block=False)
    plt.pause(0.001)

  #######################################################################################
  def _abs_project_clouds(self,
                          rcl, zcl, thetacl,
                          ntabs = 0
                          ):
    self.reset_observer()
    xobs = self.robs * np.cos(self.mydisk.thetaobs)
    yobs = self.robs * np.sin(self.mydisk.thetaobs)

    a    = - zcl / (zcl - self.zobs) # --> 0 if zobs >>> zcl
    xclp = -a * xobs + (1+a) * rcl * np.cos(thetacl)
    yclp = -a * yobs + (1+a) * rcl * np.sin(thetacl)

    return xclp, yclp
  
  #######################################################################################
  def _abs_read_clouds(self,
                       mypars,
                       ntabs = 0):
    print("\t" * ntabs + f"\tReading clouds from {self.cloud_filename}")
    cloud_table = Table.read(self.cloud_filename, format="fits")
    cloud_table.pprint()
    xclp        = np.array(cloud_table["xclp"])
    yclp        = np.array(cloud_table["yclp"])
    zcl         = np.array(cloud_table["zcl"])
    rhoindex    = np.array(cloud_table["rhoindex"])
    logrhoscale = np.array(cloud_table["logrhoscale"])
    logrho0     = np.array(cloud_table["logrho0"])
    logZ        = np.array(cloud_table["logZ"])
    vcl_los     = np.array(cloud_table["vcl"]) * (u.km/u.s)

    clouds = self.makeclouds(xclp, 
                             yclp, 
                             zcl, 
                             rhoindex, 
                             logrhoscale, 
                             logrho0, 
                             logZ, 
                             vcl_los,
                             ntabs = ntabs+1)

    return clouds

  #######################################################################################
  # Wrapper to take the parameters fed into/from scipy.optimize.minimize and unpack it into a cloud class
  def _abs_unpack(self,
                  x,
                  ntabs = 0
                  ):
    if len(x)//8 > 0:
      xclp        = np.zeros(len(x)//8)
      yclp        = np.zeros(len(x)//8)
      zcl         = np.zeros(len(x)//8)
      rhoindex    = np.zeros(len(x)//8)
      logrhoscale = np.zeros(len(x)//8)
      logrho0     = np.zeros(len(x)//8)
      logZ        = np.zeros(len(x)//8)
      vcl         = np.zeros(len(x)//8) * (u.km/u.s)
      for i in range(len(x)//8):
        xclp[i]        = x[8*i]
        yclp[i]        = x[8*i+1]
        zcl[i]         = x[8*i+2]
        rhoindex[i]    = x[8*i+3]
        logrhoscale[i] = x[8*i+4]
        logrho0[i]     = x[8*i+5]
        logZ[i]        = x[8*i+6]
        vcl[i]         = x[8*i+7] * (u.km/u.s)

      clouds = self.makeclouds(xclp, 
                               yclp, 
                               zcl, 
                               rhoindex, 
                               logrhoscale, 
                               logrho0, 
                               logZ, 
                               vcl,
                               ntabs = ntabs+1
                               )
      if self.mypars.verbose:
        self.print_clouds(clouds, 
                          ntabs = ntabs+1
                          )
    else:
      clouds = None

    return clouds

  #######################################################################################
  def _abs_write_clouds(self,
                   clouds,
                   ntabs = 0
                   ):
    (rcl, zcl, thetacl, logrhoscale, rhoindex, logrho0, logZ, vcl_los) = self.grab_cloud_pars(clouds,
                                                                                              ntabs = ntabs+1
                                                                                              )
    xclp, yclp = self._abs_project_clouds(rcl, 
                                          zcl, 
                                          thetacl,
                                          ntabs = ntabs+1
                                          )
    
    print("\t" * ntabs + f"\tWriting clouds to {self.cloud_filename}")
    cloud_table = Table(data=[xclp, yclp, zcl, rhoindex, logrhoscale, logrho0, logZ, vcl_los],
                        names=["xclp","yclp","zcl","rhoindex","logrhoscale","logrho0","logZ","vcl"])
    cloud_table.pprint()
    cloud_table.write(self.cloud_filename, 
                      format="fits", 
                      overwrite=True
                      )
    
    return
  
  #######################################################################################
  def _add_to_cheb(self,
                   cloud,
                   ntabs = 0
                   ):
    print("\t" * ntabs + "\t\tExpanding Chebyshev fits with new Cloudy run...")
    (lognuFnu, log_ion_parm) = self._calc_ion_parm(cloud,
                                                   ntabs = ntabs+1)

    self.cheb_log_ion_parm_list = np.append(self.cheb_log_ion_parm_list, 
                                            log_ion_parm
                                            )

    # Iterate through ions and the ion fraction...
    ionfrac = np.zeros(cloud.iondensity.shape)
    for i in range(self.myatoms.nion):
      logZdum = cloud.logZ
      if self.myatoms.anum[self.myatoms.idx[i]] < 3:
        logZdum = 0.0
      logelemabund = self.myatoms.abund[self.myatoms.idx[i]]-12+logZdum
      ionfrac[:,i] = cloud.iondensity[:,i] / (cloud.density * np.power(10.0, 
                                                                       logelemabund
                                                                       )
                                                                       )

    if self.cheb_ionfrac_list.size > 0:
      self.cheb_ionfrac_list = np.append(self.cheb_ionfrac_list, 
                                         ionfrac, 
                                         axis=0
                                         )
      self.cheb_temperature_list = np.append(self.cheb_temperature_list, 
                                             cloud.temperature.to(u.K).value
                                             )
    else:
      self.cheb_ionfrac_list = np.copy(ionfrac)
      self.cheb_degree = np.ones(self.myatoms.idx.size, 
                                 dtype=np.int16
                                 )
      self.cheb_coeff_list = np.zeros((self.myatoms.idx.size,
                                       1
                                       )
                                       )

      self.cheb_temperature_list = np.copy(cloud.temperature.to(u.K).value)
      self.cheb_log_temperature_degree = 1
      self.cheb_log_temperature_coeff = np.zeros(self.cheb_log_temperature_degree)

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
          coeff,res = chebyshev.chebfit(xnorm[ion_mask],
                                        np.log10(self.cheb_ionfrac_list[ion_mask,i]), 
                                        self.cheb_degree[i], 
                                        full=True
                                        )
          if res[0].size > 0:
            fstat = (oldchisq / olddeg )  / (np.squeeze(res[0]) / self.cheb_degree[i])
            p_value = Ftest.sf(fstat, 
                               olddeg, 
                               self.cheb_degree[i]
                               )
            if p_value < 0.48:
              self.cheb_degree[i] += 1
              done = False
            else:
              self.cheb_degree[i] = np.max(np.array([1, self.cheb_degree[i]-1]))
              coeff,res = chebyshev.chebfit(xnorm[ion_mask],
                                            np.log10(self.cheb_ionfrac_list[ion_mask,i]), 
                                            self.cheb_degree[i], 
                                            full=True
                                            )
              done = True
          else:
            self.cheb_degree[i] = np.max(np.array([1, self.cheb_degree[i]-1]))
            coeff,res = chebyshev.chebfit(xnorm[ion_mask],
                                          np.log10(self.cheb_ionfrac_list[ion_mask,i]), 
                                          self.cheb_degree[i], 
                                          full=True
                                          )
            done = True
          olddeg = self.cheb_degree[i]
          oldchisq = np.squeeze(res[0])

          if coeff.size > self.cheb_coeff_list.shape[1]:
            dum = np.copy(self.cheb_coeff_list)
            self.cheb_coeff_list = np.zeros((self.myatoms.idx.size, 
                                             coeff.size  
                                             )
                                             )
            self.cheb_coeff_list[:,:dum.shape[1]] = np.copy(dum)
            self.cheb_coeff_list[i,:] = 0.0
          self.cheb_coeff_list[i,:coeff.size] = np.copy(coeff)


    done = False
    oldchisq = 9.99e+99
    olddeg = self.cheb_log_temperature_degree
    while not done:
      coeff,res = chebyshev.chebfit(xnorm,
                                    np.log10(self.cheb_temperature_list), 
                                    self.cheb_log_temperature_degree, 
                                    full=True
                                    )
      if res[0].size > 0:
        fstat = (oldchisq / olddeg )  / (np.squeeze(res[0]) / self.cheb_log_temperature_degree)
        p_value = Ftest.sf(fstat, 
                           olddeg, 
                           self.cheb_log_temperature_degree
                           )
        if p_value < 0.48:
          self.cheb_log_temperature_degree += 1
          done = False
        else:
          self.cheb_log_temperature_degree = np.max(np.array([1, self.cheb_log_temperature_degree-1]))
          coeff,res = chebyshev.chebfit(xnorm,
                                        np.log10(self.cheb_temperature_list), 
                                        self.cheb_log_temperature_degree, 
                                        full=True
                                        )
          done = True
      else:
        self.cheb_log_temperature_degree = np.max(np.array([1, self.cheb_log_temperature_degree-1]))
        coeff,res = chebyshev.chebfit(xnorm,
                                      np.log10(self.cheb_temperature_list), 
                                      self.cheb_log_temperature_degree, 
                                      full=True
                                      )
        done = True
      olddeg = self.cheb_log_temperature_degree
      oldchisq = np.squeeze(res[0])

      self.cheb_log_temperature_coeff = np.copy(coeff)


    sort_index = np.argsort(self.cheb_log_ion_parm_list)
    self.cheb_log_ion_parm_list = self.cheb_log_ion_parm_list[sort_index]
    self.cheb_ionfrac_list = self.cheb_ionfrac_list[sort_index,:]
    self.cheb_temperature_list = self.cheb_temperature_list[sort_index]

    self.cheb_dlog_ion_parm_list = np.zeros(self.cheb_log_ion_parm_list.size)
    self.cheb_dlog_ion_parm_list[1:-1] = 0.5 * (self.cheb_log_ion_parm_list[2:] - self.cheb_log_ion_parm_list[:-2])
    self.cheb_dlog_ion_parm_list[0] = self.cheb_dlog_ion_parm_list[1]
    self.cheb_dlog_ion_parm_list[-1] = self.cheb_dlog_ion_parm_list[-2]


    chebfile  = self.mypars.datapath+f"/Cloudy_runs/Sbh{self.mypars.sbh}-MBH{np.log10(self.mypars.mbh / const.M_sun):.2f}"
    chebfile += f"-Mdot{(self.mypars.mdot/(const.M_sun/u.year)).decompose()}-alpha{self.mypars.viscosity_alpha}_ionfracs_lox{self.myatoms.minlox}.fits"
    datatab = Table(data=[self.cheb_log_ion_parm_list, self.cheb_ionfrac_list, self.cheb_temperature_list], names=["log_ion_parm_list", "ionfrac_list", "temperature_list"])

    datatab2 = Table(data=[self.cheb_degree,self.cheb_coeff_list], names=["DEGREE","COEFFS"])

    try:
      datatab3 = Table(data=[self.cheb_log_temperature_coeff], names=["COEFFS"])
    except TypeError:
      print("\t" * ntabs + self.cheb_log_temperature_degree)
      print("\t" * ntabs + self.cheb_log_temperature_coeff)
      input("Making this table barfed...")

    datatab.write( chebfile, format="fits", overwrite=True)
    datatab2.write(chebfile, format="fits", append=True)
    datatab3.write(chebfile, format="fits", append=True)

  #######################################################################################
  def _build_modwave(self,
                     wres = 0.01 * u.Angstrom,
                     ntabs = 0
                     ):
    nwave = np.int64(((np.max(self.obswave) - np.min(self.restwave)) / wres).decompose())
    print("\t" * ntabs + f"Building model wavelength ranges from {np.min(self.restwave)} to {np.max(self.obswave)} in {nwave} {wres}-bins")
    self.wavelength = np.linspace(np.min(self.restwave.to(u.Angstrom).value),
                                  np.max(self.obswave.to(u.Angstrom).value),
                                  nwave
                                  ) * u.Angstrom

    print("\t" * ntabs + "Calculating velocity ranges for posterity")
    self.species_velocity = calcvel(self.wavelength,
                                    self.myatoms.wave,
                                    ntabs = ntabs+1
                                    ).to(u.km/u.s)

  #######################################################################################
  def _calc_ion_parm(self,
                     cloud,
                     ntabs = 0
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
                                        robs       = None,
                                        thetaobs   = None,
                                        zobs       = None,
                                        wavelength = None,
                                        debug      = False,
                                        lograd     = False,
                                        noplot     = False,
                                        ntabs = 0
                                        ):
    fu = (u.erg / (u.s * u.cm * u.cm * u.Hz))
    t0 = tm.time()
    if wavelength is None:
      wavelength = self.wavelength
    totflux     = np.zeros(wavelength.shape) * fu
    unabsflux   = np.zeros(wavelength.shape) * fu
    if clouds is not None:
      self.cld_totflux = np.zeros((wavelength.size, 
                                   len(clouds)
                                   )
                                   ) * fu

    if robs is None:
      robs = self.robs
    if thetaobs is None:
      thetaobs = 0.0
    if zobs is None:
      zobs = self.zobs

    self.reset_observer(robs     =    robs,
                        thetaobs = thetaobs,
                        zobs      =    zobs,
                        ntabs = ntabs+1
                        )
    robs_vec = np.array([robs * np.cos(thetaobs),
                        robs * np.sin(thetaobs),
                        zobs
                        ])

    if self.mypars.verbose:
      print("\t" * ntabs + f"\tSetting up integration for spectral synthesis with observer at {robs_vec}")

    done = False
    scale = 1
    while not done:
      done = True
      # Set up Gauss-Legendre grid for integration
      if self.mypars.verbose:
        print("\t" * ntabs + f"\t\tDetermining Gauss-Legendre positions and weights for a {scale * self.mypars.gaussleg_nr} x {scale * self.mypars.gaussleg_ntheta} grid ({tm.time()-t0})")
      gaussleg_y_r,     gaussleg_w_r     = np.polynomial.legendre.leggauss(scale * self.mypars.gaussleg_nr)     # Cylindrical radius (normalized)
      gaussleg_y_theta, gaussleg_w_theta = np.polynomial.legendre.leggauss(scale * self.mypars.gaussleg_ntheta) # Azimuhtal angle (normalized)

      plt.ion()

      separate_theta = False
      pool_tuple_input = []
      for rdx in range(scale*self.mypars.gaussleg_nr):
        if lograd:
          rdisk = self.mydisk.rstar[0] * np.power(self.mydisk.rstar[-1]/self.mydisk.rstar[0], 
                                                  (gaussleg_y_r[rdx] + 1)/2
                                                  ) # Units rg (log)
        else:
          rdisk = self.mydisk.rstar[0] + (self.mydisk.rstar[-1] - self.mydisk.rstar[0]) * (gaussleg_y_r[rdx] + 1) / 2 # Units rg (linear)
        
        if separate_theta:
          for tdx in range(scale*self.mypars.gaussleg_ntheta):
            thetadisk = np.array([np.pi * (gaussleg_y_theta[tdx] + 1.)]) # Azimuthal angle

            pool_tuple_input.append((rdisk,
                                     thetadisk,
                                     clouds,
                                     robs_vec,
                                     gaussleg_y_r[rdx],
                                     gaussleg_w_r[rdx],
                                     gaussleg_w_theta[tdx],
                                     wavelength,
                                     lograd
                                    )
                                   )
          dumstr = "sightlines"
        else:
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
          dumstr = "annuli"

      if self.mypars.verbose:
        print("\t" * ntabs + f"\t\tIntegrating across disk with at most {self.mypars.nproc} processors...({tm.time()-t0})")

      with Pool(self.mypars.nproc) as pool:
        output_flux_pool_tuple = pool.starmap_async(self._flux_annulus, 
                                                    pool_tuple_input
                                                    )
        output_flux_pool_tuple.wait()

      if self.mypars.verbose:
        print("\t" * ntabs + f"\t\tIntegration complete... ({tm.time()-t0})")

      min_impact_parameters_all_sightlines = None
      plot_nit = 0
      for (fluxrtnu, cld_optical_depth, min_impact_parameter) in tqdm(output_flux_pool_tuple.get(), desc=f"\t\t\tAssembling {dumstr}", ncols=0):
        tmp_flux_sum = np.sum(fluxrtnu, axis=-1)  # Sum over sightlines
        unabsflux += tmp_flux_sum
        
        if clouds is not None:
          for clddx in range(len(clouds)):
            self.cld_totflux[:,clddx] += np.sum(fluxrtnu * np.exp(-cld_optical_depth[clddx,:,:]), 
                                                axis=-1
                                                )
          totflux += np.sum(fluxrtnu * np.exp(-np.sum(cld_optical_depth, axis=0)), 
                            axis=-1
                            )
        else:
          totflux += tmp_flux_sum # Sum over sightlines
        
        if min_impact_parameter is not None:
          try:
            min_impact_parameters_all_sightlines = np.append(min_impact_parameters_all_sightlines, 
                                                             [min_impact_parameter], 
                                                             axis=0)
          except ValueError:
            min_impact_parameters_all_sightlines = np.array([min_impact_parameter])
        

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
          print("\t" * ntabs + prtstr)
          print("\t" * ntabs + f"\tIncreasing scale to {scale}")
          if scale*(self.mypars.gaussleg_nr + self.mypars.gaussleg_ntheta) > 8000:
            print("\t" * ntabs + f"\tThis is would be too expensive - bailing")
            done = True

    if clouds is not None: # and not noplot:
      self._abs_plot(totflux, unabsflux,
                     vcl = self.grab_cloud_pars(clouds)[-1], # = vcl
                     clouds = clouds,
                     ntabs = ntabs+1
                     )
      plt.pause(0.001)

    return totflux,unabsflux # shape = wavelength.shape

  #######################################################################################
  def _flux_annulus(self,
                    rdisk, thetadisk, # Disk parameters. rdisk is scalar, thetadisk is 1D nd.array
                    clouds,           # Clouds parameters (list of AbsClouds)
                    robs_vec,         # Observer parameters (nd.array with shape (3,) )
                    gaussleg_y_r, gaussleg_w_r, gaussleg_w_theta, # Integration parameters [y = radial/log-radial location, w = weights]
                    wavelength,       # Spectral parameters (i.e., wavelengths in a 1D nd.array)
                    lograd,           # boolean for quadrature method
                    ntabs = 0
                    ):
    fu = (u.erg / (u.s * u.cm * u.cm * u.Hz))

    t0 = tm.time()

    zt1cs = CubicSpline(self.mydisk.rstar, 
                        self.mydisk.zt1
                        )
    Tt1cs = CubicSpline(self.mydisk.rstar, 
                        self.mydisk.tempt1.to(u.K).value
                        )

    frequency = wavelength.to(u.Hz, equivalencies=u.spectral())

    rdisk_vec = np.array([rdisk * np.cos(thetadisk),
                          rdisk * np.sin(thetadisk),
                          np.broadcast_to(zt1cs(rdisk), (thetadisk.size,))
                          ])

    dzt1dr = zt1cs(rdisk, 
                   nu=1
                   )
    Tt1    = Tt1cs(rdisk) * u.K

    gradZ_vec = np.array([-dzt1dr * np.cos(thetadisk),
                          -dzt1dr * np.sin(thetadisk),
                          np.ones(thetadisk.size)
                          ])
    gradZmag = np.sqrt(np.sum(gradZ_vec*gradZ_vec, 
                              axis=0
                              )
                      )

    R_vec = np.broadcast_to(robs_vec, rdisk_vec.T.shape).T - rdisk_vec
    Rmag = np.sqrt(np.sum(R_vec*R_vec, 
                          axis=0
                          )
                  )
    if(np.any(Rmag == 0)):
      input("Rmag is zero")

    cosbeta = np.sum(R_vec * gradZ_vec, 
                     axis=0
                     ) / (Rmag * gradZmag)
    cosbeta_mask = cosbeta >  0

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
    Bnu_mask = cosbeta_mask[None,:] & (const.h * doppler_unshift_freq / (const.k_B * Tt1) < 670.74) # This is to prevent underflows... 670.74 was just determined through trial and error
    Bnu[Bnu_mask] = BlackBody().evaluate(doppler_unshift_freq[Bnu_mask],Tt1,1) # Emitted at doppler_unshift_freq, observed at gaussleg_freq


    tmp_fluxrtnu =  np.pi * (Bnu[:,cosbeta_mask] * u.sr) * gaussleg_w_r * gaussleg_w_theta[cosbeta_mask] * (doppler_beam_fac[cosbeta_mask]**3) * cosbeta[cosbeta_mask] / Rmag[cosbeta_mask]**2
    fluxrtnu = np.zeros(Bnu.shape) * fu
    if lograd:
      fluxrtnu[:,cosbeta_mask]  = ((self.mydisk.rstar[0])**2) * (np.log(self.mydisk.rstar[-1]/self.mydisk.rstar[0])) * np.power(self.mydisk.rstar[-1]/self.mydisk.rstar[0],
                                                                                                                                gaussleg_y_r + 1) * tmp_fluxrtnu
    else:
      fluxrtnu[:,cosbeta_mask] = (self.mydisk.rstar[-1] - self.mydisk.rstar[0]) * rdisk * tmp_fluxrtnu / 2.0

    try:
      self.t_flux_annulus_pre_tau += tm.time() - t0
    except:
      self.t_flux_annulus_pre_tau = tm.time() - t0

    t0 = tm.time()
    cld_optical_depth, min_impact_parameter = self._abs_all_optical_depth(clouds,           # List of AbsClouds
                                                                          rdisk_vec, R_vec, # 2D nd.arrays with shapes (3,thetadisk.size)
                                                                          wavelength,       # 1d nd.array
                                                                          ntabs = ntabs+1
                                                                          ) # shape = (wavelength.size, thetadisk.size)
    try:
      self.t_flux_annulus_post_tau += tm.time() - t0
    except:
      self.t_flux_annulus_post_tau = tm.time() - t0

    return fluxrtnu, cld_optical_depth, min_impact_parameter

  #######################################################################################
  def _read_cheb_files(self,
                       ntabs = 0
                       ):
    chebfile  = self.mypars.datapath+f"/Cloudy_runs/Sbh{self.mypars.sbh}-MBH{np.log10(self.mypars.mbh / const.M_sun):.2f}"
    chebfile += f"-Mdot{(self.mypars.mdot/(const.M_sun/u.year)).decompose()}-alpha{self.mypars.viscosity_alpha}_ionfracs_lox{self.myatoms.minlox}.fits"
    if os.path.exists(chebfile):
      datatab1 = Table.read(chebfile, hdu=1)
      datatab2 = Table.read(chebfile, hdu=2)
      datatab3 = Table.read(chebfile, hdu=3)

      self.cheb_log_ion_parm_list = datatab1["log_ion_parm_list"]
      self.cheb_ionfrac_list      = datatab1["ionfrac_list"]
      self.cheb_temperature_list  = datatab1["temperature_list"]
      self.cheb_coeff_list        = datatab2["COEFFS"]
      self.cheb_degree            = datatab2["DEGREE"]
      self.cheb_log_temperature_coeff = datatab3["COEFFS"]
      self.cheb_log_temperature_degree = self.cheb_log_temperature_coeff.size-1

      sort_index = np.argsort(self.cheb_log_ion_parm_list)
      self.cheb_log_ion_parm_list = self.cheb_log_ion_parm_list[sort_index]
      self.cheb_ionfrac_list = self.cheb_ionfrac_list[sort_index,:]
      self.cheb_temperature_list = self.cheb_temperature_list[sort_index]

      self.cheb_dlog_ion_parm_list = np.zeros(self.cheb_log_ion_parm_list.size)
      self.cheb_dlog_ion_parm_list[1:-1] = 0.5 * (self.cheb_log_ion_parm_list[2:] - self.cheb_log_ion_parm_list[:-2])
      self.cheb_dlog_ion_parm_list[0] = self.cheb_dlog_ion_parm_list[1]
      self.cheb_dlog_ion_parm_list[-1] = self.cheb_dlog_ion_parm_list[-2]
    else:
      self.cheb_log_ion_parm_list  = np.array([])
      self.cheb_ionfrac_list       = np.array([])
      self.cheb_temperature_list   = np.array([])
      self.cheb_coeff_list         = np.array([])
      self.cheb_degree             = np.array([])
      self.cheb_log_temperature_coeff  = np.array([])
      self.cheb_log_temperature_degree = 0
      self.cheb_dlog_ion_parm_list = np.array([])

  #######################################################################################
  # Solver for self.mywind._EULER
  def _wnd_calcstreamline_relativistic(self, 
                                       dtime = 10.0 * u.s, 
                                       mindt = 1.0 * u.s, 
                                       vres = 2000.0 * (u.km/u.s), 
                                       minr_rg = 100.0, 
                                       plotstream=False, 
                                       mupdate = True,
                                       ntabs = 0
                                       ):

    t1 = tm.time() * u.s
    print("\t" * ntabs + 'Welcome to your friendly neighborhood relativistic fluid dynamics solver...')

    self.mywind.squiggle = self._wnd_read_squiggle(ntabs = ntabs+1)

    # --- Iterative Solver ---
    if plotstream:
      plt.ion()
      plt.figure()
      plt.pause(0.001)
    t0      = tm.time() * u.s
    dvmax   = 1.0 * (u.km/u.s)
    done    = False
    titeration = tm.time() * u.s
    tplt = tm.time() * u.s
    print("\t" * ntabs + 'Beginning hydrodynamics simulation...')
    while not done:
      # --- Velocity matters ---
      vmag = np.sqrt(self.mywind.v_rsph*self.mywind.v_rsph + self.mywind.v_thetasph*self.mywind.v_thetasph + self.mywind.v_phi*self.mywind.v_phi)
      vmag[(vmag/const.c).decompose() >= 1] = 0.99 * const.c
      self.mywind.lorentz_factor = 1. / np.sqrt(1.0 - ((vmag / const.c).decompose())**2)

      trf = tm.time() * u.s
      self.mywind.rf,self.mywind.z0_for_rf = self.mywind._get_rf()
      trf = tm.time() * u.s - trf

      # -- Radiation force, pressure and gas pressure ---
      self.mywind.P_total = self.mywind._P_gas(ntabs = ntabs+1) + self.mywind._P_rad(ntabs = ntabs+1)

      # Thermodynamics
      self.mywind.specific_enthalpy = 1 + self.mywind.adiabatic_index/(self.mywind.adiabatic_index-1.) * (self.mywind.P_total)/(self.mywind.mass_density*const.c**2)

      for arrstr,arr in [("P_total", self.mywind.P_total),
                         ("mass density", self.mywind.mass_density)
                         ]:
        if not self.mywind._sanity_check(arrstr,arr):
          input("Insane")

      # Residuals
      tres = tm.time() * u.s
      poolinput = [(dtime, 'rho'),
                   (dtime,  'vr'),
                   (dtime, 'vth'),
                   (dtime, 'vph')
                   ]
      if self.mypars.nproc > 1:
        list_of_euler_linarray_tuples = []
        with Pool(self.mypars.nproc) as pool:
          list_of_euler_linarray_tuples = pool.starmap(self.mywind._EULER, poolinput)

        if len(list_of_euler_linarray_tuples) == 4:
          drho = list_of_euler_linarray_tuples[0]
          dvr  = list_of_euler_linarray_tuples[1]
          dvth = list_of_euler_linarray_tuples[2]
          dvph = list_of_euler_linarray_tuples[3]
        else:
          input("EULER barfed")
      else:
        for (dthing, thing) in poolinput:
          dthing = self.mywind._EULER(dtime, thing)

      tres = tm.time() * u.s - tres

      # Is our step small enough to remain physical? If so, update velocity and density fields
      tupdate = tm.time() * u.s
      rho_tmp = np.copy(self.mywind.mass_density)
      tau_es = (self.mywind.mass_density * const.sigma_T.cgs * self.mywind.DRR / const.u.cgs).decompose()
      where_density_changed = self.mywind.boundary_mask & (np.fabs(drho) / rho_tmp > 0.1) & (tau_es < 0.7) & (rho_tmp / const.u.cgs > 1.0e-5 / u.cm**3)
      if (np.max(np.fabs([dvr[self.mywind.boundary_mask].to(u.km/u.s),dvth[self.mywind.boundary_mask].to(u.km/u.s),dvph[self.mywind.boundary_mask].to(u.km/u.s)])) < vres.to(u.km/u.s).value) and np.all(drho[self.mywind.boundary_mask] < rho_tmp[self.mywind.boundary_mask]):
        # Relax towards enforcing continuity (conservative update)
        self.mywind.mass_density[self.mywind.boundary_mask] -= drho[self.mywind.boundary_mask]
        self.mywind.v_rsph[      self.mywind.boundary_mask] += dvr[ self.mywind.boundary_mask]
        self.mywind.v_thetasph[  self.mywind.boundary_mask] += dvth[self.mywind.boundary_mask]
        self.mywind.v_phi[       self.mywind.boundary_mask] += dvph[self.mywind.boundary_mask]
        self.mywind.tottime                                 += dtime

        self.mywind.number_density[self.mywind.boundary_mask] = self.mywind.mass_density[self.mywind.boundary_mask] / const.u.cgs

        # Update the force multiplier grid
        if mupdate and np.sum(where_density_changed) > 0:
          self._wnd_force_multiplier(where_density_changed,
                                     ntabs = ntabs+1
                                     )

        dtime *= np.pi
        if self.mywind.tottime > 1.0e+8 * u.year:
          done = True
      else:
        dtime /= np.exp(1.0)

      vmag = np.sqrt(self.mywind.v_rsph*self.mywind.v_rsph + self.mywind.v_thetasph*self.mywind.v_thetasph + self.mywind.v_phi*self.mywind.v_phi)
      dvmag = (self.mywind.v_rsph * dvr + self.mywind.v_thetasph * dvth + self.mywind.v_phi * dvph) / vmag
      vminpred = vmag + mindt * dvmag / dtime
      where_velocity_bad = self.mywind.boundary_mask & (vminpred > const.c)

      # Is the timestep too small? Do we need to mask additional bins?
      if dtime < mindt:
        maxdv = np.max(np.fabs([dvr[self.mywind.boundary_mask].to(u.cm/u.s), 
                                dvth[self.mywind.boundary_mask].to(u.cm/u.s), 
                                dvph[self.mywind.boundary_mask].to(u.cm/u.s)
                                ]
                                )
                                ) * (u.cm/u.s)
        self.mywind.boundary_mask = (self.mywind.boundary_mask & \
                                     (np.fabs(dvr) < maxdv)    & \
                                     (np.fabs(dvth) < maxdv)   & \
                                     (np.fabs(dvph) < maxdv)   & \
                                     (vminpred < const.c)      & \
                                     (rho_tmp * dtime/-drho > mindt)
                                     )
      tupdate = tm.time() * u.s - tupdate

      # --- Output ---
      v_r_val  = (self.mywind.v_rsph[    :-1, :-1].to(u.km/u.s)).value
      v_th_val = (self.mywind.v_thetasph[:-1, :-1].to(u.km/u.s)).value
      v_ph_val = (self.mywind.v_phi[     :-1, :-1].to(u.km/u.s)).value

      dvr_val  = (dvr[ :-1, :-1].to(u.km/u.s)).value
      dvth_val = (dvth[:-1, :-1].to(u.km/u.s)).value
      dvph_val = (dvph[:-1, :-1].to(u.km/u.s)).value

      f_rad_r_val  = (self.mywind._f_rad_r()  / self.mywind.mass_density )[:-1,:-1].decompose(bases=u.cgs.bases).to(u.km/u.s**2).value
      f_rad_th_val = (self.mywind._f_rad_th() / self.mywind.mass_density )[:-1,:-1].decompose(bases=u.cgs.bases).to(u.km/u.s**2).value

      rgg_val = (-self.mywind.mass_density * self.mywind.lorentz_factor * const.G.cgs * self.mypars.mbh / self.mywind.RR**2)[:-1,:-1].value

      if (tm.time() * u.s - tplt > 30 * u.s) and plotstream:
        plt.clf()
        for (pnum,title,colarr) in [(1, r'$\log |v_r|$/[km s$^{-1}$]',                                                        np.log10(np.fabs(v_r_val))),
                                    (2, r'$\log |v_\theta|$/[km s$^{-1}$]',                                                  np.log10(np.fabs(v_th_val))),
                                    (3, r'$\log |v_\phi|$/ [km s$^{-1}$]',                                                   np.log10(np.fabs(v_ph_val))),
                                    (4, r'$\log r_\mathrm{f}/$[r$_g$]',                                 np.log10(self.mywind.rf[:-1,:-1]/self.mydisk.rg)),
                                    (5, r'$\Delta v_\mathrm{r}$ [km s$^{-1}]$',                                                                  dvr_val),
                                    (6, r'$\Delta v_\mathrm{\theta}$ [km s$^{-1}]$',                                                            dvth_val),
                                    (7, r'$\Delta v_\mathrm{\phi}$ [km s$^{-1}]$',                                                              dvph_val),
                                    (8, r'$\xi$',                                                                          self.mywind.squiggle[:-1,:-1]),
                                    (9, r'$\log |f_\mathrm{rad,r}/\rho|$/[km s$^{-2}]$',                                  np.log10(np.fabs(f_rad_r_val))),
                                    (10, r'$\log |f_\mathrm{rad,\theta}/\rho|$/[km s$^{-2}]$',                           np.log10(np.fabs(f_rad_th_val))),
                                    (11, r'$\log |\Delta \rho/\rho|$',                                 np.log10(np.fabs(drho[:-1,:-1]/rho_tmp[:-1,:-1]))),
                                    (12, r'$\log \rho/$[cgs]',                                 np.log10(self.mywind.mass_density[:-1,:-1].value+1.0e-17)),
                                    (13, r'$M_\mathrm{r}$',                                                                  self.mywind.Mrgrid[:-1,:-1]),
                                    (14, r'$M_\mathrm{\theta}$',                                                         self.mywind.Mthetagrid[:-1,:-1]),
                                    (16, r'Boundary Mask',                                                            self.mywind.boundary_mask[:-1,:-1])
                                   #(16, r'Boundary/Shield Mask',                      self.mywind.in_shield[:-1,:-1]*self.mywind.boundary_mask[:-1,:-1])
                                   #(11, r'$-\gamma g$ [km s$^{-2}]$',                                                                                    rgg_val),
                                   #(13, r'$\log |M_\mathrm{r}|$',      np.log10(    np.fabs(self.mywind.Mrgrid[:-1,:-1]) + 0.1*np.min(np.extract(self.mywind.Mrgrid > 0.0, self.mywind.Mrgrid)))),
                                   #(14, r'$\log |M_\mathrm{\theta}|$', np.log10(np.fabs(self.mywind.Mthetagrid[:-1,:-1]) + 0.1*np.min(np.extract(self.mywind.Mrgrid > 0.0, self.mywind.Mrgrid)))),
                                    ]:
          plt.subplot(4,4,pnum)
          plt.title(title)
          plt.pcolormesh((self.mywind.RRCYL / self.mydisk.rg).value, 
                         (   self.mywind.ZZ / self.mydisk.rg).value, 
                         colarr,  
                         shading='flat')
          plt.xlabel(r'r ($r_g$)')
          plt.ylabel(r'z ($r_g$)')
          plt.plot(self.mydisk.rstar, 
                   self.mydisk.diskheight
                   )
          plt.plot(self.mydisk.rstar, 
                   self.mydisk.zt1
                   )
          for pltr in np.logspace(np.log10(self.mydisk.rstar[0]),np.log10(self.mydisk.rstar[-1]),num=10):
            plt.plot(pltr * np.cos(self.mywind.thetasph_grid), 
                     pltr * np.sin(self.mywind.thetasph_grid), 
                     'k:', 
                     alpha=0.1)
          for pltth in (np.linspace(0.0, np.pi/2.0, num=90) * u.rad):
            plt.plot(self.mydisk.rstar * np.cos(pltth), 
                     self.mydisk.rstar * np.sin(pltth), 
                     'k:', 
                     alpha=0.1
                     )
          plt.scatter((self.mywind.RRCYL[where_density_changed].flatten() / self.mydisk.rg).value, 
                      (self.mywind.ZZ[where_density_changed].flatten() / self.mydisk.rg).value, 
                      c='r', 
                      s=2, 
                      alpha=1)
          plt.scatter((self.mywind.RRCYL[where_velocity_bad].flatten() / self.mydisk.rg).value, 
                      (self.mywind.ZZ[where_velocity_bad].flatten() / self.mydisk.rg).value, 
                      c='m', 
                      s=2, 
                      alpha=1)
          plt.xlim(left = self.mydisk.rstar[0]) #, right = 3.0e+3)
          plt.ylim(bottom = 0.3) #, top = 3.0e+3)
          plt.xscale("log")
          plt.yscale("log")
          plt.colorbar()
          plt.tight_layout()
        tplt = tm.time() * u.s
      dvmax = np.max(np.fabs(np.array([dvr_val,dvth_val,dvph_val]))) * (u.km/u.s)
      pltstr  = f' Simulated time: {self.mywind.tottime:e} \n Time step: {dtime:e} \n'
      pltstr += f' Time since last write/plot: {self.mywind._mcgv_timer(t0):.0f}/{self.mywind._mcgv_timer(tplt):.0f}\n'
      pltstr += f' Run time: {self.mywind._mcgv_timer(t1)} \n'
      pltstr += f' Max change in velocity: {(dvmax/const.c).decompose():.2e} c \n'
      pltstr +=  ' Max '+r'$\Delta\rho/\rho$: '+f'{np.max((drho[self.mywind.boundary_mask]/rho_tmp[self.mywind.boundary_mask]).decompose()):.2e} \n'
      pltstr += f' Number of cells with '+r'$|\Delta\rho|/\rho>0.1$: '+f'{np.sum(where_density_changed)} \n'
      pltstr += f' Number of simulated cells: {np.sum(self.mywind.boundary_mask)}' # \n'
      #pltstr += f' {trf:.2f} {tres:.2f} {tupdate:.2f} {tm.time() * u.s - titeration:.2f}'
      titeration = tm.time() * u.s
      if plotstream:
        plt.annotate(pltstr,(0.52,0.05),xycoords='figure fraction',fontsize=14,color='w',backgroundcolor='b')
        plt.show(block=False)
        plt.pause(0.01)

      sane = True
      for arrstr,arr in [('squiggle',                             self.mywind.squiggle),
                         ('f_rad_r',                            self.mywind._f_rad_r()),
                         ('f_rad_th',                          self.mywind._f_rad_th()),
                          ('P_total',                              self.mywind.P_total),
                         ('rho',   self.mywind.mass_density[self.mywind.boundary_mask]),
                         ('lorentz_factor',                 self.mywind.lorentz_factor),
                         ('rf',              self.mywind.rf[self.mywind.boundary_mask]),
                         ('v_r',         self.mywind.v_rsph[self.mywind.boundary_mask]),
                         ('v_theta', self.mywind.v_thetasph[self.mywind.boundary_mask]),
                         ('v_phi',        self.mywind.v_phi[self.mywind.boundary_mask]),
                         ('dvr',                                                   dvr),
                         ('dvth',                                                 dvth),
                         ('dvph',                                                 dvph)
                         ]:
        arrsanity = self.mywind._sanity_check(arrstr,arr)
        if not arrsanity:
          sane = False
      if not sane:
        input("We've gone insane...")

      if done or (tm.time() * u.s - t0 > 300. * u.s):
        datatab = Table(data=(self.mywind.RR,self.mywind.TT,self.mywind.v_rsph,self.mywind.v_thetasph,self.mywind.v_phi,self.mywind.mass_density,self.mywind.boundary_mask), 
                        names=['r2D','theta2D','vr2D','vtheta2D','vphi2D','rho2D','boundary_mask']
                        )
        table_hdu = fits.BinTableHDU(data=datatab)
        table_hdu.header['SIMTIME'] = (self.mywind.tottime.value,'Simulated time (s)')
        hdul = fits.HDUList([fits.PrimaryHDU(), table_hdu])
        hdul.writeto(self.mywind.windfile, 
                     overwrite=True
                     )
        datatab2 = Table([self.mywind.column_density_table_grid],
                         names = ['column_density_grid'] 
                         )
        datatab2.write(self.mywind.windfile,
                       format = 'fits',
                       append = True)
        t0 = tm.time() * u.s

  #######################################################################################
  def _wnd_force_multiplier(self,
                            which_grid_cells,
                            ntabs = 0
                            ):

    bm          = self.mywind.boundary_mask[ which_grid_cells].flatten()
    rcell       = self.mywind.RRCYL[         which_grid_cells].flatten()
    zcell       = self.mywind.ZZ[            which_grid_cells].flatten()
    num_density = self.mywind.number_density[which_grid_cells].flatten()
    temperature = self.mywind.temperature[   which_grid_cells].flatten()
    thickness   = self.mywind.DRR[           which_grid_cells].flatten()
    in_shield   = self.mywind.in_shield[     which_grid_cells].flatten()

    mrg = np.zeros(rcell.size)
    mrt = np.zeros(rcell.size)
    column_density_arrays = np.zeros((rcell.size, self.myatoms.photo_Z.size)) / u.cm**2

    pool_tuple_input = []
    for cdx in range(rcell.size):
      rcell_vec = np.array([rcell[cdx] / self.mydisk.rg, 
                            0, 
                            zcell[cdx] / self.mydisk.rg
                            ])
      pool_tuple_input.append( ( rcell_vec, num_density[cdx], thickness[cdx], ntabs+1 ) )

    with Pool(self.mypars.nproc) as pool:
        descstr = "\t" * ntabs + "Updating force multipliers"
        with tqdm(total=rcell.size, ncols=0, desc=descstr) as pbar:
          pool_tuple_output = pool.starmap_async(self._wnd_force_multiplier_onecell, 
                                                 pool_tuple_input
                                                 )
          nproc_left = rcell.size
          while not pool_tuple_output.ready():
            if pool_tuple_output._number_left < nproc_left:
              pbar.update(nproc_left-pool_tuple_output._number_left)
              nproc_left = pool_tuple_output._number_left
            tm.sleep(1)

    for cdx in range(rcell.size):
          #mrg_cdx,mrt_cdx,temperature_cdx,ionization_parameter_cdx,column_density_table_cdx = self._wnd_force_multiplier_onecell(*pool_tuple_input[cdx])
          mrg_cdx,mrt_cdx,temperature_cdx,ionization_parameter_cdx,column_density_table_cdx = (pool_tuple_output.get())[cdx]
          mrg[cdx] = mrg_cdx
          mrt[cdx] = mrt_cdx
          temperature[cdx] = temperature_cdx
          try:
            column_density_arrays[cdx,:] = column_density_table_cdx
          except:
            print(f"Unable to equate column_density_arrays[cdx,:] = {column_density_arrays[cdx,:]} ")
            print(f" with column_density_table_cdx = {column_density_table_cdx}")
          in_shield[cdx] = ionization_parameter_cdx >= 60

    self.mywind.Mrgrid[     which_grid_cells] = mrg
    self.mywind.Mthetagrid[ which_grid_cells] = mrt
    self.mywind.temperature[which_grid_cells] = temperature
    self.mywind.in_shield[  which_grid_cells] = in_shield
    self.mywind.column_density_table_grid[which_grid_cells,:] = column_density_arrays

    data = Table(data=[self.mywind.Mrgrid,self.mywind.Mthetagrid], 
                 names=["Mrgrid","Mthetagrid"]
                 )
    data.write(self.mywind.forcemultfile, 
               format="fits", 
               overwrite=True
               )

  #######################################################################################
  def _wnd_force_multiplier_onecell(self,
                                    rcell_vec, 
                                    num_density,
                                    thickness,
                                    ntabs = 0
                                    ):
    fu  = u.erg / (u.s * u.cm * u.cm * u.Hz)
    tol = 1.0e-7
    freq_lo = 3.040e-9 * (const.Ryd).to(u.Hz, equivalencies=u.spectral())
    freq_hi = 1.0e+8   *        u.eV.to(u.Hz, equivalencies=u.spectral()) * u.Hz
    frequency  = np.logspace(np.log10(freq_lo.to(u.Hz).value),
                             np.log10(freq_hi.to(u.Hz).value),
                             num=500
                             ) * u.Hz
    dfreq      = np.power(10.0, 
                          np.linspace(np.log10(freq_lo.to(u.Hz).value),
                                      np.log10(freq_hi.to(u.Hz).value),
                                      num=500
                                      )
                          ) * u.Hz

    gaussleg_y_r,     gaussleg_w_r     = np.polynomial.legendre.leggauss(self.mypars.gaussleg_nr)     # Cylindrical radius (normalized)
    gaussleg_y_theta, gaussleg_w_theta = np.polynomial.legendre.leggauss(self.mypars.gaussleg_ntheta) # Azimuhtal angle (normalized)

    thetadisk = np.pi * (gaussleg_y_theta + 1.) # Azimuthal angle

    # The spherical unit vectors
    theta        = np.arctan2(rcell_vec[0], rcell_vec[2])
    phi          = 0
    rsph_hat     = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi),  np.cos(theta)])
   #phisph_hat   = np.array([               -np.sin)phi),                 np.cos(phi),              0])
    thetasph_hat = np.array([np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)])

    rcell_vec_mag_sq = np.sum(rcell_vec*rcell_vec)
    f_grav_bh_mag  = - const.G.cgs * self.mypars.mbh / (self.mydisk.rg * self.mydisk.rg * rcell_vec_mag_sq) # This is mssing rcell_vec
    f_grav_bh = (f_grav_bh_mag * rcell_vec/np.sqrt(rcell_vec_mag_sq)).decompose(bases=u.cgs.bases)

    f_grav_disk = np.squeeze(self.mydisk.diskgravity(rcell_vec[0] * self.mydisk.rg, rcell_vec[2] * self.mydisk.rg))
    f_grav_disk = np.array([f_grav_disk[0],
                            0,
                            f_grav_disk[1]
                            ]) * (u.cm/u.s**2)

    totfnu = (np.zeros(frequency.shape) + 1.0e-100) * fu
    # Start with the X-ray corona
    R_vec = rcell_vec - self.mycorona.position_vec
    fluxrtnu = np.squeeze(self.mycorona.fnu_lamppost(frequency, 
                                                     rcell_vec[0], 
                                                     rcell_vec[2],
                                                     ntabs = ntabs+1
                                                     )
                          )

    if True:
      shield_cells = self.mywind.shield_poke_sightline(self.mycorona.position_vec,
                                                       R_vec,
                                                       ntabs = ntabs+1
                                                       )
      shield_optical_depth = self.mywind.shield_optical_depth(shield_cells,
                                                              frequency.to(u.eV, 
                                                                           equivalencies=u.spectral()
                                                                           ),
                                                              ntabs = ntabs+1
                                                              )
    else:
      shield_optical_depth = np.zeros(frequency.shape)

    totfnu += fluxrtnu * np.exp(-shield_optical_depth )

    fnurt_dnu = fluxrtnu * dfreq
    nufnurt = np.sum(fnurt_dnu, axis=0) # Integral over frequency

    f_elec_scat_mag  = (const.sigma_T.cgs * nufnurt / (const.c.cgs * const.u.cgs)).to(u.cm/u.s**2)
    f_elec_scat = f_elec_scat_mag * R_vec / np.sqrt(np.sum(R_vec*R_vec))

    # Add in disk annuli
    #for rdx in tqdm(range(self.mypars.gaussleg_nr), desc="\t"*ntabs+f"{rcell_vec} Looping through disk annuli", ncols=0):
    for rdx in range(self.mypars.gaussleg_nr):
      rdisk = self.mydisk.rstar[0] * np.power(self.mydisk.rstar[-1]/self.mydisk.rstar[0], 
                                              (gaussleg_y_r[rdx] + 1)/2
                                              ) # Units rg (log)
      rdisk_vecs = np.array([rdisk * np.cos(thetadisk),
                             rdisk * np.sin(thetadisk),
                             np.interp(rdisk, self.mydisk.rstar, self.mydisk.zt1) * np.ones(thetadisk.shape)
                             ] 
                             )
      R_vecs = rcell_vec[:,None] - rdisk_vecs
      R_hats = R_vecs / np.sqrt(np.sum(R_vecs*R_vecs, axis=0))[None,:]

      fluxrtnu = self._flux_annulus(rdisk,
                                    thetadisk,
                                    None,
                                    rcell_vec,
                                    gaussleg_y_r[rdx],
                                    gaussleg_w_r[rdx],
                                    gaussleg_w_theta,
                                    frequency.to(u.Angstrom, equivalencies=u.spectral()),
                                    True,
                                    ntabs = ntabs+1
                                    )[0]

      for tdx in range(thetadisk.size):
        if False:
          shield_cells = self.mywind.shield_poke_sightline(rdisk_vecs[:,tdx],
                                                           R_vecs[:,tdx],
                                                           ntabs = ntabs+1)
          shield_optical_depth = self.mywind.shield_optical_depth(shield_cells,
                                                                  frequency.to(u.eV, 
                                                                               equivalencies=u.spectral()
                                                                               ),
                                                                  ntabs = ntabs+1
                                                                  )
        else:
          shield_optical_depth = np.zeros(frequency.shape)

        fluxrtnu[:,tdx] *= np.exp(-shield_optical_depth)

      fnurt_dnu = fluxrtnu * dfreq[:,None]
      nufnurt = np.sum(fnurt_dnu, 
                       axis=0
                       ) # Integral over frequency
      totfnu += np.sum(fluxrtnu, 
                       axis=-1
                       ) # Sum over azimuthal angles

      f_elec_scat_mag  = (const.sigma_T.cgs * nufnurt / (const.c.cgs * const.u.cgs)).to(u.cm/u.s**2)
      f_elec_scat_vecs = f_elec_scat_mag[None,:] * R_hats
      f_elec_scat     += np.sum(f_elec_scat_vecs, axis=-1)

    f_elec_scat_mag = np.sqrt(np.sum(f_elec_scat*f_elec_scat))

    csflux = CubicSpline(frequency.to(u.Hz).value,
                         totfnu.to(fu).value
                         )
    csiflux = csflux.integrate((0.1 * ((u.Ry)/const.h).to(u.Hz)).value, 
                               (1000 * ((u.Ry)/const.h).to(u.Hz)).value
                               ) * fu * u.Hz
    lgxi = np.log10((4 * np.pi * csiflux / num_density).to(u.erg * u.cm / u.s).value)

    # Need to run cloudy to get the gas temperature and line emissions
    cloudy_sim = cloudy(0,                 # 0 = emission, 1 = absorption
                        self.mypars,       # instance of readpars
                        self.myatoms,      # instance of atomic class
                        frequency, 
                        totfnu, # ionizing spectrum
                        rhoindex=0.0, 
                        logrhoscale=np.log10(thickness.to(u.cm).value), 
                        logrho0=np.log10(num_density.to(u.cm**-3).value), # density parameters
                        logZ = 0.0,
                        rstar = rcell_vec[0], zstar = rcell_vec[2],
                        verbose = False,
                        ntabs = ntabs+1
                        )
    temperature_cdx = cloudy_sim.temperature
    ionization_parameter_cdx = cloudy_sim.ionization_parameter
    column_density_table_cdx = cloudy_sim.column_density_array

    nit = 0
    if lgxi <= 5:
      vth = np.sqrt(2 * const.k_B * temperature_cdx / const.m_p).to(u.cm/u.s)
      # Iteratively determine the force multiplier:
      # fm -> lSob -> lgt -> fm -|
      # ^------------------------|
      done = False
      # fm is the magnitudes of the force multiplier in each direction being sampled.
      fm = np.ones((self.mypars.gaussleg_nr,
                    self.mypars.gaussleg_ntheta))
      while not done:
        nit += 1
        oldfm = np.copy(fm)

        try:
          ftot = f_grav_bh[None,None,:] + f_grav_disk[None,None,:] + f_elec_scat[None,None,:] * fm[:,:,None]
        except:
          print("\t" * ntabs + f"f_grav_bh = {f_grav_bh}")
          print("\t" * ntabs + f"f_grav_disk = {f_grav_disk}")
          print("\t" * ntabs + f"f_elec_scat = {f_elec_scat}")
          print("\t" * ntabs + f"fm = {fm}")
          input("paused for unit conversion error...")
        lSob = (vth * vth / np.sqrt(np.sum(ftot*ftot, axis=-1))).decompose()
        lgt  = np.log10((const.sigma_T * num_density * lSob).decompose())

        fm = np.power(10.0, self.mywind.fmultgridfunc((lgt,lgxi)))

        if np.any(np.fabs(fm/oldfm - 1) < tol):
          done = True
        else:
          if nit > 500:
            print("\t" * ntabs + f"\t\t\t{rcell_vec} {np.log10((csiflux * 4 * np.pi * np.sum(rcell_vec*rcell_vec) * self.mydisk.rg**2).to(u.erg/u.s).value):.3f} {nit:3d}  {lgxi:e}  {lSob} {lgt}  {fm}")
            input("Pause")

      fm_vec = np.zeros((3,))
      for rdx in range(self.mypars.gaussleg_nr):
        rdisk = self.mydisk.rstar[0] * np.power(self.mydisk.rstar[-1]/self.mydisk.rstar[0], (gaussleg_y_r[rdx] + 1)/2) # Units rg (log)

        rdisk_vecs = np.array([rdisk * np.cos(thetadisk),
                               rdisk * np.sin(thetadisk),
                               np.interp(rdisk, self.mydisk.rstar, self.mydisk.zt1) * np.ones(thetadisk.shape)
                               ] 
                               )

        R_vecs = rcell_vec[:,None] - rdisk_vecs
        R_hats = R_vecs / np.sqrt(np.sum(R_vecs*R_vecs, axis=0))[None,:]

        fm_vec += np.sum(fm[rdx,None,:] * R_hats, axis=-1)

      mrg_cdx = np.sum(fm_vec * rsph_hat)
      mrt_cdx = np.sum(fm_vec * thetasph_hat)

    else:
      mrg_cdx = np.sum(f_elec_scat * rsph_hat)     / f_elec_scat_mag
      mrt_cdx = np.sum(f_elec_scat * thetasph_hat) / f_elec_scat_mag

    return mrg_cdx,mrt_cdx,temperature_cdx,ionization_parameter_cdx,column_density_table_cdx

  #######################################################################################
  def _time_string(self, t0):
    if t0 > 1 * u.hour:
      return f"{t0.to(u.hour):.3f}"
    elif t0 > 1 * u.minute:
      return f"{t0.to(u.minute):.3f}"
    else:
      return f"{t0:.3f}"

  #######################################################################################
  # --- Squiggle is the UV fraction of flux from the disk below the cell point --
  def _wnd_read_squiggle(self,
                         ntabs = 0):
    squigglefile = self.mypars.datapath+f"Sbh{self.mypars.sbh}-MBH{np.log10(self.mypars.mbh / const.M_sun):.2f}-Mdot{(self.mypars.mdot/(const.M_sun/u.year)).decompose()}-alpha{self.mypars.viscosity_alpha}-squiggle_{self.mypars.nr}x{self.mypars.wind_ntheta}.fits"
    t1 = tm.time() * u.s
    if self.mypars.verbose:
      print("\t" * ntabs + f"\t\tLooking for {squigglefile}")
    if os.path.exists(squigglefile):
      print("\t" * ntabs + "\t\t\tReading "+squigglefile)
      data = Table.read(squigglefile, format="fits")
      return np.array(data['squiggle'])
    else:
      print("\t" * ntabs + f"\t\tComputing squiggle (UV fraction from disk)   ({self.mywind._mcgv_timer(t1)})")
      squiggle   = np.zeros(self.mywind.RR.shape)
      frequency  = np.logspace(13,19,num=3000) * u.Hz
      dfreq      = np.power(10.0, np.linspace(13,19,num=3000)) * u.Hz
      in_uv      = (frequency > (3000.0 * u.Angstrom).to(u.Hz, equivalencies = u.spectral())) & (frequency < (0.25 * u.keV).to(u.Hz, equivalencies = u.spectral()))
      for rdx in tqdm(range(self.mypars.nr), desc="\t\tComputing squiggle", ncols=0):
        rlo = np.max([self.mydisk.rstar[rdx] - 0.5*self.mydisk.drstar[rdx],0.0])

        cells_above_rstar = ((self.mywind.RRCYL / self.mydisk.rg).value > rlo) & \
            ((self.mywind.RRCYL / self.mydisk.rg).value < self.mydisk.rstar[rdx] + 0.5*self.mydisk.drstar[rdx]) & \
                ((self.mywind.ZZ/self.mydisk.rg).value > self.mydisk.zt1[rdx])

        self.mydisk.robs = (self.mywind.RRCYL[cells_above_rstar]).value.flatten() / self.mydisk.rg
        self.mydisk.zobs = (   self.mywind.ZZ[cells_above_rstar]).value.flatten() / self.mydisk.rg
        fluxrt = self.mydisk.fnudiskannulus(frequency, rdx) # fluxrt.shape = (frequency.size,self.mydisk.ntheta[rdx],self.mydisk.robs.size)
        dfreq_arr = np.broadcast_to(dfreq, fluxrt.T.shape).T
        squiggle[cells_above_rstar] = np.sum(fluxrt[in_uv,0,:] * dfreq_arr[in_uv,0,:], axis=0) /  (np.sum(fluxrt[:,0,:] * dfreq_arr[:,0,:], axis=0) + 1.0e-50 * (u.erg / (u.s * u.cm * u.cm * u.Hz)))
        if not self._sanity_check('squiggle',squiggle):
          input("Squiggle went insane...")
      data = Table(data=[squiggle], names=["squiggle"])
      data.write(squigglefile, format="fits")
      return squiggle

  #######################################################################################
  def _sanity_check(self,
                    arrstr,
                    arr,
                    ntabs = 0
                    ):
    sanity = True
    if not np.all(np.isfinite(arr)):
      print("\t" * ntabs + f'\t\t\tNaN values in {arrstr} = {arr}')
      sanity = False
    return sanity

  #######################################################################################
  def fitabs(self,
             ntabs = 0):
    plt.ion()

    fu = (u.erg / (u.s * u.cm * u.cm * u.Hz))
    print("\t" * ntabs + f"Calculating spectrum from initial parameters")
    self.printpars()
    # Reset observer location...
    self.reset_observer()
    # Initial Chisq...
    (totflux,unabsflux) = self._calculate_absorbed_flux_gaussleg(self.clouds)
    self.bestfit = totflux/unabsflux
    self._abs_plot(totflux, unabsflux)
    plt.pause(0.01)
    chisq = np.sum(self._abs_chisq(totflux, unabsflux))
    print("\t" * ntabs + f"Initial chisq = {chisq}")

    if self.clouds is not None:
      (rcl, zcl, thetacl, logrhoscale, rhoindex, logrho0, logZ, vcl) = self.grab_cloud_pars(self.clouds)
      xclp,yclp = self._abs_project_clouds(rcl, zcl, thetacl)

    # General -- self.clouds should preserve the best fit, whereas clouds is used for experimentation (adding new clouds, optimizing, comparison with self.clouds)
    list_of_bad_velocities = np.array([]) * (u.km/u.s)
    if self.clouds is not None:
      list_of_bad_velocities = np.copy(vcl)
    res = None
    oldres = None
    done = False
    minfirst = self.mypars.minfirst
    while not done:
      print("\t" * ntabs + "Determining what velocity to put a new component...")
      tot_chisq_spec = np.zeros(self.velocity.size)
      for i in range(self.mypars.anum.size):
        velocity_mask = (self.obsvel[i,-1] > self.velocity) & (self.obsvel[i,0] < self.velocity) & \
          (np.interp(self.velocity, self.obsvel[i,:], self.bigew, left=0, right=0) > np.interp(self.velocity, self.obsvel[i,:], self.bigsew, left=0, right=0))
        for bad_velocity in list_of_bad_velocities:
          velocity_mask = velocity_mask & ((self.velocity < bad_velocity - self.mypars.vres) | (self.velocity > bad_velocity + self.mypars.vres))
        tot_chisq_spec[velocity_mask] += np.interp(self.velocity[velocity_mask], self.obsvel[i,:], self.chisq_spec[i,:])

      potential_bad_vel = np.extract(tot_chisq_spec == np.max(tot_chisq_spec), self.velocity)[0]

      if self.mypars.add_clouds and not minfirst:
        print("\t" * ntabs + f"\t... and adding it at {potential_bad_vel} with badness {np.max(tot_chisq_spec)}")
        try:
          xclp        = np.append(                  xclp, 1.0                                  )
          yclp        = np.append(                  yclp, 1.0                                  )
          rhoindex    = np.append(              rhoindex, 1.0 + 1.5 * np.random.rand(1)        )
          logrho0     = np.append(               logrho0, 2.5 + 1.5 * np.random.rand(1)        )
          logrhoscale = np.append(           logrhoscale, 19.0 - logrho0[-1]                   )
          logZ        = np.append(                  logZ, 0.5                                  )
          vcl         = np.append(vcl.to(u.km/u.s).value, potential_bad_vel.to(u.km/u.s).value ) * (u.km/u.s)

          zcl_prop = (np.random.rand(1) * 0.5 * u.kpc / self.mydisk.rg).decompose()
          while (zcl_prop < 10.0**(1.5 + logrhoscale[-1]) * u.cm / self.mydisk.rg) or (zcl_prop > 10.0**(3.0 + logrhoscale[-1]) * u.cm / self.mydisk.rg):
            zcl_prop = (np.random.rand(1) *  u.kpc / self.mydisk.rg).decompose()
          zcl         = np.append(                   zcl, zcl_prop)

        except:
          logrho0     = 2.5 + 1.5 * np.random.rand(1)
          logrhoscale = 19.0 - logrho0
          xclp        = np.array([ 1.0])
          yclp        = np.array([ 1.0])
          logZ        = np.array([ 0.5])
          vcl         = np.array([potential_bad_vel.to(u.km/u.s).value]) * (u.km/u.s)
          rhoindex    = 1.0 + 1.5 * np.random.rand(1)

          zcl = (np.random.rand(1) * 0.5 * u.kpc / self.mydisk.rg).decompose()
          while (zcl < 10.0**(1.5 + logrhoscale[-1]) * u.cm / self.mydisk.rg) or (zcl > 10.0**(3.0 + logrhoscale[-1]) * u.cm / self.mydisk.rg):
            zcl = (np.random.rand(1) * u.kpc / self.mydisk.rg).decompose()

        clouds = self.makeclouds(xclp, yclp, zcl,
                                 rhoindex, logrhoscale, logrho0, logZ,
                                 vcl
                                 )
        self._abs_write_clouds(clouds)

      else:
        print("\t" * ntabs + f"\t... but not actually adding it...sigh... (would have been {potential_bad_vel})")
        clouds = self.clouds

      self.reset_observer()

      print("\t" * ntabs + "#" * 50)
      print("\t" * ntabs + f"Beginning optimization with {len(clouds)} clouds...")
      self._abscall_t0 = tm.time() * u.s
      oldchisq = chisq
      t0 = tm.time()
      if self.mypars.mcmin:
        mcminimize_clouds, chisq = self._abs_mcminimize(clouds,
                                                        totflux = totflux,
                                                        unabsflux = unabsflux
                                                        )
        x = self._abs_pack(mcminimize_clouds)
      else:
        if not res is None:
          oldres = res.copy()

        xinit = self._abs_pack(clouds)
        xbounds = self._abs_bounds(clouds)
        res = least_squares(self._abs_chisqfunc,
                            xinit,
                            bounds=xbounds,
                            jac="2-point",
                            callback=self._abs_callback,
                            diff_step = 0.001
                            )

      print("\t" * ntabs + f"Optimized (in {tm.time()-t0} seconds)!  Cleaning up...") 
      try:
        if res.success:
          print("\t" * ntabs + "\tSupposedly, the least-squares fit was successful")
          chisq = np.sum(self._abs_chisqfunc(res.x))
          clouds = copy.deepcopy(self._chisqfunc_clouds)
        else:
          print("\t" * ntabs + "\tSomething barfed with the least-squares fit")
          input("Paused")
      except AttributeError:
        if self.mypars.mcmin:
          clouds = mcminimize_clouds
      print("\t" * ntabs + "\t","#"*20)
      print("\t" * ntabs + f"\tChi^2 = {chisq} vs previous Chi^2 = {oldchisq}")
      # Was adding this cloud a statistically significant improvement in the fit?
      # We need to run an F-test
      dof = 0
      for i in range(self.mypars.anum.size):
        velocity_mask = (self.obsvel[i,:] > self.velocity[0]) & (self.obsvel[i,:] < self.velocity[-1])
        dof += np.sum(velocity_mask)
      dof -= 7 * len(clouds)
      old_dof = dof - 7
      F_stat = (oldchisq / old_dof) / (chisq / dof)

      p_value = Ftest.sf(F_stat, dof, old_dof)
      print("\t" * ntabs + f"\tF-stat = {F_stat} --> probablility that the new and old fits are statistically consistent {p_value}")
      if p_value < self.mypars.F_test_prob or not self.mypars.add_clouds or minfirst:
        print("\t" * ntabs + "\t\tKEEPING NEW FIT!")
        minfirst = False
        self.clouds = copy.deepcopy(clouds)
        self._abs_write_clouds(self.clouds)
        self.reset_observer()
        if not self.mypars.mcmin:
          chisq = np.sum(self._abs_chisq(*self._calculate_absorbed_flux_gaussleg(self.clouds)
                                         )
                         )
      else:
        print("\t" * ntabs + "\t\tRESETTING BACK TO OLD FIT AND FLAGGING VELOCITY!")
        list_of_bad_velocities = np.append(list_of_bad_velocities.to(u.km/u.s).value,
                                           potential_bad_vel.to(u.km/u.s).value) * (u.km/u.s)
        clouds = copy.deepcopy(self.clouds)
        (rcl, zcl, thetacl, logrhoscale, rhoindex, logrho0, logZ, vcl) = self.grab_cloud_pars(clouds)
        xclp,yclp = self._abs_project_clouds(rcl, zcl, thetacl)
        chisq = oldchisq
  
      # Do we need another cloud?
      if np.max(self.chisq_spec) < self.mypars.maxchi or not self.mypars.add_clouds:
        done = True
      
      #input("Paused in fitabs for inspection of output. How'd I do?")

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
      logZ        = np.array([cld.logZ                    for cld in clouds])
      vcl         = np.array([cld.vlos.to(u.km/u.s).value for cld in clouds]) * (u.km/u.s)
    else:
      rcl         = np.array([])
      zcl         = np.array([])
      thetacl     = np.array([])
      logrhoscale = np.array([])
      rhoindex    = np.array([])
      logrho0     = np.array([])
      logZ        = np.array([])
      vcl         = np.array([]) * (u.km/u.s)

    return rcl, zcl, thetacl, logrhoscale, rhoindex, logrho0, logZ, vcl

  #######################################################################################
  def makeclouds(self,
                 xclp, yclp, zcl,
                 rhoindex, logrhoscale, logrho0, logZ,
                 vcl,
                 ntabs = 0
                 ):
    fu = (u.erg / (u.s * u.cm * u.cm * u.Hz))

    if xclp.size > 0:
      if self.mypars.verbose:
        print("\t" * ntabs + f"Computing cloud positions (apparent positions given as ({xclp}, {yclp}, {zcl})....")
        print("\t" * (ntabs+2) + f"with observer located at r,z = {self.robs}, {self.zobs}")
      xcl, ycl = self._abs_deproject_clouds(xclp, yclp, zcl)
      rcl     = np.sqrt(xcl * xcl + ycl * ycl)
      thetacl = np.arctan2(ycl, xcl)
      if not isinstance(thetacl, Quantity):
        thetacl *= u.rad
      thetacl = thetacl.to(u.deg)

      if self.mypars.verbose: print("\t"*ntabs +  "Making clouds!")
      clouds = []
      ncl    = rcl.size
      for i in range(ncl):
        t0 = tm.time()
        if self.mypars.verbose:
          print("\t"*ntabs +  "#" * 50)
          print("\t"*(ntabs+1) + f"{i} Making a cloud with the following parameters: xclp = {xclp[i]} yclp = {yclp[i]}")
          print("\t"*(ntabs+2) + f"--> xcl = {xcl[i]:.3f}   ycl = {ycl[i]:.3f} zcl={zcl[i]:.3f}")
          print("\t"*(ntabs+2) + f"              --> r = {rcl[i]:.3f}  theta = {thetacl[i]:.3f}")
          print("\t"*(ntabs+2) + f"rhoindex = {rhoindex[i]}  logrhoscale = {logrhoscale[i]} logrho0 = {logrho0[i]}")
          print("\t"*(ntabs+2) + f"log Z = {logZ[i]}  vcl_los = {vcl[i]}")
        clouds.append(AbsCloud(self.mypars.datapath, self.mydisk, self.mycorona, self.myatoms,
                               rcl[i], zcl[i], thetacl[i], rhoindex=rhoindex[i], logrhoscale=logrhoscale[i], logrho0=logrho0[i], logZ=logZ[i], vcl_los=vcl[i]))
        if self.mypars.verbose:
          print("\t"*(ntabs+1) +  f"{i} Determining ionizing spectrum")
        cloudy_rootname = f"ABS-rho0{logrho0[i]}-index{rhoindex[i]}-scale{logrhoscale[i]}-logZ{logZ[i]}-zcl{zcl[i]}"
        clouds[i].calcionspec(cloudyfileroot = cloudy_rootname, 
                              structure_only = True,
                              ntabs = ntabs+1
                              )
        totflux = self._calculate_absorbed_flux_gaussleg(None,
                                                         robs = rcl[i], 
                                                         thetaobs = thetacl[i], 
                                                         zobs = zcl[i],
                                                         wavelength = clouds[i].ionspecfreq.to(u.Angstrom, equivalencies=u.spectral()),
                                                         lograd = True, 
                                                         noplot = True
                                                         )[0]
        try:
          dum = self._spectrum_scale
        except AttributeError:
          print("\t" * (ntabs+1) +  "Setting spectral scale 'cause I'm stupid and can't integrate...")
          clouds[i].calcionspec(cloudyfileroot = cloudy_rootname, 
                                structure_only = False,
                                ntabs = ntabs+1
                                )
          corona_flux = np.squeeze(self.mycorona.fnu_lamppost(clouds[i].ionspecfreq,
                                                              rcl[i],
                                                              zcl[i]
                                                              )
                                  )
          self._spectrum_scale = np.max(clouds[i].ionspecfreq*clouds[i].ionspecflux) / np.max(clouds[i].ionspecfreq*totflux)
          while np.any(self._spectrum_scale * totflux > clouds[i].ionspecflux):
            self._spectrum_scale *= 0.999
          print("\t" * (ntabs+2) +  f"Scale set to {self._spectrum_scale}")

        corona_flux = np.squeeze(self.mycorona.fnu_lamppost(clouds[i].ionspecfreq,rcl[i],zcl[i]))
        totflux = self._spectrum_scale * totflux + corona_flux
        clouds[i].ionspecflux = np.where(totflux < 1.0e-100 * fu, 1.0e-100 * fu, totflux)

        if self.mypars.verbose:
          print("\t" * (ntabs+1) +  f"{i} Resolving ionization structure {self.mypars.cloudypath}")
        # This is for creating clouds[i].radius, clouds[i].density, clouds[i].temperature, clouds[i].iondensity arrays
        clouds[i].getcloudy(self.mypars.cloudypath, 
                            verbose = self.mypars.verbose, 
                            runcloudy = False, 
                            softenning = self.mypars.softenning)
        (lognuFnu, log_ion_parm) = self._calc_ion_parm(clouds[i])
        sdx = np.argsort(log_ion_parm)
        if self.mypars.verbose:
          print("\t" * (ntabs+2) +  f"Log ionization parameter in range ({np.min(log_ion_parm)}, {np.max(log_ion_parm)})")
        

        # Need to fill the clouds[i].iondensity array
        try:
          dum = self.cheb_log_ion_parm_list
        except AttributeError:
          print("\t" * (ntabs+2) +  "OOPS - FORGOT TO READ CHEBYSHEV TABLES...")
          self._read_cheb_files()

        if self.cheb_log_ion_parm_list.size > 0:
          lnmin = np.min(self.cheb_log_ion_parm_list)
          lnmax = np.max(self.cheb_log_ion_parm_list)
        else:
          print("\t" * (ntabs+2) +  "OOPS - CHEBYSHEV TABLES DON'T EXIST...")
          lnmin = np.min(log_ion_parm) + 10.0
          lnmax = np.max(log_ion_parm) - 10.0

        interp_dlog_ion_parm = np.zeros(log_ion_parm.size)
        try:
          if self.cheb_log_ion_parm_list.size > 0:
            interp_dlog_ion_parm = np.interp(log_ion_parm, 
                                             self.cheb_log_ion_parm_list, 
                                             self.cheb_dlog_ion_parm_list,
                                             left = np.max(self.cheb_dlog_ion_parm_list),
                                             right = np.max(self.cheb_dlog_ion_parm_list)
                                             )
        except:
          pass

        # If we have ionization parameters outside of the Chebyshev range, run Cloudy and expand the range
        target_dlog_ion_parm = 0.1 #np.median(self.cheb_dlog_ion_parm_list) + np.std(self.cheb_dlog_ion_parm_list)
        if np.any(log_ion_parm > lnmax) or np.any(log_ion_parm < lnmin) or np.any(interp_dlog_ion_parm >  target_dlog_ion_parm):
          if np.any(log_ion_parm > lnmax) or np.any(log_ion_parm < lnmin):
            prtstr = f" which is outside the range ({lnmin},{lnmax})   ({np.any(log_ion_parm > lnmax)}, {np.any(log_ion_parm < lnmin)})"
            print("\t" * (ntabs+3) + prtstr)
          if (np.any(log_ion_parm > lnmax) or np.any(log_ion_parm < lnmin)) and np.any(interp_dlog_ion_parm > target_dlog_ion_parm):
            print("\t" * (ntabs+4) + "and")
          if np.any(interp_dlog_ion_parm > target_dlog_ion_parm):
            missing_index = interp_dlog_ion_parm > np.median(self.cheb_dlog_ion_parm_list)
            prtstr = f"which is in a gap ({np.min(log_ion_parm[missing_index])}, {np.max(log_ion_parm[missing_index])}), with "
            prtstr += f"{np.sum(interp_dlog_ion_parm[missing_index] > target_dlog_ion_parm )} bins with > {target_dlog_ion_parm}"
            print("\t" * (ntabs+3) + prtstr)
          clouds[i].getcloudy(self.mypars.cloudypath, 
                              verbose = self.mypars.verbose, 
                              runcloudy = True, 
                              softenning = self.mypars.softenning)
          (lognuFnu, log_ion_parm) = self._calc_ion_parm(clouds[i])
          sdx = np.argsort(log_ion_parm)
          self._add_to_cheb(clouds[i])
          lnmin = np.min(self.cheb_log_ion_parm_list)
          lnmax = np.max(self.cheb_log_ion_parm_list)
        
        xnorm = 2 * (log_ion_parm - lnmin)/(lnmax-lnmin) - 1
        if self.mypars.verbose:
          print("\t" * (ntabs+2) +  f"Chebyshev xnorm in range ({np.min(xnorm)}, {np.max(xnorm)})")
        for iondx in range(self.myatoms.nion):
          ion_parm_mask1 = self.cheb_ionfrac_list[:,iondx] > 0
          if np.any(ion_parm_mask1):
            ion_parm_mask2 = (log_ion_parm < np.max(self.cheb_log_ion_parm_list[ion_parm_mask1])) & (xnorm >= -1.0) & (xnorm <= 1.0)

            logelemabund = np.zeros(log_ion_parm.size)
            if self.myatoms.anum[self.myatoms.idx[iondx]] > 1:
              logelemabund += clouds[i].logZ + self.myatoms.abund[self.myatoms.idx[iondx]] - 12.0
            logionfrac   = np.zeros(log_ion_parm.size)
            if np.any(ion_parm_mask2):
              logionfrac[ion_parm_mask2] += chebyshev.chebval(xnorm[ion_parm_mask2],
                                                              self.cheb_coeff_list[iondx,:self.cheb_degree[iondx]+1]
                                                              )
              logionfrac[~ion_parm_mask2] += chebyshev.chebval(-1.0,
                                                               self.cheb_coeff_list[iondx,:self.cheb_degree[iondx]+1]
                                                               )

            logionfrac = np.where(logionfrac > 0.0, 
                                  0.0, 
                                  logelemabund+logionfrac
                                  )

            clouds[i].iondensity[sdx,iondx] = clouds[i].density[sdx] * np.power(10.0,logionfrac)

          if not np.all(np.isfinite(clouds[i].iondensity)):
            print("\t" * ntabs +  "ERROR IN GETTING ION DENSITIES:")
            print("\t" * ntabs +  f"logionfrac = {logionfrac}")
            print("\t" * ntabs +  f"cloud density = {clouds[i].density}")
            print("\t" * ntabs +  f"cloud ion density = {clouds[i].iondensity}")
            input("Stopped in quasar.makeclouds")

        clouds[i].temperature = 10.0**chebyshev.chebval(xnorm,
                                                        self.cheb_log_temperature_coeff
                                                        ) * u.K
        if self.mypars.verbose:
          print("\t" * (ntabs+2) + f"Temperature in range ({np.min(clouds[i].temperature)}, {np.max(clouds[i].temperature)})")

        if self.mypars.verbose:
          print("\t" * (ntabs+1) + f"{i} Cloud took {tm.time()-t0} s")

    else:
      clouds = None

    return clouds

  #######################################################################################
  def printpars(self,
                ntabs = 0
                ):
    print("\t" * ntabs + "-" * 70)
    print("\t" * ntabs + f"Observer: zqso = {self.mypars.zqso}, inclination = {self.mypars.inclination}, coordinates = {self.skycoord.to_string('hmsdms')}")

    print("\t" * ntabs + f"Black hole: mass {self.mypars.mbh.to(u.Msun):e}, spin {self.mypars.sbh}")
    print("\t" * ntabs + f"            Rg = {self.mydisk.rg:e} = {self.mydisk.rg.to(u.AU)}")

    print("\t" * ntabs + f"Accretion disk: accretion rate {self.mypars.mdot.to(u.Msun/u.yr)}, viscosity parameter {self.mypars.viscosity_alpha}")
    if not self.mywind is None:
      print("\t" * ntabs + f"                Eddington ratio {self.mywind.Eddington_ratio}")
    print("\t" * ntabs + f"                Zone 1/2 boundary (pressure) {self.mydisk.x1} rg  Zone 2/3 (opacity) boundary {self.mydisk.x2} rg")
    print("\t" * ntabs + f"                Inner radius {self.mydisk.rstar[0]} rg, Outer radius {self.mydisk.rstar[-1]} rg, nr = {self.mypars.nr}")

    print("\t" * ntabs + f"Lamp post: location {self.mycorona.lamp_r_cyl}, {self.mycorona.lamp_z}")
    prtstr  = f"           spectrum L_nu_2keV = {self.mycorona.lamp_L_nu_2keV} @ nu_xo_2keV = {self.mycorona.lamp_nu_2keV}, "
    prtstr += f" alpha_x = {self.mycorona.lamp_alpha_x}, E_c = {(const.h * self.mycorona.cutoff_freq).to(u.keV)}"
    print("\t" * ntabs + prtstr)

    if not self.mywind is None:
      print("\t" * ntabs + f"Wind: grid (nr,ntheta) = ({self.mywind.nr},{self.mywind.ntheta})")

    if not self.clouds is None:
      self.print_clouds(self.clouds)
    print("\t" * ntabs + "-" * 70)

    return

  #######################################################################################
  def print_clouds(self,
                   clouds,
                   ntabs = 0):
    print("\t" * ntabs + "\t" * ntabs + "Absorbing clouds:")
    print("\t" * ntabs + "\t" * ntabs + "        num rcl          zcl          thetacl           rhoindex     rhoscale/rg  logrho0      logZ         median T     vlos")
    for i in range(len(clouds)):
      prtstr  = "\t" * ntabs + f"\t {i:2d} {clouds[i].rcl:e} {clouds[i].zcl:e} {clouds[i].thetacl:e} {clouds[i].rhoindex:e} "
      prtstr += f"{(10.0**clouds[i].logrhoscale * u.cm)/self.mydisk.rg:e} {clouds[i].logrho0:e} {clouds[i].logZ:e} {np.median(clouds[i].temperature):e} {clouds[i].vlos:e}"
      print("\t" * ntabs + prtstr)

    return

  #######################################################################################
  def print_diff_clouds(self,
                        clouds1,
                        clouds2,
                        ntabs = 0):
    print("\t" * ntabs + "\t" * ntabs + "Differences (new cloud - old cloud):")
    print("\t" * ntabs + "\t" * ntabs + "        num  drcl          dzcl          dthetacl         drhoindex     drhoscale/rg  dlogrho0      dlogZ        dvlos")
    for i in range(len(clouds1)):
      prtstr  = "\t" * ntabs + f"\t {i:2d} {clouds2[i].rcl-clouds1[i].rcl:13.6e} {clouds2[i].zcl-clouds1[i].zcl:13.6e} {clouds2[i].thetacl-clouds1[i].thetacl:13.6e} "
      prtstr += f"{clouds2[i].rhoindex-clouds1[i].rhoindex:13.6e} {((10.0**clouds2[i].logrhoscale-10.0**clouds1[i].logrhoscale) * u.cm)/self.mydisk.rg:13.6e} "
      prtstr += f"{clouds2[i].logrho0-clouds1[i].logrho0:13.6e} {clouds2[i].logZ-clouds1[i].logZ:13.6e} {clouds2[i].vlos-clouds1[i].vlos:13.6e}"
      print("\t" * ntabs + prtstr)

    return

  #######################################################################################
  def readspec(self,
               nsigma = 1.1,
               ntabs = 0
               ):

    self.mydata = hstqso(self.mypars.datapath,
                         self.skycoord,
                         self.mypars.qfileroot,
                         self.mypars.zqso,
                         redospline=self.mypars.redospline)

    self.mydata.combspec(verbose=True)

    self.mydata.contfit()

    self.obswave, self.obsflux, self.obsivar = self.mydata.bin_spec(self.mydata.totwave, self.mydata.totflux, self.mydata.totivar)

    self.obswave *= u.Angstrom
    self.obsflux *= u.erg/u.s/u.cm**2/u.Hz
    
    self.restwave = self.obswave/(1 + self.mypars.zqso)
    self.obsferr  = 1.0/np.sqrt(self.obsivar)
    self.obsferr *= u.erg/u.s/u.cm**2/u.Hz

    print("\t" * ntabs + "Normalizing continuum")
    self.normobsflux = self.obsflux.value/self.mydata.continuum(self.obswave)
    self.normobsferr = self.obsferr.value/self.mydata.continuum(self.obswave)

    self.lilew, self.lilsew, self.bigew, self.bigsew = self.mydata.ew_spec(self.restwave,
                                                                           self.obsflux,
                                                                           self.obsferr
                                                                           )
    self.bigew_mask = self.bigew > nsigma * self.bigsew

    self.wavelength = None

    print("\t" * ntabs + "Computing velocities")
    self.obsvel = np.empty((self.myatoms.wave.size,
                            self.obswave.size
                            )
                           ) * (u.km/u.s)
    for i in range(self.myatoms.wave.size):
      self.obsvel[i,:] = np.squeeze(calcvel(self.restwave,
                                            self.myatoms.wave[i]
                                            )
                                    )


  #######################################################################################
  def refresh_clouds(self,
                     clouds
                    ):
    rcl, zcl, thetacl, logrhoscale, rhoindex, logrho0, logZ, vcl = self.grab_cloud_pars(clouds)
    xclp, yclp = self._abs_project_clouds(rcl, zcl, thetacl)

    return self.makeclouds(xclp, yclp, zcl,
                           rhoindex, logrhoscale, logrho0, logZ,
                           vcl,
                           ntabs = 1
                           )

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
  #######################################################################################
  #######################################################################################
  #######################################################################################
  #######################################################################################
  def _solve_euler_cylindrical(self,
                               dtime = 10.0 * u.s, 
                               mindt = 1.0 * u.s, 
                               vres = 2000.0 * (u.km/u.s), 
                               minr_rg = 100.0, 
                               plotstream=False, 
                               mupdate = True,
                               ntabs = 0
                               ):
    mdu = u.g / u.cm**3
    t1 = tm.time() * u.s
    print("\t" * ntabs + 'Welcome to your friendly neighborhood relativistic fluid dynamics solver...')

    # --- Iterative Solver ---
    if plotstream:
      plt.ion()
      self.mywind.plotgrid(np.zeros_like(self.mywind.v_R),
                           np.zeros_like(self.mywind.v_Z),
                           np.zeros_like(self.mywind.v_phi),
                           np.zeros_like(self.mywind.mass_density),
                           self.mywind.mass_density,
                           -31.0 * u.s,
                           0.0 * u.s,
                           0.0 * u.s,
                           0.0 * u.s,
                           np.zeros(self.mywind.RR.shape, dtype="bool"),
                           np.zeros(self.mywind.RR.shape, dtype="bool"),
                           plotstream = plotstream
                           )
      plt.pause(1.0)
    t0      = tm.time() * u.s
    dvmax   = 1.0 * (u.km/u.s)
    done    = False
    titeration = tm.time() * u.s
    tplt = tm.time() * u.s
    nit = 0
    print("\t" * ntabs + 'Beginning hydrodynamics simulation...')
    while not done:
      # --- Velocity matters ---
      vmag = np.sqrt(self.mywind.v_R*self.mywind.v_R + self.mywind.v_Z*self.mywind.v_Z + self.mywind.v_phi*self.mywind.v_phi)
      vmag[(vmag/const.c).decompose() >= 1] = 0.99 * const.c
      self.mywind.lorentz_factor = 1. / np.sqrt(1.0 - ((vmag / const.c).decompose())**2)

      # -- Radiation force, pressure and gas pressure ---
      try:
        self.mywind.P_total = self.mywind._P_gas(ntabs = ntabs+1) + self.mywind._P_rad_cylindrical(ntabs = ntabs+1)
      except:
        print(f"P_total = {self.mywind._P_gas(ntabs = ntabs+1)} + {self.mywind._P_rad_cylindrical(ntabs = ntabs+1)}")

      # Thermodynamics
      self.mywind.specific_enthalpy = 1 + self.mywind.adiabatic_index/(self.mywind.adiabatic_index-1.) * (self.mywind.P_total)/(self.mywind.mass_density*const.c**2 + 1.0e-100 * (u.erg / u.cm**3))

      for arrstr,arr in [(  "adiabatic index", self.mywind.adiabatic_index  ),
                         (          "P_total", self.mywind.P_total          ),
                         (     "mass density", self.mywind.mass_density     ),
                         ("specific enthalpy", self.mywind.specific_enthalpy)
                         ]:
        if not self.mywind._sanity_check(arrstr,arr, function="quasar._solve_euler_cylindrical"):
          input("Insane")

      # Residuals
      tres = tm.time() * u.s
      poolinput = [(dtime, 'rho'),
                   (dtime,  'vR'),
                   (dtime,  'vZ'),
                   (dtime, 'vph')
                   ]
      if self.mypars.nproc > 1:
        list_of_euler_linarray_tuples = []
        with Pool(self.mypars.nproc) as pool:
          list_of_euler_linarray_tuples = pool.starmap(self.mywind._EULER_cylindrical, poolinput)

        if len(list_of_euler_linarray_tuples) == 4:
          drho = list_of_euler_linarray_tuples[0]
          dvR  = list_of_euler_linarray_tuples[1]
          dvZ  = list_of_euler_linarray_tuples[2]
          dvph = list_of_euler_linarray_tuples[3]
        else:
          input("EULER barfed")
      else:
        for (dthing, thing) in poolinput:
          dthing = self.mywind._EULER(dtime, thing)

      tres = tm.time() * u.s - tres

      # Is our step small enough to remain physical? If so, update velocity and density fields
      tupdate = tm.time() * u.s
      rho_tmp = np.where(self.mywind.mass_density.to(mdu).value > 1.0e-20, 
                         self.mywind.mass_density.to(mdu).value, 
                         1.0e-20) * mdu
      tau_es = (self.mywind.mass_density * const.sigma_T.cgs * self.mywind.DRR / const.u.cgs).decompose()
      if np.sum(self.mywind.boundary_mask) > 0:
        where_density_changed = self.mywind.boundary_mask & \
                                (np.fabs(drho) / rho_tmp > 0.1) & \
                                (tau_es < 0.7) & \
                                (rho_tmp / const.u.cgs > 1.0e-5 / u.cm**3)

        if (np.max(np.fabs([dvR[self.mywind.boundary_mask].to(u.km/u.s),
                            dvZ[self.mywind.boundary_mask].to(u.km/u.s),
                            dvph[self.mywind.boundary_mask].to(u.km/u.s)])) < vres.to(u.km/u.s).value) and \
            np.all(rho_tmp[self.mywind.boundary_mask]+drho[self.mywind.boundary_mask] > 1.0e-5 * const.u.cgs / u.cm**3 ):
          # Relax towards enforcing continuity (conservative update)
          self.mywind.mass_density[self.mywind.boundary_mask] += drho[self.mywind.boundary_mask]
          self.mywind.v_R[         self.mywind.boundary_mask] += dvR[ self.mywind.boundary_mask]
          self.mywind.v_Z[         self.mywind.boundary_mask] += dvZ[ self.mywind.boundary_mask]
          self.mywind.v_phi[       self.mywind.boundary_mask] += dvph[self.mywind.boundary_mask]
          self.mywind.tottime                                 += dtime

          md_zero = self.mywind.mass_density < 0
          self.mywind.mass_density[md_zero] = 0.0 * mdu

          self.mywind.mass_density = np.where(self.mywind.mass_density < 0 * mdu, 0 * mdu, self.mywind.mass_density)
          self.mywind.number_density[self.mywind.boundary_mask] = self.mywind.mass_density[self.mywind.boundary_mask] / const.u.cgs

          # Update the force multiplier grid
          if mupdate and np.sum(where_density_changed) > 0:
            self._wnd_force_multiplier_cylindrical(where_density_changed,
                                                   ntabs = ntabs+1
                                                   )

          dtime *= np.pi
          if self.mywind.tottime > 1.0e+8 * u.year:
            done = True
        else:
          alldv = np.array([dvR[self.mywind.boundary_mask].to(u.km/u.s),
                            dvZ[self.mywind.boundary_mask].to(u.km/u.s),
                            dvph[self.mywind.boundary_mask].to(u.km/u.s)
                            ])
          dtime /= np.exp(1.0)

        vmag = np.sqrt(self.mywind.v_R*self.mywind.v_R + self.mywind.v_Z*self.mywind.v_Z + self.mywind.v_phi*self.mywind.v_phi)
        dvmag = (self.mywind.v_R * dvR + self.mywind.v_Z * dvZ + self.mywind.v_phi * dvph) / vmag
        vminpred = vmag + mindt * dvmag / dtime
        where_velocity_bad = self.mywind.boundary_mask & (vminpred > const.c)

        # Is the timestep too small? Do we need to mask additional bins?
        if (dtime < mindt) | np.any(drho / rho_tmp < -1.0):
        #  maxdv = np.max(np.fabs([dvR[self.mywind.boundary_mask].to(u.cm/u.s), 
        #                          dvZ[self.mywind.boundary_mask].to(u.cm/u.s), 
        #                          dvph[self.mywind.boundary_mask].to(u.cm/u.s)
        #                          ]
        #                          )
        #                          ) * (u.cm/u.s)
          self.mywind.boundary_mask = (self.mywind.ZZ / self.mydisk.rg > self.mydisk.zt1[:,None]) & \
                                      ( drho / rho_tmp > -1.0 )
          nit = 0
        #  self.mywind.boundary_mask = ((     self.mywind.ZZ - self.mywind.z0[:,None] >  0     ) & \
        #                               (                                np.fabs(dvR) < maxdv  ) & \
        #                               (                                np.fabs(dvZ) < maxdv  ) & \
        #                               (                               np.fabs(dvph) < maxdv  ) & \
        #                               (                                    vminpred < const.c) #& \
        #                               #(rho_tmp * dtime/-(drho + 1.0e-100 * (u.g/u.cm**3)) > mindt  )
        #                               )
        #  if np.sum(self.mywind.boundary_mask) == 0:
        #    print(np.sum(self.mywind.boundary_mask                    ),
        #          np.sum(self.mywind.ZZ - self.mywind.z0[:,None] >  0 ),
        #          np.sum(np.fabs(dvR) < maxdv                         ),
        #          np.sum(np.fabs(dvZ) < maxdv                         ),
        #          np.sum(np.fabs(dvph) < maxdv                        ),
        #          np.sum(vminpred < const.c                           ) #,
        #          #np.sum(rho_tmp * dtime/-drho > mindt                )
        #          )
        #    for arrstr,arr in [("rho_tmp", rho_tmp), 
        #                       (  "dtime",   dtime), 
        #                       (   "drho",    drho)
        #                       ]:
        #      if not self.mywind._sanity_check(arrstr,arr):
        #        input("Paused for insantiy")
        #    #print(f"{rho_tmp * dtime/-(drho + 1.0e-100 * (u.g/u.cm**3))} > {mindt}")
        #    input("Check on boundary mask conditionals...")
        else:
          nit += 1
          if nit >= 5:
            self.mywind.boundary_mask = (self.mywind.ZZ / self.mydisk.rg > self.mydisk.zt1[:,None])
            nit = 0

        tupdate = tm.time() * u.s - tupdate

        # --- Output ---
        self.mywind.plotgrid(dvR, 
                             dvZ, 
                             dvph,
                             drho,
                             rho_tmp,
                             tplt,
                             t0,
                             t1,
                             dtime,
                             where_density_changed,
                             where_velocity_bad,
                             plotstream = plotstream
                             )

        sane = True
        for arrstr,arr in [('g_rad_R',                            self.mywind._g_rad_R()),
                           ('g_rad_Z',                            self.mywind._g_rad_Z()),
                           ('P_total',                               self.mywind.P_total),
                           ('rho',   self.mywind.mass_density[self.mywind.boundary_mask]),
                           ('lorentz_factor',                 self.mywind.lorentz_factor),
                           ('v_R',            self.mywind.v_R[self.mywind.boundary_mask]),
                           ('v_Z',            self.mywind.v_Z[self.mywind.boundary_mask]),
                           ('v_phi',        self.mywind.v_phi[self.mywind.boundary_mask]),
                           ('dvR',                                                   dvR),
                           ('dvZ',                                                   dvZ),
                           ('dvph',                                                 dvph)
                           ]:
          arrsanity = self.mywind._sanity_check(arrstr,arr)
          if not arrsanity:
            sane = False
        if not sane:
          input("We've gone insane...")

        if done or (tm.time() * u.s - t0 > 300. * u.s):
          self.mywind.write_wind()
          t0 = tm.time() * u.s
      else:
        print("I ain't got no cells in the boudnary to evolve!")

  #######################################################################################
  def _wnd_force_multiplier_cylindrical(self,
                            which_grid_cells,
                            ntabs = 0
                            ):
    bm          = self.mywind.boundary_mask[ which_grid_cells].flatten()
    rcell       = self.mywind.RR[            which_grid_cells].flatten()
    zcell       = self.mywind.ZZ[            which_grid_cells].flatten()
    num_density = self.mywind.number_density[which_grid_cells].flatten()
    temperature = self.mywind.temperature[   which_grid_cells].flatten()
    thickness   = self.mywind.DRR[           which_grid_cells].flatten()

    MR_grid = np.zeros(rcell.size)
    MZ_grid = np.zeros(rcell.size)
    column_density_arrays = np.zeros((rcell.size, self.myatoms.photo_Z.size)) / u.cm**2

    pool_tuple_input = []
    for cdx in range(rcell.size):
      rcell_vec = np.array([rcell[cdx] / self.mydisk.rg, 
                            0, 
                            zcell[cdx] / self.mydisk.rg
                            ])
      pool_tuple_input.append( ( rcell_vec, num_density[cdx], thickness[cdx], ntabs+1 ) )

    descstr = "\t" * ntabs + "Updating force multipliers"
    with Pool(self.mypars.nproc) as pool, tqdm(total=rcell.size, ncols=0, desc=descstr) as pbar:
          pool_tuple_output = pool.starmap_async(self._wnd_force_multiplier_onecell_cylindrical, 
                                                 pool_tuple_input
                                                 )
          nproc_left = rcell.size
          while not pool_tuple_output.ready():
            if pool_tuple_output._number_left < nproc_left:
              pbar.update(nproc_left-pool_tuple_output._number_left)
              nproc_left = pool_tuple_output._number_left

    for cdx in range(rcell.size):
          MR_grid_cdx,MZ_grid_cdx,temperature_cdx,ionization_parameter_cdx,column_density_table_cdx = (pool_tuple_output.get())[cdx]
          MR_grid[cdx] = MR_grid_cdx
          MZ_grid[cdx] = MZ_grid_cdx
          temperature[cdx] = temperature_cdx
          try:
            column_density_arrays[cdx,:] = column_density_table_cdx
          except:
            print(f"Unable to equate column_density_arrays[cdx,:] = {column_density_arrays[cdx,:]} ")
            print(f" with column_density_table_cdx = {column_density_table_cdx}")

    self.mywind.MRgrid[                   which_grid_cells  ] = MR_grid
    self.mywind.MZgrid[                   which_grid_cells  ] = MZ_grid
    self.mywind.temperature[              which_grid_cells  ] = temperature
    self.mywind.column_density_table_grid[which_grid_cells,:] = column_density_arrays

    data = Table(data=[self.mywind.MRgrid,self.mywind.MZgrid], 
                 names=["MRgrid","MZgrid"]
                 )
    data.write(self.mywind.forcemultfile, 
               format="fits", 
               overwrite=True
               )

  #######################################################################################
  def _wnd_force_multiplier_onecell_cylindrical(self,
                                    rcell_vec, 
                                    num_density,
                                    thickness,
                                    ntabs = 0
                                    ):
    fu  = u.erg / (u.s * u.cm * u.cm * u.Hz)
    tol = 1.0e-7
    freq_lo = 3.040e-9 * (const.Ryd).to(u.Hz, equivalencies=u.spectral())
    freq_hi = 1.0e+8   *        u.eV.to(u.Hz, equivalencies=u.spectral()) * u.Hz
    frequency  = np.logspace(np.log10(freq_lo.to(u.Hz).value),
                             np.log10(freq_hi.to(u.Hz).value),
                             num=500
                             ) * u.Hz
    dfreq      = np.power(10.0, 
                          np.linspace(np.log10(freq_lo.to(u.Hz).value),
                                      np.log10(freq_hi.to(u.Hz).value),
                                      num=500
                                      )
                          ) * u.Hz

    gaussleg_y_r,     gaussleg_w_r = np.polynomial.legendre.leggauss(self.mypars.gaussleg_nr)     # Cylindrical radius (normalized)
    gaussleg_y_phi, gaussleg_w_phi = np.polynomial.legendre.leggauss(self.mypars.gaussleg_ntheta) # Azimuhtal angle (normalized)

    phidisk = np.pi * (gaussleg_y_phi + 1.) # Azimuthal angle

    # The cylindrical unit vectors
    theta        = np.arctan2(rcell_vec[0], rcell_vec[2])
    phi_cell     = 0
    rcyl_hat     = np.array([ np.cos(phi_cell), np.sin(phi_cell),  0])
    phicyl_hat   = np.array([-np.sin(phi_cell), np.cos(phi_cell),  0])
    z_hat        = np.array([                0,                0,  1])

    which_cell = (rcell_vec[0] * self.mydisk.rg > self.mywind.RR - 0.5 * self.mywind.DRR) & (rcell_vec[0] * self.mydisk.rg < self.mywind.RR + 0.5 * self.mywind.DRR) & \
                 (rcell_vec[2] * self.mydisk.rg > self.mywind.ZZ - 0.5 * self.mywind.DZZ) & (rcell_vec[2] * self.mydisk.rg < self.mywind.ZZ + 0.5 * self.mywind.DZZ)

    try:
      f_grav_bh = np.array([(self.mywind.BH_gR[which_cell])[0].to(u.cm/u.s**2).value,
                            0,
                            (self.mywind.BH_gZ[which_cell])[0].to(u.cm/u.s**2).value
                            ]) * (u.cm/u.s**2)

      f_grav_disk = np.array([(self.mywind.disk_gR[which_cell])[0].to(u.cm/u.s**2).value,
                              0,
                              (self.mywind.disk_gZ[which_cell])[0].to(u.cm/u.s**2).value
                              ]) * (u.cm/u.s**2)
    except IndexError:
      print(f"rcell_vec = {rcell_vec}")
      print(f"n(which_cell) = {np.sum(which_cell)}")
      input("Check on which_cell.. did we find the cell?")

    totfnu = (np.zeros(frequency.shape) + 1.0e-100) * fu
    # Start with the X-ray corona
    R_vec = rcell_vec - self.mycorona.position_vec
    fluxrtnu = np.squeeze(self.mycorona.fnu_lamppost(frequency, 
                                                     rcell_vec[0], 
                                                     rcell_vec[2],
                                                     ntabs = ntabs+1
                                                     )
                          )

    if True:
      shield_cells = self.mywind.shield_poke_sightline(self.mycorona.position_vec,
                                                       R_vec,
                                                       ntabs = ntabs+1
                                                       )
      shield_optical_depth = self.mywind.shield_optical_depth(shield_cells,
                                                              frequency.to(u.eV, 
                                                                           equivalencies=u.spectral()
                                                                           ),
                                                              ntabs = ntabs+1
                                                              )
    else:
      shield_optical_depth = np.zeros(frequency.shape)

    totfnu += fluxrtnu * np.exp(-shield_optical_depth )

    fnurt_dnu = fluxrtnu * dfreq
    nufnurt = np.sum(fnurt_dnu, axis=0) # Integral over frequency

    f_elec_scat_mag  = (const.sigma_T.cgs * nufnurt / (const.c.cgs * const.u.cgs)).to(u.cm/u.s**2)
    f_elec_scat = f_elec_scat_mag * R_vec / np.sqrt(np.sum(R_vec*R_vec))

    # Add in disk annuli
    #for rdx in tqdm(range(self.mypars.gaussleg_nr), desc="\t"*ntabs+f"{rcell_vec} Looping through disk annuli", ncols=0):
    for rdx in range(self.mypars.gaussleg_nr):
      rdisk = self.mydisk.rstar[0] * np.power(self.mydisk.rstar[-1]/self.mydisk.rstar[0], 
                                              (gaussleg_y_r[rdx] + 1)/2
                                              ) # Units rg (log)
      rdisk_vecs = np.array([rdisk * np.cos(phidisk),
                             rdisk * np.sin(phidisk),
                             np.interp(rdisk, self.mydisk.rstar, self.mydisk.zt1) * np.ones(phidisk.shape)
                             ] 
                             )
      R_vecs = rcell_vec[:,None] - rdisk_vecs
      R_hats = R_vecs / np.sqrt(np.sum(R_vecs*R_vecs, axis=0))[None,:]

      fluxrtnu = self._flux_annulus(rdisk,
                                    phidisk,
                                    None,
                                    rcell_vec,
                                    gaussleg_y_r[rdx],
                                    gaussleg_w_r[rdx],
                                    gaussleg_w_phi,
                                    frequency.to(u.Angstrom, equivalencies=u.spectral()),
                                    True,
                                    ntabs = ntabs+1
                                    )[0]

      for tdx in range(phidisk.size):
        if False:
          shield_cells = self.mywind.shield_poke_sightline(rdisk_vecs[:,tdx],
                                                           R_vecs[:,tdx],
                                                           ntabs = ntabs+1)
          shield_optical_depth = self.mywind.shield_optical_depth(shield_cells,
                                                                  frequency.to(u.eV, 
                                                                               equivalencies=u.spectral()
                                                                               ),
                                                                  ntabs = ntabs+1
                                                                  )
        else:
          shield_optical_depth = np.zeros(frequency.shape)

        fluxrtnu[:,tdx] *= np.exp(-shield_optical_depth)

      fnurt_dnu = fluxrtnu * dfreq[:,None]
      nufnurt = np.sum(fnurt_dnu, 
                       axis=0
                       ) # Integral over frequency
      totfnu += np.sum(fluxrtnu, 
                       axis=-1
                       ) # Sum over azimuthal angles

      f_elec_scat_mag  = (const.sigma_T.cgs * nufnurt / (const.c.cgs * const.u.cgs)).to(u.cm/u.s**2)
      f_elec_scat_vecs = f_elec_scat_mag[None,:] * R_hats
      f_elec_scat     += np.sum(f_elec_scat_vecs, axis=-1)

    f_elec_scat_mag = np.sqrt(np.sum(f_elec_scat*f_elec_scat))

    csflux = CubicSpline(frequency.to(u.Hz).value,
                         totfnu.to(fu).value
                         )
    csiflux = csflux.integrate((0.1 * ((u.Ry)/const.h).to(u.Hz)).value, 
                               (1000 * ((u.Ry)/const.h).to(u.Hz)).value
                               ) * fu * u.Hz
    lgxi = np.log10((4 * np.pi * csiflux / (num_density + 1.0e-30 / u.cm**3 )).to(u.erg * u.cm / u.s).value)

    if (lgxi <= 5) and (np.log10(num_density.to(u.cm**-3).value > -5)):
      # Need to run cloudy to get the gas temperature and line emissions
      cloudy_sim = cloudy(0,                 # 0 = emission, 1 = absorption
                          self.mypars,       # instance of readpars
                          self.myatoms,      # instance of atomic class
                          frequency, 
                          totfnu, # ionizing spectrum
                          rhoindex=0.0, 
                          logrhoscale=np.log10(thickness.to(u.cm).value), 
                          logrho0=np.log10(num_density.to(u.cm**-3).value), # density parameters
                          logZ = 0.0,
                          rstar = rcell_vec[0], zstar = rcell_vec[2],
                          verbose = False,
                          ntabs = ntabs+1
                          )
      temperature_cdx = cloudy_sim.temperature
      ionization_parameter_cdx = cloudy_sim.ionization_parameter
      column_density_table_cdx = cloudy_sim.column_density_array

      nit = 0
      vth = np.sqrt(2 * const.k_B * temperature_cdx / const.m_p).to(u.cm/u.s)
      # Iteratively determine the force multiplier:
      # fm -> lSob -> lgt -> fm -|
      # ^------------------------|
      done = False
      # fm is the magnitudes of the force multiplier in each direction being sampled.
      fm = np.ones((self.mypars.gaussleg_nr,
                    self.mypars.gaussleg_ntheta))
      while not done:
        nit += 1
        oldfm = np.copy(fm)

        try:
          ftot = f_grav_bh[None,None,:] + f_grav_disk[None,None,:] + f_elec_scat[None,None,:] * fm[:,:,None]
        except:
          print("\t" * ntabs + f"f_grav_bh = {f_grav_bh}")
          print("\t" * ntabs + f"f_grav_disk = {f_grav_disk}")
          print("\t" * ntabs + f"f_elec_scat = {f_elec_scat}")
          print("\t" * ntabs + f"fm = {fm}")
          input("paused for unit conversion error...")
        lSob = (vth * vth / np.sqrt(np.sum(ftot*ftot, axis=-1))).decompose()
        lgt  = np.log10((const.sigma_T * num_density * lSob).decompose())

        fm = np.power(10.0, self.mywind.fmultgridfunc((lgt,lgxi)))

        if np.any(np.fabs(fm/oldfm - 1) < tol):
          done = True
        else:
          if nit > 500:
            print("\t" * ntabs + f"\t\t\t{rcell_vec} {np.log10((csiflux * 4 * np.pi * np.sum(rcell_vec*rcell_vec) * self.mydisk.rg**2).to(u.erg/u.s).value):.3f} {nit:3d}  {lgxi:e}  {lSob} {lgt}  {fm}")
            input("Pause")

      fm_vec = np.zeros((3,))
      for rdx in range(self.mypars.gaussleg_nr):
        rdisk = self.mydisk.rstar[0] * np.power(self.mydisk.rstar[-1]/self.mydisk.rstar[0], (gaussleg_y_r[rdx] + 1)/2) # Units rg (log)

        rdisk_vecs = np.array([rdisk * np.cos(phidisk),
                               rdisk * np.sin(phidisk),
                               np.interp(rdisk, self.mydisk.rstar, self.mydisk.zt1) * np.ones(phidisk.shape)
                               ] 
                               )

        R_vecs = rcell_vec[:,None] - rdisk_vecs
        R_hats = R_vecs / np.sqrt(np.sum(R_vecs*R_vecs, axis=0))[None,:]

        fm_vec += np.sum(fm[rdx,None,:] * R_hats, axis=-1)

      MRgrid_cdx = np.sum(fm_vec * rcyl_hat)
      MZgrid_cdx = np.sum(fm_vec * z_hat)

    else:
      MRgrid_cdx = np.sum(f_elec_scat * rcyl_hat) / f_elec_scat_mag
      MZgrid_cdx = np.sum(f_elec_scat *    z_hat) / f_elec_scat_mag
      temperature_cdx = np.power((csiflux / const.sigma_sb).decompose(), 0.25)
      ionization_parameter_cdx = 1.0e+10
      column_density_table_cdx = np.zeros(self.myatoms.photo_Z.size) /  u.cm**2

    return MRgrid_cdx,MZgrid_cdx,temperature_cdx,ionization_parameter_cdx,column_density_table_cdx
  

  #######################################################################################
