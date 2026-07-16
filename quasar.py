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
from astropy.modeling.models import BlackBody
from astropy.table           import Table
from multiprocessing         import Pool
from numpy.polynomial        import chebyshev
from scipy.interpolate       import CubicSpline
from scipy.optimize          import least_squares
from scipy.special           import wofz
from scipy.stats             import f         as Ftest
from tqdm                    import tqdm

from AbsCloud                import AbsCloud
from atomic                  import atomic
from corona                  import corona
from doppler                 import calcvel,calcwave
from hstqso                  import hstqso
from mcgv                    import mcgv
from ntdisk                  import ntdisk
from readpars                import readpars

class Quasar:
  def __init__(self,mypars):
    self.mypars = mypars

    ###############################################################################
    print("#" * 50)
    print("Grabbing atomic data")
    self.myatoms = atomic(self.mypars.datapath,900 * u.Angstrom,3000 * u.Angstrom,
                          minlox = self.mypars.minlox
                          )

    nvel = np.int16((self.mypars.vhi-self.mypars.vlo)/self.mypars.vres)
    self.velocity   = np.linspace(start = self.mypars.vlo,
                                  stop  = self.mypars.vhi,
                                  num   = nvel
                                  )

    fu = (u.erg / (u.s * u.cm * u.cm * u.Hz))
    ###############################################################################
    print("Setting observer")
    self.skycoord    = SkyCoord(ra  = self.mypars.raqso,
                                dec = self.mypars.decqso)

    ###############################################################################
    print(f"Initializing disk with {self.mypars.nr} annuli from {self.mypars.rlo} to {self.mypars.rhi} rg")
    self.mydisk = ntdisk(self.mypars.sbh, self.mypars.mbh,
                         self.mypars.mdot, self.mypars.alpha,
                         self.mypars.inclination,
                         self.mypars.nr, self.mypars.rlo, self.mypars.rhi,
                         self.mypars.datapath,
                         dtheta_fac = self.mypars.dtheta_fac)
    comove_dist = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7).comoving_distance(self.mypars.zqso)
    self.robs =  (comove_dist * np.sin(self.mypars.inclination) / self.mydisk.rg).decompose()
    self.zobs = self.robs / np.tan(self.mypars.inclination)
    self.reset_observer()
    
    print("\tCalculating disk")
    self.mydisk.makedisk()
    print("\tDetermining disk photosphere")
    self.mydisk.photosphere()

    ###############################################################################
    print("Initializing corona")
    self.mycorona = corona(self.mydisk)
    self.mycorona.activate_lamppost()

    ###############################################################################
    if mypars.calcwind:
      print("Initializing wind...")
      self.mywind = mcgv(self.mydisk, self.mycorona, self.myatoms, 90, self.mypars.datapath)
      forcemultfile  = self.mypars.datapath+f"Sbh{self.mydisk.sbh}-MBH{np.log10(self.mydisk.mbh / const.M_sun):.2f}"
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
    if mypars.calcabscl:
      print("Initializing absorbing clouds")
      self.reset_observer()
      self.cloud_filename = self.mypars.datapath + self.mypars.abscloudfile
      print(f"\tLooking for {self.cloud_filename}")
      if os.path.exists(self.cloud_filename):
        print(f"\t\tFound it!")
        self.clouds = self._abs_read_clouds(mypars)
      else:
        self.clouds = None
      self.bestfit = None
    else:
      self.clouds = None

  #######################################################################################
  def _abs_all_optical_depth(self,
                             clouds,           # List of AbsClouds
                             rdisk_vec, R_vec, # 2D nd.arrays with shapes (3,thetadisk.size)
                             wavelength        # 1d nd.array
                             ):

    if clouds is not None:
      t0 = tm.time()

      rcl, zcl, thetacl, logrhoscale, rhoindex, logrho0, logZ, vcl  = self.grab_cloud_pars(clouds)
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
                                                      wavelength        # 1D nd.array
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
      abs_lower_bounds.append(                                         cld.logZ-2.0 ) # log Z
      abs_lower_bounds.append(   (cld.vlos - 100.0 * (u.km/u.s)).to(u.km/u.s).value ) # vlos

      abs_upper_bounds.append(                             self.mydisk.rstar[-1] ) # xcl
      abs_upper_bounds.append(                             self.mydisk.rstar[-1] ) # ycl
      abs_upper_bounds.append(        (1.0 * u.kpc / self.mydisk.rg).decompose() ) # zcl
      abs_upper_bounds.append(                                               5.0 ) # rhoindex
      abs_upper_bounds.append(                             cld.logrhoscale + 2.5 ) # logrhoscale
      abs_upper_bounds.append(                                 cld.logrho0 + 4.0 ) # logrho0
      abs_upper_bounds.append(                                      cld.logZ+3.0 ) # log Z
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

    (totflux, unabsflux) = self._calculate_absorbed_flux_gaussleg(clouds, self.mypars, lograd = True)
    chisq = np.sum(self._abs_chisq(totflux, unabsflux))
    self.bestfit = totflux/unabsflux

    print('\n')
    print('\t\t' + '#'*20)
    self.print_clouds(clouds, ntabs=2)
    try:
      print(f"\t\tChi^2 = {chisq}  ({tm.time() * u.s - self._abscall_t0:e} since last callback)")
      self._abs_plot(totflux, unabsflux, vcl = self.grab_cloud_pars(clouds)[-1])
    except NameError:
      print(f"\t\tChi^2 = {chisq}")
      self._abs_plot(totflux, unabsflux, vcl = self.grab_cloud_pars(clouds)[-1])

    self._abscall_t0 = tm.time() * u.s

    # Write out clouds to a file so that we can pick up where we left off...
    self._abs_write_clouds(clouds)
    print('\t\t' + '#'*20)
    print('\n')
 
  #######################################################################################
  # Compute the chisq, summing across all max(f-lambda) species that are covered in the velocity range specified
  def _abs_chisq(self,
                 totflux, unabsflux
                 ):
    try:
      chisq_spec = np.zeros((self.mypars.anum.size, self.obswave.size))
      cosflux    = self.mydata._lsf_convolve(self.wavelength, totflux.value)/unabsflux.value
      for i in range(self.mypars.anum.size):
        myatoms_index = self.myatoms.getspecies(self.mypars.anum[i],
                                                self.mypars.ion[i]
                                                )[self.mypars.trandx[i]]
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
      chisq_spec = np.zeros((self.mypars.anum.size, self.wavelength.size))

    self.chisq_spec = chisq_spec
    return chisq_spec

  #######################################################################################
  # Wrapper for minimization purposes - the function that is to be minimized
  def _abs_chisqfunc(self,
                     x
                     ):
    self._chisqfunc_clouds = self._abs_unpack(x)

    # Observer coordinates (reset here for sanity)
    self.reset_observer()

    (totflux, unabsflux) = self._calculate_absorbed_flux_gaussleg(self._chisqfunc_clouds, noplot=True)
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
                      totflux = None,
                      unabsflux = None
                      ):

    ncl = len(clouds)
    self.reset_observer()
    if totflux is None:
      (totflux, unabsflux) = self._calculate_absorbed_flux_gaussleg(clouds,
                                                                    )
    chisq = np.sum(self._abs_chisq(totflux,
                                   unabsflux
                                   )
                   )
    self.bestfit = totflux/unabsflux
    x = self._abs_pack(clouds)

    better_clouds = copy.deepcopy(clouds)
    niter = self.mypars.maxiter * x.size
    rng = np.random.default_rng()
    good_direction = False
    check_negative_direction = False
    step_size = self.mypars.initstep
    while niter > 0 and step_size > self.mypars.minstep:
      self.reset_observer()
      (rcl, zcl, thetacl, logrhoscale, rhoindex, logrho0, logZ, vcl) = self.grab_cloud_pars(better_clouds)
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
      print(f"\tProposing changes to clouds [step size = {step_size/self.mypars.minstep} x minstep (={self.mypars.minstep}), good_direction = {good_direction}, check_negative_direction = {check_negative_direction}]...")
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
        print(f"\t\tResetting heights for clouds {np.arange(ncl)[zmask]}: {zcl[zmask]} < {zmin[zmask]}")
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
                                         vcl
                                         )
      self.reset_observer()

      print("\tTrying the following changes:")
      self.print_diff_clouds(better_clouds,
                             potential_clouds,
                             ntabs=2)

      # Determine chisq
      (newtotflux, newunabsflux) = self._calculate_absorbed_flux_gaussleg(potential_clouds)
      newchisq = np.sum(self._abs_chisq(newtotflux,
                                        newunabsflux
                                        )
                        )

      # If the chisq is better, then keep it and continue.
      # If it is not better, increment the iteration number and try again until the iteration number is bunk
      if newchisq < chisq:
        print(f"\n\tIMPROVED FIT! iterations remaining: {niter}    chisq = {chisq} - {chisq - newchisq}\n")
        better_clouds = copy.deepcopy(potential_clouds)
        chisq         = np.copy(newchisq)
        totflux       = np.copy(newtotflux)
        unabsflux     = np.copy(newunabsflux)
        self.bestfit  = totflux/unabsflux

        if newchisq < chisq-1.0:
          niter = self.mypars.maxiter * x.size

        good_direction = True
        step_size *= 1.0 + self.mypars.dstep

        self._abs_write_clouds(better_clouds)
        self._abscall_t0 = tm.time() * u.s
      else:
        good_direction = False
        step_size *= 1.0 - self.mypars.dstep
        if not check_negative_direction:
          niter -= 1

        print(f"\n\tKeeping old fit! iterations remaining: {niter}    chisq = {chisq} + {newchisq-chisq}")

      self.print_clouds(better_clouds, ntabs=2)

    return better_clouds, chisq

  #######################################################################################
  # Absorption optical depth from sightlines (at impact_parameter) piercing one cloud
  def _abs_optical_depth(self,
                         impact_parameter, # 1D nd.array
                         cloud,            # AbsCloud
                         wavelength        # 1D nd.array
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
                           self.myatoms.wave
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
        (optical_depth_species, dvel_mask, optical_depth_bv_fill_time, optical_depth_Ntau0_fill_time, optical_depth_voigt_calc_time) = self._abs_optical_depth_single_species(*tuple_input)
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
                                        iondensity
                                        ):
    tau_const = (2 * np.sqrt(np.pi) * const.e.esu * const.e.esu / (const.m_e * const.c) ).decompose()

    t0 = tm.time()
    bvalue = np.sqrt(2 * const.k_B * temperature.to(u.K) / self.myatoms.amass[myatoms_index]) # shape (cloud.radius.size,)

    dvel = np.outer((velocity[:,myatoms_index] - vlos), # shape (velocity[:,myatoms_index].size,)
                     1 / bvalue                         # shape (cloud.radius.size,)
                    ).decompose()                       # shape (velocity[:,myatoms_index].size, cloud.radius.size)
    #optical_depth_bv_fill_time = tm.time() - t0

    dvel_mask = np.abs(velocity[:,myatoms_index] - vlos)/np.average(bvalue) < 300.0
    if np.any(dvel_mask):
      which_ion_index = (self.myatoms.ions == 100*self.myatoms.anum[myatoms_index]+self.myatoms.ion[myatoms_index])

      t0 = tm.time()
      column_density = dxarray * np.squeeze(iondensity[:,which_ion_index])[None,:]                         # shape (impact_parameter.size, cloud.radius.size)
      tau0 = (tau_const * self.myatoms.flam[myatoms_index] * column_density / bvalue[None,:] ).decompose() # shape (impact_parameter.size, cloud.radius.size)
      optical_depth_Ntau0_fill_time = tm.time() - t0

      t0 = tm.time()
      if self.myatoms.wave[myatoms_index] > 1215.66 * u.Angstrom and self.myatoms.wave[myatoms_index] < 1215.68 * u.Angstrom:  # self.myatoms.gamma[myatoms_index].value > 0:
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
                clouds
                ):
    x = np.array([])
    for cld in clouds:
      xclp, yclp = self._abs_project_clouds(cld.rcl, cld.zcl, cld.thetacl)
      x = np.append(x, [xclp, yclp, cld.zcl, cld.rhoindex, cld.logrhoscale, cld.logrho0, cld.logZ, cld.vlos.value])

    return x

  #######################################################################################
  # Make a plot of the normalized absorption spectra (observed and predicted)
  def _abs_plot(self,
                totflux, unabsflux,
                vcl = np.array([])
                ):
    plt.ion()
    plt.clf()

    plt.plot(self.velocity, np.zeros(self.velocity.size), "k--")
    for tdx in range(self.mypars.anum.size):
      myatoms_index = self.myatoms.getspecies(self.mypars.anum[tdx],
                                              self.mypars.ion[tdx]
                                              )[self.mypars.trandx[tdx]]
      plt.plot(self.velocity, np.ones(self.velocity.size) + tdx, "k--")
      #----------------------------------------------------------
      try:
        plt.step(self.obsvel[myatoms_index,:],
                 self.normobsflux + tdx,
                 c=self.plot_code[tdx],
                 label=f"{self.myatoms.specstr[myatoms_index]} "+r"$\lambda$"+f"{self.myatoms.wave[myatoms_index]:.3f}")
      except:
        pass
      #----------------------------------------------------------
      try:
        for clddx in range(self.cld_totflux.shape[1]):
          plt.plot(self.species_velocity[:,myatoms_index],
                   self.cld_totflux[:,clddx]/unabsflux + tdx,
                   self.plot_code[tdx]+":"
                  )
      except:
        pass
      #----------------------------------------------------------
      try:
        for clddx in range(self.cld_totflux.shape[1]):
          plt.plot(self.species_velocity[:,myatoms_index],
                   self.cld_totflux[:,clddx]/unabsflux + tdx,
                   self.plot_code[tdx]+":"
                  )
      except IndexError:
        print("Oops... no clouds yet")
      #----------------------------------------------------------
      try:
        plt.plot(self.species_velocity[:,myatoms_index],
                 totflux/unabsflux + tdx,
                 self.plot_code[tdx])
      except:
        pass
      #----------------------------------------------------------
      try:
        if self.bestfit is not None:
          plt.plot(self.species_velocity[:,myatoms_index],
                   self.bestfit + tdx,
                   f"{self.plot_code[tdx]}--")
      except:
        pass

    if vcl.size > 0:
      for v in vcl:
        v_nounits = v.to(u.km/u.s).value
        plt.plot(np.array([v_nounits,v_nounits]),
                 np.array([-0.2, 1.2*self.mypars.anum.size]),
                 "k--")

    plt.legend()

    title_str  = self.mypars.qname + f" zqso = {self.mypars.zqso} "
    try:
      chisq_spec = self._abs_chisq(totflux, unabsflux)
      chisq      = np.sum(chisq_spec)
      title_str += r"$\chi^2 = $" + f"{chisq:.3f}"
    except:
      pass

    plt.title(title_str)
    plt.xlim([self.velocity[0].to(u.km/u.s).value, self.velocity[-1].to(u.km/u.s).value])
    plt.ylim([-0.2, 1.2*self.mypars.anum.size])
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
  def _abs_read_clouds(self,
                       mypars):
    print(f"\tReading clouds from {self.cloud_filename}")
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
                             ntabs = 1)

    return clouds

  #######################################################################################
  # Wrapper to take the parameters fed into/from scipy.optimize.minimize and unpack it into a cloud class
  def _abs_unpack(self,
                  x
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

      clouds = self.makeclouds(xclp, yclp, zcl, rhoindex, logrhoscale, logrho0, logZ, vcl)
      if self.mypars.verbose:
        self.print_clouds(clouds, ntabs=1)
    else:
      clouds = None

    return clouds

  #######################################################################################
  def _abs_write_clouds(self,
                   clouds
                   ):
    (rcl, zcl, thetacl, logrhoscale, rhoindex, logrho0, logZ, vcl_los) = self.grab_cloud_pars(clouds)
    xclp, yclp = self._abs_project_clouds(rcl, zcl, thetacl)
    
    print(f"\tWriting clouds to {self.cloud_filename}")
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
                   cloud
                   ):
    print("\t\tExpanding Chebyshev fits with new Cloudy run...")
    (lognuFnu, log_ion_parm) = self._calc_ion_parm(cloud)

    self.cheb_log_ion_parm_list = np.append(self.cheb_log_ion_parm_list, log_ion_parm)

    # Iterate through ions and the ion fraction...
    ionfrac = np.zeros(cloud.iondensity.shape)
    for i in range(self.myatoms.nion):
      logZdum = cloud.logZ
      if self.myatoms.anum[self.myatoms.idx[i]] < 3:
        logZdum = 0.0
      logelemabund = self.myatoms.abund[self.myatoms.idx[i]]-12+logZdum
      ionfrac[:,i] = cloud.iondensity[:,i] / (cloud.density * np.power(10.0, logelemabund))

    if self.cheb_ionfrac_list.size > 0:
      self.cheb_ionfrac_list = np.append(self.cheb_ionfrac_list, ionfrac, axis=0)
      self.cheb_temperature_list = np.append(self.cheb_temperature_list, cloud.temperature.to(u.K).value)
    else:
      self.cheb_ionfrac_list = np.copy(ionfrac)
      self.cheb_degree = np.ones(self.myatoms.idx.size, dtype=np.int16)
      self.cheb_coeff_list = np.zeros((self.myatoms.idx.size,1))

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


    done = False
    oldchisq = 9.99e+99
    olddeg = self.cheb_log_temperature_degree
    while not done:
      coeff,res = chebyshev.chebfit(xnorm,np.log10(self.cheb_temperature_list), self.cheb_log_temperature_degree, full=True)
      if res[0].size > 0:
        fstat = (oldchisq / olddeg )  / (np.squeeze(res[0]) / self.cheb_log_temperature_degree)
        p_value = Ftest.sf(fstat, olddeg, self.cheb_log_temperature_degree)
        if p_value < 0.48:
          self.cheb_log_temperature_degree += 1
          done = False
        else:
          self.cheb_log_temperature_degree = np.max(np.array([1, self.cheb_log_temperature_degree-1]))
          coeff,res = chebyshev.chebfit(xnorm,np.log10(self.cheb_temperature_list), self.cheb_log_temperature_degree, full=True)
          done = True
      else:
        self.cheb_log_temperature_degree = np.max(np.array([1, self.cheb_log_temperature_degree-1]))
        coeff,res = chebyshev.chebfit(xnorm,np.log10(self.cheb_temperature_list), self.cheb_log_temperature_degree, full=True)
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


    chebfile  = self.mypars.datapath+f"/Cloudy_runs/Sbh{self.mydisk.sbh}-MBH{np.log10(self.mydisk.mbh / const.M_sun):.2f}"
    chebfile += f"-Mdot{(self.mydisk.mdot/(const.M_sun/u.year)).decompose()}-alpha{self.mydisk.alpha}_ionfracs_lox{self.myatoms.minlox}.fits"
    datatab = Table(data=[self.cheb_log_ion_parm_list, self.cheb_ionfrac_list, self.cheb_temperature_list], names=["log_ion_parm_list", "ionfrac_list", "temperature_list"])

    datatab2 = Table(data=[self.cheb_degree,self.cheb_coeff_list], names=["DEGREE","COEFFS"])

    try:
      datatab3 = Table(data=[self.cheb_log_temperature_coeff], names=["COEFFS"])
    except TypeError:
      print(self.cheb_log_temperature_degree)
      print(self.cheb_log_temperature_coeff)
      input("Making this table barfed...")

    datatab.write( chebfile, format="fits", overwrite=True)
    datatab2.write(chebfile, format="fits", append=True)
    datatab3.write(chebfile, format="fits", append=True)

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
                                        robs       = None,
                                        thetaobs   = None,
                                        zobs       = None,
                                        wavelength = None,
                                        debug      = False,
                                        lograd     = False,
                                        noplot     = False
                                        ):
    fu = (u.erg / (u.s * u.cm * u.cm * u.Hz))
    if self.mypars.verbose:
      print("\tSetting up integration for spectral synthesis")
    t0 = tm.time()
    if wavelength is None:
      wavelength = self.wavelength
    totflux     = np.zeros(wavelength.shape) * fu
    unabsflux   = np.zeros(wavelength.shape) * fu
    if clouds is not None:
      self.cld_totflux = np.zeros((wavelength.size, len(clouds))) * fu

    if robs is None:
      robs     = self.robs
    if thetaobs is None:
      thetaobs = 0.0
    if zobs is None:
      zobs     = self.zobs

    self.reset_observer(robs     =    robs,
                        thetaobs = thetaobs,
                        zobs      =    zobs
                        )
    try:
      robs_vec = np.array([robs * np.cos(thetaobs),
                           robs * np.sin(thetaobs),
                           zobs
                           ])
    except TypeError:
      print(f"Tried to make robs_vec using robs = {robs}   thetaobs = {thetaobs}   zobs = {zobs}")
      input("paused")

    if debug:
      print(f"robs_vec = {robs_vec}")

    done = False
    scale = 1
    while not done:
      done = True
      # Set up Gauss-Legendre grid for integration
      if self.mypars.verbose:
        print(f"\t\tDetermining Gauss-Legendre positions and weights for a {scale * self.mypars.gaussleg_nr} x {scale * self.mypars.gaussleg_ntheta} grid ({tm.time()-t0})")
      gaussleg_y_r,     gaussleg_w_r     = np.polynomial.legendre.leggauss(scale * self.mypars.gaussleg_nr)     # Cylindrical radius (normalized)
      gaussleg_y_theta, gaussleg_w_theta = np.polynomial.legendre.leggauss(scale * self.mypars.gaussleg_ntheta) # Azimuhtal angle (normalized)

      plt.ion()

      separate_theta = False
      pool_tuple_input = []
      for rdx in range(self.mypars.gaussleg_nr):
        if lograd:
          rdisk = self.mydisk.rstar[0] * np.power(self.mydisk.rstar[-1]/self.mydisk.rstar[0], (gaussleg_y_r[rdx] + 1)/2) # Units rg (log)
        else:
          rdisk = self.mydisk.rstar[0] + (self.mydisk.rstar[-1] - self.mydisk.rstar[0]) * (gaussleg_y_r[rdx] + 1) / 2 # Units rg (linear)
        
        if separate_theta:
          for tdx in range(self.mypars.gaussleg_ntheta):
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
        print(f"\t\tIntegrating across disk with at most {self.mypars.nproc} processors...({tm.time()-t0})")

      with Pool(self.mypars.nproc) as pool:
        output_flux_pool_tuple = pool.starmap_async(self._flux_annulus, pool_tuple_input)
        output_flux_pool_tuple.wait()

      if self.mypars.verbose:
        print(f"\t\tIntegration complete... ({tm.time()-t0})")

      min_impact_parameters_all_sightlines = None
      plot_nit = 0
      for (fluxrtnu, cld_optical_depth, min_impact_parameter) in tqdm(output_flux_pool_tuple.get(), desc=f"\t\t\tAssembling {dumstr}", ncols=0):
        tmp_flux_sum = np.sum(fluxrtnu, axis=-1)  # Sum over sightlines
        unabsflux += tmp_flux_sum
        
        if clouds is not None:
          for clddx in range(len(clouds)):
            self.cld_totflux[:,clddx] += np.sum(fluxrtnu * np.exp(-cld_optical_depth[clddx,:,:]), axis=-1)
          totflux += np.sum(fluxrtnu * np.exp(-np.sum(cld_optical_depth, axis=0)), axis=-1)
        else:
          totflux += tmp_flux_sum # Sum over sightlines
        
        if min_impact_parameter is not None:
          try:
            min_impact_parameters_all_sightlines = np.append(min_impact_parameters_all_sightlines, 
                                                             [min_impact_parameter], 
                                                             axis=0)
          except ValueError:
            min_impact_parameters_all_sightlines = np.array([min_impact_parameter])
        
        if not noplot and plot_nit % 10 < 2:
          self._abs_plot(totflux, unabsflux,
                         vcl = self.grab_cloud_pars(clouds)[-1] # = vcl
                         )
          plt.pause(0.001)
          plot_nit += 1
 

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
          if np.any(scale*np.array([self.mypars.gaussleg_nr, self.mypars.gaussleg_ntheta]) > 1000):
            print(f"\t\t\tThis is would be too expensive - bailing")
            done = True

    if clouds is not None:
      self._abs_plot(totflux, unabsflux,
                     vcl = self.grab_cloud_pars(clouds)[-1] # = vcl
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
                    lograd            # boolean for quadrature method
                    ):
    fu = (u.erg / (u.s * u.cm * u.cm * u.Hz))

    t0 = tm.time()

    zt1cs = CubicSpline(self.mydisk.rstar, self.mydisk.zt1)
    Tt1cs = CubicSpline(self.mydisk.rstar, self.mydisk.tempt1.to(u.K).value)

    frequency = wavelength.to(u.Hz, equivalencies=u.spectral())

    rdisk_vec = np.array([rdisk * np.cos(thetadisk),
                          rdisk * np.sin(thetadisk),
                          np.broadcast_to(zt1cs(rdisk), (thetadisk.size,))
                          ])

    dzt1dr = zt1cs(rdisk, nu=1)
    Tt1    = Tt1cs(rdisk) * u.K

    gradZ_vec = np.array([-dzt1dr * np.cos(thetadisk),
                          -dzt1dr * np.sin(thetadisk),
                          np.ones(thetadisk.size)
                          ])
    gradZmag = np.sqrt(np.sum(gradZ_vec*gradZ_vec, axis=0))

    R_vec = np.broadcast_to(robs_vec, rdisk_vec.T.shape).T - rdisk_vec
    Rmag = np.sqrt(np.sum(R_vec*R_vec, axis=0))
    if(np.any(Rmag == 0)):
      input("Rmag is zero")

    #cosbeta = np.array([np.dot(R_vec[:,i], gradz_vec[:,i]) for i in range(thetadisk.size)]) / (gradzmag * Rmag)
    cosbeta = np.sum(R_vec * gradZ_vec, axis=0) / (Rmag * gradZmag)
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


    tmp_fluxrtnu=  np.pi * (Bnu[:,cosbeta_mask] * u.sr) * gaussleg_w_r * gaussleg_w_theta[cosbeta_mask] * (doppler_beam_fac[cosbeta_mask]**3) * cosbeta[cosbeta_mask] / Rmag[cosbeta_mask]**2
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
                                                                          wavelength        # 1d nd.array
                                                                          ) # shape = (wavelength.size, thetadisk.size)
    try:
      self.t_flux_annulus_post_tau += tm.time() - t0
    except:
      self.t_flux_annulus_post_tau = tm.time() - t0

    return fluxrtnu, cld_optical_depth, min_impact_parameter

  #######################################################################################
  def _read_cheb_files(self):
    chebfile  = self.mypars.datapath+f"/Cloudy_runs/Sbh{self.mydisk.sbh}-MBH{np.log10(self.mydisk.mbh / const.M_sun):.2f}"
    chebfile += f"-Mdot{(self.mydisk.mdot/(const.M_sun/u.year)).decompose()}-alpha{self.mydisk.alpha}_ionfracs_lox{self.myatoms.minlox}.fits"
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
  def fitabs(self):
    plt.ion()

    fu = (u.erg / (u.s * u.cm * u.cm * u.Hz))
    print(f"Calculating spectrum from initial parameters")
    self.printpars()
    # Reset observer location...
    self.reset_observer()
    # Initial Chisq...
    (totflux,unabsflux) = self._calculate_absorbed_flux_gaussleg(self.clouds)
    self.bestfit = totflux/unabsflux
    self._abs_plot(totflux, unabsflux)
    plt.pause(0.01)
    chisq = np.sum(self._abs_chisq(totflux, unabsflux))
    print(f"Initial chisq = {chisq}")

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
      print("Determining what velocity to put a new component...")
      tot_chisq_spec = np.zeros(self.velocity.size)
      for i in range(self.mypars.anum.size):
        velocity_mask = (self.obsvel[i,-1] > self.velocity) & (self.obsvel[i,0] < self.velocity) & \
          (np.interp(self.velocity, self.obsvel[i,:], self.bigew, left=0, right=0) > np.interp(self.velocity, self.obsvel[i,:], self.bigsew, left=0, right=0))
        for bad_velocity in list_of_bad_velocities:
          velocity_mask = velocity_mask & ((self.velocity < bad_velocity - self.mypars.vres) | (self.velocity > bad_velocity + self.mypars.vres))
        tot_chisq_spec[velocity_mask] += np.interp(self.velocity[velocity_mask], self.obsvel[i,:], self.chisq_spec[i,:])

      potential_bad_vel = np.extract(tot_chisq_spec == np.max(tot_chisq_spec), self.velocity)[0]

      if self.mypars.add_clouds and not minfirst:
        print(f"\t... and adding it at {potential_bad_vel} with badness {np.max(tot_chisq_spec)}")
        try:
          xclp        = np.append(                  xclp, 0.0                                                                     )
          yclp        = np.append(                  yclp, 0.0                                                                     )
          rhoindex    = np.append(              rhoindex, 1.0 + 1.5 * np.random.rand(1)                                           )
          logrho0     = np.append(               logrho0, 2.5 + 1.5 * np.random.rand(1)                                           )
          #logrhoscale = np.append(           logrhoscale, 0.5 + 2.5 * np.random.rand(1) + np.log10(self.mydisk.rg.to(u.cm).value) )
          logrhoscale = np.append(           logrhoscale, 19.0 - logrho0[-1] )
          logZ        = np.append(                  logZ, 0.0                                                                     )
          vcl         = np.append(vcl.to(u.km/u.s).value, potential_bad_vel.to(u.km/u.s).value) * (u.km/u.s)

          #zcl_prop = (10.0**(2.6 + logrhoscale[-1]) * u.cm / self.mydisk.rg).decompose()
          zcl_prop = (np.random.rand(1) * 0.5 * u.kpc / self.mydisk.rg).decompose()
          while (zcl_prop < 10.0**(1.5 + logrhoscale[-1]) * u.cm / self.mydisk.rg) or (zcl_prop > 10.0**(3.0 + logrhoscale[-1]) * u.cm / self.mydisk.rg):
            zcl_prop = (np.random.rand(1) *  u.kpc / self.mydisk.rg).decompose()
          zcl         = np.append(                   zcl, zcl_prop)
        except:
          xclp        = np.array([ 0.0])
          yclp        = np.array([ 0.0]) 
          logZ        = np.array([ 0.0])

          vcl         = np.array([potential_bad_vel.to(u.km/u.s).value]) * (u.km/u.s)

          rhoindex    = 1.0 + 1.5 * np.random.rand(1)
          logrho0     = 2.5 + 1.5 * np.random.rand(1)
          #logrhoscale = 0.5 + 2.5 * np.random.rand(1) + np.log10(self.mydisk.rg.to(u.cm).value)
          logrhoscale = 19.0 - logrho0

          #zcl = (10.0**(2.6 + logrhoscale) * u.cm / self.mydisk.rg).decompose()
          zcl = (np.random.rand(1) * 0.5 * u.kpc / self.mydisk.rg).decompose()
          while (zcl < 10.0**(1.5 + logrhoscale[-1]) * u.cm / self.mydisk.rg) or (zcl > 10.0**(3.0 + logrhoscale[-1]) * u.cm / self.mydisk.rg):
            zcl = (np.random.rand(1) * u.kpc / self.mydisk.rg).decompose()

        clouds = self.makeclouds(xclp, yclp, zcl,
                                 rhoindex, logrhoscale, logrho0, logZ,
                                 vcl
                                 )
        self._abs_write_clouds(clouds)

      else:
        print(f"\t... but not actually adding it...sigh... (would have been {potential_bad_vel})")
        clouds = self.clouds

      self.reset_observer()

      print("#" * 50)
      print(f"Beginning optimization with {len(clouds)} clouds...")
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
        res = least_squares(self._abs_chisqfunc,
                            self._abs_pack(clouds),
                            bounds=self._abs_bounds(clouds),
                            jac="2-point",
                            callback=self._abs_callback,
                            diff_step = 0.001
                            )

      print(f"Optimized (in {tm.time()-t0} seconds)!  Cleaning up...") 
      try:
        if res.success:
          print("\tSupposedly, the least-squares fit was successful")
          chisq = np.sum(self._abs_chisqfunc(res.x))
          clouds = copy.deepcopy(self._chisqfunc_clouds)
        else:
          print("\tSomething barfed with the least-squares fit")
          input("Paused")
      except AttributeError:
        if self.mypars.mcmin:
          clouds = mcminimize_clouds
      print("\t","#"*20)
      print(f"\tChi^2 = {chisq} vs previous Chi^2 = {oldchisq}")
      # Was adding this cloud a statistically significant improvement in the fit?
      # We need to run F-test
      dof = 0
      for i in range(self.mypars.anum.size):
        velocity_mask = (self.obsvel[i,:] > self.velocity[0]) & (self.obsvel[i,:] < self.velocity[-1])
        dof += np.sum(velocity_mask)
      dof -= 7 * len(clouds)
      old_dof = dof - 7
      F_stat = (oldchisq / old_dof) / (chisq / dof)

      p_value = Ftest.sf(F_stat, dof, old_dof)
      print(f"\tF-stat = {F_stat} --> probablility that the new and old fits are statistically consistent {p_value}")
      if p_value < self.mypars.F_test_prob or not self.mypars.add_clouds or minfirst:
        print("\t\tKEEPING NEW FIT!")
        minfirst = False
        self.clouds = copy.deepcopy(clouds)
        self._abs_write_clouds(self.clouds)
        self.reset_observer()
        if not self.mypars.mcmin:
          chisq = np.sum(self._abs_chisq(*self._calculate_absorbed_flux_gaussleg(self.clouds)
                                         )
                         )
      else:
        print("\t\tRESETTING BACK TO OLD FIT AND FLAGGING VELOCITY!")
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
        print("\t"*ntabs, 
              f"Computing cloud positions (apparent positions given as ({xclp}, {yclp}, {zcl})....")
        print("\t"*ntabs, 
              f"\t\twith observer located at r,z = {self.robs}, {self.zobs}")
      xcl, ycl = self._abs_deproject_clouds(xclp, yclp, zcl)
      rcl     = np.sqrt(xcl * xcl + ycl * ycl)
      thetacl = np.arctan2(ycl, xcl)
      if not isinstance(thetacl, Quantity):
        thetacl *= u.rad
      thetacl = thetacl.to(u.deg)

      if self.mypars.verbose: print("\t"*ntabs, "Making clouds!")
      clouds = []
      ncl    = rcl.size
      for i in range(ncl):
        t0 = tm.time()
        if self.mypars.verbose:
          print("\t"*ntabs, "#" * 50)
          print("\t"*ntabs, f"\t{i} Making a cloud with the following parameters: xclp = {xclp[i]} yclp = {yclp[i]}")
          print("\t"*ntabs, "\t"*7 + f"--> xcl = {xcl[i]:.3f}   ycl = {ycl[i]:.3f} zcl={zcl[i]:.3f}")
          print("\t"*ntabs, "\t"*7 + f"              --> r = {rcl[i]:.3f}  theta = {thetacl[i]:.3f}")
          print("\t"*ntabs, "\t"*7 + f"rhoindex = {rhoindex[i]}  logrhoscale = {logrhoscale[i]} logrho0 = {logrho0[i]}")
          print("\t"*ntabs, "\t"*7 + f"log Z = {logZ[i]}  vcl_los = {vcl[i]}")
        clouds.append(AbsCloud(self.mypars.datapath, self.mydisk, self.mycorona, self.myatoms,
                               rcl[i], zcl[i], thetacl[i], rhoindex=rhoindex[i], logrhoscale=logrhoscale[i], logrho0=logrho0[i], logZ=logZ[i], vcl_los=vcl[i]))
        if self.mypars.verbose:
          print("\t"*ntabs, f"\t{i} Determining ionizing spectrum")
        cloudy_rootname = f"ABS-rho0{logrho0[i]}-index{rhoindex[i]}-scale{logrhoscale[i]}-logZ{logZ[i]}-zcl{zcl[i]}"
        clouds[i].calcionspec(cloudyfileroot = cloudy_rootname, structure_only = True)
        totflux = self._calculate_absorbed_flux_gaussleg(None,
                                                         robs = rcl[i], thetaobs = thetacl[i], zobs = zcl[i],
                                                         wavelength = clouds[i].ionspecfreq.to(u.Angstrom, equivalencies=u.spectral()),
                                                         lograd = True, noplot = True
                                                         )[0]
        try:
          dum = self._spectrum_scale
        except AttributeError:
          print("\t"*ntabs, "\tSetting spectral scale 'cause I'm stupid and can't integrate...")
          clouds[i].calcionspec(cloudyfileroot = cloudy_rootname, structure_only = False)
          corona_flux = np.squeeze(self.mycorona.fnu_lamppost(clouds[i].ionspecfreq,rcl[i],zcl[i]))
          self._spectrum_scale = np.max(clouds[i].ionspecfreq*clouds[i].ionspecflux) / np.max(clouds[i].ionspecfreq*totflux)
          while np.any(self._spectrum_scale * totflux > clouds[i].ionspecflux):
            self._spectrum_scale *= 0.999
          print("\t"*ntabs, f"\t\tScale set to {self._spectrum_scale}")

        corona_flux = np.squeeze(self.mycorona.fnu_lamppost(clouds[i].ionspecfreq,rcl[i],zcl[i]))
        totflux = self._spectrum_scale * totflux + corona_flux
        clouds[i].ionspecflux = np.where(totflux < 1.0e-100 * fu, 1.0e-100 * fu, totflux)

        if self.mypars.verbose:
          print("\t"*ntabs, f"\t{i} Resolving ionization structure {self.mypars.cloudypath}")
        # This is for creating clouds[i].radius, clouds[i].density, clouds[i].temperature, clouds[i].iondensity arrays
        clouds[i].getcloudy(self.mypars.cloudypath, 
                            verbose = self.mypars.verbose, 
                            runcloudy = False, 
                            softenning = self.mypars.softenning)
        (lognuFnu, log_ion_parm) = self._calc_ion_parm(clouds[i])
        sdx = np.argsort(log_ion_parm)
        #if self.mypars.verbose:
        #  print("\t"*ntabs, f"\t\tLog ionization parameter in range ({np.min(log_ion_parm)}, {np.max(log_ion_parm)})")
        

        # Need to fill the clouds[i].iondensity array
        try:
          dum = self.cheb_log_ion_parm_list
        except AttributeError:
          print("\t"*ntabs, "\t\tOOPS - FORGOT TO READ CHEBYSHEV TABLES...")
          self._read_cheb_files()

        if self.cheb_log_ion_parm_list.size > 0:
          lnmin = np.min(self.cheb_log_ion_parm_list)
          lnmax = np.max(self.cheb_log_ion_parm_list)
        else:
          print("\t"*ntabs, "\t\tOOPS - CHEBYSHEV TABLES DON'T EXIST...")
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
          prtstr = "\t"*ntabs + f"\t\tNeed ionization parameter range ({np.min(log_ion_parm)}, {np.max(log_ion_parm)})"
          print(prtstr)
          if np.any(log_ion_parm > lnmax) or np.any(log_ion_parm < lnmin):
            prtstr = f" \t\t\twhich is outside the range ({lnmin},{lnmax})   ({np.any(log_ion_parm > lnmax)}, {np.any(log_ion_parm < lnmin)})"
            print(prtstr)
          if (np.any(log_ion_parm > lnmax) or np.any(log_ion_parm < lnmin)) and np.any(interp_dlog_ion_parm > target_dlog_ion_parm):
            print("\t\t\t\tand")
          if np.any(interp_dlog_ion_parm > target_dlog_ion_parm):
            missing_index = interp_dlog_ion_parm > np.median(self.cheb_dlog_ion_parm_list)
            prtstr = f"\t\t\twhich is in a gap ({np.min(log_ion_parm[missing_index])}, {np.max(log_ion_parm[missing_index])}), with "
            prtstr += f"{np.sum(interp_dlog_ion_parm[missing_index] > target_dlog_ion_parm )} bins with > {target_dlog_ion_parm}"
            print(prtstr)
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
          print("\t"*ntabs, f"\t\tChebyshev xnorm in range ({np.min(xnorm)}, {np.max(xnorm)})")
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

            logionfrac = np.where(logionfrac > 0.0, 0.0, logelemabund+logionfrac)

            clouds[i].iondensity[sdx,iondx] = clouds[i].density[sdx] * np.power(10.0,logionfrac)

          if not np.all(np.isfinite(clouds[i].iondensity)):
            print("\t"*ntabs, "ERROR IN GETTING ION DENSITIES:")
            print("\t"*ntabs, f"logionfrac = {logionfrac}")
            print("\t"*ntabs, f"cloud density = {clouds[i].density}")
            print("\t"*ntabs, f"cloud ion density = {clouds[i].iondensity}")
            input("Stopped in quasar.makeclouds")

        clouds[i].temperature = 10.0**chebyshev.chebval(xnorm,
                                                        self.cheb_log_temperature_coeff
                                                        ) * u.K
        if self.mypars.verbose:
          print("\t"*ntabs, f"\t\tTemperature in range ({np.min(clouds[i].temperature)}, {np.max(clouds[i].temperature)})")



        if self.mypars.verbose:
          print("\t"*ntabs, f"\t{i} Cloud took {tm.time()-t0} s")


    else:
      clouds = None

    return clouds

  #######################################################################################
  def printpars(self):
    print("-" * 70)
    print(f"Observer: zqso = {self.mypars.zqso}, inclination = {self.mypars.inclination}, coordinates = {self.skycoord.to_string('hmsdms')}")

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
    print("\t" * ntabs + "        num rcl          zcl          thetacl           rhoindex     rhoscale/rg  logrho0      logZ         median T     vlos")
    for i in range(len(clouds)):
      prtstr  = "\t" * ntabs + f"\t {i:2d} {clouds[i].rcl:e} {clouds[i].zcl:e} {clouds[i].thetacl:e} {clouds[i].rhoindex:e} "
      prtstr += f"{(10.0**clouds[i].logrhoscale * u.cm)/self.mydisk.rg:e} {clouds[i].logrho0:e} {clouds[i].logZ:e} {np.median(clouds[i].temperature):e} {clouds[i].vlos:e}"
      print(prtstr)

    return

  #######################################################################################
  def print_diff_clouds(self,
                        clouds1,
                        clouds2,
                        ntabs = 0):
    print("\t" * ntabs + "Differences (new cloud - old cloud):")
    print("\t" * ntabs + "        num  drcl          dzcl          dthetacl         drhoindex     drhoscale/rg  dlogrho0      dlogZ        dvlos")
    for i in range(len(clouds1)):
      prtstr  = "\t" * ntabs + f"\t {i:2d} {clouds2[i].rcl-clouds1[i].rcl:13.6e} {clouds2[i].zcl-clouds1[i].zcl:13.6e} {clouds2[i].thetacl-clouds1[i].thetacl:13.6e} "
      prtstr += f"{clouds2[i].rhoindex-clouds1[i].rhoindex:13.6e} {((10.0**clouds2[i].logrhoscale-10.0**clouds1[i].logrhoscale) * u.cm)/self.mydisk.rg:13.6e} "
      prtstr += f"{clouds2[i].logrho0-clouds1[i].logrho0:13.6e} {clouds2[i].logZ-clouds1[i].logZ:13.6e} {clouds2[i].vlos-clouds1[i].vlos:13.6e}"
      print(prtstr)

    return

  #######################################################################################
  def readspec(self,
               nsigma = 1.1
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
