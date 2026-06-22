import numpy              as np
import matplotlib.pyplot  as plt
import os
import time               as tm

from astropy                 import constants   as const
from astropy                 import units       as u
from astropy.table           import Table
from multiprocessing         import Pool
from scipy.interpolate       import CubicSpline
from scipy.special           import erf,erfc
from tqdm                    import tqdm

from cloudy  import cloudy
from corona  import corona

# This class is about characterizing one absorbing cloud and having the physical conditions.
# Eventually, this will be about setting up and running Cloudy and reading in the output.
# But before we do that, we can still set up some ofthe architecture forthe clouds.
#
class AbsCloud:
  def __init__(self, basedir, mydisk, mycorona, myatoms,
               rcl, zcl, thetacl,                       # position of the cloud
               rhoindex=-2.0, logrhoscale=16, logrho0=2, # density parameters
               vcl_los=0.0
               ):
    self.basedir     = basedir
    self.mydisk      = mydisk
    self.mycorona    = mycorona
    self.myatoms     = myatoms
    self.rcl         = rcl
    self.zcl         = zcl
    self.thetacl     = thetacl
    self.rhoindex    = rhoindex
    self.logrhoscale = logrhoscale
    self.logrho0     = logrho0
    self.vlos        = vcl_los
    self.cloudyran   = False

    if not 0.1 * self.zcl > (10.0**self.logrhoscale * u.cm / self.mydisk.rg):
      print("AbsCloud WARNING: Your cloud size is not much less than the height above the disk! Is this physical?")
      print(f"\t\tResetting zcl = {self.zcl} --> {10.0*self.zcl}")
      self.zcl = 10 * np.abs(self.zcl)

  #####################################################################
  def calcionspec(self,
                  cloudyfileroot = '',
                  structure_only = False,
                  verbose = False
                  ):
    fu = u.erg / (u.s * u.cm * u.cm * u.Hz)
    sedfilename = self.basedir+'Cloudy_runs/'+cloudyfileroot+'.sed'
    if verbose:
      print(f"\t\tLooking for {sedfilename}...")
    if os.path.exists(sedfilename):
      if verbose:
        print(f"\t\t\t...Found! Reading in file...")
      datatab = Table.read(sedfilename, format="ascii.basic", guess=False, data_start=1, names=["freq","flux"])
      RydinHz = (const.Ryd).to(u.Hz, equivalencies=u.spectral())
      self.ionspecfreq = datatab["freq"] * RydinHz
      self.ionspecflux = datatab["flux"] * fu
    else:
      if verbose:
        print(f"\t\t\t...Not found! Calculating...")
      # Calculate the ionizing spectrum for the cloud
      self.ionspecfreq = np.logspace(13,20,num=3000) * u.Hz
      self.ionspecflux = (np.zeros(3000) + 1.0e-100) * fu
      if not structure_only:
        self.mydisk.robs = self.rcl
        self.mydisk.zobs = self.zcl
        fracerr = np.inf
        for r in tqdm(range(self.mydisk.rstar.size), desc=f"\t\t\tIntegrating disk:", ncols=0):
          oldspec = np.where(self.ionspecflux < 1.0e-100 * fu, 1.0e-100 * fu, self.ionspecflux)

          fluxdiskannulusdivided = self.mydisk.fnudiskannulus(self.ionspecfreq,r)
          fluxdiskannulus        = np.sum(fluxdiskannulusdivided[:,:,0],axis=1)
          self.ionspecflux      += fluxdiskannulus

          fracerr = np.sum(self.ionspecflux/oldspec - 1.0)
          if (r > 0.2 * self.mydisk.rstar.size) and (fracerr < 1.0e-3):
            break

        corona_flux = np.squeeze(self.mycorona.fnu_lamppost(self.ionspecfreq, self.rcl, self.zcl))
        self.ionspecflux += corona_flux
        self.ionspecflux = np.where(self.ionspecflux < 1.0e-100 * fu, 1.0e-100 * fu, self.ionspecflux)

  #####################################################################
  def getcloudy(self,
                cloudypath,
                verbose = False,
                runcloudy = False):
    if runcloudy:
      cloud  = cloudy(self.basedir, cloudypath, 1,                 # 0 = emission, 1 = absorption
                      self.myatoms,
                      self.ionspecfreq, self.ionspecflux,            # ionizing spectrum
                      self.rhoindex, self.logrhoscale, self.logrho0, # density parameters
                      zstar = self.zcl,
                      verbose = verbose
                      )

      self.cloudyran = True
      self.depth  = np.copy(cloud.depth)
      self.radius = np.max(self.depth)-self.depth

      self.density     = np.copy(cloud.density)
      self.temperature = np.copy(cloud.temperature)
      self.iondensity  = np.copy(cloud.iondens)    # This should actually be an array of shape (self.depth.size,self.myatoms.nion)

    else: 
      self.depth = np.logspace(0,self.logrhoscale-0.1,num=100) * u.cm
      self.radius = np.max(self.depth)-self.depth

      self.ddepth = np.zeros_like(self.depth)
      self.ddepth[1:-1] = 0.5 * (self.depth[2:] - self.depth[:-2])
      self.ddepth[0] = self.ddepth[1]
      self.ddepth[-1] = self.ddepth[-2]

      # Density - use formula for the globule in Hazy
      self.density = 10.0**(self.logrho0) * np.power(1 - self.depth.value/(10.0**self.logrhoscale), -self.rhoindex) / u.cm**3

      # Temperature - scale with density using the ideal gas law
      self.temperature = (self.density[0] * 1.0e+4 * u.K)  / self.density

      # Iondensity - Use density, but scaled for the abundance of the element stored in self.myatoms.abund.
      # Shape will be (self.depth.size,self.myatoms.nion)
      self.iondensity = np.zeros((self.depth.size,self.myatoms.nion)) * (u.cm**-3)

    self.dr = np.zeros(self.depth.shape) * u.cm
    self.dr[1:-1] = 0.5 * (self.depth[2:] - self.depth[:-2])
    self.dr[0] = self.dr[1]
    self.dr[-1] = self.dr[-2]


  #####################################################################
  # Calculate the optical depth as a function of wavelength/frequency for a single sightline
  # Equation 71 from our documentation
  def runsightline(self,
                   impact_parameter,
                   wavelength
                   ):
    opticaldepth = np.zeros(wavelength.size)
    radius       = np.max(self.depth)-self.depth
    zonearray    = np.where(radius > impact_parameter)[0]
    #print(f"\t\t\t\tAt impact parameter {impact_parameter}, have to loop through {zonearray.size} zones")
    for i in range(self.myatoms.nion):
      ionindex = self.myatoms.idx[i]

      for zone in zonearray:
        dx = np.squeeze(self.dr[zone] * np.sqrt(1 - np.square(impact_parameter / radius[zone])))
      
        N = self.iondensity[zone,i] * dx
        b = np.squeeze((np.sqrt(2 * const.k_B * self.temperature[zone] / self.myatoms.amass[ionindex])).decompose())
        
        tau0   = ((np.sqrt(np.pi) * np.square(const.e.esu)) / (const.m_e * const.c) * self.myatoms.f[ionindex] * self.myatoms.wave[ionindex] * (N / b)).decompose()
        tauion = np.squeeze(self.myatoms.tauion(wavelength,self.myatoms.anum[ionindex],self.myatoms.ion[ionindex],tau0,b,self.vlos))

        opticaldepth += 2 * tauion

    return opticaldepth

  # Version of runsightline that accepts an array of impact_parameters
  def runsightline2D(self,
                     impact_parameter,
                     wavelength
                     ):
    restwavelength = wavelength * np.sqrt((1.0 - (self.vlos/const.c).decompose()) / (1.0 + (self.vlos/const.c).decompose()))
    opticaldepth = np.zeros((wavelength.size,impact_parameter.size))
    radius = np.max(self.depth)-self.depth
    for ipdx in range(impact_parameter.size):

      # See if we can replace zonearray and trdx with a zone_mask and a trans_mask
      #zone_mask = radius > impact_parameter[ipdx]
      #dxarray = np.squeeze(self.dr[zone_mask] * np.sqrt(1 - np.square(impact_parameter[ipdx] / radius[zone_mask])))
      #trans_mask = (self.myatoms.wave > restwavelength[0]) & (self.myatoms.wave < restwavelength[-1])
      #anumion_mask = (self.myatoms.anum[self.myatoms.idx] == self.myatoms.anum[trans_mask]) & (self.myatoms.ion[self.myatoms.idx] == self.myatoms.ion[trans_mask])
      #nz = np.int16(np.sum(zone_mask))
      #nt = np.int16(np.sum(trans_mask))
      #print(nz,nt,self.myatoms.species[trans_mask],self.myatoms.anum[self.myatoms.idx[anumion_mask]],self.myatoms.ion[self.myatoms.idx[anumion_mask]])
      #idx2d_array = np.int16(np.broadcast_to(np.extract(trans_mask, range(self.myatoms.wave.size)), (nz,nt)))
      #dx2d_array = np.broadcast_to(dxarray, (nt,nz)).T
      #N2d_array = self.iondensity[zone_mask,anumion_mask] * dx2d_array
      #b2d_array = np.sqrt(2 * const.k_B * np.outer(self.temperature[zone_mask], 1.0 / self.myatoms.amass[trans_mask])).to(u.km/u.s)
      #flambda_2d_array = np.broadcast_to(self.myatoms.f[trans_mask] * self.myatoms.wave[trans_mask], (nz,nt))
      #tau0_2d_array = ((np.sqrt(np.pi) * np.square(const.e.esu)) / (const.m_e * const.c)) * flambda_2d_array * (N2d_array / b2d_array)
      #with np.nditer([idx2d_array,tau0_2d_array, b2d_array.value], flags=['multi_index']) as it:
      #  opticaldepth[:,ipdx] += 2 * np.sum(self.myatoms.tausingle(wavelength,np.int16(it[0]),it[1],it[2] * (u.km/u.s),self.vlos), axis=1)

      zonearray = np.where(radius > impact_parameter[ipdx])[0]
      if zonearray.size > 0:
        dxarray   = np.squeeze(self.dr[zonearray] * np.sqrt(1 - np.square(impact_parameter[ipdx] / radius[zonearray])))
        for i in range(self.myatoms.nion):
          ionindex = self.myatoms.idx[i]
          trdx = np.extract((self.myatoms.anum == self.myatoms.anum[ionindex]) & (self.myatoms.ion == self.myatoms.ion[ionindex]) & (self.myatoms.wave > restwavelength[0]) & (self.myatoms.wave < restwavelength[-1]),
                            range(self.myatoms.wave.size))
          if trdx.size > 0:
            Narray    = self.iondensity[zonearray,i] * dxarray
            barray    = np.squeeze((np.sqrt(2 * const.k_B * self.temperature[zonearray] / self.myatoms.amass[ionindex])).decompose())
            tau0array = ((np.sqrt(np.pi) * np.square(const.e.esu)) / (const.m_e * const.c) * self.myatoms.f[ionindex] * self.myatoms.wave[ionindex] * (Narray / barray)).decompose()
            tauion    = self.myatoms.tauion_old(wavelength,
                                            self.myatoms.anum[ionindex],
                                            self.myatoms.ion[ionindex],
                                            tau0array,
                                            barray,
                                            np.ones(zonearray.size) * self.vlos)
            opticaldepth[:,ipdx] += 2 * np.sum(tauion, axis = 1)

    return opticaldepth


  # Version of runsightline that accepts an array of impact_parameters
  def runsightline2D_new(self,
                         impact_parameter,
                         wavelength,
                         nproc  = 1,
                         nparse = 30
                         ):
    restwavelength = wavelength * np.sqrt((1.0 - (self.vlos/const.c).decompose()) / (1.0 + (self.vlos/const.c).decompose()))
    opticaldepth = np.zeros((wavelength.size,impact_parameter.size))

    radius2D = np.broadcast_to(self.radius.to(u.cm), (impact_parameter.size,self.depth.size)) * u.cm
    drad2D   = np.broadcast_to(self.dr.to(u.cm),     (impact_parameter.size,self.depth.size)) * u.cm

    impact_parameter2D = np.broadcast_to(impact_parameter.to(u.cm), (self.depth.size,impact_parameter.size)).T * u.cm
    zone_mask = radius2D > impact_parameter2D

    dxarray2D = np.zeros((impact_parameter.size,self.depth.size)) * u.cm
    dxarray2D[zone_mask] = drad2D[zone_mask] * np.sqrt(1 - np.square(impact_parameter2D[zone_mask] / radius2D[zone_mask]))

    # Which transitions are available for the rest wavelength range given?
    trans_mask = (self.myatoms.wave > restwavelength[0]) & (self.myatoms.wave < restwavelength[-1])
    # I need the indices of the strongest(=first) transition for each ion that is covered (for a given ion, transitions are listed in order of decreasing strength)
    ions = np.unique(100*self.myatoms.anum + self.myatoms.ion)
    anumion_mask = np.isin(ions,100*self.myatoms.anum[trans_mask] + self.myatoms.ion[trans_mask])
    nai = np.int16(np.sum(anumion_mask))
    aidx = np.int16(np.extract(anumion_mask, range(self.myatoms.nion)))

    N3D_array        = np.zeros((impact_parameter.size, self.depth.size, self.myatoms.nion)) /  u.cm**2
    b3D_array        = np.ones( (impact_parameter.size, self.depth.size, self.myatoms.nion)) * (u.km/u.s)
    flambda_3d_array = np.zeros((impact_parameter.size, self.depth.size, self.myatoms.nion)) * (u.Angstrom)

    for ipdx in range(impact_parameter.size):
      zdx  = np.int16(np.extract(zone_mask[ipdx,:], range(self.depth.size)))
      zmask = zone_mask[ipdx,:]
      N3D_array[       ipdx,zmask,:][:,anumion_mask] = self.iondensity[zmask,:][:,anumion_mask] * (np.broadcast_to(dxarray2D[ipdx,zdx].to(u.cm), (nai,zdx.size)).T * u.cm)
      b3D_array[       ipdx,zmask,:][:,anumion_mask] = np.sqrt(2 * const.k_B * np.outer(self.temperature[zdx], 1 / self.myatoms.amass[self.myatoms.idx[anumion_mask]])).to(u.km/u.s)
      flambda_3d_array[ipdx,zmask,:][:,anumion_mask] = np.broadcast_to(self.myatoms.f[self.myatoms.idx[anumion_mask]] * self.myatoms.wave[self.myatoms.idx[anumion_mask]].to(u.Angstrom), (zdx.size, nai)) * u.Angstrom
      #for zdx in np.int16(np.extract(zone_mask[ipdx,:], range(self.depth.size))):
      #    N3D_array[       ipdx,zdx,anumion_mask] = self.iondensity[zdx,anumion_mask] * dxarray2D[ipdx,zdx]
      #    b3D_array[       ipdx,zdx,anumion_mask] = np.sqrt(2 * const.k_B * self.temperature[zdx] / self.myatoms.amass[self.myatoms.idx[anumion_mask]]).to(u.km/u.s)
      #    flambda_3d_array[ipdx,zdx,anumion_mask] = self.myatoms.f[self.myatoms.idx[anumion_mask]] * self.myatoms.wave[self.myatoms.idx[anumion_mask]]

    tau0_3d_array = ((np.sqrt(np.pi) * np.square(const.e.esu)) / (const.m_e * const.c)) * flambda_3d_array * (N3D_array / b3D_array)

    if nproc == 1:
      for atom_ion_dx in np.extract(anumion_mask, range(self.myatoms.nion)):
        nimp = 0
        while nimp < impact_parameter.size:
          nimp_lo = nimp
          nimp_hi = np.int16(np.min([impact_parameter.size,nimp_lo+nparse]))
          tau = self.myatoms.tauion_old(wavelength,
                                        self.myatoms.anum[self.myatoms.idx[atom_ion_dx]],
                                        self.myatoms.ion[ self.myatoms.idx[atom_ion_dx]],
                                        tau0_3d_array[    nimp_lo:nimp_hi,:,atom_ion_dx].flatten(),
                                        b3D_array[        nimp_lo:nimp_hi,:,atom_ion_dx].flatten(),
                                        np.ones(b3D_array[nimp_lo:nimp_hi,:,atom_ion_dx].shape).flatten() * self.vlos
                                        ) * 2
          opticaldepth[:,nimp_lo:nimp_hi] += np.sum(tau.reshape(wavelength.size,nimp_hi-nimp_lo,self.depth.size),axis=2)
          nimp += nparse
    else:
      which_ions = np.extract(anumion_mask, range(self.myatoms.nion))

      nimp = 0
      t1 = tm.time()
      time_per_ip = 0.0
      old_time_per_ip = time_per_ip
      while (nimp < impact_parameter.size):
        if old_time_per_ip < time_per_ip:
          nparse *= 2
        else:
          nparse = nparse // 3
        with Pool(which_ions.size) as pool:
          nimp_lo = nimp
          if nimp_lo < impact_parameter.size:
            nimp_hi = np.int16(np.min([impact_parameter.size,nimp_lo+nparse]))
          print(f"\t\t\tStarting integration for patches {nimp_lo} - {nimp_hi} ({nparse})  {tm.time()-t1} s  ({time_per_ip}) s / sightline")
          pool_input_tuple = []
          for pdx in range(which_ions.size):
              atom_ion_dx = which_ions[pdx]
              pool_input_tuple.append((wavelength,
                                       self.myatoms.anum[self.myatoms.idx[atom_ion_dx]],
                                       self.myatoms.ion[ self.myatoms.idx[atom_ion_dx]],
                                       tau0_3d_array[    nimp_lo:nimp_hi,:,atom_ion_dx].flatten(),
                                       b3D_array[        nimp_lo:nimp_hi,:,atom_ion_dx].flatten(),
                                       np.ones(b3D_array[nimp_lo:nimp_hi,:,atom_ion_dx].shape).flatten() * self.vlos) )

          tau_output_tuple = pool.starmap(self.myatoms.tauion_old, pool_input_tuple)

          for pdx in range(which_ions.size):
            opticaldepth[:,nimp_lo:nimp_hi] += np.sum(tau_output_tuple[pdx].reshape(wavelength.size,nimp_hi-nimp_lo,self.depth.size),axis=2)

          old_time_per_ip = time_per_ip
          time_per_ip = (tm.time() - t1) / (nimp_hi - nimp_lo)
          t1 = tm.time()

        nimp = nimp_hi

    return opticaldepth
