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
  def __init__(self, 
               mypars,         # Instance of readpars class
               mydisk,         # Instance of ntdisk class
               mycorona,       # Instance of corona class
               myatoms,        # Instance of atomic class
               rcl,            # position of the cloud - cylindrical r component (units of rg)
               zcl,            # position of the cloud - z component (units of rg)
               thetacl,        # position of the cloud - cylindrical theta/azimuthal component
               rhoindex=-2.0,  # density parameters - globule power-law index
               logrhoscale=16, # density parameters - logarithmic scale/rg
               logrho0=2,      # density parameters - logarithmic number density at surfance
               logZ = 0.0,     # Abundance parameters - log of metallicity in solar units
               vcl_los=0.0,    # Line-of-sight velocity of cloud (km/s)
               ntabs = 0
               ):
    self.mypars      = mypars
    self.mydisk      = mydisk
    self.mycorona    = mycorona
    self.myatoms     = myatoms
    self.rcl         = rcl
    self.zcl         = zcl
    self.thetacl     = thetacl
    self.rhoindex    = rhoindex
    self.logrhoscale = logrhoscale
    self.logrho0     = logrho0
    self.logZ        = logZ
    self.vlos        = vcl_los
    self.cloudyran   = False

    if not 0.1 * self.zcl > (10.0**self.logrhoscale * u.cm / self.mydisk.rg):
      print("\t" * ntabs + "AbsCloud WARNING: Your cloud size is not much less than the height above the disk! Is this physical?")
      print("\t" * (ntabs+1) + f"\tResetting zcl = {self.zcl} --> {10.0*self.zcl}")
      self.zcl = 10 * np.abs(self.zcl)

  #####################################################################
  def calcionspec(self,
                  cloudyfileroot = '',
                  structure_only = False,
                  verbose = False,
                  ntabs = 0
                  ):
    fu = u.erg / (u.s * u.cm * u.cm * u.Hz)
    sedfilename = self.mypars.datapath+'Cloudy_runs/'+cloudyfileroot+'.sed'
    if verbose:
      print("\t" * ntabs + f"Looking for {sedfilename}...")
    if os.path.exists(sedfilename):
      if verbose:
        print("\t" * (ntabs+1) + f"...Found! Reading in file...")
      datatab = Table.read(sedfilename, format="ascii.basic", guess=False, data_start=1, names=["freq","flux"])
      RydinHz = (const.Ryd).to(u.Hz, equivalencies=u.spectral())
      self.ionspecfreq = datatab["freq"] * RydinHz
      self.ionspecflux = datatab["flux"] * fu
    else:
      if verbose:
        print("\t" * (ntabs+1) + f"...Not found! Calculating...")
      # Calculate the ionizing spectrum for the cloud
      self.ionspecfreq = np.logspace(12.0,24.5,num=5000) * u.Hz
      self.ionspecflux = (np.zeros(5000) + 1.0e-100) * fu
      if not structure_only:
        self.mydisk.robs = self.rcl
        self.mydisk.zobs = self.zcl
        fracerr = np.inf
        for r in tqdm(range(self.mydisk.rstar.size), desc="\t" * ntabs + f"Integrating disk", ncols=0):
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

    self.ionspecdfreq       = np.copy(self.ionspecfreq)
    self.ionspecdfreq[1:-1] = 0.5 * (self.ionspecfreq[2:] - self.ionspecfreq[:-2])
    self.ionspecdfreq[ 0]   = self.ionspecdfreq[ 1]
    self.ionspecdfreq[-1]   = self.ionspecdfreq[-2]

  #####################################################################
  def getcloudy(self,
                cloudypath,
                softenning = 0.1,
                verbose = False,
                runcloudy = False
                ):
    if runcloudy:
      cloud  = cloudy(self.mypars.datapath, cloudypath, 1,                 # 0 = emission, 1 = absorption
                      self.myatoms,
                      self.ionspecfreq, self.ionspecflux,            # ionizing spectrum
                      rhoindex=self.rhoindex, logrhoscale=self.logrhoscale, logrho0=self.logrho0, # density parameters
                      logZ=self.logZ,
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
      self.depth = np.logspace(0,self.logrhoscale-softenning,num=100) * u.cm
      self.radius = np.max(self.depth)-self.depth

      self.ddepth = np.zeros_like(self.depth)
      self.ddepth[1:-1] = 0.5 * (self.depth[2:] - self.depth[:-2])
      self.ddepth[0] = self.ddepth[1]
      self.ddepth[-1] = self.ddepth[-2]

      # Density - use formula for the globule in Hazy
      norm_depth = 1.0 - self.depth/(10.0**self.logrhoscale * u.cm)
      norm_depth[norm_depth <= 0] = 1.0 - (10.0**(-softenning))
      self.density = 10.0**(self.logrho0) * np.power(norm_depth, -self.rhoindex) / u.cm**3

      # Temperature - scale with density using the ideal gas law, assuming gas pressure balance
      self.temperature = np.ones(self.density.size) * 100.0 * u.K

      # Iondensity - Shape will be (self.depth.size,self.myatoms.nion)
      self.iondensity = np.zeros((self.depth.size,self.myatoms.nion)) * (u.cm**-3)

    self.dr       = np.zeros(self.depth.shape) * u.cm
    self.dr[1:-1] = 0.5 * (self.depth[2:] - self.depth[:-2])
    self.dr[ 0]   = self.dr[ 1]
    self.dr[-1]   = self.dr[-2]
