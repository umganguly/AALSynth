import numpy                  as np
import os
import subprocess
import matplotlib.pyplot as plt

from astropy                 import constants as const
from astropy                 import units     as u
from astropy.io              import fits, ascii
from astropy.table           import Table

# The cloudy class is meant to be an interface to running and reading Cloudy
class cloudy:
  def __init__(self, basedir, cloudypath, Abs_or_Em,    # 0 = emission, 1 = absorption
               myatoms,                                 # instance of atomic class
               ionspecfreq, ionspecflux,                # ionizing spectrum
               rhoindex=0.0, logrhoscale=16, logrho0=2, # density parameters
               contemp = 1.0e+6, rstar = 10.0, zstar = 10.0,
               verbose = False):

    self.slash = "/"
    #if os.name == 'nt':
    #  self.slash = "\\"

    self.Abs_or_Em = Abs_or_Em
    
    self.basedir = basedir
    if basedir[-1] == '/' or basedir[-1] == '\\':
      self.basedir = basedir[:-1]

    os.chdir(self.basedir+self.slash+"Cloudy_runs")
    self.myatoms = myatoms
    lognuFnu = np.interp((const.Ryd).to(u.Hz, equivalencies=u.spectral()), ionspecfreq, np.log10((ionspecfreq * ionspecflux).value))
    if Abs_or_Em == 0: # Emission - Has not been developed yet
      rootname = f"EM-hden{logrho0:.2f}-nuFnu{lognuFnu:.2f}-rstar{rstar:.2f}-zstar{zstar:.2f}"
    else: # Absorption
      rootname = f"ABS-rho0{logrho0}-index{rhoindex}-scale{logrhoscale}-zcl{zstar}"
      #rootname = f"ABS-rho0{logrho0}-index{rhoindex}-scale{logrhoscale}-nuFnu{lognuFnu}"
      #rootname = f"ABS-rho0{logrho0:.2f}-index{rhoindex:.2f}-scale{logrhoscale:.2f}-nuFnu{lognuFnu:.2f}"
      #rootname = f"ABS"

    fitsfile = self.basedir+self.slash+"Cloudy_runs"+self.slash+f"{rootname}.fits"
      
    if verbose: print(f"\t\t\tLooking for cloudy in {cloudypath}")
    if os.path.exists(f"{cloudypath}/cloudy.exe"):
      if verbose: print(f"                                      {cloudypath}/cloudy.exe exists!")
      # For the parameters given, do we need to run Cloudy or do we have files already?

      if not os.path.exists(fitsfile): # if the Cloudy files don't exist, then we need to run Cloudy
        if verbose: print(f"                Generating {rootname}.in file for Cloudy input")
        # Write out the SED file:
        with open(self.basedir+self.slash+"Cloudy_runs"+self.slash+rootname+".sed", "w") as f:
          f.write(f"# SED for {rootname}\n")
          RydinHz = (const.Ryd).to(u.Hz, equivalencies=u.spectral())
          for i in range(ionspecfreq.size):
            f.write(f"{ionspecfreq[i].value/RydinHz.value} {ionspecflux[i].value}\n")

        # Write out the commands for Cloudy:
        with open(self.basedir+self.slash+"Cloudy_runs"+self.slash+rootname+".in", "w") as f:
          f.write(f"table SED \"{rootname}.sed\"\n")
          f.write(f"nuF(nu) = {lognuFnu:.2f}\n")
          f.write(f"table HM05 redshift 0.4\n")
          f.write("CMB redshift 0.4\n")
          f.write("Cosmic rays background\n")
          f.write("stop temperature 3 K linear\n")
          f.write("iterate\n")
          f.write("print last iteration\n")
          f.write(f"set save prefix \"{rootname}\"\n")
          f.write(f"save overview \".ovr\" last iteration\n")
          if self.Abs_or_Em == 0: # Line-emitting gas
            f.write(f"hden {logrho0}\n")
            f.write(f"constant temperature {contemp}\n")
            f.write(f"stop thickness {logrhoscale}\n")
            f.write(f"save lines, array \".lin\" last iteration \n")
          else: # Absorbing clouds
            f.write(f"globule density={logrho0}, depth={logrhoscale}, power={rhoindex}\n") # Density law
            f.write(f"stop thickness {logrhoscale-0.1}\n")

          for el in np.unique(self.myatoms.anum): # Elemental/ionic number densities
            (elemname, elemcode) = self.myatoms.cloudyelem(el)
            f.write(f"save element {elemname} \".{elemcode}\" density last\n")

        if not os.path.exists(rootname+".out") or os.path.getsize(rootname+".out") < 10000:
          # To run a C program in Python:
          if verbose: print("\t\t\tRunning Cloudy...")
          try:
            subprocess.run([f"{cloudypath}cloudy.exe", f"{rootname}.in"], capture_output=True, text=True, check=True)
            self.cloudyran = True
          except subprocess.CalledProcessError as e:
            print(f"\t\t\tcloudy.__init__: Execution error: {e.returncode}")
            print("\t\t\tcloudy.__init__: STDOUT:", e.output)
            print("\t\t\tcloudy.__init__: STDERR:", e.stderr)
            self.cloudyran = False
            input("paused: Why did Cloudy crash?")
            
          cloudy_output = subprocess.run(["tail", "--lines=10", f"{rootname}.out"], capture_output=True)
          for clo in cloudy_output.stdout.splitlines():
            print(f"\t\t\t{clo.decode('utf-8')}")

      else:
        if verbose: print(f"                                      {rootname} files exist!")
        self.cloudyran = True

    elif verbose:
      print("\t\t\tcloudy.__init__: Can't find Cloudy!")

    if os.path.exists(self.basedir+self.slash+"Cloudy_runs"+self.slash+f"{rootname}.ovr") and self.cloudyran:
      self._writeabsfits(rootname)
      self._cleanup_abs(rootname)

    if os.path.exists(fitsfile):
      print(f"\t\t\tReading in {fitsfile}")
      self._readcloudy(rootname)
    else:
      self.cloudyran = False

    if os.getcwd() == self.basedir+self.slash+"Cloudy_runs":
      os.chdir("../")

  ######################################################
  def _cleanup_abs(self,rootname):
    if os.getcwd() == self.basedir:
      os.chdir("Cloudy_runs")

    subprocess.run(["rm", f"{rootname}.in"])
    subprocess.run(["rm", f"{rootname}.out"])
    subprocess.run(["rm", f"{rootname}.ovr"])
    for el in np.unique(self.myatoms.anum): # Elemental/ionic number densities
      (elemname, elemcode) = self.myatoms.cloudyelem(el)
      subprocess.run(["rm", f"{rootname}.{elemcode}"])

    if os.getcwd() == self.basedir+self.slash+"Cloudy_runs":
      os.chdir("../")

  ######################################################
  def _readcloudy(self,rootname):
    # Now we need to read in the Cloudy outputs...
    if os.path.exists(self.basedir+self.slash+"Cloudy_runs"+self.slash+f"{rootname}.fits"):
      data = Table.read(self.basedir+self.slash+"Cloudy_runs"+self.slash+f"{rootname}.fits", format="fits")
      self.depth       = np.copy(data['depth']) * u.cm
      self.temperature = np.copy(data['temperature']) * u.K
      self.density     = np.copy(data['density']) / u.cm**3
      self.iondens     = np.copy(data['iondens']) / u.cm**3

      #if self.Abs_or_Em == 0: # Line-emitting gas
        #self.linarray = ascii.read(f"{rootname}.lin", format='commented_header', header_start=0, data_start=1, delimiter='\t', guess=False)

      self.cloudyran = True
    else:
      self.cloudyran = False
      self.depth = np.array([])

  ######################################################
  def _writeabsfits(self,rootname):
    if os.getcwd() == self.basedir:
      os.chdir("Cloudy_runs")
    # Now we need to read in the Cloudy outputs...
    if os.path.exists(f"{rootname}.ovr"):
      ovrtable = ascii.read(f"{rootname}.ovr", format='commented_header', header_start=0, data_start=1)
      depth       = ovrtable['depth'] * u.cm
      temperature = ovrtable['Te']  * u.K
      density     = ovrtable['hden']  / u.cm**3

      iondens     = np.zeros((depth.size,self.myatoms.nion))  / u.cm**3
      anumarray   = np.unique(self.myatoms.anum)
      for a in range(anumarray.size):
        el = anumarray[a]
        (elemname, elemcode) = self.myatoms.cloudyelem(el)
        ionarray = np.unique(np.extract(self.myatoms.anum == el, self.myatoms.ion))
        if os.path.exists(rootname+f".{elemcode}"):
          elemtable = ascii.read(rootname+f".{elemcode}", format='commented_header', header_start=0, data_start=1)
          for ion in ionarray:
            ionindex1 = self.myatoms.getspecies(el,ion)[0]
            ionindex2 = np.extract(self.myatoms.idx == ionindex1, range(self.myatoms.nion))[0]
            iondens[:, ionindex2 ] = elemtable[ self.myatoms.plusstr[ ionindex1 ] ] / u.cm**3

      self.data = Table(data  = [ depth,  temperature,  density,  iondens],
                        names = ["depth","temperature","density","iondens"])
      self.data.write(self.basedir+self.slash+"Cloudy_runs"+self.slash+f"{rootname}.fits", format="fits", overwrite=True)

    if os.getcwd() == self.basedir+self.slash+"Cloudy_runs":
      os.chdir("../")



