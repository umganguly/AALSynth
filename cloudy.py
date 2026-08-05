import numpy             as np
import os
import re
import matplotlib.pyplot as plt
import subprocess
import sys

from astropy                 import constants as const
from astropy                 import units     as u
from astropy.io              import fits, ascii
from astropy.table           import Table

# The cloudy class is meant to be an interface to running and reading Cloudy
class cloudy:
  def __init__(self, Abs_or_Em,                 # 0 = emission, 1 = absorption
               mypars,                                  # instance of readpars
               myatoms,                                 # instance of atomic class
               ionspecfreq, ionspecflux,                # ionizing spectrum
               rhoindex=0.0, logrhoscale=16, logrho0=2, # density parameters
               logZ = 0.0,
               rstar = 10.0, zstar = 10.0,
               verbose = False,
               ntabs = 0
               ):

    self.mypars = mypars
    self.myatoms = myatoms
    self.Abs_or_Em = Abs_or_Em

    os.chdir(self.mypars.datapath+"Cloudy_runs")
    lognuFnu = np.interp((const.Ryd).to(u.Hz, equivalencies=u.spectral()), ionspecfreq, np.log10((ionspecfreq * ionspecflux).value))
    if Abs_or_Em == 0: # Emission - Has not been developed yet
      self.rootname = f"EM-hden{logrho0:.2f}-nuFnu{lognuFnu:.2f}-rstar{rstar:.2f}-zstar{zstar:.2f}"
    else: # Absorption
      self.rootname = f"ABS-rho0{logrho0}-index{rhoindex}-scale{logrhoscale}-logZ{logZ}-zcl{zstar}"

    fitsfile = self.rootname+".fits"

    if verbose: print("\t" * ntabs + f"Looking for cloudy in {self.mypars.cloudypath}")
    if os.path.exists(f"{self.mypars.cloudypath}/cloudy.exe"):
      if verbose: print("\t" * ntabs + f"                      {self.mypars.cloudypath}/cloudy.exe exists!")
      # For the parameters given, do we need to run Cloudy or do we have files already?

      if not os.path.exists(fitsfile): # if the Cloudy files don't exist, then we need to run Cloudy
        if verbose: print("\t" * ntabs + f"Generating {self.rootname}.in file for Cloudy input")
        # Write out the SED file:
        with open(self.mypars.datapath+f"Cloudy_runs/{self.rootname}.sed", "w") as f:
          f.write(f"# SED for {self.rootname}\n")
          RydinHz = (const.Ryd).to(u.Hz, equivalencies=u.spectral())
          fu = u.erg / (u.s * u.cm * u.cm * u.Hz)
          for i in range(ionspecfreq.size):
            if self.Abs_or_Em == 0:
              writestr = f"{(ionspecfreq[i].to(u.Hz)/RydinHz).value:.15e} {ionspecflux[i] * ionspecfreq[i] / np.max(ionspecflux * ionspecfreq):.15e} nuFnu\n"
              f.write(writestr)
            else:
              f.write(f"{(ionspecfreq[i].to(u.Hz)/RydinHz).value:.15e} {ionspecflux[i].to(fu).value:.15e}\n")
          f.write("*****\n")

        # Write out the commands for Cloudy:
        with open(self.mypars.datapath+f"Cloudy_runs/{self.rootname}.in", "w") as f:
          f.write(f"table SED \"{self.rootname}.sed\"\n")
          f.write(f"nuF(nu) = {lognuFnu:.2f}\n")
          f.write(f"table HM05 redshift 0.4\n")
          f.write("CMB redshift 0.4\n")
          f.write("Cosmic rays background\n")

          f.write(f"metals {logZ} log\n")

          f.write("stop temperature 3 K linear\n")
          f.write("iterate\n")
          f.write("print last iteration\n")

          f.write(f"set save prefix \"{self.rootname}\"\n")
          f.write(f"save overview \".ovr\" last iteration\n")

          if self.Abs_or_Em == 0: # Line-emitting gas
            f.write(f"hden {logrho0}\n")
            f.write(f"stop thickness {logrhoscale}\n")
            f.write(f"save lines, array \".lin\" last iteration \n")
            f.write(f"save species column densities \".col\" all last iteration\n")
          else: # Absorbing clouds
            f.write(f"globule density={logrho0}, depth={logrhoscale}, power={rhoindex}\n") # Density law
            f.write(f"stop thickness {logrhoscale-0.1}\n")

            for el in np.unique(self.myatoms.anum): # Elemental/ionic number densities
              (elemname, elemcode) = self.myatoms.cloudyelem(el)
              f.write(f"save element {elemname} \".{elemcode}\" density last\n")

        if not os.path.exists(self.rootname+".out") or os.path.getsize(self.rootname+".out") < 10000:
          # To run a C program in Python:
          if verbose: print("\t" * ntabs + "Running Cloudy...")
          try:
            subprocess.run([f"{self.mypars.cloudypath}cloudy.exe", f"{self.rootname}.in"], capture_output=True, text=True, check=True)
            self.cloudyran = True
          except subprocess.CalledProcessError as e:
            print("\n\n")
            print("\t" * ntabs + f"Error with Cloudy sim for {self.rootname}?")
            print("\t" * ntabs + f"cloudy.__init__: Execution error: {e.returncode}")
            print("\t" * ntabs + "cloudy.__init__: STDOUT:", e.output)
            print("\t" * ntabs + "cloudy.__init__: STDERR:", e.stderr)
            print("\n\n")
            self.cloudyran = False
            self.plotsed()
            input(f"Error: Why did Cloudy crash for {self.rootname}?")
            
          cloudy_output = subprocess.run(["tail", "--lines=10", f"{self.rootname}.out"], capture_output=True)
          if verbose:
            for clo in cloudy_output.stdout.splitlines():
              print("\t" * ntabs + f"{clo.decode('utf-8')}")

      else:
        if verbose: print("\t" * ntabs + f"{self.rootname} files exist!")
        self.cloudyran = True

    elif verbose:
      print("\t" * ntabs + "cloudy.__init__: Can't find Cloudy!")

    ovrname = self.mypars.datapath+f"Cloudy_runs/{self.rootname}.ovr"
    if os.path.exists(ovrname):
      if self.Abs_or_Em == 0: # Line-emitting gas
        self._writeemfits()
      else:
        self._writeabsfits()
      self._cleanup()

    fitsfile = self.mypars.datapath+f"Cloudy_runs/{self.rootname}.fits"
    if os.path.exists(fitsfile):
      if verbose:
        print("\t" * ntabs + f"Reading in {fitsfile}")
      self._readcloudy()
    else:
      print("\t" * ntabs + f"Could not find the fits file {fitsfile}")
      self.cloudyran = False

    if os.getcwd() == self.mypars.datapath+"Cloudy_runs":
      os.chdir("../")

  ######################################################
  def _cleanup(self):

    try:
      subprocess.run(["rm", f"{self.mypars.datapath}Cloudy_runs/{self.rootname}.in"])
      subprocess.run(["rm", f"{self.mypars.datapath}Cloudy_runs/{self.rootname}.out"])
      subprocess.run(["rm", f"{self.mypars.datapath}Cloudy_runs/{self.rootname}.ovr"])
    except:
      pass

    if self.Abs_or_Em == 0: # Line-emitting gas
      try:
        subprocess.run(["rm", f"{self.mypars.datapath}Cloudy_runs/{self.rootname}.lin"])
        subprocess.run(["rm", f"{self.mypars.datapath}Cloudy_runs/{self.rootname}.col"])
      except:
        pass
    else:
      for el in np.unique(self.myatoms.anum): # Elemental/ionic number densities
        (elemname, elemcode) = self.myatoms.cloudyelem(el)
        try:
          subprocess.run(["rm", f"{self.mypars.datapath}Cloudy_runs/{self.rootname}.{elemcode}"])
        except:
          pass

  ######################################################
  def _readcloudy(self):
    # Now we need to read in the Cloudy outputs...
    fitsname = self.mypars.datapath+f"Cloudy_runs/{self.rootname}.fits"
    if os.path.exists(fitsname):
      if self.Abs_or_Em == 0: # Line-emitting gas
        with fits.open(fitsname) as hdul:
          self.ionization_parameter = float(hdul[1].header['ION_PARM'] )
          self.temperature          = float(hdul[1].header['TEMPERAT'] ) * u.K
          self.density              = float(hdul[1].header[ 'DENSITY'] ) / u.cm**3
          self.line_array           = hdul[1].data
          self.column_density_array = hdul[2].data / u.cm**2
      else: # Absorbng gas
        data = Table.read(fitsname, format="fits")
        self.depth       = np.copy(data['depth']) * u.cm
        self.temperature = np.copy(data['temperature']) * u.K
        self.density     = np.copy(data['density']) / u.cm**3
        self.iondens     = np.copy(data['iondens']) / u.cm**3

      self.cloudyran = True
    else:
      self.cloudyran = False
      self.depth = np.array([])

  ######################################################
  def _writeabsfits(self):
    if os.getcwd() == self.mypars.datapath:
      os.chdir("Cloudy_runs")
    # Now we need to read in the Cloudy outputs...
    if os.path.exists(f"{self.rootname}.ovr"):
      ovrtable = ascii.read(f"{self.rootname}.ovr", format='commented_header', header_start=0, data_start=1)
      depth       = ovrtable['depth'] * u.cm
      temperature = ovrtable['Te']  * u.K
      density     = ovrtable['hden']  / u.cm**3

      iondens     = np.zeros((depth.size,self.myatoms.nion))  / u.cm**3
      anumarray   = np.unique(self.myatoms.anum)
      for a in range(anumarray.size):
        el = anumarray[a]
        (elemname, elemcode) = self.myatoms.cloudyelem(el)
        ionarray = np.unique(np.extract(self.myatoms.anum == el, self.myatoms.ion))
        if os.path.exists(self.rootname+f".{elemcode}"):
          elemtable = ascii.read(self.rootname+f".{elemcode}", format='commented_header', header_start=0, data_start=1)
          for ion in ionarray:
            ionindex1 = self.myatoms.getspecies(el,ion)[0]
            ionindex2 = np.extract(self.myatoms.idx == ionindex1, range(self.myatoms.nion))[0]
            iondens[:, ionindex2 ] = elemtable[ self.myatoms.plusstr[ ionindex1 ] ] / u.cm**3

      self.data = Table(data  = [ depth,  temperature,  density,  iondens],
                        names = ["depth","temperature","density","iondens"])
      self.data.write(self.mypars.datapath+f"Cloudy_runs/{self.rootname}.fits", format="fits", overwrite=True)

    if os.getcwd() == self.mypars.datapath+"Cloudy_runs":
      os.chdir("../")

  ######################################################
  def _writeemfits(self):
    if os.getcwd() == self.mypars.datapath:
      os.chdir("Cloudy_runs")
    # Now we need to read in the Cloudy outputs...
    if os.path.exists(f"{self.rootname}.out"):
      pattern = re.compile(r"IONIZE PARMET:  U")
      with open(f"{self.rootname}.out", 'r') as f:
        for line in f:
          match = pattern.search(line)
          if match:
            ionization_parameter = line.split()[3]
    else:
      ionization_parameter = -999

    column_density_array = np.zeros(self.myatoms.photo_Z.size) /  u.cm**2
    if os.path.exists(f"{self.rootname}.ovr"):
      ovrtable = ascii.read(f"{self.rootname}.ovr", format='commented_header', header_start=0, data_start=1)
      temperature = np.average(ovrtable['Te'])  * u.K
      density     = np.average(ovrtable['hden'])  / u.cm**3

      line_array           = ascii.read(f"{self.rootname}.lin", format='commented_header', header_start=0, data_start=1, delimiter='\t', guess=False)

      listless = []
      input_file  = f"{self.rootname}.col"
      with open(input_file, "r", encoding="utf-8") as infile:
          for line in infile:
            cleaned_line = line.replace("^", "")
            cc_line      = cleaned_line.replace("#column density", "")
            listless.append(cc_line.split())

      column_density_dict = dict(zip(listless[0], listless[1]))

      for i in range(self.myatoms.photo_Z.size):
        specstr = self.myatoms.atomstr(self.myatoms.photo_Z[i])
        if self.myatoms.photo_Z[i] > self.myatoms.photo_N[i]:
          specstr += "+"
          if self.myatoms.photo_Z[i] - self.myatoms.photo_N[i] > 1:
            specstr += f"{self.myatoms.photo_Z[i] - self.myatoms.photo_N[i]}"
        try:
          column_density_array[i] = float(column_density_dict[specstr]) / u.cm**2
        except KeyError:
          print(f"Failing to find {specstr} in column_density_table:")
          print(column_density_dict)

    else:
      temperature = 2.7 * u.K
      density = 1.0e-10 / u.cm**3
      line_array = Table(names=('id', 'name', 'flux'))

    fitsfile = self.mypars.datapath+f"Cloudy_runs/{self.rootname}.fits"
    table_hdu = fits.BinTableHDU(data=line_array)
    table_hdu.header['ION_PARM'] = (ionization_parameter, 'Ionization Parameter (U)' )
    table_hdu.header['TEMPERAT'] = (   temperature.value,          'Temperature (K)' )
    table_hdu.header[ 'DENSITY'] = (       density.value,         'Density (cm**-3)' )
    hdul = fits.HDUList([fits.PrimaryHDU(), table_hdu])
    hdul.writeto(fitsfile, overwrite=True)
    try:
      column_density_table = Table([column_density_array],
                                   names = ['column_densities'] 
                                   )
      column_density_table.write(fitsfile, format="fits", append=True)
    except ValueError:
      print("Why ValueError?????")
      print(column_density_array)

    if os.getcwd() == self.mypars.datapath+"Cloudy_runs":
      os.chdir("../")

  ######################################################
  def plotsed(self):
    sedtab = ascii.read(self.mypars.datapath+f"Cloudy_runs/{self.rootname}.sed")

    plt.clf()
    plt.plot(sedtab['col1'], sedtab['col2]'])
    plt.xscale("log")
    plt.yscale("log")
    plt.show()