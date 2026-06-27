import numpy as np
import os
from astropy import units as u
from tqdm    import tqdm

class readpars:
    def __init__(self, inputfile):
         if os.path.exists(inputfile):
            with open(inputfile, 'r') as f:
               print(f"Reading parameters from {inputfile}")
               lines = f.readlines()
               for line in tqdm(lines):
                  if line.startswith('#'):
                      continue
                  key, value = line.split('=')
                  key = key.strip()
                  value = value.strip()
                  try:
                      value = eval(value)
                  except:
                      pass
                  setattr(self, key, value)

        ###########################################################################
         print("Defining program parameters")
         try:
            print(f"Data path = {self.datapath}")
         except AttributeError:
            print(f"Data path not defined in {inputfile}")
            if os.uname().nodename == 'astronomy':
                self.datapath    = "/mnt/data/AALSynth_data/"
            elif os.name == 'nt':
                self.datapath = "D:\\ganguly\\AALSynth2\\AALSynth_data\\"
            else:
                self.datapath = "./"
  
         try:
            print(f"Cloudy path = {self.cloudypath}")
         except AttributeError:
            print(f"Cloudy path not defined in {inputfile}")
            if os.uname().nodename == 'astronomy':
                self.cloudypath  = "/mnt/data/c23.01/source/"
            elif os.name == 'nt':
                self.cloudypath = self.datapath
            else:
                self.cloudypath = "./"

         try:
            print(f"Number of processors = {self.nproc}")
         except AttributeError:
            print(f"Number of processors not defined in {inputfile}")
            self.nproc = 20

         try:
            print(f"Verbose = {self.verbose}")
         except AttributeError:
            print(f"Verbose not defined in {inputfile}")
            self.verbose = False

         try:
            print(f"Calculate absorbing clouds = {self.calcabscl}")
         except AttributeError:
            print(f"Calculate absorbing clouds not defined in {inputfile}")
            self.calcabscl = True

         try:
            print(f"Calculate wind = {self.calcwind}")
         except AttributeError:
            print(f"Calculate wind not defined in {inputfile}")
            self.calcwind = False

        ###########################################################################
         print("Spectral synthesis parameters:")
         try:
            print(f"Min lox = {self.minlox}")
         except AttributeError:
            print(f"Min lox not defined in {inputfile}")
            self.minlox = 9.9

         try:
            self.vlo *= (u.km/u.s)
            print(f"vlo = {self.vlo}")
         except AttributeError:
            print(f"vlo not defined in {inputfile}")
            self.vlo = -2000.0 * (u.km/u.s)

         try:
            self.vhi *= (u.km/u.s)
            print(f"vhi = {self.vhi}")
         except AttributeError:
            print(f"vhi not defined in {inputfile}")
            self.vhi = -1000.0 * (u.km/u.s)

         try:
            self.vres *= (u.km/u.s)
            print(f"vres = {self.vres}")
         except AttributeError:
            print(f"vres not defined in {inputfile}")
            self.vres = 3.0 * (u.km/u.s)

         try:
            print(f"anum = {self.anum}")
         except AttributeError:
            print(f"anum not defined in {inputfile}")
            self.anum = np.array([1, 7, 8])

         try:
            print(f"ion = {self.ion}")
         except AttributeError:
            print(f"ion not defined in {inputfile}")
            self.ion = np.array([1, 5, 6])

         try:
            print(f"trandx = {self.trandx}")
         except AttributeError:
            print(f"trandx not defined in {inputfile}")
            self.trandx = np.array([0, 0, 1])

         try:
            print(f"Plot code = {self.plot_code}")
         except AttributeError:
            print(f"Plot code not defined in {inputfile}")
            self.plot_code = ["g", "r", "m"]

         try:
            print(f"Gauss-Legendre radial bins = {self.gaussleg_nr}")
         except AttributeError:
            print(f"Gauss-Legendre radial bins not included in {inputfile}")
            self.gaussleg_nr = 150

         try:
            print(f"Gauss-Legendre azimuthal bins = {self.gaussleg_ntheta}")
         except AttributeError:
            print(f"Gauss-Legendre azimuthal bins not included in {inputfile}")
            self.gaussleg_ntheta = 60

        ###########################################################################
         print("Black hole parameters:")
         try:
            print(f"Sbh = {self.sbh}")
         except AttributeError:
            print(f"Sbh not defined in {inputfile}")
            self.sbh = 0.01

         try:
            self.mbh *= u.M_sun
            print(f"Mbh = {self.mbh}")
         except AttributeError:
            print(f"Mbh not defined in {inputfile}")
            self.mbh = 1.2e+8 * u.M_sun

         ###########################################################################
         print("Accretion disk parameters:")    
         try: 
            self.mdot *= u.M_sun / u.yr
            print(f"Mdot = {self.mdot} solar masses per year")
         except AttributeError:
            print(f"Mdot not defined in {inputfile}")
            self.mdot = 0.72 * u.M_sun / u.yr

         try:
            print(f"Viscosity parameter = {self.alpha}")
         except AttributeError:
            print(f"alpha not defined in {inputfile}")
            self.alpha = 0.1

         try:
            print(f"Number of disk annuli = {self.nr}")
         except AttributeError:
            print(f"nr not defined in {inputfile}")
            self.nr = 300

         try:
            print(f"Disk inner radius = {self.rlo} rg")
         except AttributeError:
            print(f"rlo not defined in {inputfile}")
            self.rlo = 0.1

         try:
            print(f"Disk outer radius = {self.rhi} rg")
         except AttributeError:
            print(f"rhi not defined in {inputfile}")
            self.rhi = 1.0e+5

         try:
            print(f"dtheta_fac = {self.dtheta_fac}")
         except AttributeError:
            print(f"dtheta_fac not defined in {inputfile}")
            self.dtheta_fac = 0.5

         ###########################################################################
         print("Observer parameters:")
         try:
            print(f"Quasar redshift = {self.zqso}")
         except AttributeError:
            print(f"zqso not defined in {inputfile}")
            self.zqso = 0.114272

         try:
            self.raqso *= u.deg
            print(f"RAqso = {self.raqso}")
         except AttributeError:
            print(f"raqso not defined in {inputfile}")
            self.raqso = 175.19958333333 * u.deg

         try:
            self.decqso *= u.deg
            print(f"DECqso = {self.decqso}")
         except AttributeError:
            print(f"decqso not defined in {inputfile}")
            self.decqso = 46.36805555556 * u.deg

         try:
            print(f"inclination = {self.inclination}")
         except AttributeError:
            print(f"inclination not defined in {inputfile}")
            self.inclination = 20.0 * u.degree

         ###########################################################################
         print("Absorbing cloud parameters:")
         try:
            print(f"Absorbing cloud file = {self.abscloudfile}")
         except AttributeError:
            print(f"abscloudfile not defined in {inputfile}")
            self.abscloudfile = "Clouds.fits"

         ###########################################################################
         print("Data files:")
         try:
            print(f"Quasar name = {self.qname}")
         except AttributeError:
            print(f"qname not defined in {inputfile}")
            self.qname = "WISEA J114047"
         
         try:
            print(f"Quasar file root = {self.qfileroot}")
         except AttributeError:
            print(f"qfileroot not defined in {inputfile}")
            self.qfileroot = "WISEA_J114047"

         try:
            print(f"Redo spline = {self.redospline}")
         except AttributeError:
            print(f"redospline not defined in {inputfile}")
            self.redospline = False

         ###########################################################################
         print("Optimization parameters:")
         try:
            print(f"Max chi = {self.maxchi}")
         except AttributeError:
            print(f"Max chi not defined in {inputfile}")
            self.maxchi = 1.5    

         try:
            print(f"Min step = {self.minstep}")
         except AttributeError:
            print(f"Min step not defined in {inputfile}")
            self.minstep = 1.0e-5

         try:
            print(f"dstep = {self.dstep}")
         except AttributeError:
            print(f"dstep not not defined in {inputfile}")
            self.dstep = 0.15

         try:
            if self.mcmin:
               method_string = "Monte Carlo"
            else:
               method_string = "SciPy Least-Squares"
            print(f"Minimization method: {method_string}")
         except AttributeError:
            print(f"Minimization method not chosen in {inputfile}")
            self.mcmin = True

         try:
            print(f"Maximum iterations per parameter for Monto Carlo method: {self.maxiter}")
         except AttributeError:
            print(f"Maximum iterations per parameter not defined in {inputfile}")
            self.maxiter = 100

         try:
            print(f"First_try = {self.first_time}")
         except AttributeError:
            print(f"First_try not set in {inputfile}")
            self.first_time = True
