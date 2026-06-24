import numpy as np
import os
from astropy import units as u

class readpars:
    def __init__(self, inputfile):
        if os.path.exists(inputfile):
            with open(inputfile, 'r') as f:
                print(f"Reading parameters from {inputfile}")
                lines = f.readlines()
        for line in lines:
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
        except NameError:
            print(f"Data path not defined in {inputfile}")
            if os.uname().nodename == 'astronomy':
                self.datapath    = "/mnt/data/AALSynth_data/"
            elif os.name == 'nt':
                self.datapath = "D:\\ganguly\\AALSynth2\\AALSynth_data\\"
            else:
                self.datapath = "./"
  
        try:
           print(f"Cloudy path = {self.cloudypath}")
        except NameError:
            print(f"Cloudy path not defined in {inputfile}")
            if os.uname().nodename == 'astronomy':
                self.cloudypath  = "/mnt/data/c23.01/source/"
            elif os.name == 'nt':
                self.cloudypath = self.datapath
            else:
                self.cloudypath = "./"

        try:
           print(f"Number of processors = {self.nproc}")
        except NameError:
           print(f"Number of processors not defined in {inputfile}")

        try:
           print(f"Verbose = {self.verbose}")
        except NameError:
           print(f"Verbose not defined in {inputfile}")
           self.verbose = False

        try:
           print(f"Calculate wind = {self.calcwind}")
        except NameError:
           print(f"Calculate wind not defined in {inputfile}")
           self.calcwind = False

        try:
           print(f"Calculate absorbing clouds = {self.calcabscl}")
        except NameError:
           print(f"Calculate absorbing clouds not defined in {inputfile}")
           self.calcabscl = True

        ###########################################################################
        print("Spectral synthesis parameters:")
        try:
            print(f"Min lox = {self.minlox}")
        except NameError:
            print(f"Min lox not defined in {inputfile}")
            self.minlox = 9.9

        try:
           print(f"Max chi = {self.maxchi}")
        except NameError:
           print(f"Max chi not defined in {inputfile}")
           self.maxchi = 1.5    

        try:
           print(f"Min step = {self.minstep}")
        except NameError:
           print(f"Min step not defined in {inputfile}")
           self.minstep = 1.0e-5

        try:
           self.vlo *= (u.km/u.s)
           print(f"vlo = {self.vlo}")
        except NameError:
           print(f"vlo not defined in {inputfile}")
           self.vlo = -2000.0 * (u.km/u.s)

        try:
           self.vhi *= (u.km/u.s)
           print(f"vhi = {self.vhi}")
        except NameError:
           print(f"vhi not defined in {inputfile}")
           self.vhi = -1000.0 * (u.km/u.s)

        try:
           self.vres *= (u.km/u.s)
           print(f"vres = {self.vres}")
        except NameError:
           print(f"vres not defined in {inputfile}")
           self.vres = 3.0 * (u.km/u.s)

        ###########################################################################
        print("Black hole parameters:")
        try:
           print(f"Sbh = {self.sbh}")
        except NameError:
           print(f"Sbh not defined in {inputfile}")
           self.sbh = 0.01

        try:
           print(f"Mbh = {self.mbh}")
        except NameError:
           print(f"Mbh not defined in {inputfile}")
           self.mbh = 1.2e+8

        ###########################################################################
        print("Accretion disk parameters:")    
        try: 
           print(f"Mdot = {self.mdot} solar masses per year")
        except NameError:
           print(f"Mdot not defined in {inputfile}")
           self.mdot = 1.0e+5

        try:
           print(f"Viscosity parameter = {self.alpha}")
        except NameError:
           print(f"alpha not defined in {inputfile}")
           self.alpha = 0.1

        try:
           print(f"Number of disk annuli = {self.nr}")
        except NameError:
           print(f"Nr not defined in {inputfile}")
           self.nr = 300

        try:
           print(f"Disk inner radius = {self.rlo} rg")
        except NameError:
           print(f"rlo not defined in {inputfile}")
           self.rlo = 0.1

        try:
           print(f"Disk outer radius = {self.rhi} rg")
        except NameError:
           print(f"rhi not defined in {inputfile}")
           self.rhi = 1.0e+5

        try:
           print(f"dtheta_fac = {self.dtheta_fac}")
        except NameError:
           print(f"dtheta_fac not defined in {inputfile}")
           self.dtheta_fac = 0.5

        ###########################################################################
        print("Observer parameters:")
        try:
            print(f"Quasar redshift = {self.zqso}")
        except NameError:
            print(f"zqso not defined in {inputfile}")
            self.zqso = 0.114272

        try:
            self.raqso *= u.deg
            print(f"RAqso = {self.raqso}")
        except NameError:
            print(f"raqso not defined in {inputfile}")
            self.raqso = 175.19958333333 * u.deg

        try:
            self.decqso *= u.deg
            print(f"DECqso = {self.decqso}")
        except NameError:
            print(f"decqso not defined in {inputfile}")
            self.decqso = 46.36805555556 * u.deg

        try:
            print(f"inclination = {self.inclination}")
        except NameError:
            print(f"inclination not defined in {inputfile}")
            self.inclination = 20.0 * u.degree

        ###########################################################################
        print("Absorbing cloud parameters:")
        try:
            print(f"Absorbing cloud file = {self.abscloudfile}")
        except NameError:
            print(f"abscloudfile not defined in {inputfile}")
            self.abscloudfile = "Clouds.fits"
