import os
import numpy             as np

from astropy.io            import ascii
from doppler               import calcvel
from doppler               import calcwave
from astropy               import units as u
from astropy.modeling.functional_models import Voigt1D


# The atomic class defines the database of atomic transitions from Verner, Barthel, & Tytler and Morton.
######################################################
# User inputs for __init__ to define the class:
#        self.basedir = string location of database
#        wlo          = Lowest  wavelength for the spectrum (Angstroms)
#        whi          = Highest wavelength for the spectrum (Angstroms)
#        minlox       = Minimum Line Observability Index (Hellsten et al) to keep transitions
#
######################################################
# __init__ sets the following:
#        self.species = Species name (no spaces)
#        self.wave    = Transition wavelength (Angstroms)
#        self.f       = Transition oscillator strength
#        self.gamma   = Transition natural broadening parameter
#        self.ip      = Ion ionization potential
#        self.anum    = Atomic number
#        self.ion     = Ion stange (1=neutral, 2=singly ionized, etc.)
#        self.abund   = Elemental abundance (12 + [X/H]) from Asplund et al.
#        self.amass   = Atomic mass (u)
#        self.lox     = Line Observability Index = self.abund + np.log10(self.f*self.wave + 1.0e-30)
#        self.nion    = Number of unique ions in database
#        self.idx     = Index of the first occurence of each unique ion in the database (coresponding to the strongest transition)
#        self.specstr = Species name (w/ spaces)
#        self.flam    = self.f * self.wave
#
###############################################################################################
#    def getspecies:
#        Input: anum = Atomic Number
#               ion  = Ion stage
#        Return: array of indices corresponding to all transition of that anum and ion
#
###############################################################################################
#    def atomstr:
#        Input: anum = Atomic Number
#        Return: String of the element period table symbol
#
###############################################################################################
#    def ionstr
#        Input: inum = Ion stage
#        Return: String with ocrresponding Roman numberal (for spectroscopist notation)
#
###############################################################################################
#    def tauion:
#        Input wave = wavelength array (Angstroms)
#              anum = Atomic number
#              ion  = Ion stage
#              tau0 = Opitcal depths at line center
#              b    = b-parameters (typcally thermal broadening, km/s)
#              v    = cloud velocities (km/s)
#        Return: line optical depth array of shape (wave.size,nabs) where nabs is the number of clouds
#
###############################################################################################

class atomic:
    ###############################################################################################

    def __init__(self,basedir,wlo,whi,minlox=8.5):
        datfile = "atoms4.dat"
        self.minlox = minlox
        if (os.path.exists(basedir+"/"+datfile)):
            data = ascii.read(basedir+"/"+datfile,format='no_header')
            self.species = np.copy(data['col1'])
            self.wave    = np.copy(data['col2']) * u.Angstrom
            self.f       = np.copy(data['col3'])
            self.gamma   = np.copy(data['col4']) / u.s #units????
            self.ip      = np.copy(data['col5']) * u.eV
            self.anum    = np.copy(data['col6'])
            self.ion     = np.copy(data['col7'])
            self.abund   = np.copy(data['col8'])
            self.amass   = np.copy(data['col9']) * u.u

            self.lox = self.abund + np.log10(self.f*self.wave.value + 1.0e-30)

            ldx = np.extract((self.lox > minlox) & (self.wave > wlo) & (self.wave < whi) & (self.ip >= 13.5 * u.eV), range(self.wave.size))

            self.species = self.species[ldx]
            self.wave    = self.wave[ldx]
            self.f       = self.f[ldx]
            self.gamma   = self.gamma[ldx]
            self.ip      = self.ip[ldx]
            self.anum    = self.anum[ldx]
            self.ion     = self.ion[ldx]
            self.abund   = self.abund[ldx]
            self.amass   = self.amass[ldx]
            self.lox     = self.lox[ldx]
            self.lsdx    = np.argsort(self.lox)

            ions,idx = np.unique(100*self.anum + self.ion, return_index=True)
            self.ions = ions
            self.nion = ions.size
            self.idx = idx

            self.specstr = np.copy(self.species)
            self.plusstr = np.copy(self.species)
            for i in range(self.species.size):
                self.specstr[i] = self.atomstr(self.anum[i])+' '+self.ionstr(self.ion[i])
                plusses = ''
                if self.ion[i] == 1:
                    plusses = ''
                else:
                    plusses += '+'
                    if self.ion[i] > 2:
                        plusses += f"{self.ion[i]-1}"
                self.plusstr[i]  = self.atomstr(self.anum[i])+plusses

            self.flam = self.f * self.wave

        else:
            print(basedir+'/'+datfile," does not exist")

    ###############################################################################################

    def getspecies(self,anum,ion):
        return np.extract((self.anum == anum) & (self.ion == ion), range(self.anum.size))

    ###############################################################################################

    def atomstr(self,anum):
      if anum == 1:
        return 'H'
      elif anum == 2:
        return 'He'
      elif anum == 3:
        return 'Li'
      elif anum == 4:
        return 'Be'
      elif anum == 5:
        return 'B'
      elif anum == 6:
        return 'C'
      elif anum == 7:
        return 'N'
      elif anum == 8:
        return 'O'
      elif anum == 9:
        return 'F'
      elif anum == 10:
        return 'Ne'
      elif anum == 11:
        return 'Na'
      elif anum == 12:
        return 'Mg'
      elif anum == 13:
        return 'Al'
      elif anum == 14:
        return 'Si'
      elif anum == 15:
        return 'P'
      elif anum == 16:
        return 'S'
      elif anum == 17:
        return 'Cl'
      elif anum == 18:
        return 'Ar'
      elif anum == 19:
        return 'K'
      elif anum == 20:
        return 'Ca'
      elif anum == 21:
        return 'Sc'
      elif anum == 22:
        return 'Ti'
      elif anum == 23:
        return 'V'
      elif anum == 24:
        return 'Cr'
      elif anum == 25:
        return 'Mn'
      elif anum == 26:
        return 'Fe'
      elif anum == 27:
        return 'Co'
      elif anum == 28:
        return 'Ni'
      elif anum == 29:
        return 'Cu'
      elif anum == 30:
        return 'Zn'
      else:
        return 'nope'

    def ionstr(self,inum):
        if inum == 1:
            return 'I'
        elif inum == 2:
            return 'II'
        elif inum == 3:
            return 'III'
        elif inum == 4:
            return 'IV'
        elif inum == 5:
            return 'V'
        elif inum == 6:
            return 'VI'
        else:
            return 'nope'

    ###############################################################################################
    def tauion(self,wave,anum,ion,tau0,b,v):
        idx = np.where((self.anum == anum) & (self.ion == ion))[0]
        tau_profile = np.zeros(wave.size)
        for i in idx:
            vv  = np.squeeze(calcvel(wave.to(u.Angstrom), self.wave[i].to(u.Angstrom))).to(u.cm/u.s)
            dvel = ((vv - v) / b).decompose()
            dvel_mask = np.fabs(dvel) < 100.0
            if self.gamma[i].value > 0:
                a = (self.gamma[i] * self.wave[i] / b).decompose()
                V1   = Voigt1D(x_0         = 0,           # These are the parameters of a normalized Voigt profile centered at 0
                               amplitude_L = 1/(np.pi * a),
                               fwhm_L      = 2 * a,
                               fwhm_G      = 2 * Voigt1D.sqrt_ln2
                               )
                V2 = V1(dvel[dvel_mask])
                if len(V2.shape) > 1:
                    V2 = np.mean(V2, axis=1)
                tau_profile[dvel_mask] += V2 * self.flam[i]/np.max(self.flam[idx])
            else:
                tau_profile[dvel_mask] += np.exp(-np.square(dvel[dvel_mask])) * self.flam[i]/np.max(self.flam[idx])

        return tau0 * tau_profile

    def tauion_old(self,wave,anum,ion,tau0,b,v):
        nabs  = v.size
        bb    = np.broadcast_to(b.to(u.cm/u.s).value, (wave.size,nabs)) * (u.cm/u.s)
        vv    = np.broadcast_to(v.to(u.cm/u.s).value, (wave.size,nabs)) * (u.cm/u.s)
        idx = np.extract((self.anum == anum) & (self.ion == ion), range(self.anum.size))
        tau = np.zeros((wave.size,nabs))
        for i in idx:
            vvv  = np.squeeze(calcvel(wave.to(u.Angstrom), self.wave[i].to(u.Angstrom))).to(u.cm/u.s)
            dvel = (np.transpose(np.broadcast_to(vvv.value, (nabs,wave.size))) * (u.cm/u.s) - vv) / bb
            dvel_mask = np.fabs(dvel) < 100.0
            if self.gamma[i].value > 0:
                a = (self.gamma[i] * self.wave[i] / bb[dvel_mask]).decompose()
                V1   = Voigt1D(x_0         = 0,
                               amplitude_L = 1/(np.pi * a),
                               fwhm_L      = 2 * a,
                               fwhm_G      = 2 * Voigt1D.sqrt_ln2
                               )
                V2 = V1(dvel[dvel_mask])
                tau[dvel_mask] += V2 * self.flam[i]/np.max(self.flam[idx])
            else:
                tau[dvel_mask] += np.exp(-np.square(dvel[dvel_mask])) * self.flam[i]/np.max(self.flam[idx])

        return np.broadcast_to(tau0, (wave.size,nabs)) * tau


    def tausingle(self,wave,idx,tau0,b,v):
        nabs      = v.size
        bb    = np.broadcast_to(b.to(u.cm/u.s).value, (wave.size,nabs)) * (u.cm/u.s)
        vv    = np.broadcast_to(v.to(u.cm/u.s).value, (wave.size,nabs)) * (u.cm/u.s)
        tau_base  = np.zeros((wave.size,nabs))

        vvv  = np.squeeze(calcvel(wave.to(u.Angstrom), self.wave[idx].to(u.Angstrom))).to(u.cm/u.s)
        dvel      = (np.transpose(np.broadcast_to(vvv.value, (nabs,wave.size))) * (u.cm/u.s) - vv) / bb

        ion_mask = (self.anum == self.anum[idx]) & (self.ion == self.ion[idx])

        if self.gamma[idx].value > 0:
            a = (self.gamma[idx] * self.wave[idx] / bb).decompose()
            V1   = Voigt1D(x_0         = 0,
                           amplitude_L = 1/(np.pi * a),
                           fwhm_L      = 2 * a,
                           fwhm_G      = 2 * Voigt1D.sqrt_ln2
                           )
            tau_base += V1(dvel) * self.flam[idx] / np.max(self.flam[ion_mask])
        else:
            tau_base += np.exp(-np.square(dvel)) * self.flam[idx] / np.max(self.flam[ion_mask])

        return np.broadcast_to(tau0, (wave.size,nabs)) * tau_base
        
    ###############################################################################################

    def cloudyelem(self,anum):
      if anum == 1:
        return (  "hydrogen", "hydr")
      elif anum == 2:
        return (    "helium", "heli")
      elif anum == 3:
        return (   "lithium", "lith")
      elif anum == 4:
        return ( "beryilium", "bery")
      elif anum == 5:
        return (     "boron", "boro")
      elif anum == 6:
        return (    "carbon", "carb")
      elif anum == 7:
        return (  "nitrogen", "nitr")
      elif anum == 8:
        return (    "oxygen", "oxyg")
      elif anum == 9:
        return (  "fluorine", "fluo")
      elif anum == 10:
        return (      "neon", "neon")
      elif anum == 11:
        return (    "sodium", "sodi")
      elif anum == 12:
        return ( "magnesium", "magn")
      elif anum == 13:
        return (  "aluminum", "alum")
      elif anum == 14:
        return (   "silicon", "sili")
      elif anum == 15:
        return ("phosphorus", "phos")
      elif anum == 16:
        return (   "sulpher", "sulp")
      elif anum == 17:
        return (  "chlorine", "chlo")
      elif anum == 18:
        return (     "argon", "argo")
      elif anum == 19:
        return ( "potassium", "pota")
      elif anum == 20:
        return (   "calcium", "calc")
      elif anum == 21:
        return (  "scandium", "scan")
      elif anum == 22:
        return (  "titanium", "tita")
      elif anum == 23:
        return (  "vanadium", "vana")
      elif anum == 24:
        return (  "chromium", "chro")
      elif anum == 25:
        return ( "manganese", "mang")
      elif anum == 26:
        return (      "iron", "iron")
      elif anum == 27:
        return (    "cobalt", "coba")
      elif anum == 28:
        return (    "nickel", "nick")
      elif anum == 29:
        return (    "copper", "copp")
      elif anum == 30:
        return (      "zinc", "zinc")
      else:
        return (    "nuh-uh", "nope")

###############################################################################################

