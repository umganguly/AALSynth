import os

import matplotlib.pyplot  as plt
import numpy              as np
import time               as tm

from astropy                 import constants as const
from astropy                 import units as u

from ntdisk                  import ntdisk

class corona:
    ######################################################
    def __init__(self, mydisk):
        self.mydisk = mydisk

    ######################################################
    def activate_lamppost(self):
        fu = (u.erg / (u.s * u.cm * u.cm * u.Hz))

        # The location of the lamppost should be on the z-axis at a height of r_ISCO
        self.lamp_r_cyl = 0.0
        self.lamp_z = self.mydisk.rstar[0] # rg
        print(f"\tPlacing lampost at r = {self.lamp_r_cyl} rg,   z = {self.lamp_z} rg")

        self.mydisk.robs = np.array([self.lamp_r_cyl])
        self.mydisk.zobs = np.array([self.lamp_z])
        self.mydisk.thetaobs = 0.0

        frequency = np.logspace(13,20,num=100) * u.Hz
        dfreq = np.copy(frequency)
        dfreq[1:-1] = 0.5 * (frequency[2:] - frequency[:-2])
        dfreq[0] = dfreq[0]
        dfreq[-1] = dfreq[-2]

        
        tempt1 = np.copy(self.mydisk.tempt1)
        done = False
        while not done:
            oldtempt1 = np.copy(self.mydisk.tempt1)
            
            # Determine the 2500A luminosity
            frequency2500 = np.array([((const.c.to(u.Angstrom * u.Hz)) / (2500.0 * u.Angstrom)).value]) * u.Hz
            fnudisk = 0.0 * fu
            for r in range(self.mydisk.nr):
                fnudisk += np.sum(self.mydisk.fnudiskannulus(frequency2500, r))
            self.Lnu2500A = 4.0 * np.pi * (self.lamp_z * self.mydisk.rg)**2 * fnudisk

            # Lusso & Risaliti (2016, ApJ, 819, 154)
            # alpha_ox = log(Lnu2keV/Lnu2500A) / log(nu2keV/nu2500A) = -0.385 log(Lnu2keV/Lnu2500A)
            self.alpha_ox = 0.154 * np.log10(self.Lnu2500A.to(u.erg/u.s/u.Hz).value) - 3.176

            # Luminosity of corona at 2 keV
            self.lamp_L_nu_2keV = self.Lnu2500A * 10.0**(self.alpha_ox / -0.385)
            self.lamp_nu_2keV = (2.0 * u.keV / const.h).to(u.Hz)

            prtstr  = f"\tUsing Lnu(2500A) = {self.Lnu2500A:.3e} (nuLnu(2500A) = {(self.Lnu2500A * frequency2500)[0].to(u.erg/u.s):.3e}) "
            prtstr += f"--> alpha_ox = {self.alpha_ox:.3f} --> Lnu(2keV) = {self.lamp_L_nu_2keV:.3e}"
            print(prtstr)

            # Shape of the hard X-ray spectrum, adapted from Laha et al. (2025, Frontiers in Astronomy and Space Sciences)
            mdotedd         = 4 * np.pi * const.G.cgs * self.mydisk.mbh / (0.1 * const.c.cgs * (const.sigma_T.cgs/const.u.cgs))
            Eddington_ratio = (self.mydisk.mdot / mdotedd).decompose()
            if Eddington_ratio > 0.01:
                photon_index = 0.41 * np.log10(Eddington_ratio) + 2.17
            else:
                photon_index = -0.09 * np.log10(Eddington_ratio) + 1.55
            self.lamp_alpha_x = 1.0 - photon_index

            cutoff_energy = (96/13) * const.m_e * const.c * const.c / ((photon_index + 0.5)**2 - (9/4))
            self.cutoff_freq   = (cutoff_energy / const.h).to(u.Hz)

            prtstr  = f"\tX-ray spectrum: nuLnu = {(self.lamp_L_nu_2keV * self.lamp_nu_2keV).to(u.erg/u.s):.3e} "
            prtstr += f"(E / {(const.h * self.lamp_nu_2keV).to(u.keV):.3f})^{self.lamp_alpha_x:.3f} "
            prtstr += f"exp(-E/{cutoff_energy.to(u.keV):.3f})"
            print(prtstr)

            print("\tModifying tau=1 surface temperature profile of disk...")
            for i in range(self.mydisk.rstar.size):
                added_flux = np.sum(self.fnu_lamppost(frequency, self.mydisk.rstar[i], self.mydisk.zt1[i]) * dfreq)
                self.mydisk.tempt1[i] = np.power(tempt1[i]**4 + added_flux / const.sigma_sb, 0.25)

            done = True
            if np.any(np.fabs(self.mydisk.tempt1 - oldtempt1) / oldtempt1 > 1.0e-3):
                done = False

    ######################################################
    # We should insert the flux incident from the lamppost. And then we can add in the flux from each of the annuli.
    # Google says the average photon index for hard X-ray emission is Gamma = 1.84.
    # Math says that this should be a spectral index of alpha_x = 1 - Gamma = -0.84
    # L_x = L_xo * (frequency/frequency_o)^alpha_x
    # In principle, we should actually use a truncated power law: ~ E^-Gamma exp(-E/Ec) Ec = cutoff energy
    # From Shaban et al. 2022
    # lamp_nu_2keV = (3.8 * u.keV)/const.h.to(u.keV/u.Hz)  # [0.5 - 7] keV band - split the difference
    # lamp_L_xo = (10.0**45.9) * (u.erg/u.s) / lamp_nu_2keV
    def fnu_lamppost(self, frequency, robs, zobs, # robs and zobs in units of rg.. they should be 1D numpy arrays
                     ):
        fu = (u.erg / (u.s * u.cm * u.cm * u.Hz))

        # Also, what is the range in frequencies where this is valid? From 10^16 Hz ?
        lamp_freq_mask = frequency > 1e+16 * u.Hz

        flux      = np.zeros((frequency.size,robs.size)) * fu
        if np.sum(lamp_freq_mask) > 0:
            # So, flux in the frequency range will be L_x / [4 pi (\vec{r_grid_point} - \vec{r_lampost})^2]
            r_cell_to_lamp_sq = ((robs - self.lamp_r_cyl)**2 + (zobs - self.lamp_z)**2) * self.mydisk.rg**2

            freq_2d   = np.broadcast_to(frequency.to(u.Hz).value, (robs.size, frequency.size)).T * u.Hz
            r_cell_to_lamp_sq_2d = np.broadcast_to(r_cell_to_lamp_sq.to(u.cm**2).value, (frequency.size, robs.size)) * u.cm**2
            flux[lamp_freq_mask,:] = (self.lamp_L_nu_2keV / (4.0 * np.pi * r_cell_to_lamp_sq_2d[lamp_freq_mask,:])) * \
                np.power(freq_2d[lamp_freq_mask,:]/self.lamp_nu_2keV, self.lamp_alpha_x) * np.exp(-freq_2d[lamp_freq_mask,:] / self.cutoff_freq)

        return flux
