import numpy             as np
import matplotlib.pyplot as plt
import os
import sys
import time              as tm

from astropy                 import constants as const
from astropy                 import units as u
from astropy.cosmology       import Planck18
from astropy.io              import fits
from astropy.table           import Table
from multiprocessing         import Pool
from scipy.interpolate       import RegularGridInterpolator, CubicSpline
from tqdm                    import tqdm

from cloudy import cloudy

class mcgv:
    ######################################################
    def __init__(self, 
                 mydisk, 
                 mycorona, 
                 myatoms, 
                 mypars,
                 ntabs = 0
                 ):
        self.mydisk   = mydisk
        self.mycorona = mycorona
        self.myatoms  = myatoms
        self.mypars   = mypars
        mdotedd       = 4 * np.pi * const.G.cgs * self.mypars.mbh / (0.1 * const.c.cgs * (const.sigma_T.cgs/const.u.cgs))
        self.Eddington_ratio = (self.mypars.mdot / mdotedd).decompose()

        rootname = self.mypars.datapath+f"Sbh{self.mypars.sbh}"
        rootname += f"-MBH{np.log10(self.mypars.mbh / const.M_sun):.2f}"
        rootname += f"-Mdot{(self.mypars.mdot/(const.M_sun/u.year)).decompose()}"
        rootname += f"-alpha{self.mypars.viscosity_alpha}"

        self.R = self.mydisk.rstar * self.mydisk.rg.to(u.cm)
        self.Z = np.logspace(-0.5, 
                                3.0, 
                                self.mypars.wind_ntheta
                                ) * self.mydisk.rg.to(u.cm)

        self.windfile  = rootname + f"-wind_R{self.mypars.nr}xZ{self.mypars.wind_ntheta}.fits"
        print("\t" * ntabs + f"Looking for {self.windfile}")
        self.bounded = False
        if os.path.exists(self.windfile):
            self.read_wind(ntabs = ntabs+1)
        else:
            print("\t" * (ntabs+1) + f"Not found. Initializing R-Z mesh")
            self.RR, self.ZZ = np.meshgrid(self.R,
                                            self.Z,
                                            indexing="ij")
            self.tottime = 0. * u.s

        self.DRR = np.zeros_like(self.RR)
        self.DRR[1:-1,:] = 0.5 * (self.RR[2:,:] - self.RR[:-2,:])
        self.DRR[ 0,:] = self.DRR[ 1,:]
        self.DRR[-1,:] = self.DRR[-2,:]

        self.DZZ = np.zeros_like(self.ZZ)
        self.DZZ[:,1:-1] = 0.5 * (self.ZZ[:,2:] - self.ZZ[:,:-2])
        self.DZZ[:, 0] = self.DZZ[:, 1]
        self.DZZ[:,-1] = self.DZZ[:,-2]

        self.rcell_vecs = np.array([self.RR / self.mydisk.rg, 
                                    np.zeros(self.RR.shape), 
                                    self.ZZ  / self.mydisk.rg
                                    ]) # shape = (3,) + self.RR.shape # units rg

        print("\t" * ntabs + f"Setting computational domain")
        self.z0   = self.mydisk.zt1 * self.mydisk.rg.to(u.cm)
        self.boundary_mask = (self.ZZ / self.mydisk.rg > self.mydisk.zt1[:,None]) & \
                                (self.RR / self.mydisk.rg > self.mydisk.rstar[0])
        self.in_disk       = (self.ZZ / self.mydisk.rg < self.mydisk.zt1[:,None]) & \
                                (self.RR / self.mydisk.rg > self.mydisk.rstar[0])
        print("\t" * (ntabs+1) + f"{np.sum(self.boundary_mask)} cells above disk")
        print("\t" * (ntabs+1) + f"{np.sum(self.in_disk)} cells inside disk")

        # --- Thermodynamics ---
        print("\t" * ntabs + f"Setting initial thermodynamics...")
        if not self.bounded:
            pool_tuple_input = []
            zz = self.ZZ / self.mydisk.rg
            for rzdx in range(self.R.size):
                pool_tuple_input.append((zz[rzdx,:], rzdx))

            with Pool(self.mypars.nproc) as pool, tqdm(total=self.R.size, ncols=0, desc="\t"*(ntabs+1) + "Fetching number densities") as pbar:
                pool_tuple_output = pool.starmap_async(self.mydisk.verticaldensity, 
                                                       pool_tuple_input, 
                                                       chunksize=1
                                                       )
                nproc_left = self.R.size
                while not pool_tuple_output.ready():
                    if pool_tuple_output._number_left < nproc_left:
                        pbar.update(nproc_left-pool_tuple_output._number_left)
                        nproc_left = pool_tuple_output._number_left
            self.number_density = np.zeros(self.RR.shape) / u.cm**3
            for rzdx in range(self.R.size):
                self.number_density[rzdx,:] = pool_tuple_output.get()[rzdx]
            mdu = u.g / u.cm**3
            #self.number_density[self.boundary_mask] = np.where(self.number_density[self.boundary_mask].to(u.cm**-3).value < 1.0e+10, #(1.66e-14 * mdu / const.u).to(u.cm**-3).value,
            #                               1.0e+10, # (1.66e-14 * mdu / const.u).to(u.cm**-3).value,
            #                               self.number_density[self.boundary_mask].to(u.cm**-3).value) * u.cm**-3

            with Pool(self.mypars.nproc) as pool, tqdm(total=self.R.size, ncols=0, desc="\t"*(ntabs+1) + "Fetching temperatures") as pbar:
                pool_tuple_output = pool.starmap_async(self.mydisk.verticaltemperature, 
                                                       pool_tuple_input, 
                                                       chunksize=1
                                                       )
                nproc_left = self.R.size
                while not pool_tuple_output.ready():
                    if pool_tuple_output._number_left < nproc_left:
                        pbar.update(nproc_left-pool_tuple_output._number_left)
                        nproc_left = pool_tuple_output._number_left
            self.temperature = np.zeros(self.RR.shape) * u.K
            for rzdx in range(self.R.size):
                self.temperature[rzdx,:] = pool_tuple_output.get()[rzdx]
            too_cold = self.temperature < Planck18.Tcmb(self.mypars.zqso)
            if np.sum(too_cold) > 0:
                self.temperature[too_cold] = Planck18.Tcmb(self.mypars.zqso)

            self.column_density_table_grid = np.zeros((self.mypars.nr,
                                                        self.mypars.wind_ntheta,
                                                        self.myatoms.photo_Z.size
                                                        )) / u.cm**2

        self.drho = np.zeros_like(self.number_density) * u.g
        self.adiabatic_index = 5./3.
        self.specific_enthalpy = None

        if not self.bounded:
            print("\t" * ntabs + "Grabbing components of BH gravity")
            self.BH_gR = np.zeros(self.RR.shape) * (u.cm / u.s**2)
            self.BH_gZ = np.zeros(self.RR.shape) * (u.cm / u.s**2)
            s2 = (self.RR**2 + self.ZZ**2)
            s  = np.sqrt(s2)
            try:
                s3 = s2 * s + 1.0e-100 * u.cm**3
            except:
                print(f"s3 = {s2} * {s} + {1.0e-100 * u.cm**3}")
                input("paused")
            try:
                self.BH_gR = (-const.G.cgs * self.mypars.mbh * self.RR / s3).to(u.cm/u.s**2)
                self.BH_gZ = (-const.G.cgs * self.mypars.mbh * self.ZZ / s3).to(u.cm/u.s**2)
            except:
                print(f"{const.G.cgs} * {self.mypars.mbh} * {const.M_sun} * {self.RR} / {s3}")
                print(f"{const.G.cgs} * {self.mypars.mbh} * {const.M_sun} * {self.ZZ} / {s3}")
                input("paused")

            print("\t" * ntabs + "Grabbing components of the disk gravity")
            self.disk_gR = np.zeros(self.RR.shape) * (u.cm / u.s**2)
            self.disk_gZ = np.zeros(self.RR.shape) * (u.cm / u.s**2)
            disk_gR, disk_gZ = self.mydisk.diskgravity(self.RR.ravel(),
                                                        self.ZZ.ravel(),
                                                        ntabs = ntabs+1
                                                        )
            self.disk_gR += disk_gR.reshape(self.RR.shape)
            self.disk_gZ += disk_gZ.reshape(self.RR.shape)

        print("\t" * ntabs + f"Initializing velocity field")
        # --- Initial Fields ---
        # Use the thermal speed = v_rms = sqrt(3 k T / m) at the tau=1 surface
        # What is the direction from the tau=1 surface? Normal to it?
        # self.mydisk.dzdr gives the slope of the tau=1 surface....
        # slope of normal is -1/self.mydisk.dzdr
        # so normal is at an angle = - np.arctan(self.mydisk.dzdr)
        if not self.bounded:
            v_rms      = np.sqrt(3. * const.k_B.cgs * self.temperature / const.u.cgs).decompose(bases=u.cgs.bases)
            self.v_Z   = v_rms
            self.v_R   = np.zeros_like(self.v_Z)
            self.v_phi = np.sqrt(const.G * self.mypars.mbh / (self.RR)).decompose(bases=u.cgs.bases)

        self.dv_R   = np.zeros_like(self.v_R)
        self.dv_Z   = np.zeros_like(self.v_Z)
        self.dv_phi = np.zeros_like(self.v_phi)

        print("\t" * ntabs + "Setting force multiplier functions")
        ######################################################
        # Dannen, Randall C.; Proga, Daniel; Kallman, Timothy R.; Waters, Tim 2019ApJ...882...99D
        # These tables are formatted to be easy to parse (see the python example below),
        # and a C++ interface to simulation codes is provided. The first entry in each
        # table is N_xi, the number of photoionization parameter values. The remainder
        # of the first row contains the N_xi values of log10(xi). The remainder of the
        # first column is all the log10 values of the optical depth parameter, t. The
        # entries corresponding to a given (t,xi) pair are the values of log10(M), where
        # M is the force multiplier.
        #
        # fmultarray[0,:] = log10(xi)    xi = 4 np.pi Fx / nH     Fx = 0.1-1000 Ryd integrated flux
        # fmultarray[:,0] = log10(t)      t = rho sigma_e vth / (dv_l/dl) = optical depth parameter
        fmultfile = self.mypars.datapath+"DPKW19_tables/DPKW19_tables/AGN1_Fmult.dat"
        self.fmultarray = np.genfromtxt(fmultfile)
        

        # Call as fmultgridfunc((lgt,lgxi))
        self.fmultgridfunc = RegularGridInterpolator((self.fmultarray[1:,0],self.fmultarray[0,1:]), self.fmultarray[1:,1:], bounds_error=False, fill_value=0)

        self.forcemultfile  = rootname + f"-fmultgrid_R{self.mypars.nr}xZ{self.mypars.wind_ntheta}.fits"
        print("\t" * ntabs + f"Looking for {self.forcemultfile}")
        if os.path.exists(self.forcemultfile):
            print("\t" * ntabs + f"\tReading {self.forcemultfile}")
            data = Table.read(self.forcemultfile, format="fits")
            self.MRgrid = np.array(data['MRgrid'])
            self.MZgrid = np.array(data['MZgrid'])
        else:
            print("\t" * ntabs + "\tNot found...")
            self.MRgrid = np.zeros(self.RR.shape)
            self.MZgrid = np.zeros(self.RR.shape)

        #Mtot_mag_grid = np.sqrt(self.MRgrid*self.MRgrid + self.MZgrid*self.MZgrid)
        #fm_mask = self.boundary_mask & (Mtot_mag_grid == 0)
        #self.MRgrid[fm_mask] = self.RR[fm_mask] / np.sqrt(self.RR[fm_mask] * self.RR[fm_mask] + self.ZZ[fm_mask] * self.ZZ[fm_mask])
        #self.MRgrid[fm_mask] = self.ZZ[fm_mask] / np.sqrt(self.RR[fm_mask] * self.RR[fm_mask] + self.ZZ[fm_mask] * self.ZZ[fm_mask])

        #self.number_density[self.boundary_mask] = 1.0e+4 / u.cm**3

        print("\t" * ntabs + "Commiting None-sequitters")
        self.lorentz_factor = np.ones(self.RR.shape)

        if not self.bounded:
            print("\t" * ntabs + "Writing initial wind")
            self.write_wind()

        print("\t" * ntabs + "Initializing grid plot...")
        plt.gcf()
        if not plt.isinteractive():
            print("\t" * (ntabs+1) + "Turning on interactive plotting...")
            plt.ion()
        plt.clf()
        plt.pause(5)
        self.plot_grid(self.number_density * const.u,
                       -31.0 * u.s,
                       0.0 * u.s,
                       0.0 * u.s,
                       0.0 * u.s,
                       np.zeros(self.RR.shape, dtype="bool"),
                       np.zeros(self.RR.shape, dtype="bool"),
                       plotstream = True
                       )
        print("\t" * (ntabs+1) + "You should have a plot now...")
        plt.pause(5)

    ######################################################
    # Relativistic Euler equation residuals
    def _EULER_cylindrical(self, 
                          dtime, 
                          which_residual # = ['rho','vR','vZ','vTH']
                          ):  
        match which_residual:
            case 'rho': # Continuity
                # Mass flux in r, theta directions ---
                massflux_R = const.u * self.number_density * self.lorentz_factor * self.v_R # g / cm**2 / s
                massflux_Z = const.u * self.number_density * self.lorentz_factor * self.v_Z # g / cm**2 / s

                dR_mdot_RdR         = np.zeros((self.mypars.nr,self.mypars.wind_ntheta)) * (u.g / u.cm**3 / u.s)
                dR_mdot_RdR[1:-1,:] = ( ( (self.RR[2:, :]) * massflux_R[ 2:,  :] ) - ( (self.RR[:-2, :]) * massflux_R[ :-2, :  ] ) ) / (2 * self.DRR[1:-1, :] * self.RR[1:-1, :] + 1.0e-100 * u.cm**2)   # g / cm**3 / s

                dmdot_dZ         = np.zeros((self.mypars.nr,self.mypars.wind_ntheta)) * (u.g / u.cm**3 / u.s)
                dmdot_dZ[:,1:-1] = ( massflux_Z[ :, 2:] - massflux_Z[:  , :-2] ) / (2 * self.DZZ[:,1:-1] + 1.0e-100 * u.cm) # g / cm**3 / s

                cont = dR_mdot_RdR + dmdot_dZ

                drho = -dtime * cont
                drho[self.in_disk] = 0.0 * (u.g/u.cm**3)

                for arrstr,arr in [(     "mass density", const.u * self.number_density[self.boundary_mask] ),
                                   (   "lorentz factor", self.lorentz_factor[self.boundary_mask]           ),
                                   (              "v_R", self.v_R[self.boundary_mask]                      ),
                                   (              "v_Z", self.v_Z[self.boundary_mask]                      ),
                                   (       "massflux_R", massflux_R[self.boundary_mask]                    ),
                                   (       "massflux_Z", massflux_Z[self.boundary_mask]                    ),
                                   (      "dR_mdot_RdR", dR_mdot_RdR[self.boundary_mask]                   ),
                                   (         "dmdot_dZ", dmdot_dZ[self.boundary_mask]                      ),
                                   (             "cont", cont                                              )
                                   ]:
                    if not self._sanity_check(arrstr,arr, function="mcgv._EULER_cylindrical: drho"):
                        input("Insane")

                return drho

            case 'vR': # Radial Momentum
                dv_R_dR         = np.zeros_like(self.v_R) / np.ones_like(self.DRR)
                dv_R_dR[1:-1,:] = ( self.v_R[2:, :] - self.v_R[:-2, :] ) / (2 * self.DRR[1:-1, :] + 1.0e-100 * u.cm)

                dv_R_dZ         = np.zeros_like(self.v_R) / np.ones_like(self.ZZ)
                dv_R_dZ[:,1:-1] = ( self.v_R[:, 2:] - self.v_R[:, :-2]) / (2 * self.DZZ[:,1:-1] + 1.0e-100 * u.cm)

                dP_dR         = np.zeros((self.mypars.nr,self.mypars.wind_ntheta)) * (u.dyne / u.cm**2) / np.ones_like(self.DRR)
                dP_dR[1:-1,:] = (self.P_total[2:,:] - self.P_total[:-2,:]) / (2 * self.DRR[1:-1, :] + 1.0e-100 * u.cm)

                lhs_R   = np.zeros((self.mypars.nr,self.mypars.wind_ntheta)) * (u.dyne/u.cm**3)
                rhs_R   = np.zeros((self.mypars.nr,self.mypars.wind_ntheta)) * (u.dyne/u.cm**3)

                lhs_R[self.boundary_mask] = \
                    const.u * self.number_density[self.boundary_mask] * self.specific_enthalpy[self.boundary_mask] * self.lorentz_factor[self.boundary_mask]**2 * \
                    ( self.v_R[self.boundary_mask] * dv_R_dR[self.boundary_mask] + self.v_Z[self.boundary_mask] * dv_R_dZ[self.boundary_mask] - self.v_phi[self.boundary_mask]**2 / self.RR[self.boundary_mask] ) \
                    + dP_dR[self.boundary_mask]

                for arrstr,arr in [(     "mass density", const.u * self.number_density[self.boundary_mask] ),
                                   ("specific enthalpy", self.specific_enthalpy[self.boundary_mask]        ),
                                   (   "lorentz factor", self.lorentz_factor[self.boundary_mask]           ),
                                   (              "v_R", self.v_R[self.boundary_mask]                      ),
                                   (          "dv_R_dr", dv_R_dR[self.boundary_mask]                       ),
                                   (              "v_Z", self.v_Z[self.boundary_mask]                      ),
                                   (          "dv_R_dz", dv_R_dZ[self.boundary_mask]                       ),
                                   (            "v_phi", self.v_phi[self.boundary_mask]                    ),
                                   (               "RR", self.RR[self.boundary_mask]                       ),
                                   (            "dP_dR", dP_dR[self.boundary_mask]                         ),
                                   (            "lhs_R",lhs_R                                              )
                                   ]:
                    if not self._sanity_check(arrstr,arr, function="mcgv._EULER_cylindrical"):
                        input("Insane")

                try:
                    aRtot = self.BH_gR[self.boundary_mask] + self.disk_gR[self.boundary_mask] + self._g_rad_R()[self.boundary_mask]
                except:
                    print(f"aRtot = {self.BH_gR[self.boundary_mask]} + {self.disk_gR[self.boundary_mask]} + {self._g_rad_R()[self.boundary_mask]}")
                    input("paused")
                rhs_R[self.boundary_mask] = (const.u * self.number_density[self.boundary_mask] * self.lorentz_factor[self.boundary_mask] * aRtot).decompose(bases=u.cgs.bases)

                dvR = (dtime * (rhs_R - lhs_R)
                       / (const.u * self.number_density * self.specific_enthalpy * self.lorentz_factor**2 + 1e-8 * (u.g/u.cm**3))
                       ).decompose(bases=u.cgs.bases)
                dvR[self.in_disk] = 0.0 * (u.cm/u.s)

                if not self._sanity_check("rhs_R",rhs_R):
                    input("Insane")

                return dvR

            case 'vZ': # Polar equation
                dv_Z_dR         = np.zeros_like(self.v_Z) / np.ones_like(self.DRR)
                dv_Z_dR[1:-1,:] = ( self.v_Z[2:, :] - self.v_Z[:-2, :] ) / (2 * self.DRR[1:-1, :] + 1.0e-100 * u.cm)

                dv_Z_dZ         = np.zeros_like(self.v_Z) / np.ones_like(self.ZZ)
                dv_Z_dZ[:,1:-1] = ( self.v_Z[:, 2:] - self.v_Z[:, :-2]) / (2 * self.DZZ[:,1:-1] + 1.0e-100 * u.cm)

                dP_dZ         = np.zeros((self.mypars.nr,self.mypars.wind_ntheta)) * (u.dyne / u.cm**2) / np.ones_like(self.DZZ)
                dP_dZ[1:-1,:] = (self.P_total[2:,:] - self.P_total[:-2,:]) / (2 * self.DZZ[1:-1, :] + 1.0e-100 * u.cm)

                lhs_Z   = np.zeros((self.mypars.nr,self.mypars.wind_ntheta)) * (u.dyne/u.cm**3)
                rhs_Z   = np.zeros((self.mypars.nr,self.mypars.wind_ntheta)) * (u.dyne/u.cm**3)

                lhs_Z[self.boundary_mask] = \
                    const.u * self.number_density[self.boundary_mask] * self.specific_enthalpy[self.boundary_mask] * self.lorentz_factor[self.boundary_mask]**2 * \
                    ( self.v_R[self.boundary_mask] * dv_Z_dR[self.boundary_mask] + self.v_Z[self.boundary_mask] * dv_Z_dZ[self.boundary_mask] ) \
                    + dP_dZ[self.boundary_mask]

                aZtot = (self.BH_gZ[self.boundary_mask] + self.disk_gZ[self.boundary_mask] + self._g_rad_Z()[self.boundary_mask]) 
                rhs_Z[self.boundary_mask] = (const.u * self.number_density[self.boundary_mask] * self.lorentz_factor[self.boundary_mask] * aZtot).decompose(bases=u.cgs.bases)

                dvZ = (dtime * (rhs_Z - lhs_Z)
                       / (const.u * self.number_density * self.specific_enthalpy * self.lorentz_factor**2 + 1e-8 * (u.g/u.cm**3))
                       ).decompose(bases=u.cgs.bases)
                dvZ[self.in_disk] = 0.0 * (u.cm/u.s)

                return dvZ

            case 'vph': # Azimuthal equation (w/ angular momentum conservation)
                ell = self.specific_enthalpy * self.lorentz_factor * self.RR * self.v_phi # cm**2 / s

                RDvRell = self.RR * const.u * self.number_density * self.lorentz_factor * self.v_R * ell # dyne      = g cm / s**2 = cm (g / cm**3) (cm / s) (cm**2 / s)
                DvZell  =           const.u * self.number_density * self.lorentz_factor * self.v_Z * ell # dyne / cm = g    / s**2 =    (g / cm**3) (cm / s) (cm**2 / s)

                dRDvRell_dR         = np.zeros_like(RDvRell)   / np.ones_like(self.DRR)      # dyne / cm
                dRDvRell_dR[1:-1,:] = ( RDvRell[2:,:] - RDvRell[:-2,:] ) / (2 * self.DRR[1:-1, :] + 1.0e-100 * u.cm)
            
                dDvZell_dZ         = np.zeros_like(DvZell)   / np.ones_like(self.ZZ)         # dyne / cm**2
                dDvZell_dZ[:,1:-1] = ( DvZell[:,2:] - DvZell[:,:-2] ) / (2 * self.DZZ[:, 1:-1] + 1.0e-100 * u.cm)

                lhs_phi = np.zeros((self.mypars.nr,self.mypars.wind_ntheta)) * (u.dyne/u.cm**2)
                rhs_phi = np.zeros((self.mypars.nr,self.mypars.wind_ntheta)) * (u.dyne/u.cm**2) # If there were any torque terms...

                lhs_phi[self.boundary_mask] = ( (dRDvRell_dR[self.boundary_mask] / self.RR[self.boundary_mask]) + \
                                               dDvZell_dZ[self.boundary_mask] 
                                               ) # dyne / cm**2 = g cm / (s**2 cm**2) = g / (s**2 cm)
               #rhs_phi[self.boundary_mask]  = self.RR * const.u * self.number_density * self.lorentz_factor * "self._g_rad_phi" if there were a g_rad_phi
               #                               cm (g / cm**3) (cm / s**2) = dyne / cm**2

                dell = (dtime * (rhs_phi - lhs_phi) / (const.u * self.number_density * self.specific_enthalpy * self.lorentz_factor**2 + 1e-8 * (u.g/u.cm**3))
                        ).decompose(bases=u.cgs.bases)

                ell_hRc_ratio = ell / (self.specific_enthalpy * self.RR * const.c)
                dvph = ( dell / (self.specific_enthalpy * self.RR * self.lorentz_factor) / (1 + ell_hRc_ratio * ell_hRc_ratio)
                    ).decompose(bases=u.cgs.bases)
                dvph[self.in_disk] = 0.0 * (u.cm/u.s)

                return dvph

    ######################################################
    # Cylindrical R- and Z- components of the radiative force per unit mass
    def _g_rad_R(self,
                 ntabs = 0
                 ):
        return (const.G.cgs * self.mypars.mbh / (self.RR + 1.0e-100 * u.cm)**2) * self.Eddington_ratio * self.MRgrid

    def _g_rad_Z(self,
                 ntabs = 0
                 ):
        return (const.G.cgs * self.mypars.mbh / (self.RR + 1.0e-100 * u.cm)**2) * self.Eddington_ratio * self.MZgrid

    ######################################################
    def _mcgv_timer(self, 
                    t1
                    ):
        if tm.time() * u.s - t1 < 300 * u.s:
            return tm.time() * u.s - t1
        elif (tm.time() * u.s - t1).to(u.minute) < 60 * u.min:
            return (tm.time() * u.s - t1).to(u.minute)
        else:
            return (tm.time() * u.s - t1).to(u.hour)

    ######################################################
    def _mcgv_time(self, 
                    t1
                    ):
        if t1 < 300 * u.s:
            return t1
        elif t1.to(u.minute) < 60 * u.min:
            return t1.to(u.minute)
        elif t1.to(u.hour) < 24 * u.hour:
            return t1.to(u.hour)
        elif t1.to(u.d) < (1.0 * u.yr).to(u.d):
            return t1.to(u.d)
        else:
            return t1.to(u.yr)

    ######################################################
    # Ideal gas law for gas pressure
    def _P_gas(self,
               ntabs = 0
               ):
        return const.k_B.cgs * const.u * self.number_density * self.temperature / const.u.cgs

    ######################################################
    def _sanity_check(self,
                      arrstr,
                      arr,
                      function = None,
                      ntabs = 0
                      ):
        sanity = True
        if not np.all(np.isfinite(arr)):
            if function is not None:
                dumstr = f"{function}"
            else:
                dumstr = "?"
            print("\t" * ntabs + f'\t\t\tNaN values in {arrstr} = {arr} (from {dumstr})')
            sanity = False
        return sanity

    ######################################################
    def plot_grid(self,
                  rho_tmp,
                  tplt,
                  t0,
                  t1,
                  dtime,
                  where_density_changed,
                  where_velocity_bad,
                  plotstream = False
                  ):
        mdu = u.g / u.cm**3
        
        v_R_val  = (self.v_R[  :-1, :-1].to(u.km/u.s)).value + sys.float_info.epsilon
        v_Z_val  = (self.v_Z[  :-1, :-1].to(u.km/u.s)).value + sys.float_info.epsilon
        v_ph_val = (self.v_phi[:-1, :-1].to(u.km/u.s)).value + sys.float_info.epsilon

        dvR_val  = (self.dv_R[  :-1, :-1].to(u.km/u.s)).value + sys.float_info.epsilon
        dvZ_val  = (self.dv_Z[  :-1, :-1].to(u.km/u.s)).value + sys.float_info.epsilon
        dvph_val = (self.dv_phi[:-1, :-1].to(u.km/u.s)).value + sys.float_info.epsilon

        g_R_val = (self._g_rad_R() + self.BH_gR + self.disk_gR)[:-1,:-1].to(u.km/u.s**2).value + sys.float_info.epsilon
        g_Z_val = (self._g_rad_Z() + self.BH_gZ + self.disk_gZ)[:-1,:-1].to(u.km/u.s**2).value + sys.float_info.epsilon

        num_dens_val = self.number_density[:-1,:-1].to(u.cm**-3).value + sys.float_info.epsilon
        temp_val     = self.temperature[:-1,:-1].to(u.K).value         + sys.float_info.epsilon

        dlnrho = self.drho[:-1,:-1]/(rho_tmp[:-1,:-1] + sys.float_info.epsilon * mdu)

        logMR = np.log10(np.fabs(self.MRgrid[:-1,:-1]) + sys.float_info.epsilon)
        logMZ = np.log10(np.fabs(self.MZgrid[:-1,:-1]) + sys.float_info.epsilon)

        if (tm.time() * u.s - tplt > 30 * u.s) and plotstream:
            fig = plt.gcf()
            fig.clf()
            for (pnum,title,colarr) in [( 1, r'$\log |v_R|$/[km s$^{-1}$]',      np.log10(np.fabs(v_R_val))),
                                        ( 2, r'$\log |v_Z|$/[km s$^{-1}$]',      np.log10(np.fabs(v_Z_val))),
                                        ( 3, r'$\log |v_\phi|$/ [km s$^{-1}$]', np.log10(np.fabs(v_ph_val))),
                                        ( 4, r'Boundary Mask',                  self.boundary_mask[:-1,:-1]),
                                        ( 5, r'$\Delta v_R$ [km s$^{-1}$]',                         dvR_val),
                                        ( 6, r'$\Delta v_Z$ [km s$^{-1}$]',                         dvZ_val),
                                        ( 7, r'$\Delta v_\mathrm{\phi}$ [km s$^{-1}$]',            dvph_val),
                                        ( 8, r'$\Delta \rho/\rho$',                                  dlnrho),
                                        ( 9, r'$g_\mathrm{R}$ [km s$^{-2}$]',                       g_R_val),
                                        (10, r'$g_\mathrm{Z}$ [km s$^{-2}$]',                       g_Z_val),
                                        (11, r'$\log T/[K]$',                            np.log10(temp_val)),
                                        (12, r'$\log n/$[cm$^{-3}$]',                np.log10(num_dens_val)),
                                        (13, r'$\log |M_\mathrm{R}|$',                                logMR),
                                        (14, r'$\log |M_\mathrm{Z}|$',                                logMZ)
                                        ]:
                self.plot_panel(pnum,
                                title,
                                colarr,
                                where_density_changed,
                                where_velocity_bad
                                )
            fig.tight_layout()

        dvmax = np.max(np.fabs(np.array([dvR_val,dvZ_val,dvph_val]))) * (u.km/u.s)
        pltstr  = f' Simulated time: {self._mcgv_time(self.tottime):e} \n Time step: {self._mcgv_time(dtime):e} \n'
        pltstr += f' Time since last write/plot: {self._mcgv_timer(t0):.0f}/{self._mcgv_timer(tplt):.0f}\n'
        pltstr += f' Run time: {self._mcgv_timer(t1)} \n'
        pltstr += f' Max change in velocity: {(dvmax/const.c).decompose():.2e} c \n'

        pltstr +=  r'$\Delta\rho/\rho$ range: '
        pltstr += f'{np.min((self.drho[self.boundary_mask]/(rho_tmp[self.boundary_mask])).decompose()):.2e} to '
        pltstr += f'{np.max((self.drho[self.boundary_mask]/(rho_tmp[self.boundary_mask])).decompose()):.2e} \n'
        pltstr += f' Number of cells with '+r'$|\Delta\rho|/\rho>0.1$: '+f'{np.sum(np.fabs(self.drho[self.boundary_mask])/rho_tmp[self.boundary_mask] > 0.1)} \n'

        pltstr += f' Number of simulated cells: {np.sum(self.boundary_mask)}' # \n'
        titeration = tm.time() * u.s
        if plotstream:
          plt.annotate(pltstr,(0.77,0.02),xycoords='figure fraction',fontsize=14,color='w',backgroundcolor='b')
          plt.pause(1)

    ######################################################
    def plot_panel(self,
                   pnum,
                   title,
                   colarr,
                   where_density_changed,
                   where_velocity_bad
                   ):
        fig = plt.gcf()
        ax = plt.subplot(4,4,pnum)
        ax.clear()

        ax.set_title(title)
        cmap = plt.get_cmap('viridis').copy()
        cmap.set_under('black')
        cmap.set_over('red')
        mesh = ax.pcolormesh((self.RR / self.mydisk.rg).value, 
                            (self.ZZ / self.mydisk.rg).value, 
                            colarr,
                            vmin = np.max([np.min(colarr),np.average(colarr) - 3.0 * np.std(colarr)]),
                            vmax = np.min([np.max(colarr),np.average(colarr) + 3.0 * np.std(colarr)]),
                            cmap = 'viridis',
                            shading='flat'
                            )
        fig.colorbar(mesh, 
                    ax=ax
                    )
        #print(f"Number of figure axes: {len(fig.axes)}    idx = {idx}")
        if pnum == 13:
            ax.set_xlabel(r'R ($r_g$)')
            ax.set_ylabel(r'Z ($r_g$)')
        ax.plot(self.mydisk.rstar, 
                    self.mydisk.diskheight
                    )
        ax.plot(self.mydisk.rstar, 
                    self.mydisk.zt1
                    )
        # Plotting RZ grid
        for pltr in self.R:
            ax.plot(pltr * np.ones(self.Z.size) / self.mydisk.rg, 
                    self.Z / self.mydisk.rg, 
                    'k:', 
                    alpha=0.1)
        for pltz in self.Z:
            ax.plot(self.R / self.mydisk.rg, 
                    pltz * np.ones(self.R.size) / self.mydisk.rg, 
                    'k:', 
                    alpha=0.1
                    )

        ax.scatter((self.RR[where_density_changed].flatten() / self.mydisk.rg).value, 
                    (self.ZZ[where_density_changed].flatten() / self.mydisk.rg).value, 
                    c='c', 
                    s=2, 
                    alpha=1)
        ax.scatter((self.RR[where_velocity_bad].flatten() / self.mydisk.rg).value, 
                    (self.ZZ[where_velocity_bad].flatten() / self.mydisk.rg).value, 
                    c='m', 
                    s=2, 
                    alpha=1)
        ax.set_xlim(left = self.mydisk.rstar[0]) #, right = 3.0e+3)
        ax.set_ylim(bottom = 0.3) #, top = 3.0e+3)
        ax.set_xscale("log")
        ax.set_yscale("log")

    ######################################################
    def projectvlos(self,
                    ntabs = 0
                    ):
        # The (Cartesian) vector pointing to Theo is
        r_Theo = np.array([self.mydisk.robs * np.cos(self.mydisk.thetaobs),
                           self.mydisk.robs * np.sin(self.mydisk.thetaobs),
                           self.mydisk.zobs])

        
        # Need to take the dot product of the velocity vector field with the direction of Theo (from each of the cells!)
        # self.RR and self.ZZ provide the R and Z coordinates for every cell w/o the phi part.
        # self.vR, self.vphi, and self.vZ provide the cylindrical components of the velocity vectors, but only for phi=0
        # Problem - we don't have a grid in phi... how to determine the 3D field from the rotation about the z-axis?
        # Use the self.mydisk.ntheta (cylindrical theta) to grid in phi.
        nphi = np.int16(np.max(self.mydisk.ntheta))
        phi = np.linspace(0,2*np.pi,nphi)
        cosphi = np.cos(phi)
        sinphi = np.sin(phi)
        # The vlos scalar field should have a shape (self.mypars.nr,self.ntheta,nphi)
        self.vlos = np.empty((self.mypars.nr,self.ntheta,nphi))
        # Want to convert cylindrical (vR,vphi,vZ) to cartesian (vx,vy,vz)
        # https://en.wikipedia.org/wiki/Vector_fields_in_cylindrical_and_spherical_coordinates says howto do this.
        vX = self.v_R[:,:,None] * cosphi[None,None,:] - self.v_phi[:,:,None] * sinphi[None,None,:]
        vY = self.v_R[:,:,None] * sinphi[None,None,:] + self.v_phi[:,:,None] * sinphi[None,None,:]
        vZ = self.v_Z[:,:,None] * np.ones(phi.shape)[None,None,:]

        # Vector from each cell to TheO
        RX = self.mydisk.robs * np.cos(self.mydisk.thetaobs) - self.RR[:,:,None] * cosphi[None,None,:]
        RY = self.mydisk.robs * np.sin(self.mydisk.thetaobs) - self.RR[:,:,None] * sinphi[None,None,:]
        RZ = self.mydisk.zobs                                - self.ZZ[:,:,None] * np.ones(phi.shape)[None,None,:]

        self.vlos = - (vX * RX + vY * RY + vZ * RZ) / np.sqrt(RX * RX + RY * RY + RZ * RZ)

    ######################################################
    def read_wind(self,
                  ntabs = 0
                  ):
        print("\t" * (ntabs+1) + f"Reading {self.windfile}")
        with fits.open(self.windfile, hdu=1) as hdul:
            self.tottime       = hdul[1].header['SIMTIME'] * u.s
            self.RR            = hdul[1].data['R2D'] * u.cm
            self.ZZ            = hdul[1].data['Z2D'] * u.cm
            self.v_R           = hdul[1].data['vR2D'] * (u.cm/u.s)
            self.v_Z           = hdul[1].data['vZ2D'] * (u.cm/u.s)
            self.v_phi         = hdul[1].data['vphi2D'] * (u.cm/u.s)
            self.number_density  = hdul[1].data['rho2D'] * (u.cm**-3)
            self.temperature   = hdul[1].data['T2D'] * (u.K)
            self.BH_gR         = hdul[1].data['BH_gR'] * (u.cm/u.s**2)
            self.BH_gZ         = hdul[1].data['BH_gZ'] * (u.cm/u.s**2)
            self.disk_gR       = hdul[1].data['disk_gR'] * (u.cm/u.s**2)
            self.disk_gZ       = hdul[1].data['disk_gZ'] * (u.cm/u.s**2)
            self.boundary_mask = hdul[1].data['boundary_mask']


            self.bounded = True

            try:
                self.column_density_table_grid = hdul[2].data['column_density_grid'] / u.cm**2
            except:
                self.column_density_table_grid = np.zeros(self.RR.shape + (self.myatoms.photo_Z.size,)) / u.cm**2

    ######################################################
    # Need to use R_vec to extract cells that are intercepted and determine optical depth attentuating the X-rays
    # A = self.mycorona_position_vec + a x R_vec (a = 0..1) parameterizes the sightline
    # D = A - rcell_vecs = vector from rcell_vecs to a point on A
    # Want a that minimizes the magnitude of D:
    # D^2 = (A - rcell_vecs)*(A - rcell_vecs) = A*A + rcell_vecs*rcell_vecs - 2 A * rcell_vecs
    #     = self.mycorona_position_vec*self.mycorona_position_vec + a^2 x R_vec*R_vec + 2 a self.mycorona_position_vec*R_vec 
    #                                       + rcell_vecs*rcell_vecs - 2 self.mycorona_position_vec*rcell_vecs - 2 a x R_vec*rcell_vecs
    # 2D (dD/da) = 2a R_vec*R_vec + 2 self.mycorona_position_vec*R_vec - 2 R_vec*rcell_vecs = 0 to minimize
    # a = (rcell_vecs - self.mycorona_position_vec) * R_vec  / (R_vec * R_vec)
    def shield_optical_depth_v2(self,
                                rdisk_vecs, # should be shape (3,nphi)   photon origin
                                rcell_vec,   # should be shape (3,)     photon destination
                                energy,
                                ntabs = 0
                                ):
        try:
            n_sightlines = rdisk_vecs.shape[1]
        except:
            n_sightlines = 1

        R_vecs = rcell_vec[:,None] - rdisk_vecs # (3,n_sightlines)

        # shapes:            (3,nr,nz)               (3,n_sightlines)
       #gg = self.rcell_vecs[:,:,:,None] - rdisk_vecs[:,None,None,:]                                        # (3,nr,nz,n_sightlines)
       #a = np.sum(gg * R_vecs[:,None,None,:], axis=0 ) / np.sum(R_vecs * R_vecs, axis=0)[None,None,:]      # (  nr,nz,n_sightlines)
       #D = rdisk_vecs[:,None,None,:] + a[None,:,:,:] * R_vecs[:,None,None,:] - self.rcell_vecs[:,:,:,None] # (3,nr,nz,n_sightlines)
       #Dmag = np.sqrt(np.sum( D * D, axis=0))                                                              # (  nr,nz,n_sightlines)

        shield_optical_depth = np.zeros((energy.size, n_sightlines))
        for sdx in range(n_sightlines):

            # shapes:          (3,nr,nz)               (3,n_sightlines)
            gg = self.rcell_vecs[:,:,:] - rdisk_vecs[:,None,None,sdx]                                           # (3,nr,nz)
            a = np.sum(gg * R_vecs[:,None,None,sdx], axis=0 ) / np.sum(R_vecs * R_vecs, axis=0)[None,None,sdx]      # (  nr,nz)
            D = rdisk_vecs[:,None,None,sdx] + a[None,:,:] * R_vecs[:,None,None,sdx] - self.rcell_vecs[:,:,:] # (3,nr,nz)
            Dmag = np.sqrt(np.sum( D * D, axis=0))                                                              # (  nr,nz)

            #              |         "along" sightline         |
           #shield_cells = (a[:,:,sdx] > 0) & (a[:,:,sdx] < 1) & (Dmag[:,:,sdx] < self.DRR / self.mydisk.rg)
            #                                                  | intersecting sightline                    |
            #              |         "along" sightline |
            shield_cells = (a[:,:] > 0) & (a[:,:] < 1) & (Dmag[:,:] < self.DRR / self.mydisk.rg)
            #                                          | intersecting sightline                    |

            if np.sum(shield_cells) > 0:
                for shield_cell_column_densities in self.column_density_table_grid[shield_cells,:]:
                    if np.any(shield_cell_column_densities > 0):
                        which_ions = shield_cell_column_densities > 0

                        big_energy = np.broadcast_to(energy.to(u.eV).value, (np.sum(which_ions), energy.size)) * u.eV
                        energy_mask = (big_energy > self.myatoms.photo_E_th[which_ions,None]) & (big_energy < self.myatoms.photo_E_max[which_ions,None])

                        if np.any(energy_mask):
                            big_x  = np.zeros(big_energy.shape)
                            big_y  = np.zeros(big_energy.shape)
                            big_aa = np.zeros(big_energy.shape)
                            big_bb = np.zeros(big_energy.shape)
                            big_cc = np.zeros(big_energy.shape)

                            big_x[energy_mask] = (big_energy / (self.myatoms.photo_E_0[which_ions,None] - self.myatoms.photo_y_0[which_ions,None]))[energy_mask]
                            big_y[energy_mask] = (np.sqrt(big_x**2 + self.myatoms.photo_y_w[which_ions,None]**2))[energy_mask]

                            big_aa[energy_mask] = ((big_x-1)**2 + self.myatoms.photo_y_w[which_ions,None]**2)[energy_mask]
                            big_bb[energy_mask] = (np.power(big_y+1.0e-30, 0.5*(self.myatoms.photo_p[which_ions,None]-11)))[energy_mask]
                            big_cc[energy_mask] = (np.power(1 + np.sqrt(big_y / self.myatoms.photo_y_a[which_ions,None]), self.myatoms.photo_p[which_ions,None]))[energy_mask]

                            big_cross_section = np.zeros(big_energy.shape)
                            big_cross_section[energy_mask] = (self.myatoms.photo_sig_0[which_ions,None] * big_aa * big_bb * big_cc)[energy_mask]

                            big_optical_depth = (big_cross_section * shield_cell_column_densities[which_ions,None]).decompose().value

                            shield_optical_depth[:,sdx] += np.sum(big_optical_depth, axis=0)

        return shield_optical_depth

    ######################################################
    def shield_optical_depth(self,
                             shield_cells,
                             energy,
                             ntabs = 0
                             ):
        shield_optical_depth = np.zeros(energy.shape)

        for shield_cell_column_densities in self.column_density_table_grid[shield_cells,:]:

            if np.any(shield_cell_column_densities > 0):
                which_ions = shield_cell_column_densities > 0

                big_energy = np.broadcast_to(energy.to(u.eV).value, (np.sum(which_ions), energy.size)) * u.eV
                energy_mask = (big_energy > self.myatoms.photo_E_th[which_ions,None]) & (big_energy < self.myatoms.photo_E_max[which_ions,None])

                if np.any(energy_mask):
                    big_x  = np.zeros(big_energy.shape)
                    big_y  = np.zeros(big_energy.shape)
                    big_aa = np.zeros(big_energy.shape)
                    big_bb = np.zeros(big_energy.shape)
                    big_cc = np.zeros(big_energy.shape)

                    big_x[energy_mask] = (big_energy / (self.myatoms.photo_E_0[which_ions,None] - self.myatoms.photo_y_0[which_ions,None]))[energy_mask]
                    big_y[energy_mask] = (np.sqrt(big_x**2 + self.myatoms.photo_y_w[which_ions,None]**2))[energy_mask]

                    big_aa[energy_mask] = ((big_x-1)**2 + self.myatoms.photo_y_w[which_ions,None]**2)[energy_mask]
                    big_bb[energy_mask] = (np.power(big_y+1.0e-30, 0.5*(self.myatoms.photo_p[which_ions,None]-11)))[energy_mask]
                    big_cc[energy_mask] = (np.power(1 + np.sqrt(big_y / self.myatoms.photo_y_a[which_ions,None]), self.myatoms.photo_p[which_ions,None]))[energy_mask]

                    big_cross_section = np.zeros(big_energy.shape)
                    big_cross_section[energy_mask] = (self.myatoms.photo_sig_0[which_ions,None] * big_aa * big_bb * big_cc)[energy_mask]

                    big_optical_depth = (big_cross_section * shield_cell_column_densities[which_ions,None]).decompose().value

                    shield_optical_depth += np.sum(big_optical_depth, axis=0)

        return shield_optical_depth

    ######################################################
    # Need to use R_vec to extract cells that are intercepted and determine optical depth attentuating the X-rays
    # A = self.mycorona_position_vec + a x R_vec (a = 0..1) parameterizes the sightline
    # D = A - rcell_vecs = vector from rcell_vecs to a point on A
    # Want a that minimizes the magnitude of D:
    # D^2 = (A - rcell_vecs)*(A - rcell_vecs) = A*A + rcell_vecs*rcell_vecs - 2 A * rcell_vecs
    #     = self.mycorona_position_vec*self.mycorona_position_vec + a^2 x R_vec*R_vec + 2 a self.mycorona_position_vec*R_vec 
    #                                       + rcell_vecs*rcell_vecs - 2 self.mycorona_position_vec*rcell_vecs - 2 a x R_vec*rcell_vecs
    # 2D (dD/da) = 2a R_vec*R_vec + 2 self.mycorona_position_vec*R_vec - 2 R_vec*rcell_vecs = 0 to minimize
    # a = (rcell_vecs - self.mycorona_position_vec) * R_vec  / (R_vec * R_vec)
    def shield_poke_sightline(self,
                              rorigin_vec,
                              R_vec,
                              ntabs = 0
                              ):
        gg = self.rcell_vecs - rorigin_vec[:,None,None]
        a = np.sum(gg * R_vec[:,None,None], axis=0 ) / np.sum(R_vec * R_vec)
        D = rorigin_vec[:,None,None] + a * R_vec[:,None,None] - self.rcell_vecs
        Dmag = np.sqrt(np.sum( D * D, axis=0))

        #                             | "along" sightline |
       #shield_cells = self.in_shield & (a > 0) & (a < 1) & (Dmag < self.DRR / self.mydisk.rg)
        shield_cells =                  (a > 0) & (a < 1) & (Dmag < self.DRR / self.mydisk.rg)
        #                                                 | intersecting sightline

        return shield_cells

    ######################################################
    def update_fm(self,
                  result,
                  dtime = 0.0 * u.s
                  ):

        rcell_vec_cdx,MR_grid_cdx,MZ_grid_cdx,temperature_cdx,ionization_parameter_cdx,column_density_table_cdx = result

        which_cell = self.which_cell(rcell_vec_cdx)

        self.MRgrid[                   which_cell  ] = MR_grid_cdx
        self.MZgrid[                   which_cell  ] = MZ_grid_cdx
        self.temperature[              which_cell  ] = temperature_cdx
        self.column_density_table_grid[which_cell,:] = column_density_table_cdx

        try:
            self._update_iteration += 1
        except:
            self._update_iteration = 1

        if self._update_iteration % self.mypars.nproc == 0:
            plt.clf()
            self.plot_grid(const.u * self.number_density,
                           -31.0 * u.s,
                           0.0 * u.s,
                           0.0 * u.s,
                           dtime,
                           np.zeros(self.RR.shape, dtype="bool"),
                           np.zeros(self.RR.shape, dtype="bool"),
                           plotstream = True
                           )
            plt.pause(1)
            self.write_fm()
            self.write_wind()

    ######################################################
    def which_cell(self,
                   rcell_vec,
                   ntabs = 0
                   ):
        return (rcell_vec[0] * self.mydisk.rg > self.RR - 0.5 * self.DRR) & (rcell_vec[0] * self.mydisk.rg < self.RR + 0.5 * self.DRR) & \
               (rcell_vec[2] * self.mydisk.rg > self.ZZ - 0.5 * self.DZZ) & (rcell_vec[2] * self.mydisk.rg < self.ZZ + 0.5 * self.DZZ)


    ######################################################
    def write_fm(self):
        data = Table(data=[self.MRgrid,self.MZgrid], 
                    names=["MRgrid","MZgrid"]
                    )
        data.write(self.forcemultfile, 
                format="fits", 
                overwrite=True
                )

    ######################################################
    def write_wind(self):
        datatab = Table(data=(self.RR,self.ZZ,
                              self.v_R,self.v_Z,self.v_phi,
                              self.number_density,self.temperature,
                              self.BH_gR,self.BH_gZ,
                              self.disk_gR,self.disk_gZ,
                              self.boundary_mask), 
                        names=['R2D','Z2D','vR2D','vZ2D','vphi2D','rho2D','T2D','BH_gR','BH_gZ','disk_gR','disk_gZ','boundary_mask']
                          )

        table_hdu = fits.BinTableHDU(data=datatab)
        table_hdu.header['SIMTIME'] = (self.tottime.value,'Simulated time (s)')
        hdul = fits.HDUList([fits.PrimaryHDU(), table_hdu])
        hdul.writeto(self.windfile, 
                     overwrite=True
                       )
        datatab2 = Table([self.column_density_table_grid],
                          names = ['column_density_grid'] 
                        )
        datatab2.write(self.windfile,
                       format = 'fits',
                       append = True)



    ######################################################
    def extract_emission_line(self,
                              linestr
                              ):
        nvel = 1000
        velocity = np.linspace(np.min(self.vlos),
                               np.max(self.vlos),
                               nvel)
        fnu = np.zeros(nvel)

        dvel = (np.max(self.vlos) - np.min(self.vlos)) / nvel

        dir_path = self.mypars.datapath+"Cloudy_runs"
        EM_file_list = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]

        for vdx in range(nvel):
            which_cells = (self.vlos[:,:,0] > (velocity[vdx] - 0.5 * dvel)) & \
                          (self.vlos[:,:,0] < (velocity[vdx] + 0.5 * dvel))
            if np.sum(which_cells) > 0:
                subRR = self.RR[which_cells].ravel() / self.mywind.rg
                subZZ = self.ZZ[which_cells].ravel() / self.mywind.rg
                for wdx in range(subRR.size):
                    sub_cloudy_fitsname = f"-rstar{subRR[wdx]:.2f}-zstar{subZZ[wdx]:.2f}.fits"
                    cloudy_fitsname = next((s for s in EM_file_list if sub_cloudy_fitsname in s), None)
                    if cloudy_fitsname is not None:
                        with fits.open(cloudy_fitsname) as hdul:
                            line_array = hdul[1].data
                        emlin = np.extract(self.linarray['ID'] == linestr, self.linarray)
                        if emlin.size > 0:
                            fnu[vdx] += emlin[0][2]

        return velocity, fnu
 
    ######################################################
    # Produces the dumstr for printing out a line with H I, N V, and C IV emission lines
    # Also packs self.emissiongrid
    #def _getprint("\t" * ntabs + self,i,j,gridx,gridy):
    #    lyalin = np.extract(self.linarray['ID'] == 'H  1                1215.67A', self.linarray)
    #    dumstr = f"            {self.mydisk.rstar[i]} {self.theta[j].to(u.degree)} {gridx[i,j]} {gridy[i,j]} {self.lognuFnugrid[i,j]}"
    #    if lyalin.size > 0:
    #        self.emissiongrid[i,j,0] = lyalin[0][2]
    #        dumstr += f"   H I: {lyalin[0][2]}"
    #        lyblin = np.extract(self.linarray['ID'] == 'H  1                1025.72A', self.linarray)
    #        if lyblin.size > 0:
    #            self.emissiongrid[i,j,1] = lyblin[0][2]
    #            dumstr += f" {lyblin[0][2]}

    #    nvb = np.extract(self.linarray['ID'] == 'N  5                1238.82A', self.linarray)
    #    if nvb.size > 0:
    #        self.emissiongrid[i,j,2] = nvb[0][2]
    #        dumstr += "   N V: "
    #        dumstr += f"{nvb[0][2]}"
    #        nvr = np.extract(self.linarray['ID'] == 'N  5                1242.80A', self.linarray)
    #        if nvr.size > 0:
    #            self.emissiongrid[i,j,3] = nvr[0][2]
    #            dumstr += f" {nvr[0][2]}"
                                    
    #    civb = np.extract(self.linarray['ID'] == 'C  4                1548.19A', self.linarray)
    #    if civb.size > 0:
    #        self.emissiongrid[i,j,4] = civb[0][2]
    #        dumstr += f"   C IV: {civb[0][2]}"
    #        civr = np.extract(self.linarray['ID'] == 'C  4                1550.77A', self.linarray)
    #        if civr.size > 0:
    #            self.emissiongrid[i,j,5] = civr[0][2]
    #            dumstr += f" {civr[0][2]}"

    #    return dumstr
