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
from scipy.special           import kn
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
        self.ref_velocity = np.sqrt(const.G.cgs * self.mypars.mbh / (self.mydisk.rstar[0] * self.mydisk.rg)).to(u.cm/u.s)
        self.ref_accel    = (self.ref_velocity * self.ref_velocity / self.mydisk.rg).to(u.cm/u.s**2)

        rootname = self.mypars.datapath+f"Sbh{self.mypars.sbh}"
        rootname += f"-MBH{np.log10(self.mypars.mbh / const.M_sun):.2f}"
        rootname += f"-Mdot{(self.mypars.mdot/(const.M_sun/u.year)).decompose()}"
        rootname += f"-alpha{self.mypars.viscosity_alpha}"

        self.R = np.logspace(np.log10(self.mydisk.rstar[0]),
                             np.log10(self.mydisk.rstar[-1]),
                             self.mypars.wind_nr
                            )
        self.Z = np.logspace(-0.5, 
                             3.0, 
                             self.mypars.wind_nz
                            )

        self.windfile  = rootname + f"-wind_R{self.mypars.wind_nr}xZ{self.mypars.wind_nz}.fits"
        print("\t" * ntabs + f"Looking for {self.windfile}")
        self.bounded = False
        if os.path.exists(self.windfile):
            self.read_wind(ntabs = ntabs+1)
        else:
            print("\t" * (ntabs+1) + f"Not found. Initializing R-Z mesh")
            self.RR, self.ZZ = np.meshgrid(self.R,
                                            self.Z,
                                            indexing="ij")
            print("\t" * (ntabs+1) + f"Created RR and ZZ mesh grids of shape {self.RR.shape}")
            self.tottime = 0. * u.s

        self.DRR = np.zeros_like(self.RR)
        self.DRR[1:-1,:] = 0.5 * (self.RR[2:,:] - self.RR[:-2,:])
        self.DRR[ 0,:] = self.DRR[ 1,:]
        self.DRR[-1,:] = self.DRR[-2,:]

        self.DZZ = np.zeros_like(self.ZZ)
        self.DZZ[:,1:-1] = 0.5 * (self.ZZ[:,2:] - self.ZZ[:,:-2])
        self.DZZ[:, 0] = self.DZZ[:, 1]
        self.DZZ[:,-1] = self.DZZ[:,-2]

        self.rcell_vecs = np.array([self.RR, 
                                    np.zeros(self.RR.shape), 
                                    self.ZZ
                                    ]) # shape = (3,) + self.RR.shape # units rg

        print("\t" * ntabs + f"Setting computational domain")
        zt1_grid = np.interp(self.RR, 
                             self.mydisk.rstar, 
                             self.mydisk.zt1
                            )
        self.boundary_mask = (self.ZZ > zt1_grid) & \
                             (self.RR > self.mydisk.rstar[0]) & \
                             (np.sqrt(self.RR*self.RR+self.ZZ*self.ZZ) > self.mypars.minrg_bound)
        self.in_disk       = (self.ZZ < zt1_grid) & \
                                (self.RR > self.mydisk.rstar[0])
        print("\t" * (ntabs+1) + f"{np.sum(self.boundary_mask)} cells above disk")
        print("\t" * (ntabs+1) + f"{np.sum(self.in_disk)} cells inside disk")

        # --- Thermodynamics ---
        print("\t" * ntabs + f"Setting initial thermodynamics...")
        if not self.bounded:
            pool_tuple_input = []
            for rzdx in range(self.R.size):
                pool_tuple_input.append((self.ZZ[rzdx,:], 
                                         rzdx
                                         ))

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
            number_density = np.zeros(self.RR.shape) / u.cm**3
            for rzdx in range(self.R.size):
                number_density[rzdx,:] = pool_tuple_output.get()[rzdx]
            mdu = u.g / u.cm**3

            self.ref_number_density = np.max(number_density)
            self.ln_number_density = np.log(number_density / self.ref_number_density + sys.float_info.epsilon)

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

            self.column_density_table_grid = np.zeros((self.mypars.wind_nr,
                                                        self.mypars.wind_nz,
                                                        self.myatoms.photo_Z.size
                                                        )) / u.cm**2

        #self.dln_number_density = np.zeros(self.ln_number_density.shape)
        self.adiabatic_index = 5./3.

        for arrstr,arr in [("number density", self.ref_number_density * np.exp(self.ln_number_density) ),
                           (   "temperature", self.temperature                                         )
                          ]:
            if not self._sanity_check(arrstr,arr, function="mcgv.__init__"):
                input("Insane")

        for arrstr,arr in [("number density", self.ref_number_density * np.exp(self.ln_number_density) ),
                           (   "temperature", self.temperature                                         )
                          ]:
            if not self._sanity_check(arrstr,arr, function="mcgv.__init__"):
                input("Insane")

        if not self.bounded:
            print("\t" * ntabs + "Grabbing components of BH gravity")
            self.BH_gR = np.zeros(self.RR.shape)
            self.BH_gZ = np.zeros(self.RR.shape)
            s2 = (self.RR**2 + self.ZZ**2)
            s  = np.sqrt(s2)
            try:
                s3 = s2 * s + sys.float_info.epsilon
            except:
                print(f"s3 = {s2} * {s} + {sys.float_info.epsilon}")
                input("paused")
            try:
                self.BH_gR = (-const.G.cgs * self.mypars.mbh * self.RR / (s3 * self.mydisk.rg**2 * self.ref_accel)).decompose()
                self.BH_gZ = (-const.G.cgs * self.mypars.mbh * self.ZZ / (s3 * self.mydisk.rg**2 * self.ref_accel)).decompose()
            except:
                print(f"{(const.G.cgs * self.mypars.mbh / (self.mydisk.rg**2 * self.ref_accel)).decompose()} * {self.RR} / ({s3}")
                print(f"{(const.G.cgs * self.mypars.mbh / (self.mydisk.rg**2 * self.ref_accel)).decompose()} * {self.ZZ} / ({s3}")
                input("paused")

            print("\t" * ntabs + "Grabbing components of the disk gravity")
            self.disk_gR = np.zeros(self.RR.shape)
            self.disk_gZ = np.zeros(self.RR.shape)
            disk_gR, disk_gZ = self.mydisk.diskgravity(self.RR.ravel() * self.mydisk.rg,
                                                        self.ZZ.ravel() * self.mydisk.rg,
                                                        ntabs = ntabs+1
                                                        )
            self.disk_gR += disk_gR.reshape(self.RR.shape) / self.ref_accel
            self.disk_gZ += disk_gZ.reshape(self.RR.shape) / self.ref_accel

        print("\t" * ntabs + f"Initializing velocity field")
        # --- Initial Fields ---
        # Work with u = gamma v, instead of just v in order to unbound the physically acceptable range:
        #    u_i = gamma v_i
        #    gamma = sqrt(1 + sum(u_i**2)/c**2)
        # Use the thermal speed = v_rms = sqrt(3 k T / m) at the tau=1 surface
        # What is the direction from the tau=1 surface? Normal to it?
        # self.mydisk.dzdr gives the slope of the tau=1 surface....
        # slope of normal is -1/self.mydisk.dzdr
        # so normal is at an angle = - np.arctan(self.mydisk.dzdr)
        if not self.bounded:
            self.u     = np.zeros((3,) + self.RR.shape)   # Convention for (3,) = (r, z, phi)

            init_mask = (self.ZZ < 1.2 * zt1_grid)

            self.tottime       = 0.0 * u.s
        else:
            init_mask = self.in_disk

        # Boundary or initial conditions
        self._enforce_boundary_conditions(ntabs = ntabs+1)

        self.du = np.zeros_like(self.u)

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

        self.forcemultfile  = rootname + f"-fmultgrid_R{self.mypars.wind_nr}xZ{self.mypars.wind_nz}.fits"
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

        self.MRgrid[self.in_disk] = 1.0
        self.MZgrid[self.in_disk] = 1.0

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
        self.plot_grid(self.ref_number_density * np.exp(self.ln_number_density) * const.u,
                       -31.0 * u.s,
                       0.0 * u.s,
                       0.0 * u.s,
                       0.0 * u.s,
                       np.ones(self.RR.shape, dtype="bool"),
                       plotstream = True,
                       ntabs = ntabs+1
                       )

        print("\t" * (ntabs+1) + "You should have a plot now...")
        plt.pause(5)


    ######################################################
    def _div_dR(self,
                A,
                ntabs = 0
                ):

        for arrstr,arr in [(  "A", A        ),
                           ("DRR", self.DRR )
                          ]:
            if not self._sanity_check(arrstr,
                                      arr, 
                                      function="mcgv._div_R",
                                      ntabs = ntabs+1
                                      ):
                input("Insane")

        dA_dR         = np.zeros_like(A)
        dA_dR[1:-1,:] = (  A[ 2:,  :]  - A[:-2, :] ) / (2 * self.DRR[1:-1, :] + sys.float_info.epsilon)

        return dA_dR

    ######################################################
    def _div_dZ(self,
                A,
                ntabs = 0
                ):

        for arrstr,arr in [(  "A", A        ),
                           ("DZZ", self.DZZ )
                          ]:
            if not self._sanity_check(arrstr,
                                      arr, 
                                      function="mcgv._div_Z",
                                      ntabs = ntabs+1
                                      ):
                input("Insane")

        dA_dZ         = np.zeros_like(A)
        dA_dZ[1:-1,:] = ( A[ 2:,  :] - A[:-2, :] ) / (2 * self.DZZ[1:-1, :] + sys.float_info.epsilon)

        return dA_dZ

    ######################################################
    def _enforce_boundary_conditions(self,
                                     ntabs = 0
                                     ):
        for arrstr,arr in [("1.0/self.mydisk.dzdr", 1.0/self.mydisk.dzdr),
                           ("temperature", self.temperature ),
                           ("RR", self.RR),
                           ("ZZ", self.ZZ)
                          ]:
            if not self._sanity_check(arrstr,
                                      arr, 
                                      function="mcgv._enforce_boundary_conditions",
                                      ntabs = ntabs+1
                                      ):
                input("Insane")

        zt1_grid = np.interp(self.RR, 
                             self.mydisk.rstar, 
                             self.mydisk.zt1
                            )
        dzdr_grid = np.interp(self.RR, 
                              self.mydisk.rstar, 
                              self.mydisk.dzdr
                            )
        init_mask = (self.ZZ < 1.2 * zt1_grid)

        v_angle = -np.arctan(1.0/dzdr_grid)
        thermal_parameter = (const.k_B.cgs * self.temperature / (const.u.cgs * const.c.cgs * const.c.cgs)).decompose()
        average_v_mj = 2 * const.c.cgs * thermal_parameter * (thermal_parameter + 1) * (np.exp(-1/(thermal_parameter + sys.float_info.epsilon))  + sys.float_info.epsilon) / (kn(2, 1/(thermal_parameter + sys.float_info.epsilon))  + sys.float_info.epsilon)
        if np.any(np.isnan(average_v_mj)):
            average_v_mj = np.sqrt(3 * const.k_B.cgs * self.temperature / const.u.cgs).to(u.cm/u.s)

        ss = np.sqrt(self.RR * self.RR + self.ZZ * self.ZZ) * self.mydisk.rg

        v_R = average_v_mj[init_mask] * np.sin((np.broadcast_to(v_angle, self.RR.T.shape).T)[init_mask])
        v_Z = average_v_mj[init_mask] * np.cos((np.broadcast_to(v_angle, self.RR.T.shape).T)[init_mask])
        v_P = np.sqrt(const.G * self.mypars.mbh / (ss[init_mask] + sys.float_info.epsilon * u.cm) ).decompose(bases=u.cgs.bases)
        vsq = v_R * v_R + v_Z * v_Z + v_P * v_P
        lorentz_factor = 1.0 / np.sqrt(1. - vsq / (const.c.cgs * const.c.cgs))

        self.u[:, init_mask] = np.array([v_R.to(u.cm/u.s) / self.ref_velocity, 
                                         v_Z.to(u.cm/u.s) / self.ref_velocity, 
                                         v_P.to(u.cm/u.s) / self.ref_velocity
                                         ]) * lorentz_factor

        if np.any(np.isnan(self.u)):
            input("paused in mcgv._enforce_boundary_conditions for de-nanification")

    ######################################################
    def _div_dR(self,
                A,
                ntabs = 0
                ):

        for arrstr,arr in [(  "A", A        ),
                           ("DRR", self.DRR )
                          ]:
            if not self._sanity_check(arrstr,
                                      arr, 
                                      function="mcgv._div_R",
                                      ntabs = ntabs+1
                                      ):
                input("Insane")

        dA_dR         = np.zeros_like(A) / u.cm
        dA_dR[1:-1,:] = (  A[ 2:,  :]  - A[:-2, :] ) / (2 * self.DRR[1:-1, :] + sys.float_info.epsilon * u.cm)

        return dA_dR

    ######################################################
    def _div_dZ(self,
                A,
                ntabs = 0
                ):

        for arrstr,arr in [(  "A", A        ),
                           ("DZZ", self.DZZ )
                          ]:
            if not self._sanity_check(arrstr,
                                      arr, 
                                      function="mcgv._div_Z",
                                      ntabs = ntabs+1
                                      ):
                input("Insane")

        dA_dZ         = np.zeros_like(A) / u.cm
        dA_dZ[1:-1,:] = ( A[ 2:,  :] - A[:-2, :] ) / (2 * self.DZZ[1:-1, :] + sys.float_info.epsilon * u.cm)

        return dA_dZ

    ######################################################
    def _enforce_boundary_conditions(self,
                                     ntabs = 0
                                     ):
        for arrstr,arr in [("1.0/self.mydisk.dzdr", 1.0/self.mydisk.dzdr),
                           ("temperature", self.temperature ),
                           ("RR", self.RR),
                           ("ZZ", self.ZZ)
                          ]:
            if not self._sanity_check(arrstr,
                                      arr, 
                                      function="mcgv._enforce_boundary_conditions",
                                      ntabs = ntabs+1
                                      ):
                input("Insane")


        init_mask = (self.ZZ/self.mydisk.rg < 1.2 * self.mydisk.zt1[:,None])

        v_angle = -np.arctan(1.0/self.mydisk.dzdr)
        thermal_parameter = (const.k_B.cgs * self.temperature / (const.u.cgs * const.c.cgs * const.c.cgs)).decompose()
        average_v_mj = 2 * const.c.cgs * thermal_parameter * (thermal_parameter + 1) * (np.exp(-1/(thermal_parameter + sys.float_info.epsilon))  + sys.float_info.epsilon) / (kn(2, 1/(thermal_parameter + sys.float_info.epsilon))  + sys.float_info.epsilon)
        if np.any(np.isnan(average_v_mj)):
            average_v_mj = np.sqrt(3 * const.k_B.cgs * self.temperature / const.u.cgs).to(u.cm/u.s)

        ss = np.sqrt(self.RR * self.RR + self.ZZ * self.ZZ)

        v_R = average_v_mj[init_mask] * np.sin((np.broadcast_to(v_angle, self.RR.T.shape).T)[init_mask])
        v_Z = average_v_mj[init_mask] * np.cos((np.broadcast_to(v_angle, self.RR.T.shape).T)[init_mask])
        v_P = np.sqrt(const.G * self.mypars.mbh / (ss[init_mask] + sys.float_info.epsilon * u.cm) ).decompose(bases=u.cgs.bases)
        vsq = v_R * v_R + v_Z * v_Z + v_P * v_P
        lorentz_factor = 1.0 / np.sqrt(1. - vsq / (const.c.cgs * const.c.cgs))

        self.u[:, init_mask] = np.array([v_R.to(u.cm/u.s).value, 
                                         v_Z.to(u.cm/u.s).value, 
                                         v_P.to(u.cm/u.s).value
                                         ]) * lorentz_factor * (u.cm/u.s)

        if np.any(np.isnan(self.u)):
            input("paused in mcgv._enforce_boundary_conditions for de-nanification")

    ######################################################
    # Relativistic Euler equation residuals
    def _EULER_cylindrical(self, 
                           ln_number_density,
                           u,
                           which_residual, # = ['lnrho','uR','uZ','uph']
                           ntabs = 0
                           ):  
        for arrstr,arr in [("u", self.u),
                           ("ln_number_density", ln_number_density ),
                           ("RR", self.RR),
                           ("ZZ", self.ZZ)
                          ]:
            if not self._sanity_check(arrstr,
                                      arr, 
                                      function="mcgv._EULER_cylindrical",
                                      ntabs = ntabs+1
                                      ):
                input("Insane")

        u_R   = u[0,:,:]
        u_Z   = u[1,:,:]
        u_phi = u[2,:,:]

        number_density = np.exp(ln_number_density)

        specific_enthalpy = self._specific_enthalpy(self.ref_number_density * number_density)
        KE_R  = number_density * specific_enthalpy * u_R * u_R
        KE_Z  = number_density * specific_enthalpy * u_Z * u_Z
        KE_RZ = number_density * specific_enthalpy * u_R * u_Z
        KE_P  = number_density * specific_enthalpy * u_phi * u_phi

        match which_residual:
            case 'lnrho': # Continuity
                # Mass flux in r, z directions ---
                massflux_R = number_density * u_R
                massflux_Z = number_density * u_Z

                dR_mdot_RdR = self._div_dR(self.RR * massflux_R) /  self.RR
                dmdot_dZ    = self._div_dZ(massflux_Z)

                lhs_rho = (dR_mdot_RdR + dmdot_dZ) / (const.u.cgs * self.ref_number_density)
                rhs_rho = np.zeros_like(lhs_rho) # Place holder for not having continuity

                return lhs_rho - rhs_rho

            case 'uR': # Radial Momentum
                dRKE_R_dR = self._div_dR(self.RR * KE_R)
                dKE_RZ_dZ = self._div_dZ(KE_RZ)
                dP_dR     = self._div_dR(self._P_gas(self.ref_number_density * number_density,
                                                     ntabs = ntabs+1
                                                     ) / (const.u * self.ref_number_density * self.ref_velocity**2))

                lhs_R   = np.zeros((self.mypars.wind_nr,self.mypars.wind_nz))
                rhs_R   = np.zeros((self.mypars.wind_nr,self.mypars.wind_nz))

                lhs_R[self.boundary_mask] = \
                    ( dRKE_R_dR[self.boundary_mask] / self.RR[self.boundary_mask] + dKE_RZ_dZ[self.boundary_mask] - KE_P[self.boundary_mask] / self.RR[self.boundary_mask] ) \
                    + dP_dR[self.boundary_mask]

                aRtot = self.BH_gR[self.boundary_mask] + self.disk_gR[self.boundary_mask] + self._g_rad_R()[self.boundary_mask]
                rhs_R[self.boundary_mask] = number_density[self.boundary_mask] * self._lorentz_factor()[self.boundary_mask] * aRtot

                return lhs_R - rhs_R

            case 'uZ': # Vertical Momentum
                dRKE_RZ_dR = self._div_dR(self.RR * KE_RZ)
                dKE_Z_dZ   = self._div_dZ(KE_Z)
                dP_dZ      = self._div_dZ(self._P_gas(self.ref_number_density * number_density,
                                                      ntabs = ntabs+1
                                                      ) / (const.u * self.ref_number_density * self.ref_velocity**2))

                lhs_Z   = np.zeros((self.mypars.wind_nr,self.mypars.wind_nz))
                rhs_Z   = np.zeros((self.mypars.wind_nr,self.mypars.wind_nz))

                lhs_Z[self.boundary_mask] = \
                    ( dRKE_RZ_dR[self.boundary_mask] / self.RR[self.boundary_mask] + dKE_Z_dZ[self.boundary_mask] ) \
                    + dP_dZ[self.boundary_mask]

                aZtot = (self.BH_gZ[self.boundary_mask] + self.disk_gZ[self.boundary_mask] + self._g_rad_Z()[self.boundary_mask]) 
                rhs_Z[self.boundary_mask] = number_density[self.boundary_mask] * self._lorentz_factor()[self.boundary_mask] * aZtot

                return lhs_Z - rhs_Z

            case 'uph': # Azimuthal equation (w/ angular momentum conservation)
                ell = specific_enthalpy * self.RR * u_phi

                RDvRell = self.RR * number_density * u_R * ell
                DvZell  =           number_density * u_Z * ell

                dRDvRell_dR = self._div_dR(RDvRell)
                dDvZell_dZ  = self._div_dZ(DvZell)

                lhs_phi = np.zeros((self.mypars.wind_nr,self.mypars.wind_nz))
                rhs_phi = np.zeros((self.mypars.wind_nr,self.mypars.wind_nz)) # If there were any torque terms...

                lhs_phi[self.boundary_mask] = ( (dRDvRell_dR[self.boundary_mask] / self.RR[self.boundary_mask]) + \
                                                 dDvZell_dZ[self.boundary_mask] 
                                               )

                return lhs_phi - rhs_phi

    ######################################################
    # Relativistic Euler equation residuals
    def _EULER_cylindrical_wrapper(self, 
                                   x,
                                   ntabs = 0
                                  ):
        ln_number_density_nodim, velocity_vector = self._EULER_unpack_vector(x)

        residuals = np.array([])
        for which_residual in ['lnrho','uR','uZ','uph']:
            residuals = np.append(residuals, 
                                  self._EULER_cylindrical(ln_number_density_nodim, 
                                                          velocity_vector, 
                                                          which_residual
                                                         ).ravel()
                                 )

        return residuals

    ######################################################
    def _EULER_pack_vector(self,
                           ln_number_density_nodim,
                           velocity_vector_nodim,
                           ntabs = 0
                           ):
        x = ln_number_density_nodim.ravel()
        for i in range(velocity_vector_nodim.shape[0]):
            x = np.append(x, velocity_vector_nodim[i,:,:].ravel())
        return x

    ######################################################
    def _EULER_unpack_vector(self,
                             x
                             ):
        ln_number_density_nodim = x[              :  self.RR.size].reshape(self.RR.shape)
        velocity_vector_r_nodim = x[  self.RR.size:2*self.RR.size].reshape(self.RR.shape)
        velocity_vector_z_nodim = x[2*self.RR.size:3*self.RR.size].reshape(self.RR.shape)
        velocity_vector_p_nodim = x[3*self.RR.size:              ].reshape(self.RR.shape)

        velocity_vector = np.array([velocity_vector_r_nodim,
                                    velocity_vector_z_nodim,
                                    velocity_vector_p_nodim
                                    ])

        return ln_number_density_nodim, velocity_vector

    ######################################################
    # Cylindrical R- and Z- components of the radiative force per unit mass
    def _g_rad_R(self,
                 ntabs = 0
                 ):
        g_rad = (const.G.cgs * self.mypars.mbh / (self.RR * self.mydisk.rg + sys.float_info.epsilon * u.cm)**2) * self.Eddington_ratio * self.MRgrid  
        return g_rad * self.mydisk.rg / self.ref_velocity**2

    def _g_rad_Z(self,
                 ntabs = 0
                 ):
        g_rad = (const.G.cgs * self.mypars.mbh / (self.RR * self.mydisk.rg + sys.float_info.epsilon * u.cm)**2) * self.Eddington_ratio * self.MZgrid 
        return g_rad * self.mydisk.rg / self.ref_velocity**2

    ######################################################
    def _lorentz_factor(self):
        u = self.u * self.ref_velocity
        return np.sqrt(1 + np.sum(u*u, axis = 0)/(const.c.cgs*const.c.cgs))

    ######################################################
    def _lorentz_factor(self):
        return np.sqrt(1 + np.sum(self.u*self.u, axis = 0)/(const.c.cgs*const.c.cgs))

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
               number_density, # with dimensions
               ntabs = 0
               ):

        for arrstr,arr in [("number density", number_density),  
                           (   "temperature", self.temperature)
                          ]:
            if not self._sanity_check(arrstr,arr, function="mcgv._P_gas"):
                input("Insane")

        return const.k_B.cgs * number_density * self.temperature

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
    def _specific_enthalpy(self,
                           number_density, # with dimensions
                           ntabs = 0
                           ):
      pressure_nodim = self._P_gas(number_density,
                                   ntabs = ntabs+1
                                  ) /(const.u * number_density*const.c**2 + sys.float_info.epsilon * (u.erg / u.cm**3))
      specific_enthalpy = 1 + pressure_nodim * self.adiabatic_index/(self.adiabatic_index-1.)

      return specific_enthalpy

    ######################################################
    def plot_grid(self,
                  rho_tmp,
                  tplt,
                  t0,
                  t1,
                  dtime,
                  euler_boundary_mask,
                  plotstream = False,
                  time_out_nit = 0,
                  ntabs = 0
                  ):
        mdu = u.g / u.cm**3

        lorentz_factor_c = self._lorentz_factor() * const.c.cgs
        v_R_val  = (self.u[0, :-1, :-1] * self.ref_velocity / lorentz_factor_c[:-1, :-1] ).decompose()
        v_Z_val  = (self.u[1, :-1, :-1] * self.ref_velocity / lorentz_factor_c[:-1, :-1] ).decompose()
        v_ph_val = (self.u[2, :-1, :-1] * self.ref_velocity / lorentz_factor_c[:-1, :-1] ).decompose()

        g_R_val = (((self._g_rad_R() + self.BH_gR + self.disk_gR)[:-1,:-1]) * self.ref_accel).to(u.km/u.s**2).value + sys.float_info.epsilon
        g_Z_val = (((self._g_rad_Z() + self.BH_gZ + self.disk_gZ)[:-1,:-1]) * self.ref_accel).to(u.km/u.s**2).value + sys.float_info.epsilon

        num_dens_val = (self.ref_number_density * np.exp(self.ln_number_density[:-1,:-1])).to(u.cm**-3).value + sys.float_info.epsilon
        temp_val     = self.temperature[:-1,:-1].to(u.K).value + sys.float_info.epsilon

        logMR = np.log10(np.fabs(self.MRgrid[:-1,:-1]) + sys.float_info.epsilon)
        logMZ = np.log10(np.fabs(self.MZgrid[:-1,:-1]) + sys.float_info.epsilon)

        if (tm.time() * u.s - tplt > 30 * u.s) and plotstream:
            fig = plt.gcf()
            fig.clf()
            for (pnum,title,colarr) in [( 1, r'$v_R$/c',                                            v_R_val),
                                        ( 2, r'$v_Z/c$',                                            v_Z_val),
                                        ( 3, r'$v_\phi/c$',                                        v_ph_val),
                                        ( 4, r'Boundary Mask',                  self.boundary_mask[:-1,:-1]),
                                        ( 5, r'$g_\mathrm{R}$ [km s$^{-2}$]',                       g_R_val),
                                        ( 6, r'$g_\mathrm{Z}$ [km s$^{-2}$]',                       g_Z_val),
                                        ( 7, r'$\log T/[K]$',                            np.log10(temp_val)),
                                        ( 8, r'$\log n/$[cm$^{-3}$]',                np.log10(num_dens_val)),
                                        ( 9, r'$\log |M_\mathrm{R}|$',                                logMR),
                                        (10, r'$\log |M_\mathrm{Z}|$',                                logMZ)
                                        ]:
                self.plot_panel(pnum,
                                title,
                                colarr,
                                euler_boundary_mask
                                )
            fig.tight_layout()

        pltstr  = f' Simulated time: {self._mcgv_time(self.tottime):e} \n Time step: {self._mcgv_time(dtime):e} \n'
        pltstr += f' Time since last write/plot: {self._mcgv_timer(t0):.0f}/{self._mcgv_timer(tplt):.0f}\n'
        pltstr += f' Run time: {self._mcgv_timer(t1)} \n'

        pltstr +=  r'$\Delta\rho/\rho$ range: '

        pltstr += f' Number of simulated cells: {np.sum(self.boundary_mask)} ({time_out_nit})'
        titeration = tm.time() * u.s
        if plotstream:
          plt.annotate(pltstr,(0.7,0.02),xycoords='figure fraction',fontsize=14,color='w',backgroundcolor='b')
          plt.pause(1)


    ######################################################
    def plot_panel(self,
                   pnum,
                   title,
                   colarr,
                   euler_boundary_mask
                   ):
        fig = plt.gcf()
        ax = plt.subplot(3,4,pnum)
        ax.clear()

        ax.set_title(title)
        cmap = plt.get_cmap('viridis').copy()
        cmap.set_under('black')
        cmap.set_over('red')
        mesh = ax.pcolormesh(self.RR, 
                            self.ZZ, 
                            colarr,
                            cmap = 'viridis',
                            shading='flat'
                            )
        fig.colorbar(mesh, 
                     ax=ax
                    )
        if pnum == 13:
            ax.set_xlabel(r'R ($r_g$)')
            ax.set_ylabel(r'Z ($r_g$)')
        ax.plot(self.mydisk.rstar, 
                    self.mydisk.diskheight,
                    'b'
                    )
        ax.plot(self.mydisk.rstar, 
                    self.mydisk.zt1,
                    'r'
                    )
        theta = np.linspace(0,np.pi/2,50)
        ax.plot(self.mypars.minrg_bound * np.cos(theta),
                self.mypars.minrg_bound * np.sin(theta),
                'r'
                )
        for pltr in self.R:
            ax.plot(pltr * np.ones(self.Z.size), 
                    self.Z, 
                    'k:', 
                    alpha=0.1)
        for pltz in self.Z:
            ax.plot(self.R, 
                    pltz * np.ones(self.R.size), 
                    'k:', 
                    alpha=0.1
                    )

        ax.scatter(self.RR[self.boundary_mask & ~euler_boundary_mask].flatten(), 
                   self.ZZ[self.boundary_mask & ~euler_boundary_mask].flatten(), 
                   c='m', 
                   s=2, 
                   alpha=1
                   )
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
        # The vlos scalar field should have a shape (self.mypars.wind_nr,self.mypars.wind_nz,nphi)
        self.vlos = np.empty((self.mypars.wind_nr,self.mydisk.ntheta,nphi))
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
            self.RR            = hdul[1].data['R2D']
            self.ZZ            = hdul[1].data['Z2D']
            self.u             = np.zeros((3,) + self.RR.shape)
            self.u[0,:]        = hdul[1].data['uR2D']
            self.u[1,:]        = hdul[1].data['uZ2D']
            self.u[2,:]        = hdul[1].data['uphi2D']
            number_density     = hdul[1].data['rho2D'] * u.cm**-3
            self.temperature   = hdul[1].data['T2D'] * (u.K)
            self.BH_gR         = hdul[1].data['BH_gR']
            self.BH_gZ         = hdul[1].data['BH_gZ']
            self.disk_gR       = hdul[1].data['disk_gR']
            self.disk_gZ       = hdul[1].data['disk_gZ']
            self.boundary_mask = hdul[1].data['boundary_mask']

            self.bounded = True

            try:
                self.column_density_table_grid = hdul[2].data['column_density_grid'] / u.cm**2
            except:
                self.column_density_table_grid = np.zeros(self.RR.shape + (self.myatoms.photo_Z.size,)) / u.cm**2

        self.ref_number_density = np.max(number_density)
        self.ln_number_density = np.log(number_density / self.ref_number_density)

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
            shield_cells = (a[:,:] > 0) & (a[:,:] < 1) & (Dmag[:,:] < self.DRR)
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
               #energy_mask = (energy[None,:]] > self.myatoms.photo_E_th[which_ions,None]) & (energy[None,:] < self.myatoms.photo_E_max[which_ions,None])

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
        shield_cells =                  (a > 0) & (a < 1) & (Dmag < self.DRR)
        #                                                 | intersecting sightline

        return shield_cells

    ######################################################
    def update_fm(self,
                  result,
                  dtime = 0.0 * u.s,
                  ntabs = 0
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
            self.plot_grid(const.u * self.ref_number_density * np.exp(self.ln_number_density),
                           -31.0 * u.s,
                           0.0 * u.s,
                           0.0 * u.s,
                           dtime,
                           np.ones(self.RR.shape, dtype="bool"),
                           plotstream = True,
                           ntabs = ntabs+1
                           )
            plt.pause(1)
            self.write_fm()
            self.write_wind()

    ######################################################
    def velocity_check(self):
        # Sanity check velocitiess:
        for v in [self.v_R, self.v_Z, self.v_phi]:
            bad_mask = (np.fabs(v)/const.c.cgs).decompose() > 1
            if np.sum(bad_mask) > 0:
                v[bad_mask] = 0.0 * (u.cm/u.s)

    ######################################################
    def which_cell(self,
                   rcell_vec,
                   ntabs = 0
                   ):
        return (rcell_vec[0] > self.RR - 0.5 * self.DRR) & (rcell_vec[0] < self.RR + 0.5 * self.DRR) & \
               (rcell_vec[2] > self.ZZ - 0.5 * self.DZZ) & (rcell_vec[2] < self.ZZ + 0.5 * self.DZZ)


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
                              self.u[0,:,:],self.u[1,:,:],self.u[2,:,:],
                              self.ref_number_density * np.exp(self.ln_number_density),
                              self.temperature,
                              self.BH_gR,self.BH_gZ,
                              self.disk_gR,self.disk_gZ,
                              self.boundary_mask), 
                        names=['R2D','Z2D','uR2D','uZ2D','uphi2D','rho2D','T2D','BH_gR','BH_gZ','disk_gR','disk_gZ','boundary_mask']
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