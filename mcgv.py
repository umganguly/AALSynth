import numpy             as np
import matplotlib.pyplot as plt
import os
import time              as tm

from astropy                 import constants as const
from astropy                 import units as u
from astropy.io              import fits
from astropy.table           import Table
from multiprocessing         import Pool
from scipy.interpolate       import RegularGridInterpolator, CubicSpline
from tqdm                    import tqdm

from cloudy import cloudy

######################################################
# The mcgv class defines routines for computing a radiatively-driven outflow from Murray, Chaing, Grossman, & Voit (1995)
######################################################
# User inputs for __init__ to define the class:
#        self.mydisk      = instance of an ntdisk class
#        self.ntheta      = number of grid opint for the theta (angle wrt z-axis)
#        self.datapath    = path for the location of DPKW19 tables for interpolating the force multiplier
#
######################################################
# __init__ computes the following:
#        self.Eddington_ratio = Eddington ratio
#        self.nr          = number of grid point in the spherical r direction (set to the same as self.mydisk.nr)
#        self.theta       = grid of spherical theta angles for computation (rad)
#        self.density     = 2D array of densities    (self.nr,self.ntheta) (cm**-3)
#        self.temperature = 2D array of temperatures (self.nr,self.ntheta) (K)
#        self.pgas        = 2D array of gas pressures (self.nr,self.ntheta) (dynes/cm**2)
#        self.pgasfunc    = function to interpolate self.pgas (dynes/cm**2)
#        self.log10densityfunc = function to interpolate log10(self.density.value) (no explicit unit, implicitly cm**-3)
#        self.fmultarray  = 2D array of force multipliers
#                           self.fmultarray[1:,0] is log t (dimensionless optical depth),
#                           self.fmultarray[0,1:] is log xi (ionization parameter)
#        self.fmultgridfunc = function to interpolate fmultarray
# __init__ allocates memory for the following:
#        self.vr          = 2D array of radial    components of velocity vector (self.nr,self.ntheta) (cm/s)
#        self.vtheta      = 2D array of polar     components of velocity vector (self.nr,self.ntheta) (cm/s)
#        self.vphi        = 2D array of azimuthal components of velocity vector (self.nr,self.ntheta) (cm/s)
#
######################################################
# calcstreamline(self, rf, plotstream=False)
#        return r, theta, vr, vtheta, vphi
#        r      = array of   radial coordinates of streamline originating from rf  (rg?)
#        theta  = array of    polar coordinates of streamline originating from rf  (rad)
#        vr     = array of    radial velocities of streamline originating from rf  (cm/s)
#        vtheta = array of     polar velocities of streamline originating from rf  (cm/s)
#        vphi   = array of azumithal velocities of streamline originating from rf  (cm/s)
#
######################################################
# fmultfunc(self, robs, zobs, verbose=False):
#        return np.fabs(np.sum(fm*rhat)), np.fabs(np.sum(fm*thetahat)) #, np.fabs(np.sum(fm*phihat))
#        fm       = force multiplier vector from which components are extracted
#        rhat     = unit vector for radial direction
#        thetahat = unit vector for polar direction
#        phihat   = unit vector for azimuthal direction

######################################################
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

        # Create grids in r and theta (spherical coordinates)
        # For r, use the same grid as the cylindrical r coordinate in myspec
        self.windfile  = rootname + f"-wind_{self.mypars.nr}x{self.mypars.wind_ntheta}.fits"

        # --- Grid ---
        self.rsph_grid = self.mydisk.rstar * self.mydisk.rg.to(u.cm)
        self.thetasph_grid = np.linspace(np.min(np.arcsin(self.mydisk.rstar[0]/self.mydisk.rstar)),   # avoid pole (anthing not directly above the disk)
                                         np.pi/2.0-1e-3,   # avoid midplane - really should avoid disk, this is masked below under Computational Domain
                                         self.mypars.wind_ntheta) * u.rad
        self.dthsph = self.thetasph_grid[1] - self.thetasph_grid[0]

        print("\t" * ntabs + f"Looking for {self.windfile}")
        self.bounded = False
        if os.path.exists(self.windfile):
            print("\t" * (ntabs+1) + f"Reading {self.windfile}")
            with fits.open(self.windfile, hdu=1) as hdul:
                self.tottime       = hdul[1].header['SIMTIME'] * u.s
                self.RR            = hdul[1].data['r2D'] * u.cm
                self.TT            = hdul[1].data['theta2D'] * u.rad
                self.v_rsph        = hdul[1].data['vr2D'] * (u.cm/u.s)
                self.v_thetasph    = hdul[1].data['vtheta2D'] * (u.cm/u.s)
                self.v_phi         = hdul[1].data['vphi2D'] * (u.cm/u.s)
                self.mass_density  = hdul[1].data['rho2D'] * (u.g/u.cm**3)
                self.boundary_mask = hdul[1].data['boundary_mask']

                self.number_density   = self.mass_density / const.u.cgs
                self.bounded = True

            with fits.open(self.windfile, hdu=2) as hdul:
                self.column_density_table_grid = hdul[1].data['column_density_grid'] / u.cm**2

        else:
            print("\t" * (ntabs+1) + f"Not found. Initializing r-theta mesh")
            # --- Meshgrid for geometry ---
            self.RR, self.TT = np.meshgrid(self.rsph_grid, self.thetasph_grid, indexing="ij")
            self.tottime = 0. * u.s


        self.DRR   = np.broadcast_to((self.mydisk.drstar * self.mydisk.rg).to(u.cm).value, self.RR.T.shape).T * u.cm
        self.RRCYL = self.RR * np.sin(self.TT)
        self.ZZ    = self.RR * np.cos(self.TT)

        self.rcell_vecs = np.array([self.RRCYL / self.mydisk.rg, 
                                    np.zeros(self.RRCYL.shape), 
                                    self.ZZ  / self.mydisk.rg
                                    ]) # shape = (3,) + self.RRCYL.shape # units rg

        self.sinth, self.costh = np.sin(self.TT), np.cos(self.TT)
        self.cotth = self.costh / (self.sinth+1e-10)

        # --- Computational Domain ---
        print("\t" * ntabs + f"Setting computational domain")
        self.z0   = self.mydisk.zt1 * self.mydisk.rg.to(u.cm)
        self.z02D = np.broadcast_to(self.z0.to(u.cm), self.ZZ.T.shape).T * u.cm
        if not self.bounded:
            ZZLO = self.ZZ - 0.5 * self.DRR * np.cos(self.TT)
            self.boundary_mask = (ZZLO > np.interp(self.RRCYL, self.rsph_grid, self.z0)) & ((self.RRCYL/self.mydisk.rg).decompose() > self.mydisk.rstar[0])
        self.in_disk =  self.ZZ + 0.5 * self.DRR * np.cos(self.TT) < np.interp(self.RRCYL, self.rsph_grid, self.z0)
        self.in_shield = np.copy(self.boundary_mask)

        # --- Dilution of central force ---
        tau = np.exp(- (self.ZZ - self.z02D)/self.z02D)
        self.modified_Gamma = self.Eddington_ratio * np.exp(-tau)

        # --- Thermodynamics ---
        print("\t" * ntabs + f"Setting initial thermodynamics...")
        if not self.bounded:
            self.number_density = self.mydisk.verticaldensity3(self.ZZ.flatten()/self.mydisk.rg, 
                                                                self.RRCYL.flatten()/self.mydisk.rg
                                                                ).reshape(self.mypars.nr,self.mypars.wind_ntheta)
            self.mass_density   = self.number_density * const.u.cgs
            self.mass_density   = np.where(np.isfinite(self.mass_density) & (self.mass_density > const.u.cgs / u.cm**3), self.mass_density, np.ones((self.mass_density.shape)) * const.u.cgs / u.cm**3)

            self.column_density_table_grid = np.zeros((self.mypars.nr,self.mypars.wind_ntheta,self.myatoms.photo_Z.size)) / u.cm**2

        self.temperature = np.ones(self.number_density.shape) * 2.7 * u.K

        self.adiabatic_index = 5./3.
        self.specific_enthaply = None

        if not os.path.exists(self.windfile):
            print("\t" * ntabs + f"Initializing velocity field")
            # --- Initial Fields ---
            # Use the thermal speed = v_rms = sqrt(3 k T / m) at the tau=1 surface
            # What is the direction from the tau=1 surface? Normal to it?
            # self.mydisk.dzdr gives the slope of the tau=1 surface....
            # slope of normal is -1/self.mydisk.dzdr
            # so normal is at an angle = - np.arctan(self.mydisk.dzdr)
            v_rms           = np.sqrt(3. * const.k_B.cgs * self.temperature / const.u.cgs).decompose(bases=u.cgs.bases)
            self.v_thetasph = - np.broadcast_to(np.cos(np.arctan(self.mydisk.zt1/self.mydisk.rstar) - np.arctan(self.mydisk.dzdr)), (self.mypars.wind_ntheta,self.mypars.nr)).T * v_rms
            self.v_rsph     = np.sqrt(v_rms * v_rms - self.v_thetasph* self.v_thetasph)
            self.v_phi      = np.sqrt(const.G * self.mypars.mbh / (self.RR)).decompose(bases=u.cgs.bases)

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
        # fmult[0,:] = log10(xi)    xi = 4 np.pi Fx / nH     Fx = 0.1-1000 Ryd integrated flux
        # fmult[:,0] = log10(t)      t = rho sigma_e vth / (dv_l/dl) = optical depth parameter
        fmultfile = self.mypars.datapath+"DPKW19_tables/DPKW19_tables/AGN1_Fmult.dat"
        self.fmultarray = np.genfromtxt(fmultfile)

        # Call as fmultgridfunc((lgt,lgxi))
        self.fmultgridfunc = RegularGridInterpolator((self.fmultarray[1:,0],self.fmultarray[0,1:]), self.fmultarray[1:,1:], bounds_error=False, fill_value=0)

        self.forcemultfile  = rootname + f"-fmultgrid_{self.mypars.nr}x{self.mypars.wind_ntheta}.fits"
        print("\t" * ntabs + f"Looking for {self.forcemultfile}")
        if os.path.exists(self.forcemultfile):
          print("\t" * ntabs + f"\tReading {self.forcemultfile}")
          data = Table.read(self.forcemultfile, format="fits")
          self.Mrgrid     = np.array(data['Mrgrid'])
          self.Mthetagrid = np.array(data['Mthetagrid'])
        else:
          print("\t" * ntabs + "\tNot found...")
          self.Mrgrid     = np.zeros(self.RR.shape)
          self.Mthetagrid = np.zeros(self.RR.shape)

        print("\t" * ntabs + "Commiting None-sequitters")
        self.squiggle = None
        self.rf = None
        self.z0_for_rf = None
        self.lorentz_factor = None

    ######################################################
    # Relativistic Euler equation residuals
    def _EULER(self, 
               dtime, 
               which_residual
               ):   #which_residual=['rho','vr','vth','vphi']
        match which_residual:
            case 'rho': # Continuity
                # Mass flux in r, theta directions ---
                massflux_r  = self.mass_density * self.lorentz_factor * self.v_rsph     # g / cm**2 / s
                massflux_th = self.mass_density * self.lorentz_factor * self.v_thetasph # g / cm**2 / s

                dr2mdot_dr         = np.zeros((self.mypars.nr,self.mypars.wind_ntheta)) * (u.g / u.cm / u.s)
                dr2mdot_dr[1:-1,:] = ( ( (self.RR[2:, :]**2) * massflux_r[ 2:,  :] ) - ( (self.RR[:-2, :]**2) * massflux_r[ :-2, :  ] ) ) / (2 * self.DRR[1:-1, :])   # g / cm / s

                dsinthmdot_dth         = np.zeros((self.mypars.nr,self.mypars.wind_ntheta)) * (u.g / u.cm**2 / u.s)
                dsinthmdot_dth[:,1:-1] = ( (  self.sinth[ :, 2:] * massflux_th[ :, 2:] ) - ( self.sinth[:, :-2]   * massflux_th[:  , :-2] ) ) / (2 * self.dthsph.to(u.rad).value) # g / cm**2 / s [ / rad]

                cont = (   dr2mdot_dr / self.RR**2 + dsinthmdot_dth /(self.RR * self.sinth) )

                drho = dtime * cont
                drho[self.in_disk] = 0.0 * (u.g/u.cm**3)

                return drho

            case 'vr': # Radial Equation
                dv_r_dr         = np.zeros_like(self.v_rsph) / np.ones_like(self.DRR)
                dv_r_dr[1:-1,:] = ( self.v_rsph[2:, :] - self.v_rsph[:-2, :] ) / (2 * self.DRR[1:-1, :])

                dv_r_dtheta         = np.zeros_like(self.v_rsph) / np.ones_like(self.TT)
                dv_r_dtheta[:,1:-1] = ( self.v_rsph[:, 2:] - self.v_rsph[:, :-2]) / (2 * self.dthsph)

                dP_dr         = np.zeros((self.mypars.nr,self.mypars.wind_ntheta)) * (u.dyne / u.cm**2) / np.ones_like(self.DRR)
                dP_dr[1:-1,:] = (self.P_total[2:,:] - self.P_total[:-2,:]) / (2 * self.DRR[1:-1, :])

                lhs_r   = np.zeros((self.mypars.nr,self.mypars.wind_ntheta)) * (u.dyne/u.cm**3)
                rhs_r   = np.zeros((self.mypars.nr,self.mypars.wind_ntheta)) * (u.dyne/u.cm**3)

                lhs_r[self.boundary_mask] = (
                    self.mass_density[self.boundary_mask] * self.specific_enthaply[self.boundary_mask] * self.lorentz_factor[self.boundary_mask]**2 *
                    (self.v_rsph[self.boundary_mask] * dv_r_dr[self.boundary_mask]
                     + self.v_thetasph[self.boundary_mask] * u.rad / self.RR[self.boundary_mask] * dv_r_dtheta[self.boundary_mask]
                     - (self.v_thetasph[self.boundary_mask]**2 + self.v_phi[self.boundary_mask]**2) / self.RR[self.boundary_mask]
                    )
                    + dP_dr[self.boundary_mask]
                )
                rhs_r[self.boundary_mask] = (-self.mass_density[self.boundary_mask] * self.lorentz_factor[self.boundary_mask] * const.G.cgs * self.mypars.mbh / (self.RR[self.boundary_mask]**2) + self._f_rad_r()[self.boundary_mask]).decompose(bases=u.cgs.bases)

                dvr = (dtime * (rhs_r - lhs_r)
                       / (self.mass_density * self.specific_enthaply * self.lorentz_factor**2 + 1e-8 * (u.g/u.cm**3))
                       ).decompose(bases=u.cgs.bases)
                dvr[self.in_disk] = 0.0 * (u.cm/u.s)

                return dvr

            case 'vth': # Polar equation
                dv_th_dr         = np.zeros_like(self.v_thetasph) / np.ones_like(self.DRR)
                dv_th_dr[1:-1,:] = (self.v_thetasph[2:,:] - self.v_thetasph[:-2,:]) / (2 * self.DRR[1:-1,:])

                dv_th_dtheta         = np.zeros_like(self.v_thetasph) / np.ones_like(self.TT)
                dv_th_dtheta[:,1:-1] = (self.v_thetasph[:,2:] - self.v_thetasph[:,:-2]) / (2 * self.dthsph)

                dP_dth         = np.zeros((self.mypars.nr,self.mypars.wind_ntheta)) * (u.dyne / u.cm**2) / np.ones_like(self.TT)
                dP_dth[:,1:-1] = (self.P_total[:,2:] - self.P_total[:,:-2]) / (2 * self.dthsph)

                # --- Theta Equation (for v_theta) ---
                lhs_th  = np.zeros((self.mypars.nr,self.mypars.wind_ntheta)) * (u.dyne/u.cm**3)
                rhs_th  = np.zeros((self.mypars.nr,self.mypars.wind_ntheta)) * (u.dyne/u.cm**3)

                lhs_th[self.boundary_mask] = (
                    self.mass_density[self.boundary_mask] * self.specific_enthaply[self.boundary_mask] * self.lorentz_factor[self.boundary_mask]**2 *
                    (self.v_rsph[self.boundary_mask] * dv_th_dr[self.boundary_mask]
                     + self.v_thetasph[self.boundary_mask] * u.rad / self.RR[self.boundary_mask] * dv_th_dtheta[self.boundary_mask]
                     + (self.v_rsph[self.boundary_mask] * self.v_thetasph[self.boundary_mask]) / self.RR[self.boundary_mask]
                     - (self.v_phi[self.boundary_mask]**2) * self.cotth[self.boundary_mask] / self.RR[self.boundary_mask]
                    )
                    + (1 * u.rad/self.RR[self.boundary_mask]) * dP_dth[self.boundary_mask]
                )
                rhs_th[self.boundary_mask] = self._f_rad_th()[self.boundary_mask]

                dvth = (dtime * (rhs_th - lhs_th)
                        / (self.mass_density * self.specific_enthaply * self.lorentz_factor**2 + 1e-8 * (u.g/u.cm**3))
                        ).decompose(bases=u.cgs.bases)
                dvth[self.in_disk] = 0.0 * (u.cm/u.s)

                return dvth

            case 'vph': # Azimuthal equation (w/ angular momentum conservation
                dv_phi_dr        = np.zeros_like(self.v_phi)   / np.ones_like(self.DRR)
                dv_phi_dr[1:-1,:] = ( self.v_phi[2:,:] - self.v_phi[:-2,:] ) / (2 * self.DRR[1:-1, :])
            
                dv_ph_dtheta         = np.zeros_like(self.v_phi)   / np.ones_like(self.TT)
                dv_ph_dtheta[:,1:-1] = ( self.v_phi[:,2:] - self.v_phi[:,:-2] ) / (2 * self.dthsph)

                lhs_phi = np.zeros((self.mypars.nr,self.mypars.wind_ntheta)) * (u.dyne/u.cm**3)

                lhs_phi[self.boundary_mask] = (
                    self.mass_density[self.boundary_mask] * self.specific_enthaply[self.boundary_mask] * self.lorentz_factor[self.boundary_mask]**2 *
                    (self.v_rsph[self.boundary_mask] * dv_phi_dr[self.boundary_mask]
                     + (self.v_thetasph[self.boundary_mask] * u.rad / self.RR[self.boundary_mask]) * dv_ph_dtheta[self.boundary_mask]
                     + (self.v_rsph[self.boundary_mask] * self.v_phi[self.boundary_mask]) / self.RR[self.boundary_mask]
                     + (self.v_thetasph[self.boundary_mask] * self.v_phi[self.boundary_mask]) * self.cotth[self.boundary_mask] / self.RR[self.boundary_mask]
                    )
                )
                # For no explicit azimuthal torque: rhs = 0
                dvph = -(dtime * lhs_phi
                        / (self.mass_density * self.specific_enthaply * self.lorentz_factor**2 + 1e-8 * (u.g/u.cm**3))
                    ).decompose(bases=u.cgs.bases)  # negative sign for conservation
                dvph[self.in_disk] = 0.0 * (u.cm/u.s)

                return dvph

    
    ######################################################
    # Spherical r- and theta- components of the radiative force per unit mass
    def _f_rad_r(self,
                 ntabs = 0
                 ):
        return ((const.G.cgs * self.mypars.mbh * self.mass_density / self.RR**2) * ( (self.rf/self.RR) * (1 - self.modified_Gamma) + self.modified_Gamma * self.Mrgrid ) )

    def _f_rad_th(self,
                  ntabs = 0
                  ):
        f_rad = ((const.G.cgs * self.mypars.mbh * self.mass_density / self.RR**2) * ( (self.rf/self.RR) * (1 - self.modified_Gamma) * self.cotth - (self.z0_for_rf/self.RR) * self.squiggle * self.Mthetagrid ) )
        if np.any(np.isfinite(f_rad) == False):
            print("\t" * ntabs + f"f_rad has Nans:")
            print("\t" * ntabs + f"mass density = {self.mass_density}")
            print("\t" * ntabs + f"RR = {self.RR}")
            print("\t" * ntabs + f"rf = {self.rf}")
            print("\t" * ntabs + f"z0_for_rf = {self.z0_for_rf}")
            print("\t" * ntabs + f"squiggle = {self.squiggle}")
            print("\t" * ntabs + f"Mthetagrid = {self.Mthetagrid}")
            input("Find it")
        return f_rad

    ######################################################
    # Use the velocity gradients to determine the footpoints
    # rf values are cylindrical-r's but assigned for every cell
    # Have to trace the directions of the velocity vectors backwards, and find where the trace intercepts the photosphere (zt1)
    # We really only need to know, for every cell, from which cell did the gas come from. Then we can trace back to the photosphere
    def _get_rf(self):
        rfr_arr = np.copy(self.RR)
        rft_arr = np.copy(self.TT)

        rfz_arr           = rfr_arr * np.cos(rft_arr.value)
        rfrcyl_arr        = rfr_arr * np.sin(rft_arr.value)
        z0_for_rfrcyl_arr = np.interp(rfrcyl_arr, self.rsph_grid, self.z0)

        vmag = np.sqrt(self.v_rsph*self.v_rsph + self.v_thetasph*self.v_thetasph)

        which_cells         = self.boundary_mask & (vmag > 0 * (u.cm/u.s))
        too_far_list        = np.copy(which_cells)
        not_far_enough_list = np.copy(which_cells)
        dt_arr        = (np.ones_like(self.RR) * u.year / u.cm).to(u.s)

        rfiteration = 0
        rfreallydone = False
        while not rfreallydone:
            rfdone = False
            while not rfdone:
                too_far_list[which_cells] = (np.fabs(dt_arr[which_cells] * self.v_rsph[which_cells]) >= 2. * self.DRR[which_cells]) | \
                    (np.fabs(dt_arr[which_cells] * self.v_thetasph[which_cells]/self.RR[which_cells]) >= 2. * self.dthsph.value)
                dt_arr[too_far_list] *= 0.99
                
                not_far_enough_list[which_cells] = (np.fabs(dt_arr[which_cells] * self.v_rsph[which_cells]) <= self.DRR[which_cells]) & \
                    (np.fabs(dt_arr[which_cells] * self.v_thetasph[which_cells]/self.RR[which_cells]) <= self.dthsph.value)
                dt_arr[not_far_enough_list] *= 1.01

                if not np.any(too_far_list[which_cells]) and not np.any(not_far_enough_list[which_cells]):
                    rfdone = True

            rfr_arr[which_cells] -= dt_arr[which_cells] * self.v_rsph[which_cells]
            rft_arr[which_cells] -= dt_arr[which_cells] * self.v_thetasph[which_cells] * u.rad / self.RR[which_cells]

            rfz_arr[which_cells]    = rfr_arr[which_cells] * np.cos(rft_arr[which_cells].value)
            rfrcyl_arr[which_cells] = rfr_arr[which_cells] * np.sin(rft_arr[which_cells].value)

            z0_for_rfrcyl_arr[which_cells] = np.interp(rfrcyl_arr[which_cells], self.rsph_grid, self.z0)

            which_cells = self.boundary_mask & (vmag > 0 * (u.cm/u.s)) & (z0_for_rfrcyl_arr < rfz_arr)

            if not np.any(which_cells):
                rfreallydone = True

            rfiteration += 1

        rf = np.copy(rfrcyl_arr)

        rf[self.in_disk] = self.RR[self.in_disk] * np.sin(self.TT[self.in_disk])

        z0_for_rf = np.zeros_like(rf)
        z0_for_rf[self.boundary_mask] = np.interp(rf[self.boundary_mask], self.rsph_grid, self.z0)
        z0_for_rf[rf == 0] = 0.0

        return rf,z0_for_rf
    
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
    # Ideal gas law for gas pressure
    def _P_gas(self,
               ntabs = 0
               ):
        return const.k_B.cgs * self.mass_density * self.temperature / const.u.cgs

    # Radiative force per unit mass for radiative pressure gradient
    def _P_rad(self,
               ntabs = 0):
        return (np.sqrt(self._f_rad_r()**2 + self._f_rad_th()**2) * self.DRR).decompose(bases=u.cgs.bases) 

    ######################################################
    def projectvlos(self,
                    ntabs = 0
                    ):
        # The (Cartesian) vector pointing to Theo is
        r_Theo = np.array([self.mydisk.robs * np.cos(self.mydisk.thetaobs),
                           self.mydisk.robs * np.sin(self.mydisk.thetaobs),
                           self.mydisk.zobs])

        # Need to take the dot product of the velocity vector field with the direction of Theo (from each of the cells!)
        # Want to convert spherical (vr,vtheta,vphi) to cartesian (vx,vy,vz)
        # https://en.wikipedia.org/wiki/Vector_fields_in_cylindrical_and_spherical_coordinates#Vector_fields_2 says howto do this.
        # Problem - we don't have a grid in phi... how to determine the 3D field from the rotation about the z-axis?
        # Use the self.mydisk.ntheta (cylindrical theta) to grid in phi.
        nphi = np.int16(np.max(self.mydisk.ntheta))
        phi = np.linspace(0,2*np.pi,nphi)
        # The vlos scalar field should have a shape (self.mypars.nr,self.ntheta,nphi)
        self.vlos = np.empty((self.mypars.nr,self.ntheta,nphi))
        for i in range(self.mypars.nr):
            for j in range(self.mypars.wind_ntheta):
                vx = self.vr[i,j] * np.sin(self.theta[i,j]) * np.cos(phi) + self.vtheta[i,j] * np.cos(self.theta[i,j]) * np.cos(phi) - self.vphi[i,j] * np.sin(self.phi)
                vy = self.vr[i,j] * np.sin(self.theta[i,j]) * np.sin(phi) + self.vtheta[i,j] * np.cos(self.theta[i,j]) * np.sin(phi) + self.vphi[i,j] * np.cos(self.phi)
                vz = self.vr[i,j] * np.cos(self.theta[i,j])               - self.vtheta[i,j] * np.sin(self.theta[i,j])
                v_all_ij_cells = np.array([vx,vy,vz])

                # We need the (unit) vector pointing from the [i,j,k] cell to Theo
                # The cells are at
                r_all_ij_cells = np.array([self.r[i] * np.sin(self.theta[i,j]) * np.cos(self.phi),
                                           self.r[i] * np.sin(self.theta[i,j]) * np.sin(self.phi),
                                           self.r[i] * np.cos(self.theta[i,j])
                                           ])
                # So, the cells-to-Theo vectors are
                r_cell = np.array([self.r[i] * np.sin(self.theta[i,j]) * np.cos(self.phi), self.r[i] * np.sin(self.theta[i,j]) * np.sin(self.phi), self.r[i] * np.cos(self.theta[i,j])])
                R = np.broadcast_to(r_Theo, (nphi,3)).T - r_cell

                self.vlos[i,j] = np.sum(v_all_ij_cells * R, axis=0)/np.sqrt(np.sum(R * R, axis=0))


    ######################################################
    def _sanity_check(self,
                      arrstr,
                      arr,
                      ntabs = 0
                      ):
        sanity = True
        if not np.all(np.isfinite(arr)):
            print("\t" * ntabs + f'\t\t\tNaN values in {arrstr} = {arr}')
            sanity = False
        return sanity

    ######################################################
    def shield_optical_depth(self,
                             shield_cells,
                             energy,
                             ntabs = 0
                             ):
        shield_optical_depth = np.zeros(energy.shape)
        #shield_cell_indices = np.argwhere(shield_cells)

        for shield_cell_column_densities in self.column_density_table_grid[shield_cells,:]:
            for k in range(self.myatoms.photo_Z.size):
                spec_column_density = shield_cell_column_densities[k]
                if spec_column_density > 0:
                    energy_mask = (energy > self.myatoms.photo_E_th[k]) & (energy < self.myatoms.photo_E_max[k])

                    x = energy[energy_mask] / (self.myatoms.photo_E_0[k] - self.myatoms.photo_y_0[k])
                    y = np.sqrt(x**2 + self.myatoms.photo_y_w[k]**2)

                    aa = (x-1)**2 + self.myatoms.photo_y_w[k]**2
                    bb = np.power(y, 0.5*(self.myatoms.photo_p[k]-11))
                    cc = np.power(1 + np.sqrt(y / self.myatoms.photo_y_a[k]), self.myatoms.photo_p[k])

                    cross_section = self.myatoms.photo_sig_0[k] * aa * bb * cc

                    try:
                        shield_optical_depth[energy_mask] += (cross_section * spec_column_density).decompose()
                    except:
                        print("\t" * ntabs + f"{shield_optical_depth[energy_mask]} += {cross_section} * {spec_column_density}")
                        input("paused")

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
                              robs_vec,
                              R_vec,
                              ntabs = 0
                              ):
        gg = self.rcell_vecs - robs_vec[:,None,None]
        a = np.sum(gg * R_vec[:,None,None], axis=0 ) / np.sum(R_vec * R_vec)
        D = robs_vec[:,None,None] + a * R_vec[:,None,None] - self.rcell_vecs
        Dmag = np.sqrt(np.sum( D * D, axis=0))

        #                             | "along" sightline |
        shield_cells = self.in_shield & (a > 0) & (a < 1) & (Dmag < self.DRR / self.mydisk.rg)
        #                                                 | intersecting sightline

        return shield_cells

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
