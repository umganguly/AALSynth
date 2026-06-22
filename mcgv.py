import numpy             as np
import matplotlib.pyplot as plt
import os
import time              as tm

from astropy                 import constants as const
from astropy                 import units as u
from astropy.io              import fits, ascii
from astropy.table           import Table
from astropy.visualization   import astropy_mpl_style, quantity_support
from multiprocessing         import cpu_count, Pool
from scipy.interpolate       import griddata, RegularGridInterpolator, CubicSpline
from tqdm                    import tqdm

from cloudy import cloudy
from corona import corona
from ntdisk import ntdisk

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
    def __init__(self, mydisk, mycorona, myatoms, ntheta, datapath): # Input is an ntdisk class
        self.mydisk      = mydisk
        self.mycorona    = mycorona
        self.myatoms     = myatoms
        self.datapath    = datapath
        mdotedd          = 4 * np.pi * const.G.cgs * self.mydisk.mbh / (0.1 * const.c.cgs * (const.sigma_T.cgs/const.u.cgs))
        self.Eddington_ratio = (self.mydisk.mdot / mdotedd).decompose()

        # Create grids in r and theta (spherical coordinates)
        # For r, use the same grid as the cylindrical r coordinate in myspec
        self.nr          = self.mydisk.nr
        self.ntheta      = ntheta

        self.windfile    = self.datapath+f"Sbh{self.mydisk.sbh}-MBH{np.log10(self.mydisk.mbh / const.M_sun):.2f}-Mdot{(self.mydisk.mdot/(const.M_sun/u.year)).decompose()}-alpha{self.mydisk.alpha}-wind_{self.nr}x{self.ntheta}.fits"

        # --- Grid ---
        #rmin, rmax = self.mydisk.rstar[0] * self.mydisk.rg, self.mydisk.rstar[-1] * self.mydisk.rg    # m
        self.r     = self.mydisk.rstar * self.mydisk.rg.to(u.cm) # = np.linspace(rmin, rmax, self.nr)
        self.theta = np.linspace(np.min(np.arcsin(self.mydisk.rstar[0]/self.mydisk.rstar)), np.pi/2.0-1e-3, self.ntheta) * u.rad  # avoid pole and midplane
        self.dr    = self.mydisk.drstar * self.mydisk.rg
        self.dth   = self.theta[1] - self.theta[0]

        print(f"\tLooking for {self.windfile}")
        self.bounded = False
        if os.path.exists(self.windfile):
            print(f"\t\tReading {self.windfile}")
            with fits.open(self.windfile) as hdul:
                self.tottime       = hdul[1].header['SIMTIME'] * u.s
                self.RR            = hdul[1].data['r2D'] * u.cm
                self.TT            = hdul[1].data['theta2D'] * u.rad
                self.v_r           = hdul[1].data['vr2D'] * (u.cm/u.s)
                self.v_theta       = hdul[1].data['vtheta2D'] * (u.cm/u.s)
                self.v_phi         = hdul[1].data['vphi2D'] * (u.cm/u.s)
                self.mass_density  = hdul[1].data['rho2D'] * (u.g/u.cm**3)
                self.boundary_mask = hdul[1].data['boundary_mask']
                self.bounded = True
        else:
            print(f"\t\tNot found. Initializing r-theta mesh")
            # --- Meshgrid for geometry ---
            self.RR, self.TT = np.meshgrid(self.r, self.theta, indexing="ij")
            self.tottime = 0. * u.s

        self.DRR   = np.broadcast_to(self.dr.to(u.cm).value, self.RR.T.shape).T * u.cm
        self.RRCYL = self.RR * np.sin(self.TT)
        self.ZZ    = self.RR * np.cos(self.TT)

        self.adiabatic_index = 5./3.

        self.RR_cyl   = (self.RRCYL / self.mydisk.rg).value
        self.ZZ_cyl   = (   self.ZZ / self.mydisk.rg).value

        self.sinth, self.costh = np.sin(self.TT), np.cos(self.TT)
        self.cotth = self.costh / (self.sinth+1e-10)

        self.Mrgrid     = np.zeros((self.nr,self.ntheta))
        self.Mthetagrid = np.zeros((self.nr,self.ntheta))

        # --- Computational Domain ---
        print(f"\tSetting computational domain")
        self.z0   = self.mydisk.zt1 * self.mydisk.rg.to(u.cm)
        self.z02D = np.broadcast_to(self.z0.to(u.cm), self.ZZ.T.shape).T * u.cm
        if not self.bounded:
            self.boundary_mask = (self.ZZ > np.interp(self.RRCYL, self.r, self.z0)) & (self.RR_cyl > self.mydisk.rstar[0])
        self.in_disk =  self.ZZ < np.interp(self.RRCYL, self.r, self.z0)

        # --- Dilution of central force ---
        tau = np.exp(- (self.ZZ - self.z02D)/self.z02D)
        self.modified_Gamma = self.Eddington_ratio * np.exp(-tau)

        print(f"\tSetting initial thermodynamics...")
        if not self.bounded:
            self.number_density = self.mydisk.verticaldensity3(self.ZZ.flatten()/self.mydisk.rg, self.RRCYL.flatten()/self.mydisk.rg).reshape(self.nr,self.ntheta)
            self.mass_density   = self.number_density * const.u.cgs
            self.mass_density   = np.where(np.isfinite(self.mass_density) & (self.mass_density > const.u.cgs / u.cm**3), self.mass_density, np.ones((self.mass_density.shape)) * const.u.cgs / u.cm**3)

        self.temperature    = self.mydisk.verticaltemperature3(self.ZZ.flatten()/self.mydisk.rg, self.RRCYL.flatten()/self.mydisk.rg).reshape(self.nr,self.ntheta)
        self.temperature    = np.where(self.temperature < 2.7 * u.K, 2.7 * np.ones_like(self.temperature), self.temperature)

        if not os.path.exists(self.windfile):
            print(f"\tInitializing velocity field")
            # --- Initial Fields ---
            # Use the thermal speed = v_rms = sqrt(3 k T / m) at the tau=1 surface
            # What is the direction from the tau=1 surface? Normal to it?
            # self.mydisk.dzdr gives the slope of the tau=1 surface....
            # slope of normal is -1/self.mydisk.dzdr
            # so normal is at an angle = - np.arctan(self.mydisk.dzdr)
            v_rms        = np.sqrt(3. * const.k_B.cgs * self.temperature / const.u.cgs).decompose(bases=u.cgs.bases)
            self.v_theta = - np.broadcast_to(np.cos(np.arctan(self.mydisk.zt1/self.mydisk.rstar) - np.arctan(self.mydisk.dzdr)), (self.ntheta,self.nr)).T * v_rms
            self.v_r     = np.sqrt(v_rms * v_rms - self.v_theta * self.v_theta)
            self.v_phi   = np.sqrt(const.G.cgs * self.mydisk.mbh.to(u.g) / (self.RR.to(u.cm)))

        print("\tSetting force multiplier functions")

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
        fmultfile = self.datapath+"DPKW19_tables/DPKW19_tables/AGN1_Fmult.dat"
        self.fmultarray = np.genfromtxt(fmultfile)

        # Call as fmultgridfunc((lgt,lgxi))
        self.fmultgridfunc = RegularGridInterpolator((self.fmultarray[1:,0],self.fmultarray[0,1:]), self.fmultarray[1:,1:], bounds_error=False, fill_value=0)

    ######################################################
    # Solver for _EULER
    def calcstreamline_relativistic(self, dtime = 10.0 * u.s, mindt = 1.0 * u.s, vres = 2000.0 * (u.km/u.s), minr_rg = 100.0, plotstream=False, mupdate = True):

        t1 = tm.time() * u.s
        print('\t\tWelcome to your friendly neighborhood relativistic fluid dynamics solver...')

        self.squiggle = self._read_squiggle()

        # --- Get the force multiplier for the (self.rstar, self.theta) grid set up in __init__ ---
        #forcemultfile = self.datapath+f"Sbh{self.mydisk.sbh}-MBH{np.log10(self.mydisk.mbh / const.M_sun):.2f}-Mdot{(self.mydisk.mdot/(const.M_sun/u.year)).decompose()}-alpha{self.mydisk.alpha}-fmultgrid_{self.nr}x{self.ntheta}.fits"
        #print(f"\t\tLooking for {forcemultfile}   ({self._mcgv_timer(t1)})")
        #if os.path.exists(forcemultfile):
        #    print("\t\t\tReading "+forcemultfile)
        #    data = Table.read(forcemultfile, format="fits")
        #    self.Mrgrid     = np.array(data['Mrgrid'])
        #    self.Mthetagrid = np.array(data['Mthetagrid'])
        #else:
        #    nproc = 10 #np.max(np.array([cpu_count()-3, 1]))
        #    nnit  = self.nr//nproc + 1
        #    print(f"\t\tNot found. Starting calculation of force multiplier grid using {nproc} cores requiring {nnit} iterations...")
        #    if nproc > 1:
        #        rdx = -1
        #        for iteration in range(nnit): #tqdm(range(nnit), desc='Force Multiplier Grid '):
        #            list_of_Mgrid_tuples = []
        #            print(f'\t\t\titeration {iteration}: radius indices {iteration*nproc} - {(iteration+1)*nproc-1}')
        #            with Pool() as pool:
        #                poolinput = []
        #                for procnum in range(nproc): # do nproc at a time
        #                    rdx += 1
        #                    if rdx < self.nr:
        #                        if np.any(self.boundary_mask[rdx,:]):
        #                            poolinput.append( (self.RRCYL[rdx,self.boundary_mask[rdx,:]]/self.mydisk.rg, self.ZZ[rdx,self.boundary_mask[rdx,:]]/self.mydisk.rg) )
        #                list_of_Mgrid_tuples = pool.starmap(self.fmultfunc, poolinput)
        #            j = -1
        #            for (mrg,mtg) in list_of_Mgrid_tuples:
        #                j += 1
        #                self.Mrgrid[iteration*nproc+j,     self.boundary_mask[iteration*nproc+j,:] ] = np.copy(mrg)
        #                self.Mthetagrid[iteration*nproc+j, self.boundary_mask[iteration*nproc+j,:] ] = np.copy(mtg)
        #            rdx = np.min(np.array([rdx,self.nr-1]))
        #            print(f"\t\t\tr[{rdx}/{self.nr}] = {r[rdx]/self.mydisk.rg} rg with {np.extract(self.boundary_mask[rdx,:], np.ones(self.ntheta)).size} angular divisions   ({self._mcgv_timer(t1)}, {self._mcgv_timer(t1 - (tm.time() * u.s - t1)*np.sum(self.boundary_mask[rdx:,:])/np.sum(self.boundary_mask[:rdx,:]))})")
        #    else:
        #        for rdx in range(self.nr):
        #            print(f"\t\t\tr[{rdx}/{self.nr}] = {r[rdx]/self.mydisk.rg} rg with {np.extract(self.boundary_mask[rdx,:], np.ones(self.ntheta)).size} angular divisions   ({self._mcgv_timer(t1)}, {self._mcgv_timer(t1 - (tm.time() * u.s - t1)*np.sum(self.boundary_mask[rdx:,:])/np.sum(self.boundary_mask[:rdx,:]))})")
        #            mrg,mrt = self.fmultfunc(self.RRCYL[rdx,self.boundary_mask[rdx,:]]/self.mydisk.rg, self.ZZ[rdx,self.boundary_mask[rdx,:]]/self.mydisk.rg)
        #            self.Mrgrid[    rdx,:] = mrg
        #            self.Mthetagrid[rdx,:] = mrt

        #    data = Table(data=[self.Mrgrid,self.Mthetagrid], names=["Mrgrid","Mthetagrid"])
        #    data.write(forcemultfile, format="fits")

        # --- Iterative Solver ---
        if plotstream:
            plt.close("all")
            plt.ion()
            plt.figure()
            plt.pause(10)
            if os.name == 'nt':
                plt.get_current_fig_manager().window.state("zoomed")
        t0      = tm.time() * u.s
        dvmax   = 1.0 * (u.km/u.s)
        done    = False
        titeration = tm.time() * u.s
        tplt = tm.time() * u.s
        while not done:
            # --- Velocity matters ---
            vmag = np.sqrt(self.v_r**2 + self.v_theta**2 + self.v_phi**2)
            vmag = np.where(vmag >= const.c.to(u.cm/u.s), 0.99 * (const.c.to(u.cm/u.s)), vmag)
            self.lorentz_factor = 1. / np.sqrt(1.0 - ((vmag / const.c).decompose())**2)

            trf = tm.time() * u.s
            self.rf,self.z0_for_rf = self._get_rf()
            trf = tm.time() * u.s - trf

            # -- Radiation force, pressure and gas pressure ---
            self.P_total = self._P_gas() + self._P_rad()

            # Thermodynamics
            self.specific_enthaply = 1 + self.adiabatic_index/(self.adiabatic_index-1.) * (self.P_total)/(self.mass_density*const.c**2)

            # Residuals
            tres = tm.time() * u.s
            nproc = 1
            if nproc > 1:
                poolinput = [(dtime, 'rho'),
                             (dtime,  'vr'),
                             (dtime, 'vth'),
                             (dtime, 'vph')
                             ]
                list_of_euler_linarray_tuples = []
                with Pool() as pool:
                    list_of_euler_linarray_tuples = pool.starmap(self._EULER, poolinput)

                if len(list_of_euler_linarray_tuples) == 4:
                    drho = list_of_euler_linarray_tuples[0]
                    dvr  = list_of_euler_linarray_tuples[1]
                    dvth = list_of_euler_linarray_tuples[2]
                    dvph = list_of_euler_linarray_tuples[3]
                else:
                    input("EULER barfed")
            else:
                drho = self._EULER(dtime, 'rho')
                dvr  = self._EULER(dtime,  'vr')
                dvth = self._EULER(dtime, 'vth')
                dvph = self._EULER(dtime, 'vph')               
            tres = tm.time() * u.s - tres

            # Is our step small enough to remain physical? If so, update velocity and density fields
            tupdate = tm.time() * u.s
            rho_tmp = np.copy(self.mass_density)
            where_density_changed = self.boundary_mask & (np.fabs(drho) / rho_tmp > 0.1)
            if (np.max(np.fabs([dvr[self.boundary_mask].to(u.km/u.s),dvth[self.boundary_mask].to(u.km/u.s),dvph[self.boundary_mask].to(u.km/u.s)])) < vres.to(u.km/u.s).value) and np.all(drho[self.boundary_mask] < rho_tmp[self.boundary_mask]):
                # Relax towards enforcing continuity (conservative update)
                self.mass_density[self.boundary_mask] -= drho[self.boundary_mask]
                self.v_r[         self.boundary_mask] += dvr[ self.boundary_mask]
                self.v_theta[     self.boundary_mask] += dvth[self.boundary_mask]
                self.v_phi[       self.boundary_mask] += dvph[self.boundary_mask]
                self.tottime                          += dtime

                # Update the force multiplier grid
                # self.fmultfunc(robs, zobs)
                if mupdate:
                    mrg,mrt = self.fmultfunc(where_density_changed)
                    self.Mrgrid[    where_density_changed] = mrg
                    self.Mthetagrid[where_density_changed] = mrt

                dtime *= np.pi
                if self.tottime > 1.0e+8 * u.year:
                    done = True
            else:
                dtime /= np.exp(1.0)

            vmag  = np.sqrt(self.v_r*self.v_r + self.v_theta*self.v_theta + self.v_phi*self.v_phi)
            dvmag = (self.v_r * dvr + self.v_theta * dvth + self.v_phi * dvph) / vmag
            vminpred = vmag + mindt * dvmag / dtime
            where_velocity_bad = self.boundary_mask & (vminpred > const.c)

            # Is the timestep too small? Do we need to mask additional bins?
            if dtime < mindt:
                maxdv = np.max(np.fabs([dvr[self.boundary_mask].to(u.cm/u.s), dvth[self.boundary_mask].to(u.cm/u.s), dvph[self.boundary_mask].to(u.cm/u.s)])) * (u.cm/u.s)
                self.boundary_mask = (self.boundary_mask &
                                 (np.fabs(dvr) < maxdv) &
                                 (np.fabs(dvth) < maxdv) &
                                 (np.fabs(dvph) < maxdv) &
                                 (vminpred < const.c) &
                                 (rho_tmp * dtime/-drho > mindt)
                                 )
            tupdate = tm.time() * u.s - tupdate

            # --- Output ---
            v_r_val  = (self.v_r[    :-1, :-1].to(u.km/u.s)).value
            v_th_val = (self.v_theta[:-1, :-1].to(u.km/u.s)).value
            v_ph_val = (self.v_phi[  :-1, :-1].to(u.km/u.s)).value

            dvr_val  = (dvr[ :-1, :-1].to(u.km/u.s)).value
            dvth_val = (dvth[:-1, :-1].to(u.km/u.s)).value
            dvph_val = (dvph[:-1, :-1].to(u.km/u.s)).value

            f_rad_r_val  = (self._f_rad_r()  / self.mass_density )[:-1,:-1].decompose(bases=u.cgs.bases).to(u.km/u.s**2).value
            f_rad_th_val = (self._f_rad_th() / self.mass_density )[:-1,:-1].decompose(bases=u.cgs.bases).to(u.km/u.s**2).value

            rgg_val = (-self.mass_density * self.lorentz_factor * const.G.cgs * self.mydisk.mbh / self.RR**2)[:-1,:-1].value

            if (tm.time() * u.s - tplt > 30 * u.s) and plotstream:
                plt.clf()
                for (pnum,title,colarr) in [(1, r'$\log |v_r|$/[km s$^{-1}$]',                                                                 np.log10(np.fabs(v_r_val))),
                                            (2, r'$\log |v_\theta|$/[km s$^{-1}$]',                                                           np.log10(np.fabs(v_th_val))),
                                            (3, r'$\log |v_\phi|$/ [km s$^{-1}$]',                                                            np.log10(np.fabs(v_ph_val))),
                                            (4, r'$\log r_\mathrm{f}/$[r$_g$]',                                                 np.log10(self.rf[:-1,:-1]/self.mydisk.rg)),
                                            (5, r'$\Delta v_\mathrm{r}$ [km s$^{-1}]$',                                                                           dvr_val),
                                            (6, r'$\Delta v_\mathrm{\theta}$ [km s$^{-1}]$',                                                                     dvth_val),
                                            (7, r'$\Delta v_\mathrm{\phi}$ [km s$^{-1}]$',                                                                       dvph_val),
                                            (8, r'$\xi$',                                                                                          self.squiggle[:-1,:-1]),
                                            (9, r'$\log |f_\mathrm{rad,r}/\rho|$/[km s$^{-2}]$',                                           np.log10(np.fabs(f_rad_r_val))),
                                            (10, r'$\log |f_\mathrm{rad,\theta}/\rho|$/[km s$^{-2}]$',                                    np.log10(np.fabs(f_rad_th_val))),
#                                        (11, r'$-\gamma g$ [km s$^{-2}]$',                                                                                    rgg_val),
                                            (11, r'$\log |\Delta \rho/\rho|$',                                          np.log10(np.fabs(drho[:-1,:-1]/rho_tmp[:-1,:-1]))),
                                            (12, r'$\log \rho/$[cgs]',                                                 np.log10(self.mass_density[:-1,:-1].value+1.0e-17)),
                                            (13, r'$\log M_\mathrm{r}$',      np.log10(    self.Mrgrid[:-1,:-1] + 0.1*np.min(np.extract(self.Mrgrid > 0.0, self.Mrgrid)))),
                                            (14, r'$\log M_\mathrm{\theta}$', np.log10(self.Mthetagrid[:-1,:-1] + 0.1*np.min(np.extract(self.Mrgrid > 0.0, self.Mrgrid)))),
                                            (16, r'Boundary Mask',                                                                            self.boundary_mask[:-1,:-1])
                                            ]:
                    plt.subplot(4,4,pnum)
                    plt.title(title)
                    plt.pcolormesh(self.RR_cyl, self.ZZ_cyl, colarr,  shading='flat')
                    plt.xlabel(r'r ($r_g$)')
                    plt.ylabel(r'z ($r_g$)')
                    plt.plot(self.mydisk.rstar, self.mydisk.diskheight)
                    plt.plot(self.mydisk.rstar, self.mydisk.zt1)
                    for pltr in np.logspace(np.log10(self.mydisk.rstar[0]),np.log10(self.mydisk.rstar[-1]),num=10):
                        plt.plot(pltr * np.cos(self.theta), pltr * np.sin(self.theta), 'k:', alpha=0.1)
                    for pltth in (np.linspace(0.0, np.pi/2.0, num=90) * u.rad):
                        plt.plot(self.mydisk.rstar * np.cos(pltth), self.mydisk.rstar * np.sin(pltth), 'k:', alpha=0.1)
                    plt.scatter(self.RR_cyl[where_density_changed].flatten(), self.ZZ_cyl[where_density_changed].flatten(), c='r', s=2, alpha=1)
                    plt.scatter(self.RR_cyl[where_velocity_bad].flatten(), self.ZZ_cyl[where_velocity_bad].flatten(), c='m', s=2, alpha=1)
                    plt.xlim(left = self.mydisk.rstar[0]) #, right = 3.0e+3)
                    plt.ylim(bottom = 0.3) #, top = 3.0e+3)
                    plt.xscale("log")
                    plt.yscale("log")
                    plt.colorbar()
                    plt.tight_layout()
                tplt = tm.time() * u.s
            dvmax = np.max(np.fabs(np.array([dvr_val,dvth_val,dvph_val]))) * (u.km/u.s)
            pltstr  = f' Simulated time: {self.tottime:e} \n Time step: {dtime:e} \n'
            pltstr += f' Time since last write/plot: {self._mcgv_timer(t0):.0f}/{self._mcgv_timer(tplt):.0f}\n'
            pltstr += f' Run time: {self._mcgv_timer(t1)} \n'
            pltstr += f' Max change in velocity: {(dvmax/const.c).decompose():.2e} c \n'
            pltstr +=  ' Max '+r'$\Delta\rho/\rho$: '+f'{np.max((drho[self.boundary_mask]/rho_tmp[self.boundary_mask]).decompose()):.2e} \n'
            pltstr += f' Number of cells with '+r'$|\Delta\rho|/\rho>0.1$: '+f'{np.sum(where_density_changed)} \n'
            pltstr += f' Number of simulated cells: {np.sum(self.boundary_mask)} \n'
            pltstr += f' {trf:.2f} {tres:.2f} {tupdate:.2f} {tm.time() * u.s - titeration:.2f}'
            titeration = tm.time() * u.s
            if plotstream:
                plt.annotate(pltstr,(0.52,0.05),xycoords='figure fraction',fontsize=14,color='w',backgroundcolor='b')

                plt.show(block=False)
                plt.pause(0.01)

            sane = True
            for arrstr,arr in [('squiggle',                         self.squiggle),
                               ('f_rad_r',                        self._f_rad_r()),
                               ('f_rad_th',                      self._f_rad_th()),
                               ('P_total',                           self.P_total),
                               ('rho',      self.mass_density[self.boundary_mask]),
                               ('lorentz_factor',             self.lorentz_factor),
                               ('rf',                 self.rf[self.boundary_mask]),
                               ('v_r',               self.v_r[self.boundary_mask]),
                               ('v_theta',       self.v_theta[self.boundary_mask]),
                               ('v_phi',           self.v_phi[self.boundary_mask]),
                               ('dvr',                                        dvr),
                               ('dvth',                                      dvth),
                               ('dvph',                                      dvph)
                               ]:
                arrsanity = self._sanity_check(arrstr,arr)
                if not arrsanity:
                    sane = False
            if not sane:
                input("We've gone insane...")

            if done or (tm.time() * u.s - t0 > 300. * u.s):
                datatab = Table(data=(self.RR,self.TT,self.v_r,self.v_theta,self.v_phi,self.mass_density,self.boundary_mask), names=['r2D','theta2D','vr2D','vtheta2D','vphi2D','rho2D','boundary_mask'])
                table_hdu = fits.BinTableHDU(data=datatab)
                table_hdu.header['SIMTIME'] = (self.tottime.value,'Simulated time (s)')
                hdul = fits.HDUList([fits.PrimaryHDU(), table_hdu])
                hdul.writeto(self.windfile, overwrite=True)
                t0 = tm.time() * u.s

    ######################################################
    def _cloudypool(self, i, j, gridx, gridy,cloudypath):
        fu = (u.erg / (u.s * u.cm * u.cm * u.Hz))
        ionspecfreq = np.logspace(13,19,num=3000) * u.Hz
        linarray = Table(data=(['no','nope','no way jose'], [1,2,3]), names=['ID', 'Grr'])
        lognuFnu = self.lognuFnugrid[i,j]
        if lognuFnu == 0.0:
            ionspecflux = (np.zeros(3000) + 1.0e-100) * fu
            for r in range(self.mydisk.rstar.size):
                # Reset observer location...
                self.mydisk.robs     = self.mydisk.rstar[i] * np.sin(self.theta[j]) # Cylindrical r
                self.mydisk.thetaobs = 0.0 # Cylindrical theta...
                self.mydisk.zobs     = self.mydisk.rstar[i] * np.cos(self.theta[j]) # z... (cylindrical or Cartesian)
                fluxdiskannulusdivided = self.mydisk.fnudiskannulus(ionspecfreq,r)
                fluxdiskannulus        = np.sum(fluxdiskannulusdivided[:,:,0],axis=1)
                ionspecflux           += fluxdiskannulus
            ionspecflux = np.where(ionspecflux < 1.0e-100 * fu, 1.0e-100 * fu, ionspecflux)

            lognuFnu = np.interp((const.Ryd).to(u.Hz, equivalencies=u.spectral()), ionspecfreq, np.log10((ionspecfreq * ionspecflux).value))
            logvden = np.log10(self.mydisk.verticaldensity_onepoint(gridx,gridy).to(u.cm**-3).value)
            if lognuFnu < 27.38 and logvden > -5.0:
                cloud = cloudy(self.datapath, cloudypath, 0,                 # 0 = emission, 1 = absorption
                               self.myatoms,
                               ionspecfreq, ionspecflux,            # ionizing spectrum
                               0.0, np.log10((self.mydisk.drstar[i] * self.mydisk.rg).to(u.cm).value), logvden,
                               contemp = self.mydisk.tempt1[j].value, rstar = gridx, zstar = gridy
                               )
                rootname = f"EM-hden{logvden:.2f}-nuFnu{lognuFnu:.2f}-rstar{gridx:.2f}-zstar{gridy:.2f}"
                if cloud.cloudyran:
                    linarray    = np.copy(cloud.linarray)
                elif os.path.exists("./Cloudy_runs/"+rootname+".lin"):
                    if os.path.getsize("./Cloudy_runs/"+rootname+".lin") > 0:
                        linarray = ascii.read(f"./Cloudy_runs/{rootname}.lin", format='commented_header', header_start=0, data_start=1, delimiter='\t', guess=False)

        return lognuFnu, linarray

    ######################################################
    # Relativistic Euler equation residuals
    def _EULER(self, dtime, which_residual):   #which_residual=['rho','VR','VTH','VPHI']
        # Continuity
        if which_residual == 'rho':
            # Mass flux in r, theta directions ---
            massflux_r  = self.mass_density * self.lorentz_factor * self.v_r     # g / cm**2 / s
            massflux_th = self.mass_density * self.lorentz_factor * self.v_theta # g / cm**2 / s

            dr2mdot_dr         = np.zeros((self.nr,self.ntheta)) * (u.g / u.cm / u.s)
            dr2mdot_dr[1:-1,:] = ( ( (self.RR[2:, :]**2) * massflux_r[ 2:,  :] ) - ( (self.RR[:-2, :]**2) * massflux_r[ :-2, :  ] ) ) / (2 * self.DRR[1:-1, :])   # g / cm / s

            dsinthmdot_dth         = np.zeros((self.nr,self.ntheta)) * (u.g / u.cm**2 / u.s)
            dsinthmdot_dth[:,1:-1] = ( (  self.sinth[ :, 2:] * massflux_th[ :, 2:] ) - ( self.sinth[:, :-2]   * massflux_th[:  , :-2] ) ) / (2 * self.dth.to(u.rad).value) # g / cm**2 / s [ / rad]

            cont = (   dr2mdot_dr / self.RR**2 + dsinthmdot_dth /(self.RR * self.sinth) )

            drho = dtime * cont
            drho[self.in_disk] = 0.0 * (u.g/u.cm**3)

            return drho

        # Radial Equation
        if which_residual == 'vr':
            dv_r_dr         = np.zeros_like(self.v_r) / np.ones_like(self.DRR)
            dv_r_dr[1:-1,:] = ( self.v_r[2:, :] - self.v_r[:-2, :] ) / (2 * self.DRR[1:-1, :])

            dv_r_dtheta         = np.zeros_like(self.v_r) / np.ones_like(self.TT)
            dv_r_dtheta[:,1:-1] = ( self.v_r[:, 2:] - self.v_r[:, :-2]) / (2 * self.dth)

            dP_dr         = np.zeros((self.nr,self.ntheta)) * (u.dyne / u.cm**2) / np.ones_like(self.DRR)
            dP_dr[1:-1,:] = (self.P_total[2:,:] - self.P_total[:-2,:]) / (2 * self.DRR[1:-1, :])

            lhs_r   = np.zeros((self.nr,self.ntheta)) * (u.dyne/u.cm**3)
            rhs_r   = np.zeros((self.nr,self.ntheta)) * (u.dyne/u.cm**3)

            lhs_r[self.boundary_mask] = (
                self.mass_density[self.boundary_mask] * self.specific_enthaply[self.boundary_mask] * self.lorentz_factor[self.boundary_mask]**2 *
                (self.v_r[self.boundary_mask] * dv_r_dr[self.boundary_mask]
                 + self.v_theta[self.boundary_mask] * u.rad / self.RR[self.boundary_mask] * dv_r_dtheta[self.boundary_mask]
                 - (self.v_theta[self.boundary_mask]**2 + self.v_phi[self.boundary_mask]**2) / self.RR[self.boundary_mask]
                )
                + dP_dr[self.boundary_mask]
            )
            rhs_r[self.boundary_mask] = (-self.mass_density[self.boundary_mask] * self.lorentz_factor[self.boundary_mask] * const.G.cgs * self.mydisk.mbh / (self.RR[self.boundary_mask]**2) + self._f_rad_r()[self.boundary_mask]).decompose(bases=u.cgs.bases)

            dvr = (dtime * (rhs_r - lhs_r)
                   / (self.mass_density * self.specific_enthaply * self.lorentz_factor**2 + 1e-8 * (u.g/u.cm**3))
                   ).decompose(bases=u.cgs.bases)
            dvr[self.in_disk] = 0.0 * (u.cm/u.s)

            return dvr

        # Polar equation
        if which_residual == 'vth':
            dv_th_dr         = np.zeros_like(self.v_theta) / np.ones_like(self.DRR)
            dv_th_dr[1:-1,:] = (self.v_theta[2:,:] - self.v_theta[:-2,:]) / (2 * self.DRR[1:-1,:])

            dv_th_dtheta         = np.zeros_like(self.v_theta) / np.ones_like(self.TT)
            dv_th_dtheta[:,1:-1] = (self.v_theta[:,2:] - self.v_theta[:,:-2]) / (2 * self.dth)

            dP_dth         = np.zeros((self.nr,self.ntheta)) * (u.dyne / u.cm**2) / np.ones_like(self.TT)
            dP_dth[:,1:-1] = (self.P_total[:,2:] - self.P_total[:,:-2]) / (2 * self.dth)

            # --- Theta Equation (for v_theta) ---
            lhs_th  = np.zeros((self.nr,self.ntheta)) * (u.dyne/u.cm**3)
            rhs_th  = np.zeros((self.nr,self.ntheta)) * (u.dyne/u.cm**3)

            lhs_th[self.boundary_mask] = (
                self.mass_density[self.boundary_mask] * self.specific_enthaply[self.boundary_mask] * self.lorentz_factor[self.boundary_mask]**2 *
                (self.v_r[self.boundary_mask] * dv_th_dr[self.boundary_mask]
                 + self.v_theta[self.boundary_mask] * u.rad / self.RR[self.boundary_mask] * dv_th_dtheta[self.boundary_mask]
                 + (self.v_r[self.boundary_mask] * self.v_theta[self.boundary_mask]) / self.RR[self.boundary_mask]
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


        # Azimuthal equation (w/ angular momentum conservation
        if which_residual == 'vph':
            dv_phi_dr        = np.zeros_like(self.v_phi)   / np.ones_like(self.DRR)
            dv_phi_dr[1:-1,:] = ( self.v_phi[2:,:] - self.v_phi[:-2,:] ) / (2 * self.DRR[1:-1, :])
            
            dv_ph_dtheta         = np.zeros_like(self.v_phi)   / np.ones_like(self.TT)
            dv_ph_dtheta[:,1:-1] = ( self.v_phi[:,2:] - self.v_phi[:,:-2] ) / (2 * self.dth)

            lhs_phi = np.zeros((self.nr,self.ntheta)) * (u.dyne/u.cm**3)

            lhs_phi[self.boundary_mask] = (
                self.mass_density[self.boundary_mask] * self.specific_enthaply[self.boundary_mask] * self.lorentz_factor[self.boundary_mask]**2 *
                (self.v_r[self.boundary_mask] * dv_phi_dr[self.boundary_mask]
                 + (self.v_theta[self.boundary_mask] * u.rad / self.RR[self.boundary_mask]) * dv_ph_dtheta[self.boundary_mask]
                 + (self.v_r[self.boundary_mask] * self.v_phi[self.boundary_mask]) / self.RR[self.boundary_mask]
                 + (self.v_theta[self.boundary_mask] * self.v_phi[self.boundary_mask]) * self.cotth[self.boundary_mask] / self.RR[self.boundary_mask]
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
    def _f_rad_r(self):
        return ((const.G.cgs * self.mydisk.mbh * self.mass_density / self.RR**2) * ( (self.rf/self.RR) * (1 - self.modified_Gamma) + self.modified_Gamma * (1.0 + self.Mrgrid ) ) )

    def _f_rad_th(self):
        return ((const.G.cgs * self.mydisk.mbh * self.mass_density / self.RR**2) * ( (self.rf/self.RR) * (1 - self.modified_Gamma) * self.cotth - (self.z0_for_rf/self.RR) * (1.0 + self.squiggle * self.Mthetagrid) ) )

    ######################################################
    # Given a selection of which_grid_cells, compute the dimensionless optical depth (t)
    #  and the ionization parameter (xi) and interpolate with self.fmultgridfunc
    # Also need to decompose the force multiplier into the [spherical] r-hat and theta-hat directions
    def fmultfunc(self, which_grid_cells, verbose=False):
        t0  = tm.time()
        vth = np.sqrt(2 * const.k_B * (5.0e+4 * u.Kelvin) / const.m_p) # proton thermal speed at 50,000 K (assume)
        # Want to revise vth to use the self.temperature array. But have to map the cylindical robs and zobs to the
        # spherical r-theta grid that defines the 2D self.temperature array.
        # or maybe it is just vth = np.sqrt(2 * const.k_B * (self.temperature[which_grid_cells] * u.Kelvin) / const.m_p)
        fu  = u.erg / (u.s * u.cm * u.cm * u.Hz)
        tol = 1.0e-7

        frequency  = np.logspace(13,20,num=3000) * u.Hz
        dfreq      = np.power(10.0, np.linspace(13,20,num=3000)) * u.Hz

        robs = self.RRCYL[which_grid_cells].flatten() / self.mydisk.rg
        zobs = self.ZZ[   which_grid_cells].flatten() / self.mydisk.rg

        if verbose: print(f"\t\tFor RR = {self.RR[which_grid_cells]}, Robs = [{np.min(robs)},{np.max(robs)}] and Zobs = [{np.min(zobs)},{np.max(zobs)}]")

        # For each cell in which_grid_cells:
        for (robs0,zobs0) in (robs,zobs):
        #     -- Compute the incident flux spectrum and line force per mass using the Gauss-Legendre
        #         quadrature technique from Smith et al.
        #        -- Add in the flux from that sightline
        #     -- Integrate the incident flux over 0.1-1000 Ryd to get Fx
        #     -- Compute the ionization parameter (xi = 4 np.pi Fx / nH --> lgxi)
        #     For each sightline in the Gauss-Legendre sample:
        #        -- Determine n-hat, and whether that sightline intersects the disk photosphere
        #        -- Compute the optical depth parameter (t = rho sigma_e vth / (dv_l/dl) --> lgt)
        #        -- Interpolate the force multipler grid self.fmultgridfunc((lgt,lgxi))
        #        -- Vector add the radiative force in the n-hat direction
           break
        # Everything below here needs to be replaced by Dani's code

        flux      = np.zeros((frequency.size,robs.size)) * fu

        if verbose: print("\t\tAdding in coronal flux")
        flux = self.mycorona.fnu_lamppost(frequency, robs, zobs)

        # Gravity is required in order to compute the Soboloev length and hence get the proper dimensionless optical depth
        # To this, we add the radiation pressure from electron scattering and then iteratively add line pressure until convergence

        # Gravity from the black hole
        # dynes per gram (cm/s^2)
        # shape (robs.size,)
        if verbose: print("\t\tAdding in BH gravity")
        fx   = -((const.G.cgs * self.mydisk.mbh / (self.mydisk.rg * self.mydisk.rg) ) * (robs / np.power(robs * robs + zobs * zobs, 3/2)))
        fy   = -((const.G.cgs * self.mydisk.mbh / (self.mydisk.rg * self.mydisk.rg) ) * (   0 / np.power(robs * robs + zobs * zobs, 3/2)))
        fz   = -((const.G.cgs * self.mydisk.mbh / (self.mydisk.rg * self.mydisk.rg) ) * (zobs / np.power(robs * robs + zobs * zobs, 3/2)))

        if np.any(np.isnan([fx.value,fy.value,fz.value])) and verbose:
            print(f"mcgv.fmultfunc: Black hole gravity NaN'd: {fx} {fy} {fz}")
            input("paused")

        # Gravity from the disk
        # dynes per gram (cm/s^2)
        if verbose: print("\t\tAdding disk gravity")
        [fdx,fdz] = self.mydisk.diskgravity(robs * self.mydisk.rg, zobs * self.mydisk.rg)
        if fdx.size > 1:
            fdx = fdx[0]
            fdz = fdz[0]


        fx += fdx
        fz += fdz

        fx_bh_plus_disk = np.copy(fx)
        fy_bh_plus_disk = np.copy(fy)
        fz_bh_plus_disk = np.copy(fz)

        fesx = np.zeros(fx.shape) * u.cm / u.s**2
        fesy = np.zeros(fy.shape) * u.cm / u.s**2
        fesz = np.zeros(fz.shape) * u.cm / u.s**2

        if np.any(np.isnan([fx.value,fy.value,fz.value])):
            print(f"mcgv.fmultfunc: Disk gravity NaN'd: {fx} {fy} {fz} {fdx} {fdz}")
            input("paused")


        if verbose: print("\t\tElectron scattering + lines")
        self.mydisk.robs = np.copy(robs)
        self.mydisk.zobs = np.copy(zobs)
        self.mydisk.thetaobs = np.zeros(robs.size) * u.rad # hmmm.... not true??? depends on x,y????
        for rdx in range(self.mydisk.rstar.size): #tqdm(range(self.mydisk.rstar.size), desc='mcgv.fmultfunc, annuli loop'):
            dtheta           = 2.0 * np.pi * u.rad / self.mydisk.ntheta[rdx]
            ntr              = int(self.mydisk.ntheta[rdx])
            theta            = np.linspace(0,2.0*np.pi,num=ntr) * u.rad
            self.mydisk.rref = rdx
            # Want to have fluxrt = self.mydisk.fnudiskannulus(frequency,r,theta) which will have shape (frequency.size,ntr,robs.size)

            if verbose: print(f"mcgv.fmultfunc: Sending annulus {rdx}/{self.mydisk.nr} to self.mydisk.fnudiskannulus at {tm.time()-t0}")
            fluxrt = self.mydisk.fnudiskannulus(frequency,rdx) # expect shape (frequency.size,ntr,robs.size)
            if verbose: print(f"mcgv.fmultfunc: Return from fnudiskannulus at {tm.time()-t0}")
            fluxr  = np.sum(fluxrt, axis=1) # expect shape (frequency.size,robs.size) so sum over ntr
            flux  += np.copy(fluxr)         # expect shape (frequency.size,robs.size)

            oldfx       = fx # Keep a copy of the force/mass due to just BH + disk gravity + inner annuli
            oldfy       = fy
            oldfz       = fz

            # shape of self.cosbeta inherited from self.fnudiskannulus is (ntr,robs.size)
            cbdx = np.extract(self.mydisk.cosbeta > 0, range(self.mydisk.cosbeta.size))
            cbdxo = np.int16(np.mod(cbdx,robs.size))
            cbdxt = np.int16((cbdx-cbdxo)/robs.size)
            # magnitude of radiation force per unit area: fluxrt/const.c.cgs
            # [fx] = erg / (sr * cm^3) = dyne / (sr cm^2)
            # direction of radiation force per unit area:

            # Components of the force per unit mass - This is just electron scattering. Need force multiplier for lines/edges
            nufnurt  = np.sum(np.multiply(fluxrt,
                                          np.transpose(np.broadcast_to(dfreq.value,
                                                                       (ntr,robs.size,dfreq.size)),
                                                       (2,0,1)
                                                       ) * u.Hz
                                          ),
                              axis=0) /(const.c.cgs * const.u.cgs)

            fesx += (const.sigma_T.cgs) * np.squeeze(np.sum(nufnurt * self.mydisk.Rx / self.mydisk.Rmag, axis=0))
            fesy += (const.sigma_T.cgs) * np.squeeze(np.sum(nufnurt * self.mydisk.Ry / self.mydisk.Rmag, axis=0))
            fesz += (const.sigma_T.cgs) * np.squeeze(np.sum(nufnurt * self.mydisk.Rz / self.mydisk.Rmag, axis=0))

            fx += (const.sigma_T.cgs) * np.squeeze(np.sum(nufnurt * self.mydisk.Rx / self.mydisk.Rmag, axis=0))
            fy += (const.sigma_T.cgs) * np.squeeze(np.sum(nufnurt * self.mydisk.Ry / self.mydisk.Rmag, axis=0))
            fz += (const.sigma_T.cgs) * np.squeeze(np.sum(nufnurt * self.mydisk.Rz / self.mydisk.Rmag, axis=0))


            fmdx = np.extract(np.max(flux, axis=0) > 0, range(robs.size))
            if fmdx.size > 0:
                if verbose: print(f"mcgv.fmultfunc: Force multiplying at {tm.time()-t0} mark")
                csflux = CubicSpline(frequency.to(u.Hz).value,
                                     flux[:,fmdx].to(fu).value)
                csiflux = csflux.integrate((0.1 * ((u.Ry)/const.h).to(u.Hz)).value, (1000 * ((u.Ry)/const.h).to(u.Hz)).value) * fu * u.Hz

                # In lgxi, need to replace r with the indices corresponding to robs...
                rrs = np.zeros(fmdx.size, dtype=np.int16)
                if fmdx.size > 1:
                    for rdx2 in range(fmdx.size):
                        rrs[rdx2] = int(np.max(np.append(np.extract(self.mydisk.rstar <= robs[fmdx[rdx2]], range(self.mydisk.nr)),0)))
                else:
                  rrs = int(np.max(np.append(np.extract(self.mydisk.rstar <= robs, range(self.mydisk.nr)),0)))

                lgxi = np.log10((4 * np.pi * csiflux / self.mydisk.verticaldensity2(self.mydisk.zobs, rrs)).value)
                lSob = (vth * vth / np.sqrt(fx * fx + fy * fy + fz * fz)).decompose()
                lgt  = np.log10((const.sigma_T * self.mydisk.verticaldensity2(zobs, rrs) * lSob).decompose())
                fm = np.power(10.0, self.fmultgridfunc((lgt,lgxi)))

                nit = 5
                while nit > 0:
                    nit -= 1
                    fx = oldfx
                    fy = oldfy
                    fz = oldfz

                    # Components of the force per unit mass
                    fx += np.squeeze(fm) * (const.sigma_T.cgs) * np.squeeze(np.sum(nufnurt * self.mydisk.Rx / self.mydisk.Rmag, axis=0))
                    fy += np.squeeze(fm) * (const.sigma_T.cgs) * np.squeeze(np.sum(nufnurt * self.mydisk.Ry / self.mydisk.Rmag, axis=0))
                    fz += np.squeeze(fm) * (const.sigma_T.cgs) * np.squeeze(np.sum(nufnurt * self.mydisk.Rz / self.mydisk.Rmag, axis=0))

                    lSob = (vth * vth / np.sqrt(fx * fx + fy * fy + fz * fz)).decompose()
                    lgt  = np.log10((const.sigma_T * self.mydisk.verticaldensity2(zobs, rrs) * lSob).decompose())
                    fm = np.power(10.0, self.fmultgridfunc((lgt,lgxi)))

        if np.any(np.isnan([fx.value,fy.value,fz.value])):
            print(f"mcgv.fmultfunc: Radiation Pressure NaN'd: {fx} {fy} {fz} {lSob} {csiflux} {fm}")
            input("paused")

        # Subtract off the force from the BH and disk gravity. That should give us the total radiation pressure force
        fx -= fx_bh_plus_disk
        fy -= fy_bh_plus_disk
        fz -= fz_bh_plus_disk

        # The force multiplier is the line radiation pressure force divided by the force due only to electron scattering
        # This is cast as a vector (3D array) with xyz components
        # NEED TO SUBTRACT ES TO GET JUST LINE PRESSURE!!! -----------------------------------------------V
        fm  = np.array([fx.value,fy.value,fz.value]) / np.sqrt(fesx*fesx + fesy*fesy + fesz*fesz).value - 1.0

        # Need spherical coords.... so define the spherical unit vectors in terms of cartesian vectors
        theta    = np.arctan2(robs,zobs)
        rhat     = np.array([       np.sin(theta), np.zeros(theta.size),        np.cos(theta)])
        thetahat = np.array([       np.cos(theta), np.zeros(theta.size),       -np.sin(theta)])
        phihat   = np.array([np.zeros(theta.size),  np.ones(theta.size), np.zeros(theta.size)])

        # The r and theta components come from the dot products between fm and rhat and thetahat
        return np.fabs(np.sum(fm*rhat)), np.fabs(np.sum(fm*thetahat)) #, np.fabs(np.sum(fm*phihat))

    ######################################################
    def getcloudy(self,cloudypath):
        fu = (u.erg / (u.s * u.cm * u.cm * u.Hz))

        emfile = f"Sbh{self.mydisk.sbh:.2f}-MBH{np.log10(self.mydisk.mbh.to(u.Msun).value):.2f}-Mdot{self.mydisk.mdot.to(u.Msun/u.year).value:.1f}-emission.fits"
        print(f"Looking for {emfile}")
        if os.path.exists(emfile):
            with fits.open(emfile) as hdu:
                emdatatab = hdu[1].data
            gridx = emdatatab['gridr']
            gridy = emdatatab['gridz']
            self.lognuFnugrid = emdatatab['lognuFnu']
            self.emissiongrid = emdatatab['emgrid']

            print(f"Read in {emfile} with max lognuFnu = {np.max(self.lognuFnugrid)} and {np.max(self.emissiongrid)}")
        else:
            print(f"nope")
            self.emissiongrid = np.zeros((self.nr,self.ntheta,6))
            self.lognuFnugrid = np.zeros((self.nr,self.ntheta))
            gridr, gridtheta = np.meshgrid(self.mydisk.rstar,self.theta, indexing='ij')
            gridx = gridr * np.sin(gridtheta)
            gridy = gridr * np.cos(gridtheta)

                    
        for i in range(self.nr):  # Spherical r
            thetadx = np.extract((self.mydisk.rstar[i] * np.cos(self.theta) >= np.interp(self.mydisk.rstar[i] * np.sin(self.theta), self.mydisk.rstar, self.mydisk.zt1)) & 
                                 (self.mydisk.rstar[i] * np.sin(self.theta) >  np.min(self.mydisk.rstar)),
                                 range(self.theta.size)
                                 )
            print(f"\t\tFor rstar[{i}]={self.mydisk.rstar[i]}, running Cloudy for {thetadx.size} theta bins at {np.min(self.theta[thetadx]).to(u.degree)} to {np.max(self.theta[thetadx]).to(u.degree)}")
            if thetadx.size > 0:
                list_of_lognuFnu_linarray_tuples = []
                with Pool() as pool:
                    poolinput = []
                    for procnum in range(thetadx.size):
                        poolinput.append( (i,thetadx[procnum], gridx[i,thetadx[procnum]], gridy[i,thetadx[procnum]], cloudypath) )
                    list_of_lognuFnu_linarray_tuples = pool.starmap(self._cloudypool, poolinput)
                j = -1
                for (lognuFnu, linarray) in list_of_lognuFnu_linarray_tuples:
                    j += 1
                    self.lognuFnugrid[i,thetadx[j]] = lognuFnu
                    self.linarray = linarray
                    print(f"\t\t{self._getprint(i,thetadx[j],gridx,gridy)}")
                emtab = Table(data=(gridx, gridy, self.lognuFnugrid, self.emissiongrid), names=['gridr','gridz','lognuFnu','emgrid'])
                emtab.write(emfile, format='fits', overwrite=True)

            harr = np.extract(self.lognuFnugrid > 0, self.emissiongrid[:,:,0])
            narr = np.extract(self.lognuFnugrid > 0, self.emissiongrid[:,:,2])
            carr = np.extract(self.lognuFnugrid > 0, self.emissiongrid[:,:,4])

            if harr.size > 0:
                red   = 1.0 - (harr-np.min(harr)) / (np.max(harr)-np.min(harr))
                green = 1.0 - (narr-np.min(narr)) / (np.max(narr)-np.min(narr))
                blue  = 1.0 - (carr-np.min(carr)) / (np.max(carr)-np.min(carr))

                gridc = [tuple(row) for row in np.array([red.T,green.T,blue.T]).T.reshape(harr.size,3)]

                plt.clf()
                plt.plot(self.mydisk.rstar, self.mydisk.diskheight)
                plt.plot(self.mydisk.rstar, self.mydisk.zt1)
                plt.scatter(np.extract(self.lognuFnugrid > 0, gridx),
                            np.extract(self.lognuFnugrid > 0, gridy),
                            c = gridc,
                            s = np.extract(self.lognuFnugrid > 0, self.lognuFnugrid))
                plt.ylim(bottom = np.min(self.mydisk.zt1))
                plt.xlim(left = np.min(self.mydisk.rstar))
                plt.xscale("log")
                plt.yscale("log")
                plt.show(block=False)
                plt.pause(0.001)
            else:
                print(f"No emission found in grid????? {np.max(self.lognuFnugrid)}")

    ######################################################
    # Can we use the velocity gradients to determine the footpoints?
    # rf values are cylindrical-r's but assigned for every cell
    # Have to trace the directions of the velocity vectors backwards, and find where the trace intercepts the photosphere (zt1)
    # We really only need to know, for every cell, from whih cell did the gas come from. Then we can trace back to the photosphere
    def _get_rf(self):
        rfr_arr = np.copy(self.RR)
        rft_arr = np.copy(self.TT)

        rfz_arr           = rfr_arr * np.cos(rft_arr.value)
        rfrcyl_arr        = rfr_arr * np.sin(rft_arr.value)
        z0_for_rfrcyl_arr = np.interp(rfrcyl_arr, self.r, self.z0)

        which_cells         = np.copy(self.boundary_mask)
        too_far_list        = np.copy(self.boundary_mask)
        not_far_enough_list = np.copy(self.boundary_mask)
        drf_arr        = (np.ones_like(self.RR) * u.year / u.cm).to(u.s)

        rfiteration = 0
        rfreallydone = False
        while not rfreallydone:
            rfdone = False
            while not rfdone:
                too_far_list[which_cells] = (np.fabs(drf_arr[which_cells] * self.v_r[which_cells]) >= 2. * self.DRR[which_cells]) | (np.fabs(drf_arr[which_cells] * self.v_theta[which_cells]/self.RR[which_cells]) >= 2. * self.dth.value)
                drf_arr[too_far_list] *= 0.99
                
                not_far_enough_list[which_cells] = (np.fabs(drf_arr[which_cells] * self.v_r[which_cells]) <= self.DRR[which_cells]) & (np.fabs(drf_arr[which_cells] * self.v_theta[which_cells]/self.RR[which_cells]) <= self.dth.value)
                drf_arr[not_far_enough_list] *= 1.01
                
                if not np.any(too_far_list[which_cells]) and not np.any(not_far_enough_list[which_cells]):
                    rfdone = True
                    
            rfr_arr[which_cells] -= drf_arr[which_cells] * self.v_r[which_cells]
            rft_arr[which_cells] -= drf_arr[which_cells] * self.v_theta[which_cells] * u.rad / self.RR[which_cells]

            rfz_arr[which_cells]    = rfr_arr[which_cells] * np.cos(rft_arr[which_cells].value)
            rfrcyl_arr[which_cells] = rfr_arr[which_cells] * np.sin(rft_arr[which_cells].value)
            z0_for_rfrcyl_arr[which_cells] = np.interp(rfrcyl_arr[which_cells], self.r, self.z0)

            which_cells = self.boundary_mask & (z0_for_rfrcyl_arr < rfz_arr)

            if not np.any(which_cells):
                rfreallydone = True

            rfiteration += 1

        rf = np.copy(rfrcyl_arr)

        rf[self.in_disk] = self.RR[self.in_disk] * np.sin(self.TT[self.in_disk])

        z0_for_rf = np.zeros_like(rf)
        z0_for_rf[self.boundary_mask] = np.interp(rf[self.boundary_mask], self.r, self.z0)
        z0_for_rf[rf == 0] = 0.0

        return rf,z0_for_rf
    
    ######################################################
    def _mcgv_timer(self, t1):
        if tm.time() * u.s - t1 < 300 * u.s:
            return tm.time() * u.s - t1
        elif (tm.time() * u.s - t1).to(u.minute) < 60 * u.min:
            return (tm.time() * u.s - t1).to(u.minute)
        else:
            return (tm.time() * u.s - t1).to(u.hour)

    ######################################################
    # Ideal gas law for gas pressure
    def _P_gas(self):
        return const.k_B.cgs * self.mass_density * self.temperature / const.u.cgs

    # Radiative force per unit mass for radiative pressure gradient
    def _P_rad(self):
        return (np.sqrt(self._f_rad_r()**2 + self._f_rad_th()**2) * self.DRR).decompose(bases=u.cgs.bases) 

    ######################################################
    def _plot_velocity_field(self):

        #plt.figure(figsize=(10, 8))
        R, Theta = np.meshgrid(self.rstar, self.theta)

        # Display intermediate variables for debugging
        #print("R:", R)
        #print("Theta:", Theta)

        vr_val     = self.vr.to(u.km/u.s).value
        vtheta_val = self.vtheta.to(u.km/u.s).value
        vphi_val   = self.vphi.to(u.km/u.s).value

        vr_plot     = vr_val.T
        vtheta_plot = vtheta_val.T

        Vx_cyl = vr_plot * np.sin(Theta.value) + vtheta_plot * np.cos(Theta.value)
        Vz_cyl = vr_plot * np.cos(Theta.value) - vtheta_plot * np.sin(Theta.value)

        # Display intermediate variables for debugging
        #print("Vx_cyl:", Vx_cyl)
        #print("Vz_cyl:", Vz_cyl)

        X_cyl = R * np.sin(Theta)
        Z_cyl = R * np.cos(Theta)

        # Display intermediate variables for debugging
        #print("X_cyl:", X_cyl)
        #print("Z_cyl:", Z_cyl)

        V_magnitude = np.sqrt(Vx_cyl**2 + Vz_cyl**2)

        non_zero_mask = V_magnitude > 0
        Vx_norm = np.zeros_like(Vx_cyl)
        Vz_norm = np.zeros_like(Vz_cyl)
        Vx_norm[non_zero_mask] = Vx_cyl[non_zero_mask] / V_magnitude[non_zero_mask]
        Vz_norm[non_zero_mask] = Vz_cyl[non_zero_mask] / V_magnitude[non_zero_mask]

        plt.quiver(X_cyl, Z_cyl, Vx_norm, Vz_norm, V_magnitude,
                   cmap='viridis', headlength=4, headwidth=3, pivot='middle', scale_units='xy', scale=0.5)
        plt.colorbar(label='Velocity Magnitude (km/s)')
        #plt.xlabel('Cylindrical Radius (rg)')
        #plt.ylabel('Height (rg)')
        #plt.title('2D Velocity Vector Field (Cylindrical r-z plane)')
        #plt.xscale('log')
        #plt.yscale('log')
        #plt.grid(True, which="both", ls="-", alpha=0.2)
        #plt.axhline(0, color='grey', linestyle='--', linewidth=0.8)

        # Add disk and photosphere plots
        #plt.plot(self.mydisk.rstar,self.mydisk.diskheight, label="disk height")
        #plt.plot(self.mydisk.rstar,self.mydisk.zt1, label="disk photosphere")
        #plt.legend()

        #plt.tight_layout()
        #plt.show(block=False)

    ######################################################
    def projectvlos(self):
        # The (Cartesian) vector pointing to Theo is
        r_Theo = np.array([self.mydisk.robs * np.cos(self.mydisk.thetaobs),
                           self.mydisk.robs * np.sin(self.mydisk.thetaobs),
                           self.mydisk.zobs])

        # Need to take the dot product of the velocity vector field with the direction of Theo (from each of the cells!)
        # Want to convert spherical (vr,vtheta,vphi) to cartesian (vx,vy,vz)
        # https://en.wikipedia.org/wiki/Vector_fields_in_cylindrical_and_spherical_coordinates#Vector_fields_2 says howto do this.
        # Problem - we don't have a grid in phi... how to determine the 3D field from the rotation about the z-axis?
        # Use the self.mydisk.ntheta (cylindrical theta) to grid in phi.
        self.nphi = np.int16(np.max(self.mydisk.ntheta))
        self.phi = np.linspace(0,2*np.pi,nphi)
        # The vlos scalar field should have a shape (self.nr,self.ntheta,nphi)
        self.vlos = np.empty((self.nr,self.ntheta,nphi))
        for i in range(self.nr):
            for j in range(self.ntheta):
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
                R = np.broadcast_to(r_Theo, (nphi,3)).T - r_cell

                self.vlos[i,j] = np.sum(v_all_ij_cells * R, axis=0)/np.sqrt(np.sum(R * R, axis=0))


##                for k in range(nphi):
##                    vx = self.vr[i,j] * np.sin(self.theta[i,j]) * np.cos(phi[k]) + self.vtheta[i,j] * np.cos(self.theta[i,j]) * np.cos(phi[k]) - self.vphi[i,j] * np.sin(self.phi[k])
##                    vy = self.vr[i,j] * np.sin(self.theta[i,j]) * np.sin(phi[k]) + self.vtheta[i,j] * np.cos(self.theta[i,j]) * np.sin(phi[k]) + self.vphi[i,j] * np.cos(self.phi[k])
##                    vz = self.vr[i,j] * np.cos(self.theta[i,j])                  - self.vtheta[i,j] * np.sin(self.theta[i,j])
##                    v_cell = np.array([vx,vy,vz])
##
##                    # We need the (unit) vector pointing from the [i,j,k] cell to Theo
##                    # The cell is at
##                    r_cell = np.array([self.r[i] * np.sin(self.theta[i,j]) * np.cos(self.phi[k]), self.r[i] * np.sin(self.theta[i,j]) * np.sin(self.phi[k]), self.r[i] * np.cos(self.theta[i,j])])
##                    # So, the cell-to-Theo vector is
##                    R = r_Theo - r_cell
##
##                    self.vlos[i,j,k] = np.sum(v_cell*R)/np.sqrt(np.sum(R*R))
                                         

    ######################################################
    # --- Squiggle is the UV fraction of flux from the disk below the cell point --
    def _read_squiggle(self, verbose=False):
        squigglefile = self.datapath+f"Sbh{self.mydisk.sbh}-MBH{np.log10(self.mydisk.mbh / const.M_sun):.2f}-Mdot{(self.mydisk.mdot/(const.M_sun/u.year)).decompose()}-alpha{self.mydisk.alpha}-squiggle_{self.nr}x{self.ntheta}.fits"
        t1 = tm.time() * u.s
        if verbose:
            print(f"\t\tLooking for {squigglefile}")
        if os.path.exists(squigglefile):
            print("\t\t\tReading "+squigglefile)
            data = Table.read(squigglefile, format="fits")
            return np.array(data['squiggle'])
        else:
            print(f"\t\tComputing squiggle (UV fraction from disk)   ({self._mcgv_timer(t1)})")
            squiggle   = np.zeros(self.RR.shape)
            frequency  = np.logspace(13,19,num=3000) * u.Hz
            dfreq      = np.power(10.0, np.linspace(13,19,num=3000)) * u.Hz
            in_uv      = (frequency > (3000.0 * u.Angstrom).to(u.Hz, equivalencies = u.spectral())) & (frequency < (0.25 * u.keV).to(u.Hz, equivalencies = u.spectral()))
            for rdx in range(self.mydisk.nr):
                rlo = np.max([self.mydisk.rstar[rdx] - 0.5*self.mydisk.drstar[rdx],0.0])
                cells_above_rstar = (self.RR_cyl > rlo) & (self.RR_cyl < self.mydisk.rstar[rdx] + 0.5*self.mydisk.drstar[rdx]) & (self.ZZ_cyl > self.mydisk.zt1[rdx])
                print(f"\t\t\trstar[{rdx}/{self.mydisk.nr}] = {self.mydisk.rstar[rdx]} has {np.sum(cells_above_rstar)} cells above it   ({self._mcgv_timer(t1)})")
                self.mydisk.robs = self.RR_cyl[cells_above_rstar].flatten()
                self.mydisk.zobs = self.ZZ_cyl[cells_above_rstar].flatten()
                fluxrt = self.mydisk.fnudiskannulus(frequency, rdx) # fluxrt.shape = (frequency.size,self.mydisk.ntheta[rdx],self.mydisk.robs.size)
                dfreq_arr = np.broadcast_to(dfreq, fluxrt.T.shape).T
                squiggle[cells_above_rstar] = np.sum(fluxrt[in_uv,0,:] * dfreq_arr[in_uv,0,:], axis=0) /  (np.sum(fluxrt[:,0,:] * dfreq_arr[:,0,:], axis=0) + 1.0e-50 * (u.erg / (u.s * u.cm * u.cm * u.Hz)))
                if not self._sanity_check('squiggle',squiggle):
                    input("Squiggle went insane...")
            data = Table(data=[squiggle], names=["squiggle"])
            data.write(squigglefile, format="fits")
            return squiggle

    ######################################################
    def _sanity_check(self,arrstr,arr):
        sanity = True
        if not np.all(np.isfinite(arr)):
            print(f'\t\t\tNaN values in {arrstr} = {arr}')
            sanity = False
        return sanity

    ######################################################
    # Produces the dumstr for printing out a line with H I, N V, and C IV emission lines
    # Also packs self.emissiongrid
    def _getprint(self,i,j,gridx,gridy):
        lyalin = np.extract(self.linarray['ID'] == 'H  1                1215.67A', self.linarray)
        dumstr = f"            {self.mydisk.rstar[i]} {self.theta[j].to(u.degree)} {gridx[i,j]} {gridy[i,j]} {self.lognuFnugrid[i,j]}"
        if lyalin.size > 0:
            self.emissiongrid[i,j,0] = lyalin[0][2]
            dumstr += f"   H I: {lyalin[0][2]}"
            lyblin = np.extract(self.linarray['ID'] == 'H  1                1025.72A', self.linarray)
            if lyblin.size > 0:
                self.emissiongrid[i,j,1] = lyblin[0][2]
                dumstr += f" {lyblin[0][2]}"

        nvb = np.extract(self.linarray['ID'] == 'N  5                1238.82A', self.linarray)
        if nvb.size > 0:
            self.emissiongrid[i,j,2] = nvb[0][2]
            dumstr += "   N V: "
            dumstr += f"{nvb[0][2]}"
            nvr = np.extract(self.linarray['ID'] == 'N  5                1242.80A', self.linarray)
            if nvr.size > 0:
                self.emissiongrid[i,j,3] = nvr[0][2]
                dumstr += f" {nvr[0][2]}"
                                    
        civb = np.extract(self.linarray['ID'] == 'C  4                1548.19A', self.linarray)
        if civb.size > 0:
            self.emissiongrid[i,j,4] = civb[0][2]
            dumstr += f"   C IV: {civb[0][2]}"
            civr = np.extract(self.linarray['ID'] == 'C  4                1550.77A', self.linarray)
            if civr.size > 0:
                self.emissiongrid[i,j,5] = civr[0][2]
                dumstr += f" {civr[0][2]}"

        return dumstr











##    ######################################################
##    # Given a cylindrical robs and zobs, compute the dimensionless optical depth (t) and the ionization parameter (xi) and interpolate with self.fmultgridfunc
##    # Also need to decompose the force multiplier into the [spherical] r-hat and theta-hat directions
##    # robs and zobs here are arrays to allow for the computation at multiple locations
##    # robs and zobs should be units of self.mydisk.rg
##    def fmultfunc(self, robs, zobs, verbose=False):
##        t0  = tm.time()
##        vth = np.sqrt(2 * const.k_B * (5.0e+4 * u.Kelvin) / const.m_p) # proton thermal speed at 50,000 K (assume)
##        fu  = u.erg / (u.s * u.cm * u.cm * u.Hz)
##        tol = 1.0e-7
##
##        frequency  = np.logspace(13,19,num=3000) * u.Hz
##        dfreq      = np.power(10.0, np.linspace(13,19,num=3000)) * u.Hz
##
##        flux      = np.zeros((frequency.size,robs.size)) * fu
##        rave      = np.zeros((frequency.size,robs.size)) * fu
##
##        # Gravity is required in order to compute the Soboloev length and hence get the proper dimensionless optical depth
##        # To this, we add the radiation pressure from electron scattering and then iteratively add line pressure until convergence
##
##        # Gravity from the black hole
##        # dynes per gram (cm/s^2)
##        # shape (robs.size,)
##        fx   = -((const.G.cgs * self.mydisk.mbh / (self.mydisk.rg * self.mydisk.rg) ) * (robs / np.power(robs * robs + zobs * zobs, 3/2)))
##        fy   = -((const.G.cgs * self.mydisk.mbh / (self.mydisk.rg * self.mydisk.rg) ) * (   0 / np.power(robs * robs + zobs * zobs, 3/2)))
##        fz   = -((const.G.cgs * self.mydisk.mbh / (self.mydisk.rg * self.mydisk.rg) ) * (zobs / np.power(robs * robs + zobs * zobs, 3/2)))
##
##        if np.any(np.isnan([fx.value,fy.value,fz.value])) and verbose:
##            print(f"mcgv.fmultfunc: Black hole gravity NaN'd: {fx} {fy} {fz}")
##            input("paused")
##
##        # Gravity from the disk
##        # dynes per gram (cm/s^2)
##        [fdx,fdz] = self.mydisk.diskgravity(robs * self.mydisk.rg, zobs * self.mydisk.rg)
##        if fdx.size > 1:
##            fdx = fdx[0]
##            fdz = fdz[0]
##
##
##        fx += fdx
##        fz += fdz
##
##        fx_bh_plus_disk = np.copy(fx)
##        fy_bh_plus_disk = np.copy(fy)
##        fz_bh_plus_disk = np.copy(fz)
##
##        fesx = np.zeros(fx.shape) * u.cm / u.s**2
##        fesy = np.zeros(fy.shape) * u.cm / u.s**2
##        fesz = np.zeros(fz.shape) * u.cm / u.s**2
##
##        if np.any(np.isnan([fx.value,fy.value,fz.value])):
##            print(f"mcgv.fmultfunc: Disk gravity NaN'd: {fx} {fy} {fz} {fdx} {fdz}")
##            input("paused")
##
##
##        self.mydisk.robs = np.copy(robs)
##        self.mydisk.zobs = np.copy(zobs)
##        self.mydisk.thetaobs = np.zeros(robs.size) * u.rad # hmmm.... not true??? depends on x,y????
##        for rdx in range(self.mydisk.rstar.size): #tqdm(range(self.mydisk.rstar.size), desc='mcgv.fmultfunc, annuli loop'):
##            dtheta           = 2.0 * np.pi * u.rad / self.mydisk.ntheta[rdx]
##            ntr              = int(self.mydisk.ntheta[rdx])
##            theta            = np.linspace(0,2.0*np.pi,num=ntr) * u.rad
##            self.mydisk.rref = rdx
##            # Want to have fluxrt = self.mydisk.fnudiskannulus(frequency,r,theta) which will have shape (frequency.size,ntr,robs.size)
##
##            if verbose: print(f"mcgv.fmultfunc: Sending annulus {rdx}/{self.mydisk.nr} to self.mydisk.fnudiskannulus at {tm.time()-t0}")
##            fluxrt = self.mydisk.fnudiskannulus(frequency,rdx) # expect shape (frequency.size,ntr,robs.size)
##            if verbose: print(f"mcgv.fmultfunc: Return from fnudiskannulus at {tm.time()-t0}")
##            fluxr  = np.sum(fluxrt, axis=1) # expect shape (frequency.size,robs.size) so sum over ntr
##            flux  += np.copy(fluxr)         # expect shape (frequency.size,robs.size)
##            rave  += np.copy(fluxr) * self.rstar[rdx]
##
##            oldfx       = fx # Keep a copy of the force/mass due to just BH + disk gravity + inner annuli
##            oldfy       = fy
##            oldfz       = fz
##
##            # shape of self.cosbeta inherited from self.fnudiskannulus is (ntr,robs.size)
##            cbdx = np.extract(self.mydisk.cosbeta > 0, range(self.mydisk.cosbeta.size))
##            cbdxo = np.int16(np.mod(cbdx,robs.size))
##            cbdxt = np.int16((cbdx-cbdxo)/robs.size)
##            # magnitude of radiation force per unit area: fluxrt/const.c.cgs
##            # [fx] = erg / (sr * cm^3) = dyne / (sr cm^2)
##            # direction of radiation force per unit area:
##
##            # Components of the force per unit mass - This is just electron scattering. Need force multiplier for lines/edges
##            nufnurt  = np.sum(np.multiply(fluxrt,
##                                          np.transpose(np.broadcast_to(dfreq.value,
##                                                                       (ntr,robs.size,dfreq.size)),
##                                                       (2,0,1)
##                                                       ) * u.Hz
##                                          ),
##                              axis=0) /(const.c.cgs * const.u.cgs)
##
##            fesx += (const.sigma_T.cgs) * np.squeeze(np.sum(nufnurt * self.mydisk.Rx / self.mydisk.Rmag, axis=0))
##            fesy += (const.sigma_T.cgs) * np.squeeze(np.sum(nufnurt * self.mydisk.Ry / self.mydisk.Rmag, axis=0))
##            fesz += (const.sigma_T.cgs) * np.squeeze(np.sum(nufnurt * self.mydisk.Rz / self.mydisk.Rmag, axis=0))
##
##            fx += (const.sigma_T.cgs) * np.squeeze(np.sum(nufnurt * self.mydisk.Rx / self.mydisk.Rmag, axis=0))
##            fy += (const.sigma_T.cgs) * np.squeeze(np.sum(nufnurt * self.mydisk.Ry / self.mydisk.Rmag, axis=0))
##            fz += (const.sigma_T.cgs) * np.squeeze(np.sum(nufnurt * self.mydisk.Rz / self.mydisk.Rmag, axis=0))
##
##
##            fmdx = np.extract(np.max(flux, axis=0) > 0, range(robs.size))
##            if fmdx.size > 0:
##                if verbose: print(f"mcgv.fmultfunc: Force multiplying at {tm.time()-t0} mark")
##                csflux = CubicSpline(frequency.value,flux[:,fmdx].value)
##                csiflux = csflux.integrate((0.1 * ((u.Ry)/const.h).to(u.Hz)).value, (1000 * ((u.Ry)/const.h).to(u.Hz)).value) * fu * u.Hz
##
##                # In lgxi, need to replace r with the indices corresponding to robs...
##                rrs = np.zeros(fmdx.size, dtype=np.int16)
##                if fmdx.size > 1:
##                    for rdx2 in range(fmdx.size):
##                        rrs[rdx2] = int(np.max(np.append(np.extract(self.mydisk.rstar <= robs[fmdx[rdx2]], range(self.mydisk.nr)),0)))
##                else:
##                  rrs = int(np.max(np.append(np.extract(self.mydisk.rstar <= robs, range(self.mydisk.nr)),0)))
##
##                lgxi = np.log10((4 * np.pi * csiflux / self.mydisk.verticaldensity2(self.mydisk.zobs, rrs)).value)
##                lSob = (vth * vth / np.sqrt(fx * fx + fy * fy + fz * fz)).decompose()
##                lgt  = np.log10((const.sigma_T * self.mydisk.verticaldensity2(zobs, rrs) * lSob).decompose())
##                fm = np.power(10.0, self.fmultgridfunc((lgt,lgxi)))
##
##                nit = 5
##                while nit > 0:
##                    nit -= 1
##                    fx = oldfx
##                    fy = oldfy
##                    fz = oldfz
##
##                    # Components of the force per unit mass
##                    fx += np.squeeze(fm) * (const.sigma_T.cgs) * np.squeeze(np.sum(nufnurt * self.mydisk.Rx / self.mydisk.Rmag, axis=0))
##                    fy += np.squeeze(fm) * (const.sigma_T.cgs) * np.squeeze(np.sum(nufnurt * self.mydisk.Ry / self.mydisk.Rmag, axis=0))
##                    fz += np.squeeze(fm) * (const.sigma_T.cgs) * np.squeeze(np.sum(nufnurt * self.mydisk.Rz / self.mydisk.Rmag, axis=0))
##
##                    lSob = (vth * vth / np.sqrt(fx * fx + fy * fy + fz * fz)).decompose()
##                    lgt  = np.log10((const.sigma_T * self.mydisk.verticaldensity2(zobs, rrs) * lSob).decompose())
##                    fm = np.power(10.0, self.fmultgridfunc((lgt,lgxi)))
##
##        if np.any(np.isnan([fx.value,fy.value,fz.value])):
##            print(f"mcgv.fmultfunc: Radiation Pressure NaN'd: {fx} {fy} {fz} {lSob} {csiflux} {fm}")
##            input("paused")
##
##        # Subtract off the force from the BH and disk gravity. That should give us the total radiation pressure force
##        fx -= fx_bh_plus_disk
##        fy -= fy_bh_plus_disk
##        fz -= fz_bh_plus_disk
##
##        # The force multiplier is the total radiation pressure force divided by the force due only to electron scattering
##        # NEED TO SUBTRACT ES TO GET JUST LINE PRESSURE!!! -----------------------------------------------V
##        fm  = np.array([fx.value,fy.value,fz.value]) / np.sqrt(fesx*fesx + fesy*fesy + fesz*fesz).value - 1.0
##
##        # Need spherical coords....
##        theta    = np.arctan2(robs,zobs)
##        rhat     = np.array([       np.sin(theta), np.zeros(theta.size),        np.cos(theta)])
##        thetahat = np.array([       np.cos(theta), np.zeros(theta.size),       -np.sin(theta)])
##        phihat   = np.array([np.zeros(theta.size),  np.ones(theta.size), np.zeros(theta.size)])
##
##        return np.fabs(np.sum(fm*rhat)), np.fabs(np.sum(fm*thetahat)) #, np.fabs(np.sum(fm*phihat))
