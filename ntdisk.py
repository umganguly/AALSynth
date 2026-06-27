import os
import numpy             as np
import matplotlib.pyplot as plt
import time              as tm

from astropy                 import constants as const
from astropy                 import units as u
from astropy.io              import fits
from astropy.table           import Table
from astropy.modeling.models import BlackBody
from astropy.visualization   import astropy_mpl_style, quantity_support
from scipy                   import special
from scipy                   import interpolate
from scipy.integrate         import solve_ivp,odeint
from scipy.interpolate       import RegularGridInterpolator, CubicSpline
from scipy.special           import gamma



######################################################
# The ntdisk class defines routines for computing a thin-disk Novikov-Thorne solution as revised by Penna et al.
######################################################
# User inputs for __init__ to define the class:
#        self.sbh         = Spin of the black hole
#        self.mbh         = Mass of the black hole                        (solar masses)
#        self.mdot        = Accretion rate                                (solar masses / year)
#        self.alpha       = Shakura-Sunyaev viscosity parameter
#        self.inclination = inclination of eh observer point              (radians)
#        self.robs        = Cylindrical r component of the observer point (rg)
#        self.nr          = Number of radial annuli
#        self.rlo         = rlo, inner radius
#        self.rhi         = rhi, outer radius of the disk                 (rg)
#        self.datapath    = datapath, string location of disk solutions
#
######################################################
# __init__ computes the following:
#        self.zobs        = z component of the observer point             (rg)
#        self.thetaobs    = azimuthal component of observer point
#        self.rg          = Gravitational radius of the black hole, GM/c^2 (g)
#        self.rms         = radius of ISCO                                (rg)
#        self.rstar       = annuli radii                       (self.nr,) (rg)
#        self.drstar      = annuli thicknesses                            (rg)
#        self.ntheta      = number of azimuthal grid points in each annulus
#
######################################################
# diskgravity(self,robs,zobs): # robs and zobs in normal units (not rg)
#        return [fr,fz]
#        fr               = force per unit mass r-comp (robs.size,) (cm/s**2, dyn/g)
#        fz               = force per unit mass z-comp (zobs.size,) (cm/s**2, dyn/g)
#
######################################################
# Want to have fluxrt = self.fnudiskannlus(frequency,r,theta) which will have shape (frequency.size,self.ntheta[r],self.robs.size)
# fnudiskannulus(self, frequency, r):
#        return fluxrt
#        fluxrt           = flux density from the r-th annulus (frequency.size,self.ntheta[r],self.robs.size) (erg / (s * cm**2 * sr * Hz))
#        frequency        = array of frequencies at which to compute flux density (Hz)
#    R are the vectors from the disk patch to the observer(s) located at (r=self.robs,theta=0 deg,z=self.zobs) [This is the xz plane]
#    The disk patch is located at (r=self.rstar[r],theta=theta,z=self.zt1[r])
#    Rx, Ry, Rz, Rr, and Rmag will have shape (ntr,self.robs.size) and in terms of self.rg
#    Rhat should have shape (ntr,self.robs.size,3)
#        self.Rx
#        self.Ry
#        self.Rz
#        self.Rr
#        self.Rmag
#
######################################################
# makedisk:
#        self.diskheight  = disk scale height                             (rg)
#        self.temperature = mid-plane disk temperature                    (K)
#        self.density     = mid-plane disk density                        (cm**-3)
#        self.x1          = radius of pressure transition (gas/radiation) (rg)
#        self.x2          = radius of opacity change (es/ff)              (rg)
#        self.forceint    = False
#
######################################################
# photosphere(self,makeplot=False):
#        self.zt1         = height of unit optical depth surface          (rg)
#        self.dzdr        = gradient of the disk scale height
#        self.tempt1      = temperature at zt1                            (K)
#        self.denst1      = density at zt1                                (cm**-3)
#
######################################################
# pltdisk(self,i=0):
#        i                = annulus index for title labelling
#
######################################################
# surfdens(self, z, i_annulus): #z0, rho0
#        return srfdense
#        z                = array of heights for which to compute the surface density (z.size,) (rg)
#        i_annulus        = annulus index below the point for the z's (scalar)
#        srfdense         = surface density at z's              (z.size,) (cm**-2)
#
######################################################
# verticaldensity(self, z, i_annulus):
#        return rho
#        z                = array of heights (rg)
#        i_annulus        = annulus index (scalar)
#        rho              = densities                           (z.size,) (cm**-3)
#
# verticaldensity2(self, z, i_annul):
#        return rho
#        z                = array of heights                    (z.size,) (rg)
#        i_annulus        = array of annulus indices            (z.size,)
#        rho              = densities                           (z.size,) (cm**-3)
#
# verticaldensity3(self, z, r):
#        return rho
#        z                = array of heights                    (z.size,) (rg)
#        r                = array of radii                      (z.size,) (rg)
#        rho              = densities                           (z.size,) (cm**-3)
#
# verticaldensity_onepoint(self,r,z):
#        return rho
#        z                = input height                         (scalar) (rg)
#        r                = input radius                         (scalar) (rg)
#        rho              = densities                            (scalar) (cm**-3)
#
######################################################
# verticaltemperature(self, z, i_annulus):
#        return temperature
#        z                = array of heights (rg)
#        i_annulus        = annulus index (scalar)
#        temperature      = temperatures                        (z.size,) (K)
#
# verticaltemperature2(self, z, i_annul):
#        return temperature
#        z                = array of heights                    (z.size,) (rg)
#        i_annulus        = array of annulus indices            (z.size,)
#        temperature      = temperatures                        (z.size,) (K)
#
# verticaltemperature3(self, z, r):
#        return temperature
#        z                = array of heights                    (z.size,) (rg)
#        r                = array of radii                      (z.size,) (rg)
#        temperature      = temperatures                        (z.size,) (K)
#
# verticaltemperature_onepoint(self,r,z):
#        return temperature
#        z                = input height                         (scalar) (rg)
#        r                = input radius                         (scalar) (rg)
#        temperature      = temperature                          (scalar) (K)
#
######################################################
class ntdisk:
    ######################################################
    def __init__(self, sbh, mbh, mdot, alpha, inclination, nr, rlo, rhi, datapath, dtheta_fac = 2.0):
        self.sbh         = sbh
        self.mbh         = mbh
        self.mdot        = mdot
        self.alpha       = alpha
        self.inclination = inclination
        self.robs        = None
        self.zobs        = None
        self.thetaobs    = 0.0 * u.rad
        self.rg          = (const.G.cgs * self.mbh / (const.c.cgs * const.c.cgs)).to(u.cm)
        self.nr          = nr
        self.rlo         = rlo
        self.rhi         = rhi
        self.rref        = 0
        self.diskheight  = np.empty(self.nr)
        self.zt1         = np.empty(self.nr)
        self.dzdr        = np.empty(self.nr)
        self.temperature = np.empty(self.nr)
        self.tempt1      = np.empty(self.nr)
        self.density     = np.empty(self.nr)
        self.denst1      = np.empty(self.nr)
        self.x1          = self.rlo
        self.x2          = self.rhi
        self.forceint    = False
        self.datapath    = datapath

        z1    = np.power(1.0+self.sbh, 1./3.) + np.power(1.0-self.sbh,1./3.)
        z1    = 1.0 + np.power(1.0-self.sbh*self.sbh,1./3.) * ( z1 )
        z2    = np.sqrt(3.0 * self.sbh * self.sbh + z1*z1)
        self.rms   = 3.0 + z2 - np.sqrt((3.0-z1)*(3+z1+2*z2))

        y0    = np.sqrt(self.rms)
        self.rstar = np.squeeze(np.logspace(np.log10(self.rms * 1.01), np.log10(self.rms * self.rhi), num = self.nr))
        self.drstar = np.copy(self.rstar)
        self.drstar[1:-1] = 0.5*(self.rstar[2:]-self.rstar[:-2])
        self.drstar[0]  = self.drstar[0]
        self.drstar[-1] = self.drstar[-2]
        ######################################################
        # Compute the annuli thicknesses
        # rstar = np.squeeze(np.logspace(np.log10(rms*1.01),6.0,num=3000))
        # 0.5*(rstar[i+1]-rstar[i-1]) = rstar[i]*0.5*(f-1/f)
        self.deltar = np.empty(self.nr)
        self.ntheta = np.empty(self.nr, dtype=np.int8)
        self.deltar[1:-1] = 0.5 * (self.rstar[2:] - self.rstar[:-2])
        self.deltar[0]  = self.deltar[1]
        self.deltar[-1] = self.deltar[-2]
        self.ntheta = np.ceil(2.0 * np.pi * self.rstar / (dtheta_fac * self.deltar))

    ######################################################
    def diskgravity(self,robs,zobs): # robs and zobs in normal units (not rg)
        fr = np.zeros(robs.size) * u.cm / u.s**2
        fz = np.zeros(robs.size) * u.cm / u.s**2
        for rdx in range(self.nr):
            mass = (self.surfdens(np.array([30.0*self.diskheight[rdx]])*self.rg,rdx))[0] * const.u.cgs * self.rstar[rdx] * self.deltar[rdx] * self.rg * self.rg * 2.0
            frr = np.zeros(robs.size) * u.cm / u.s**2
            fzr = np.zeros(robs.size) * u.cm / u.s**2

            theta = np.broadcast_to(np.linspace(0,
                                                2.0 * np.pi,
                                                num=np.int32(self.ntheta[rdx])),
                                    (robs.size, np.int32(self.ntheta[rdx])))

            x     = self.rstar[rdx] * self.rg * np.cos(theta)
            y     = self.rstar[rdx] * self.rg * np.sin(theta)

            dx    = x - np.transpose(np.broadcast_to(robs, (np.int32(self.ntheta[rdx]), robs.size))) * u.cm

            R     = np.sqrt(dx*dx + y*y + np.transpose(np.broadcast_to(zobs.value*zobs.value, (np.int32(self.ntheta[rdx]), robs.size))) * u.cm * u.cm)

            frt   = const.G.cgs * mass / (R * R) # magnitude of the force from the mass element

            frtx  = frt * ((x - np.transpose(np.broadcast_to(robs.value, (np.int32(self.ntheta[rdx]), robs.size))) * u.cm) / R)
            frtz  = frt * ((0 - np.transpose(np.broadcast_to(zobs.value, (np.int32(self.ntheta[rdx]), robs.size))) * u.cm) / R)

            frr += np.sum(frtx, axis=1)
            fzr += np.sum(frtz, axis=1)

            fr += frr
            fz += fzr

        return [fr,fz]

    ######################################################
    # Want to have fluxrt = self.fnudiskannlus(frequency,r) which will have shape (frequency.size,self.ntheta[r],self.robs.size)
    def fnudiskannulus(self,
                       frequency,
                       r
                       ):
        fu = u.erg / (u.s * u.cm * u.cm * u.Hz)
        ntr = np.int16(self.ntheta[r])

        theta1D = np.linspace(0,
                              2.0 * np.pi,
                              ntr)
        theta2D = np.broadcast_to(theta1D,
                                  (self.robs.size,ntr)
                                  ).T
        dtheta      = 2.0 * np.pi / self.ntheta[r]
        # R are the vectors from the disk patch to the observer(s) located at (r=self.robs,theta=0 deg,z=self.zobs) [This is the xz plane]
        # The disk patch is located at (r=self.rstar[self.rref],theta=theta,z=self.zt1[self.rref])
        # Rx, Ry, Rz, Rr, and Rmag will have shape (ntr,self.robs.size) and in terms of self.rg
        # Rhat should have shape (ntr,self.robs.size,3)
        #print(f"ntdisk.fnudiskannulus:     self.robs.shape = {self.robs.shape}    self.thetaobs.shape = {self.thetaobs.shape}   ntr = {ntr}   self.robs.size = {self.robs.size}")
        self.Rx   = np.broadcast_to(self.robs * np.cos(self.thetaobs), (ntr,self.robs.size)) - self.rstar[r] * np.cos(theta2D)
        self.Ry   = np.broadcast_to(self.robs * np.sin(self.thetaobs), (ntr,self.robs.size)) - self.rstar[r] * np.sin(theta2D)
        self.Rz   = np.broadcast_to(self.zobs - self.zt1[r], (ntr,self.robs.size))
        self.Rr   = np.sqrt(self.Rx * self.Rx + self.Ry * self.Ry)
        self.Rmag = np.sqrt(self.Rr * self.Rr + self.Rz * self.Rz)

        # Example from numpy:
        #   a = np.ones((1, 2, 3))
        #   np.transpose(a, (1, 0, 2)).shape
        #   (2, 1, 3)
        #
        # [self.Rx,self.Ry,self.Rz] has a shape of .... (3,ntr,self.robs.size)
        #
        self.Rhat = np.transpose([self.Rx,self.Ry,self.Rz], (1,2,0)) / np.transpose(np.broadcast_to(self.Rmag,(3,ntr,self.robs.size)), (1,2,0))

        # shape is (ntr,)
        (gradzx,gradzy,gradzz) = (-self.dzdr[r] * np.cos(theta1D),
                                  -self.dzdr[r] * np.sin(theta1D),
                                  np.ones(ntr))
        gradzr     = np.sqrt(gradzx*gradzx + gradzy*gradzy)
        gradzmag   = np.sqrt(gradzr*gradzr + gradzz*gradzz)

        fluxrt  = np.zeros((frequency.size,ntr,self.robs.size)) * fu # shape is (frequency.size,ntr,self.robs.size).. duh
        # shape is (ntr,self.robs.size)
        self.cosbeta = np.sum(self.Rhat * np.transpose(np.broadcast_to([gradzx,gradzy,gradzz], (self.robs.size,3,ntr)), (2,0,1)), axis=2) / np.transpose(np.broadcast_to(gradzmag, (self.robs.size,ntr))) # shape is (ntr,self.robs.size)
        # shape should be (ntr,self.robs.size)
        cbdx  = np.extract(self.cosbeta > 0.0, range(self.cosbeta.size))
        cbdxo = np.int16(np.mod(cbdx, self.robs.size))
        cbdxt = np.int16((cbdx-cbdxo)/self.robs.size)

        # Blackbody emitted from the tau=1 surface
        bb     = BlackBody(temperature=self.tempt1[r])
        # Doppler shift:
        betamag = np.sqrt(1.0/self.rstar[r]) # Rotational motion of disk - in (theta+90 deg)-hat direction; Need dot product with R-hat... betamag is a scalar
        # As a vector, beta = betamag (-sin theta i-hat + cos theta j-hat)
        # theta2D has shape (ntr,self.robs.size). betamag is scalar.
        betavec = betamag * np.transpose([-np.sin(theta2D), np.cos(theta2D), np.zeros(theta2D.shape)], (1,2,0)) # betavec had shape (self.ntheta[r],3)

        betadotrhat = np.sum(betavec * self.Rhat, axis=2) # shape (self.ntheta[r],self.robs.shape)

        # Relativistic beaming:
        gamma    = 1.0/np.sqrt(1- betamag * betamag) # scalar

        t0 = tm.time()
        for i in np.unique(cbdxo):
            idx = np.extract(cbdxo == i, cbdxt)
            freqprime = frequency.reshape(frequency.size,1) @ (np.sqrt((1 + betadotrhat[idx,i]) / (1 - betadotrhat[idx,i]))).reshape(1,idx.size)   # shape (frequency.size,idx.size)

            # Relativistic beaming:
            costheta = betadotrhat[idx,i] / betamag # shape (idx.size,)
            D        = 1./(gamma * (1. - betamag * costheta))   # shape (idx.size,)

            # want to return shape (frequency.size,ntr,self.robs.size)
            cosbetarm2 = np.broadcast_to(self.cosbeta[idx,i] / self.Rmag[idx,i]**2,
                                         (frequency.size,idx.size))

            bb_eval = np.zeros(freqprime.shape) * (fu / u.sr)
            bb_mask = const.h * freqprime / (const.k_B * self.tempt1[r]) < 670.74 # This is to prevent underflows
            bb_eval[bb_mask] = bb(freqprime[bb_mask])

            fluxrt[:,idx,i]  = self.rstar[r] * self.deltar[r] * dtheta * (D**3) * cosbetarm2 * u.sr * bb_eval

        return fluxrt

    ######################################################
    def makedisk(self):
        mstar    = self.mbh / (3.0 * const.M_sun.cgs)
        mdotstar = (self.mdot / (1.0e+17 * u.g / u.s)).decompose()
        print(f"\t\tM* = {mstar:e}, Mdot* = {mdotstar:e}  Rg = {self.rg:e}  mstar = {mstar:e}  mdotstar = {mdotstar:e}")
        ######################################################
        y     = np.sqrt(self.rstar)
        ######################################################
        a = 1.0 + self.sbh * self.sbh * y * y * y * y * (1.0 + 2.0/(y * y))
        b = 1.0 + self.sbh / (y * y * y)
        c = 1.0 - 3.0 / (y * y) + 2.0 * self.sbh * self.sbh / (y * y * y)
        d = 1.0 - 2 / (y * y) + self.sbh * self.sbh / (y * y * y * y)
        e = 1.0 + self.sbh * self.sbh * (4.0 - 4.0 / (y * y) + 3.0 / (y * y * y * y)) / (y * y * y * y)
        f = 1.0 - 2.0 * self.sbh / (y * y * y) + self.sbh * self.sbh / (y * y * y * y)
        g = 1.0 - 2.0 / (y * y) + self.sbh / (y * y * y)
        r = f * f / c - self.sbh * self.sbh * (g / np.sqrt(c) - 1) / (y * y)
        s = a * a * c * r / (b * b * d)
        ######################################################
        y0 = np.sqrt(self.rms)
        y1 = 2.0 * np.cos((np.arccos(self.sbh)-np.pi)/3.0)
        y2 = 2.0 * np.cos((np.arccos(self.sbh)+np.pi)/3.0)
        y3 = -2.0 * np.cos(np.arccos(self.sbh)/3.0)
        ######################################################
        q0 = b / (y * np.sqrt(c))
        q1 = y - y0
        q2 = -1.5 * self.sbh * np.log(y / y0)
        q3 = -3 * (y1 - self.sbh) * (y1 - self.sbh) * np.log((y - y1) / (y0 - y1)) / (y1 * (y1 - y2) * (y1 - y3))
        q4 = -3 * (y2 - self.sbh) * (y2 - self.sbh) * np.log((y - y2) / (y0 - y2)) / (y2 * (y2 - y1) * (y2 - y3))
        q5 = -3 * (y3 - self.sbh) * (y3 - self.sbh) * np.log((y - y3) / (y0 - y3)) / (y3 * (y3 - y1) * (y3 - y2))
        q = q0 * (q1 + q2 + q3 + q4 + q5)
        ######################################################
        diskheightinner   = (1.0e+5 * u.cm / self.rg)     * np.power(self.alpha,0)     * np.power(mstar, 0)     *          mdotstar        * np.power(r,0)   * np.power(y,0)      * np.power(a,2)      * np.power(b,-3)     * np.power(c,1/2) * np.power(d,-1)     * np.power(s,-1)     *          q
        diskheightmiddle  = (3.0e+3 * u.cm / self.rg)     * np.power(self.alpha,-1/10) * np.power(mstar, 9/10)  * np.power(mdotstar,2/10)  * np.power(r,0)   * np.power(y,21/20)  *          a         * np.power(b,-6/5)   * np.power(c,1/2) * np.power(d,-3/5)   * np.power(s,-1/2)   * np.power(q,1/5)
        diskheightouter   = ( 900.0 * u.cm / self.rg)     * np.power(self.alpha,-1/10) * np.power(mstar, 9/10)  * np.power(mdotstar,3/10)  * np.power(r,9/8) * np.power(y,0)      * np.power(a,19/20)  * np.power(b,-11/10) * np.power(c,1/2) * np.power(d,-23/40) * np.power(s,-19/40) * np.power(q,3/40)
        temperatureinner  = (4.0e+7 * u.Kelvin)           * np.power(self.alpha,-1/4)  * np.power(mstar, -1/4)  * np.power(mdotstar,0)     * np.power(r,0)   * np.power(y,-3/4)   * np.power(a,-1/2)   * np.power(b,1/2)    * np.power(c,0)   * np.power(d,0)      * np.power(s,1/4)    * np.power(q,0)
        temperaturemiddle = (3.0e+8 * u.Kelvin)           * np.power(self.alpha,-1/5)  * np.power(mstar, -3/5)  * np.power(mdotstar,2/5)   * np.power(r,0)   * np.power(y,-9/5)   * np.power(a,0)      * np.power(b,-2/5)   * np.power(c,0)   * np.power(d,-1/5)   * np.power(s,0)      * np.power(q,2/5)
        temperatureouter  = (8.0e+7 * u.Kelvin)           * np.power(self.alpha,-1/5)  * np.power(mstar, -1/2)  * np.power(mdotstar,3/10)  * np.power(r,0)   * np.power(y,-3/2)   * np.power(a,-1/10)  * np.power(b,-1/5)   * np.power(c,0)   * np.power(d,-3/20)  * np.power(s,1/20)   * np.power(q,3/10)
        densityinner      = (1.0e-4 * u.g / u.cm**3)      * np.power(self.alpha,-1)    *          mstar         * np.power(mdotstar,-2)    * np.power(r,0)   * np.power(y,3)      * np.power(a,-4)     * np.power(b,6)      * np.power(c,0)   *          d         * np.power(s,2)      * np.power(q,-2)    / const.u.cgs
        densitymiddle     = (10.0   * u.g / u.cm**3)      * np.power(self.alpha,-7/10) * np.power(mstar,-11/10) * np.power(mdotstar,2/5)   * np.power(r,0)   * np.power(y,-33/10) * np.power(a,-1)     * np.power(b,3/5)    * np.power(c,0)   * np.power(d,-1/5)   * np.power(s,1/2)    * np.power(q,2/5)   / const.u.cgs
        densityouter      = (80.0   * u.g / u.cm**3)      * np.power(self.alpha,-7/10) * np.power(mstar,-5/4)   * np.power(mdotstar,11/20) * np.power(r,0)   * np.power(y,-15/4)  * np.power(a,-17/20) * np.power(b,3/10)   * np.power(c,0)   * np.power(d,-11/40) * np.power(s,17/40)  * np.power(q,11/20) / const.u.cgs
        pratio            = (5e-5)                        * np.power(self.alpha,-1/4)  * np.power(mstar, 7/4)   * np.power(mdotstar,-2)    * np.power(r,0)   * np.power(y,21/4)   * np.power(a,-5/2)   * np.power(b,9/2)    * np.power(c,0)   *          d         * np.power(s,5/4)    * np.power(q,-2)
        tratio            = (6e-6)                                                     *          mstar         * np.power(mdotstar,-1)    * np.power(r,0)   * np.power(y,3)      * np.power(a,-1)     * np.power(b,2)      * np.power(c,0)   * np.power(d,1/2)    * np.power(s,1/2)    * np.power(q,-1)
        ######################################################
        pdx    = np.extract(pratio > 1, np.arange(0,pratio.size,1))
        tdx    = np.extract(tratio > 1, np.arange(0,tratio.size,1))
        while self.rstar[pdx[0]] < 1.1*self.rms:
            pdx = np.delete(pdx,0)
        while self.rstar[tdx[0]] < 1.1*self.rms:
            tdx = np.delete(tdx,0)
        ######################################################
        dpdx = np.copy(pdx)
        dpdx[1:pdx.size-2] = 0.5*(pdx[2:pdx.size-1]-pdx[0:pdx.size-3])
        pdx = np.extract(dpdx == 1, pdx)
        ######################################################
        dtdx = np.copy(tdx)
        dtdx[1:tdx.size-2] = 0.5*(tdx[2:tdx.size-1]-tdx[0:tdx.size-3])
        tdx = np.extract(dtdx == 1, tdx)
        ######################################################
        self.diskheight       = np.copy(diskheightinner)
        self.temperature      = np.copy(temperatureinner)
        self.density          = np.copy(densityinner)
        self.diskheight[pdx]  = np.copy(diskheightmiddle[pdx])  * self.diskheight[pdx[0]]  / diskheightmiddle[pdx[0]]
        self.temperature[pdx] = np.copy(temperaturemiddle[pdx]) * self.temperature[pdx[0]] / temperaturemiddle[pdx[0]]
        self.density[pdx]     = np.copy(densitymiddle[pdx])     * self.density[pdx[0]]     / densitymiddle[pdx[0]]
        self.diskheight[tdx]  = np.copy(diskheightouter[tdx])   * self.diskheight[tdx[0]]  / diskheightouter[tdx[0]]
        self.temperature[tdx] = np.copy(temperatureouter[tdx])  * self.temperature[tdx[0]] / temperatureouter[tdx[0]]
        self.density[tdx]     = np.copy(densityouter[tdx])      * self.density[tdx[0]]     / densityouter[tdx[0]]

        self.diskheight[-1]  = self.diskheight[-3]
        self.temperature[-1] = self.temperature[-3]
        self.density[-1]     = self.density[-3]
        self.diskheight[-2]  = self.diskheight[-3]
        self.temperature[-2] = self.temperature[-3]
        self.density[-2]     = self.density[-3]
        ######################################################
        self.x1 = self.rstar[pdx[0]]
        self.x2 = self.rstar[tdx[0]]
        print(f'\t\tPressure change at {round(self.x1)} rg ({self.x1*self.rg})  Optical depth change at {round(self.x2)} rg ({self.x2*self.rg})')

    ######################################################
    def photosphere(self,makeplot=False):
        self.zt1 = np.empty(0)
        filename = self.datapath+f"Sbh{self.sbh}-MBH{np.log10(self.mbh / const.M_sun):.2f}-Mdot{(self.mdot/(const.M_sun/u.year)).decompose()}-alpha{self.alpha}.fits"
        if os.path.isfile(filename):
            print("\tReading "+filename)
            data = Table.read(filename, format="fits")
            self.zt1 = np.array(data['zt1'])
            self.tempt1 = np.array(data['tempt1']) * u.K
            self.denst1 = np.array(data['denst1']) / u.cm**3
        else:
            txt = "{} {:10.6f} {:10.6f} {:10.6f} {:10.6f} {:10.6f} {:10.6e} {:10.6e} {}"
            print("\tWriting "+filename)
            self.zt1  = np.zeros(self.rstar.size)
            self.dzdr = np.empty(self.rstar.size)
            self.tempt1 = np.empty(self.rstar.size) * u.Kelvin
            self.denst1 = np.empty(self.rstar.size) / u.cm**3
            nit = 0
            zlo = 1.0e-4
            zhi = 1.0e+2
            plt.ion()
            for i in range(self.rstar.size):
                done = 0
                nit = 0
                while done == 0:
                    z          = np.squeeze(np.logspace(np.log10(zlo),np.log10(zhi),num=np.power(10,4+nit)))
                    rhoz       = self.verticaldensity(z, i) * const.u.cgs
                    rdx        = np.extract(rhoz * u.cm**3 / u.g <= 1.0e-5 * const.u.cgs / u.g, np.arange(0,rhoz.size,1))
                    rhoz[rdx]  = 1.0e-5 * const.u.cgs / u.cm**3
                    zp         = np.append(z,100.0*self.diskheight[i])
                    sd         = self.surfdens(zp * self.rg, i)
                    sd0        = 2.0 * np.squeeze(sd[sd.size-1])
                    sdc        = np.copy(sd[0:sd.size-1])
                    sdr        = 1 - 4*np.power(sdc/sd0,2)
                    tempz      = self.temperature[i] * np.power(sdr, 0.25)
                    tmpdx      = np.extract(tempz / u.Kelvin <= 1.0e-5, np.arange(0,tempz.size,1))
                    if tmpdx.size > 0:
                        tempz[tmpdx] = tempz[np.min(tmpdx)-1] + 1.0e-5 * u.Kelvin
                    ffopacity  = (0.64e+23 * u.cm**2 / u.g) * np.multiply((rhoz * u.cm**3 / u.g), np.power(tempz / u.Kelvin, -7/2))
                    opacity    = (ffopacity + 0.4* u.cm**2 / u.g) * const.u.cgs
                    sd = self.surfdens(z * self.rg, i)
                    dsd        = 0.5*(sd[2:sd.size]-sd[0:sd.size-2])
                    dsd        = np.insert(dsd,0,dsd[0])
                    dsd        = np.append(dsd,dsd[-1])
                    dtau       = np.multiply(opacity,dsd)
                    opticaldepth = np.zeros(dtau.size)
                    tracko = 0
                    for o in range(dtau.size):
                        opticaldepth[o] = np.sum(dtau[o:])
                        if opticaldepth[o] > 1:
                            tracko = o
                    done = 1
                    if (np.max(opticaldepth) < 1) or (np.min(opticaldepth) > 1) or (opticaldepth[tracko]/opticaldepth[tracko+1]-1 > 1.0e-2):
                        done = 0
                        zlo *= 0.99
                        zhi *= 1.01
                    if done == 0:
                        nit += 1

                    self.zt1[i:]    = z[tracko] + (z[tracko+1]-z[tracko])*(opticaldepth[tracko+1]-1)/(opticaldepth[tracko+1]-opticaldepth[tracko])
                    self.tempt1[i:] = self.verticaltemperature(self.zt1[i], i)
                    if self.zt1[i] <= self.diskheight[i]:
                        self.denst1[i:] = self.density[i] * np.exp(-np.power(self.zt1[i]/self.diskheight[i], 2))
                    else:
                        self.denst1[i:] = self.density[i] * np.exp(-self.zt1[i]/self.diskheight[i])
                    print(txt.format(i,self.rstar[i],self.zt1[i],self.zt1[i]/self.diskheight[i],opticaldepth[tracko],opticaldepth[tracko+1],self.tempt1[i],self.denst1[i],nit)+" "+str(done))

                if makeplot:
                  plt.clf()
                  self.pltdisk(i)
                  plt.pause(0.001)

            data = Table(data=[range(self.rstar.size),self.rstar,self.diskheight,self.zt1,self.temperature,self.tempt1,self.density,self.denst1],
                         names=["i","rstar","diskheight","zt1","temperature","tempt1","density","denst1"])
            data.write(filename, format="fits")

        #self.zt1cs = interpolate.splrep(self.rstar,self.zt1,s=len(self.rstar))
        #self.dzdr  = interpolate.splev(self.rstar,self.zt1cs,der=1)
        self.zt1cs = CubicSpline(self.rstar,self.zt1)
        self.dzdr  = self.zt1cs(self.rstar,nu=1)
        self.tempt1cs = CubicSpline(self.rstar,self.tempt1)

    ######################################################
    def pltdisk(self,i=0,photosphere=True,ctype='k'):
        plt.subplot(111,frameon=False)
        plt.axis('off')
        #plt.title(r'$S_{bh}$ = '+'{:.2f}'.format(self.sbh)+r' --- log $M_{bh}/M_\odot$ = '+'{:.1f}'.format(np.log10(self.mbh.to(u.solMass).value))+r' ---  $\dot{M}$ = '+'{:.1g}'.format(self.mdot.to(u.solMass/u.yr))+r' $M_\odot$/yr --- log r/rg = '+'{:.3g}'.format(np.log10(self.rstar[i]))+' --- robs/rg = {:.3g} --- zobs/rg = {:.3g}'.format(self.robs, self.zobs))

        leglabel = r'$S_{bh}$ = '+'{:.2f}'.format(self.sbh)+r', log $M_{bh}/M_\odot$ = '+'{:.1f}'.format(np.log10(self.mbh.to(u.solMass).value))+r', $\dot{M}$ = '+'{:.2g}'.format(self.mdot.to(u.solMass/u.yr).value)+r' $M_\odot$/yr'

        plt.subplot(311)
        plt.plot(self.rstar, self.diskheight,ctype, label=leglabel)
        if photosphere:
          plt.plot(self.rstar[:self.zt1.size],self.zt1,'b--')
          plt.plot([self.x1,self.x1],[np.min(self.diskheight), np.max(self.zt1)], 'm--')
          plt.plot([self.x2,self.x2],[np.min(self.diskheight), np.max(self.zt1)], 'm--')
          plt.plot([self.rstar[i], self.rstar[i]], [np.min(self.diskheight), np.max(self.zt1)], 'g--')
        #plt.plot([self.robs],[self.zobs],'ro')
        plt.ylabel(r'Disk height/$r_g$')
        plt.xlabel(r'Radial distance $r/r_g$') #  ($r_g = GM/c^2 = $'+f'{mydisk.rg:.2e}'+')')
        plt.yscale("log")
        plt.xscale("log")

        plt.subplot(312)
        plt.plot(self.rstar,self.temperature, ctype)
        if photosphere:
          plt.plot(self.rstar[:self.zt1.size],self.tempt1,'b--')
          plt.plot([self.x1,self.x1],[np.min(self.tempt1/u.K),np.max(self.temperature/u.K)], 'm--')
          plt.plot([self.x2,self.x2],[np.min(self.tempt1/u.K),np.max(self.temperature/u.K)], 'm--')
          plt.plot([self.rstar[i],self.rstar[i]],[np.min(self.tempt1/u.K),np.max(self.temperature/u.K)], 'g--')
        plt.ylabel("Disk temperature (K)")
        plt.xlabel(r'Radial distance $r/r_g$') #  ($r_g = GM/c^2 = $'+f'{mydisk.rg:.2e}'+')')
        plt.yscale("log")
        plt.xscale("log")

        plt.subplot(313)
        plt.plot(self.rstar,self.density, ctype)
        if photosphere:
          plt.plot(self.rstar[:self.zt1.size],self.denst1,'b--')
          plt.plot([self.x1,self.x1],[np.min(self.denst1*np.power(u.cm,3)),np.max(self.density*np.power(u.cm,3))], 'm--')
          plt.plot([self.x2,self.x2],[np.min(self.denst1*np.power(u.cm,3)),np.max(self.density*np.power(u.cm,3))], 'm--')
          plt.plot([self.rstar[i],self.rstar[i]],[np.min(self.denst1*np.power(u.cm,3)),np.max(self.density*np.power(u.cm,3))], 'g--')
        plt.ylabel(r'Disk density (atoms/cm$^3$)')
        plt.xlabel(r'Radial distance $r/r_g$') #  ($r_g = GM/c^2 = $'+f'{mydisk.rg:.2e}'+')')
        plt.yscale("log")
        plt.xscale("log")

    ######################################################
    def surfdens(self, z, i_annulus): #z0, rho0:
        s   = np.empty(z.size)
        udx = np.extract(z <= self.diskheight[i_annulus] * self.rg, np.arange(0,s.size,1))
        vdx = np.extract(z > self.diskheight[i_annulus] * self.rg,  np.arange(0,s.size,1))
        if udx.size > 0:
            s[udx] = np.sqrt(np.pi)*special.erf(z[udx]/(self.diskheight[i_annulus] * self.rg)) / 2.0
        if vdx.size > 0:
            s[vdx] = (np.sqrt(np.pi)*special.erf(1) / 2.0 + np.exp(-2) * (1 - np.exp(1-z[vdx]/(self.diskheight[i_annulus] * self.rg))))
        return s * self.diskheight[i_annulus]  * self.rg * self.density[i_annulus]

    ######################################################
    # This is for a range of z values at a given i_annulus. [z = array, i_annulus = scalar]
    def verticaldensity(self, z, i_annulus):
        if z.size > 1:
            rho  = self.density[i_annulus] * np.zeros(z.size)
            udx = np.extract(z <= self.diskheight[i_annulus], np.arange(0,rho.size,1))
            vdx = np.extract(z > self.diskheight[i_annulus],  np.arange(0,rho.size,1))
            if udx.size > 0:
                rho[udx] = self.density[i_annulus] * np.exp(-np.square(z[udx]/self.diskheight[i_annulus]))
            if vdx.size > 0:
                rho[vdx] = self.density[i_annulus] * np.exp(-z[vdx]/self.diskheight[i_annulus])
        else:
            if z <= self.diskheight[i_annulus]:
                rho = self.density[i_annulus] * np.exp(-np.square(z/self.diskheight[i_annulus]))
            else:
                rho = self.density[i_annulus] * np.exp(-z/self.diskheight[i_annulus])
        return rho

    # This is for a range of z values and an array of i_annulus. Assumes both have the same dimension
    def verticaldensity2(self, z, i_annul):
        rho = np.zeros(z.size) / (u.cm**3)
        if z.size > 1:
          for i in range(z.size):
              rho[i] = self.verticaldensity(z[i],i_annul[i])
        else:
              rho = self.verticaldensity(z,i_annul)

        return rho

    # This is for a range of z values and an array of r. Assumes both are 1d witht he same size
    def verticaldensity3(self, z, r):
        # Determine the annuli corresponding to the values in r. Then can call verticaldensity2
        r2d              = np.transpose(np.broadcast_to(r,                          (self.rstar.size,r.size)))
        rstar2d          =              np.broadcast_to(self.rstar-0.5*self.drstar, (r.size,self.rstar.size))
        rstar_indices_2d =              np.broadcast_to(np.arange(self.rstar.size), (r.size,self.rstar.size))
        i_annul_2d       = np.where( r2d  < rstar2d, rstar_indices_2d , self.rstar.size-1  )
        i_annul = np.min(i_annul_2d  ,  axis=1)
        rho = self.verticaldensity2(z, i_annul)
        return rho

    # This is for interpolating the density grid for a particular value of r and z
    def verticaldensity_onepoint(self,r,z):
        return np.interp(r, self.rstar, self.verticaldensity2(np.broadcast_to(z, self.rstar.shape), range(self.nr)))

    ######################################################
    def verticaltemperature(self, z, i_annulus):
        zp = np.append(z,100.0*self.diskheight[i_annulus])  * self.rg
        sd = self.surfdens(zp,i_annulus)
        sd0 = 2.0 * np.squeeze(sd[sd.size-1])
        sdc = np.copy(sd[0:sd.size-1])
        sdr = 1 - 4*np.power(sdc/sd0,2)
        return self.temperature[i_annulus] * np.power(sdr, 0.25)

    # This is for a range of z values and an array of i_annulus. Assumes both have the same dimension
    def verticaltemperature2(self, z, i_annul):
        temperature = np.zeros(z.size) * u.K
        for i in range(z.size):
            temperature[i] = self.verticaltemperature(z[i],i_annul[i])

        return temperature

    # This is for a range of z values and an array of r. Assumes both are 1d with the same size
    def verticaltemperature3(self, z, r):
        # Determine the annuli corresponding to the values in r. Then can call verticaldensity2
        r2d              = np.transpose(np.broadcast_to(r,                          (self.rstar.size,r.size)))
        rstar2d          =              np.broadcast_to(self.rstar-0.5*self.drstar, (r.size,self.rstar.size))
        rstar_indices_2d =              np.broadcast_to(np.arange(self.rstar.size), (r.size,self.rstar.size))
        i_annul_2d       = np.where( r2d  < rstar2d, rstar_indices_2d , self.rstar.size-1  )
        i_annul = np.min(i_annul_2d  ,  axis=1)
        temperature = self.verticaltemperature2(z, i_annul)
        return temperature

    # This is for interpolating the temperature grid for a particular value of r and z
    def verticaltemperature_onepoint(self,r,z):
        return np.interp(r, self.rstar, self.verticaltemperature2(np.broadcast_to(z, self.rstar.shape), np.arange(self.nr)))
    ######################################################

