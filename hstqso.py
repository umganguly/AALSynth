import matplotlib.pyplot as plt
import numpy             as np
import os
import time              as tm
import urllib

from astropy             import units       as u
from astropy             import constants   as const
from astropy.convolution import convolve, Gaussian1DKernel
from astropy.coordinates import SkyCoord
from astropy.io          import fits, ascii
from astropy.table       import Table
from astroquery.ipac.ned import Ned
from astroquery.mast     import MastMissions, Observations
from multiprocessing     import Pool
from scipy.interpolate   import CubicSpline, interp1d
from scipy.optimize      import minimize, differential_evolution
from scipy.stats         import chi2
from scipy.stats         import f           as Ftest
from tqdm                import tqdm

from atomic                  import atomic
from doppler                 import calcvel,calcwave

###############################################################################################
class hstqso:
    def __init__(self,
                 coords,
                 qfileroot,
                 zqso,
                 redospline = False
                 ):
        self.zqso        = zqso
        self.vm          = np.array([-5000,5000]) * (u.km/u.s)
        self.qfileroot   = qfileroot
        self.contfile    = qfileroot+"_cont.fits"
        self.absmaskfile = qfileroot+"_absmask.fits"
        self.x1dlistfile = qfileroot+"_x1d_list.fits"
        self.redospline  = redospline

        if os.name == 'nt':
            self.basedir = "D:\\ganguly\\AALSynth2\\Python\\mastDownload\\HST\\"
        else:
            self.basedir = "/mnt/data/AALSynth2/Python/mastDownload/HST/"
        print(f'basedir = {self.basedir}')

        self.lsfdir = self.basedir+"LSF/"

        if os.path.exists(self.basedir+self.x1dlistfile):
            with fits.open(self.basedir+self.x1dlistfile) as hdul:
                x1dtab = hdul[1].data
            self.x1dfiles = []
            for xfile in x1dtab['x1d']:
                print(f'x1dsum file = {xfile.replace("\\","/")}')
                self.x1dfiles.append(xfile)
        else:
            cosqueryresults = MastMissions(mission='hst').query_region(coords,sci_instrume="COS",select_cols=["sci_spec_1234","sci_pep_id", "sci_pi_last_name","sci_central_wavelength"])
            obs_table = Observations.query_criteria(obs_id=cosqueryresults['sci_data_set_name'], filters=["G130M","G160M"])
            data_products = Observations.get_unique_product_list(obs_table)
            filtered = Observations.filter_products(data_products, extension='fits', productSubGroupDescription='X1DSUM')
            manifest = Observations.download_products(filtered, productType="SCIENCE", extension='fits', productSubGroupDescription='X1DSUM', verbose=False)
            self.x1dfiles = []
            for xfile in manifest['Local Path']:
                print(f'x1dsum file = {xfile[19:]}')
                self.x1dfiles.append(xfile[19:])
            x1dtab = Table([self.x1dfiles], names=['x1d'])
            x1dtab.write(self.basedir+self.x1dlistfile, format='fits', overwrite=True)

    ######################################################################
    def _boxcar_smooth(self, spectrum, window_size):
        if window_size % 2 == 0:
            raise ValueError("window_size must be an odd integer for symmetric smoothing.")

        # Create the boxcar kernel
        boxcar_kernel = np.ones(window_size) / window_size

        # Convolve the spectrum with the boxcar kernel
        smoothed_spectrum = np.convolve(spectrum, boxcar_kernel, mode='same')
        return smoothed_spectrum

    ######################################################################
    def bin_spec(self, totwave, totflux, totivar, binfac = 5):
        newlen = self.totwave.size // binfac
        binwave = (totwave[:binfac*newlen].reshape(-1,binfac).sum(axis=1)/binfac).value
        binflux = (totflux[:binfac*newlen].reshape(-1,binfac).sum(axis=1)/binfac)
        binivar = (totivar[:binfac*newlen].reshape(-1,binfac).sum(axis=1)/binfac)

        return binwave, binflux, binivar

    ######################################################################
    def combspec(self,
                 verbose=True
                 ):
        self.totwave = np.array([]) * u.Angstrom
        self.totflux = np.array([]) * (u.erg/u.s/u.cm**2/u.Angstrom)
        self.totivar = np.array([]) * (u.erg/u.s/u.cm**2/u.Angstrom)**-2
        for x1d in self.x1dfiles:
            if verbose: print(f"Reading in {self.basedir+x1d.replace('\\','/')}...")
            with fits.open(self.basedir+x1d.replace('\\','/')) as hdu1:
                exptime = hdu1[1].header["EXPTIME"] * u.s
                wave = (hdu1[1].data['WAVELENGTH']).flatten() * u.Angstrom
                flux = (hdu1[1].data['FLUX']).flatten()       * (u.erg/u.s/u.cm**2/u.Angstrom)
                ferr = (hdu1[1].data['ERROR']).flatten()      * (u.erg/u.s/u.cm**2/u.Angstrom)

                fdx  = np.extract((flux != 0) & (ferr != 0), range(flux.size))
                if fdx.size > 0:
                    avesnr = np.average(flux[fdx]/ferr[fdx])
                    if verbose:
                        prtstr  = f"\tExposure time: {exptime}  Wavelength range: {np.min(wave[fdx])} - {np.max(wave[fdx])}  "
                        prtstr += f"Mean/std flux: {np.average(flux[fdx])} +/- {np.std(flux[fdx])}  <S/N> = {avesnr}"
                        print(prtstr)
                else:
                    avesnr = 0.0

                if avesnr > 0:
                    wdx  = np.argsort(wave)
                    wave = wave[wdx]
                    flux = flux[wdx]
                    ferr = ferr[wdx]

                    fdx  = np.extract((flux == 0) | (ferr == 0), range(flux.size))
                    if fdx.size > 0:
                        if verbose: print(f"     Deleting {fdx.size} / {wave.size} bins that have either flux or error == 0")
                        wave = np.delete(wave, fdx)
                        flux = np.delete(flux, fdx)
                        ferr = np.delete(ferr, fdx)
                        ivar = 1/np.square(ferr)

                    if wave.size > 0:
                        dwave       = np.zeros(wave.size)
                        dwave[1:-1] = 0.5 * (wave[2:] - wave[:-2])
                        dwave[0]    = dwave[1]
                        dwave[-1]   = dwave[-2]

                        wdx = np.argsort(wave)
                        wave = wave[wdx]
                        flux = flux[wdx]
                        ivar = ivar[wdx]

                        scflux = np.copy(flux)

                        if self.totwave.size == 0:
                            # Slight sideways detour to grab the appropriate LSF and DISTAB for the spectrum
                            param_dict = {}  # Make a dict to store what you find here

                            for hdrKeyword in [
                                "DETECTOR",
                                "OPT_ELEM",
                                "LIFE_ADJ",
                                "CENWAVE",
                                "DISPTAB",
                            ]:  # Print out the relevant values
                                try:  # For DISPTAB
                                    value = hdu1[0].header[hdrKeyword].split("$")[1]  # Save the key/value pairs to the dictionary
                                    param_dict[hdrKeyword] = value                # DISPTAB needs the split here
                                except:  # For other params
                                    value = hdu1[0].header[hdrKeyword]  # Save the key/value pairs to the dictionary
                                    param_dict[hdrKeyword] = value
                                print(f"{hdrKeyword} = {value}")  # Print the key/value pairs

                            LSF_file_name, disptab_path = self._lsf_fetch_files(*list(param_dict.values()))
                            self.new_lsf, self.new_w, self.step = self._lsf_redefine(LSF_file_name, hdu1[0].header["CENWAVE"], disptab_path, detector=hdu1[0].header["DETECTOR"])

                            # Ok, now initially the self.tot____ arrays
                            self.totwave = np.append(self.totwave, wave)
                            self.totflux = np.append(self.totflux, flux)
                            self.totivar = np.append(self.totivar, ivar)
                            self.qra     = hdu1[1].header['RA_APER']
                            self.qdec    = hdu1[1].header['DEC_APER']
                        else:
                            ogtotwave = np.copy(self.totwave)
                            ogtotflux = np.copy(self.totflux)
                            ogtotivar = np.copy(self.totivar)

                            pdx = np.extract(wave < self.totwave[0], range(wave.size))
                            if pdx.size > 0:
                                ns = 2
                                scale = np.average(self.totflux[:ns])/np.average(flux[pdx[-ns:]])
                                oldscale = 1.0
                                while np.fabs((scale-oldscale) / oldscale) > 1.0e-5 and ns < pdx.size:
                                    ns   += 1
                                    oldscale = scale
                                    scale = np.average(self.totflux[:ns])/np.average(flux[pdx[-ns:]])
                                    if ns >= pdx.size:
                                        scale = -1.0
                                if scale > 0.0:
                                    if verbose: print(f"     Prepending {pdx.size} bins with scale = {scale} {ns}")
                                    scflux[pdx] = flux[pdx] * scale
                                    self.totwave = np.append(wave[pdx],                 self.totwave)
                                    self.totflux = np.append(scflux[pdx],               self.totflux)
                                    self.totivar = np.append(ivar[pdx] / (scale*scale), self.totivar)

                            adx = np.extract(wave > self.totwave[-1], range(wave.size))
                            if adx.size > 0:
                                ns = 2
                                scale = np.average(self.totflux[-ns:])/np.average(flux[adx[:ns]])
                                oldscale = 1.0
                                while np.fabs((scale-oldscale) / oldscale) > 1.0e-5 and ns < adx.size:
                                    ns   += 1
                                    oldscale = scale
                                    scale = np.average(self.totflux[-ns:])/np.average(flux[adx[:ns]])
                                    if ns >= adx.size:
                                        scale = -1.0
                                if scale > 0.0:
                                    if verbose: print(f"     Appending {adx.size} bins with scale = {scale} = {np.average(self.totflux[-ns:])} / {np.average(flux[adx[:ns]])} {ns}")
                                    scflux[adx] = flux[adx] * scale
                                    self.totwave = np.append(self.totwave, wave[adx])
                                    self.totflux = np.append(self.totflux, scflux[adx])
                                    self.totivar = np.append(self.totivar, ivar[adx] / (scale * scale))

                            # What about gaps in totwave that can be be filled in by wave?
                            dtotwave = np.zeros(self.totwave.size)
                            dtotwave[1:-1] = 0.5 * (self.totwave[2:] - self.totwave[:-2])
                            dtotwave[0] = dtotwave[1]
                            dtotwave[-1] = dtotwave[-2]
                            for refwave in np.extract(dtotwave > 1.0, self.totwave):
                                wlo = np.max(np.extract(self.totwave < refwave, self.totwave))
                                whi = np.min(np.extract(self.totwave > refwave, self.totwave))
                                if verbose: print(f"     Gap in wavelength coverage: {wlo} -- {whi}")
                                wldx = np.extract(self.totwave < wlo, range(self.totwave.size))
                                whdx = np.extract(self.totwave > whi, range(self.totwave.size))
                                fdx = np.extract((wave > wlo) & (wave < whi) & (flux > 0), range(wave.size))
                                if fdx.size > 0:
                                    ns = 2
                                    scale = np.average(np.append(self.totflux[wldx[-ns:]], self.totflux[whdx[:ns]]))/np.average(flux[fdx])
                                    oldscale = 1.0
                                    while np.fabs((scale-oldscale) / oldscale) > 1.0e-5 and ns < np.min([wldx.size,whdx.size]):
                                        ns   += 1
                                        oldscale = scale
                                        scale = np.median(np.append(self.totflux[wldx[-ns:]], self.totflux[whdx[:ns]]))/np.median(flux[fdx])
                                        if ns >= np.min([wldx.size,whdx.size]):
                                            scale = -1.0
                                    if scale > 0.0:
                                        if verbose: print(f"     Inserting {fdx.size} bins with scale = {scale} {ns}")
                                        scflux[fdx] = flux[fdx] * scale
                                        self.totwave = np.append(self.totwave, wave[fdx])
                                        self.totflux = np.append(self.totflux, scflux[fdx])
                                        self.totivar = np.append(self.totivar, ivar[fdx] / (scale * scale))
                                        sdx = np.argsort(self.totwave)
                                        self.totwave = self.totwave[sdx]
                                        self.totflux = self.totflux[sdx]
                                        self.totivar = self.totivar[sdx]

                            # Combine spectra where there is overlap
                            ndx = np.extract(((self.totwave < wlo) | (self.totwave > whi)) & \
                                             (self.totwave > ogtotwave[0]) & (self.totwave < ogtotwave[-1]) & \
                                             (self.totwave >      wave[0]) & (self.totwave <      wave[-1]) & \
                                             ((self.totwave < 1300. * u.Angstrom) | (self.totwave > 1310. * u.Angstrom))  & \
                                             ((self.totwave < 1213. * u.Angstrom) | (self.totwave > 1218. * u.Angstrom)),
                                             range(self.totwave.size)
                                             )
                            odx = np.extract((wave > ogtotwave[0]) & (wave < ogtotwave[-1]) & \
                                             ((wave < 1300. * u.Angstrom) | (wave > 1310. * u.Angstrom)) & \
                                             ((wave < 1213. * u.Angstrom) | (wave > 1218. * u.Angstrom) ),
                                             range(wave.size)
                                             )
                            if odx.size > 0 and ndx.size > 0:
                                scale = np.average(self.totflux[ndx])/np.average(flux[odx])
                                if scale > 0.:
                                    if verbose:
                                        prtstr = f"\tCombining {odx.size} bins into {ndx.size} bins with scale = {scale} {ns}   {np.average(self.totwave[ndx])} {np.average(wave[odx])}   "
                                        prtstr += f"{np.average(self.totflux[ndx])} {np.average(flux[odx])}"
                                        print(prtstr)
                                    odx = np.extract((wave > ogtotwave[0]) & \
                                                     (wave < ogtotwave[-1]),
                                                     range(wave.size)
                                                     )
                                    ndx = np.extract(((self.totwave < wlo) | (self.totwave > whi)) & \
                                                     (self.totwave > ogtotwave[0]) & \
                                                     (self.totwave < ogtotwave[-1]) & \
                                                     (self.totwave > wave[0]) & \
                                                     (self.totwave < wave[-1]),
                                                     range(self.totwave.size)
                                                     )
                                    intflux = np.interp(self.totwave[ndx], wave[odx], flux[odx] * scale)
                                    intivar = np.interp(self.totwave[ndx], wave[odx], ivar[odx] / (scale * scale))
                                    scflux[odx] = flux[odx] * scale
                                    self.totflux[ndx] = (self.totflux[ndx] * self.totivar[ndx] + intflux * intivar) / (self.totivar[ndx] + intivar)
                                    self.totivar[ndx] += intivar

                        wdx = np.argsort(self.totwave)
                        self.totwave = self.totwave[wdx]
                        self.totflux = self.totflux[wdx]
                        self.totivar = self.totivar[wdx]


    ######################################################################
    def contfit(self):
        oldn = 0
        olda = 0
        self.absmask = np.empty((0,2))
        print(f"\tLooking for {self.basedir+self.contfile}")
        if not os.path.exists(self.basedir+self.contfile):
            if os.path.exists(self.basedir+self.absmaskfile):
                absdata = Table.read(self.basedir+self.absmaskfile)
                self.absmask = np.array(absdata['col0'])
            self.allx = np.array([])
            self.ally = np.array([])
            self.splfit()
        else:
            print(f"\tFound {self.basedir+self.contfile}")
            contdata = Table.read(self.basedir+self.contfile)
            oldn = len(contdata)
            self.allx = contdata['wave']
            self.ally = contdata['flux']
            if os.path.exists(self.basedir+self.absmaskfile):
                absdata = Table.read(self.basedir+self.absmaskfile)
                self.absmask = np.array(absdata['col0'])
                olda = self.absmask[:,0].size
                if self.redospline:
                    for i in range(self.absmask[:,0].size):
                        print(f"{i}. {self.absmask[i,0]} -- {self.absmask[i,1]}")
                    selection = input("Remove a band? (lower case n = no = exit) ")
                    while not selection == 'n':
                        rmdx = np.int16(input("Which band? (-1 for exit) "))
                        if not rmdx == -1:
                            self.absmask = np.delete(self.absmask, rmdx, 0)
                        for i in range(self.absmask[:,0].size):
                            print(f"{i}. {self.absmask[i,0]} -- {self.absmask[i,1]}")
                        selection = input("Remove another? (lower case n = no = exit) ")
            else:
                self.absmask = np.empty((0,2))
            if self.redospline:
                print("Please review continuum fit")
                self.splfit(vm=self.vm)

        newn = self.allx.size
        if newn != oldn:
            print(f"Writing new {self.basedir+self.contfile}")
            contdata = Table([self.allx,self.ally], names=('wave','flux'))
            contdata.write(self.basedir+self.contfile, format="fits", overwrite=True)
        newa = self.absmask[:,0].size
        if newa != olda:
            print(f"Writing new {self.basedir+self.absmaskfile}")
            absdata = Table([self.absmask])
            absdata.write(self.basedir+self.absmaskfile,format="fits", overwrite=True)
        self.continuum = CubicSpline(self.allx,self.ally)
        #for i in range(self.allx.size):
        #    print(f"{self.allx[i]} {self.ally[i]} {self.continuum(self.allx[i])}")
        self.allx *= u.Angstrom
        self.ally *= (u.erg/u.s/u.cm**2/u.Angstrom)

    ######################################################################
    def ew_spec(self,
                wave,
                flux,
                ferr
                ):
        dwave = np.zeros(wave.size)
        dwave[1:-1] = 0.5 * ( wave[2:] - wave[:-2])
        dwave[0] = dwave[1]
        dwave[-1] = dwave[-2]

        ew   = (1.0 - flux.value/self.continuum(wave)) * wave * u.Angstrom
        sew  = (ferr.value/self.continuum(wave)) * dwave * u.Angstrom
        EW   = self._lsf_convolve(wave,  ew.value) * u.Angstrom
        SEW  = self._lsf_convolve(wave, sew.value) * u.Angstrom

        return ew, sew, EW, SEW

    ######################################################################
    def _lsf_convolve(self, wavelength, spec):
        """
        Main function; Convolves an input spectrum - i.e. template or STIS spectrum - with the COS LSF.
        Parameters:
        wavelength (list or array): Wavelengths of the spectrum to convolve.
        spec (list or array): Fluxes or intensities of the spectrum to convolve.
        Returns:
        wave_cos (numpy.ndarray): Wavelengths of convolved spectrum.!Different length from input wvln
        final_spec (numpy.ndarray): New LSF kernel's LSF wavelengths.!Different length from input spec
        """

        # sets up new wavelength scale used in the convolution
        wavemax = np.max(wavelength.to(u.Angstrom).value)
        if wavemax.size > 1:
            wavemax = wavemax[0]
        wavemin = np.min(wavelength.to(u.Angstrom).value)
        if wavemin.size > 1:
            wavemin = wavemin[0]
        nstep = np.round((wavemax - wavemin) / self.step.to(u.Angstrom).value) - 1
        wave_cos = wavemin * u.Angstrom + np.arange(nstep) * self.step

        # resampling onto the input spectrum's wavelength scale
        interp_func = interp1d(wavelength, spec)  # builds up interpolated function from input spectrum
        spec_cos = interp_func(wave_cos)  # builds interpolated initial spectrum at COS' wavelength scale for convolution
        final_spec = interp_func(wave_cos)  # Initializes final spectrum to the interpolated input spectrum

        for i, w in enumerate(self.new_w):  # Loop through the redefined LSF kernels
            # First need to find the boundaries of each kernel's "jurisdiction": where it applies
            # The first and last elements need to be treated separately
            if i == 0:  # First kernel
                diff_wave_left = 500 * u.Angstrom
                diff_wave_right = (self.new_w[i + 1] - w) / 2.0
            elif i == len(self.new_w) - 1:  # Last kernel
                diff_wave_right = 500 * u.Angstrom
                diff_wave_left = (w - self.new_w[i - 1]) / 2.0
            else:  # All other kernels
                diff_wave_left = (w - self.new_w[i - 1]) / 2.0
                diff_wave_right = (self.new_w[i + 1] - w) / 2.0

            # splitting up the spectrum into slices around the redefined LSF kernel wvlns
            # will apply the kernel corresponding to that chunk to that region of the spectrum - its "jurisdiction"
            chunk = np.where(
                (wave_cos < w + diff_wave_right) & (wave_cos >= w - diff_wave_left)
            )[0]
            if len(chunk) == 0:
                # off the edge, go to the next chunk
                continue

            current_lsf = self.new_lsf[:, i]  # selects the current kernel

            if len(chunk) >= len(
                current_lsf
            ):  # Makes sure that the kernel is smaller than the chunk
                final_spec[chunk] = convolve(
                    spec_cos[chunk],
                    current_lsf,  # Applies the actual convolution
                    boundary="extend",
                    normalize_kernel=True,
                )

        return np.interp(wavelength, wave_cos, final_spec)  # Remember, not the same length as input spectrum data!

    ######################################################################
    def _lsf_fetch_files(self, det, grating, lpPos, cenwave, disptab):
        """
        Given all the inputs: (detector, grating, LP-POS, cenwave, dispersion table,) this will download both
        the LSF file and Disptab file you should use in the convolution and return their paths.
        Returns:
        LSF_file_name (str): filename of the new downloaded LSF file
        disptab_path (str): path to the new downloaded disptab file
        """
        COS_site_rootname = (
            "https://www.stsci.edu/files/live/sites/www/files/home/hst/instrumentation/cos/"
            "performance/spectral-resolution/_documents/"
        )  # Link to where all the files live - split into 2 lines
        if det == "NUV":  # Only one file for NUV
            LSF_file_name = "nuv_model_lsf.dat"
        elif det == "FUV":  # FUV files follow a naming pattern
            LSF_file_name = f"aa_LSFTable_{grating}_{cenwave}_LP{lpPos}_cn.dat"

        if not os.path.exists(self.lsfdir+LSF_file_name):
            LSF_file_webpath = COS_site_rootname + LSF_file_name  # Where to find file online
            urllib.request.urlretrieve(
                LSF_file_webpath, self.lsfdir+LSF_file_name
            )  # Where to save file to locally
            print(f"Downloaded LSF file to {self.lsfdir+LSF_file_name}")
        else:
            print(f"Found LSF file in {self.lsfdir+LSF_file_name}")

        # And we'll need to get the DISPTAB file as well
        disptab_path = self.lsfdir + disptab
        if not os.path.exists(disptab_path):
            urllib.request.urlretrieve(
                f"https://hst-crds.stsci.edu/unchecked_get/references/hst/{disptab}",
                disptab_path,
            )
            print(f"Downloaded DISPTAB file to {disptab_path}")
        else:
            print(f"Found DISPTAB file in {disptab_path}")

        return LSF_file_name, disptab_path

    ######################################################################
    def _lsf_get_disp_params(self, disptab, cenwave, segment, x=[]):
        """
        Helper function to redefine_lsf(). Reads through a DISPTAB file and gives relevant\
        dispersion relationship/wavelength solution over input pixels.
        Parameters:
        disptab (str): Path to your DISPTAB file.
        cenwave (str): Cenwave for calculation of dispersion relationship.
        segment (str): FUVA or FUVB?
        x (list): Range in pixels over which to calculate wvln with dispersion relationship (optional).
        Returns:
        disp_coeff (list): Coefficients of the relevant polynomial dispersion relationship
        wavelength (list; if applicable): Wavelengths corresponding to input x pixels 
        """
        with fits.open(disptab) as d:
            wh_disp = np.where(
                (d[1].data["cenwave"] == cenwave)
                & (d[1].data["segment"] == segment)
                & (d[1].data["aperture"] == "PSA")
            )[0]
            disp_coeff = d[1].data[wh_disp]["COEFF"][0] # 0 is needed as this returns nested list [[arr]]
            d_tv03 = d[1].data[wh_disp]["D_TV03"]  # Offset from WCA to PSA in Thermal Vac. 2003 data
            d_orbit = d[1].data[wh_disp]["D"]  # Current offset from WCA to PSA

        delta_d = d_tv03 - d_orbit

        if len(x):  # If given a pixel range, build up a polynomial wvln solution pix -> λ
            wavelength = np.polyval(p=disp_coeff[::-1], x=np.arange(16384))
            return disp_coeff, wavelength
        else:  # If x is empty:
            return disp_coeff

    ######################################################################
    def _lsf_read(self, filename):
        # This is the table of all the LSFs: called "lsf"
        # The first column is a list of the wavelengths corresponding to the line profile, so we set our header accordingly
        if "nuv_" in filename:  # If its an NUV file, header starts 1 line later
            ftype = "nuv"

        else:  # assume its an FUV file
            ftype = "fuv"
        hs = 0
        lsf = Table.read(self.lsfdir+filename, format="ascii", header_start=hs)

        # This is the range of each LSF in pixels (for FUV from -160 to +160, inclusive)
        # middle pixel of the lsf is considered zero ; center is relative zero
        pix = np.arange(len(lsf)) - len(lsf) // 2  # integer division to yield whole pixels

        # the column names returned as integers.
        lsf_wvlns = np.array([int(float(k)) for k in lsf.keys()])

        return lsf, pix, lsf_wvlns * u.Angstrom

    ######################################################################
    def _lsf_redefine(self, lsf_file, cenwave, disptab, detector="FUV"):
        """
        Helper function to convolve_lsf(). Converts the LSF kernels in the LSF file from a fn(pixel) -> fn(λ)\
        which can then be used by convolve_lsf() and re-bins the kernels.
        Parameters:
        lsf_file (str): path to your LSF file
        cenwave (str): Cenwave for calculation of dispersion relationship
        disptab (str): path to your DISPTAB file
        detector (str): FUV or NUV?
        Returns:
        new_lsf (numpy.ndarray): Remapped LSF kernels.
        new_w (numpy.ndarray): New LSF kernel's LSF wavelengths.
        step (float): first order coefficient of the FUVA dispersion relationship; proxy for Δλ/Δpixel.
        """

        if detector == "FUV":
            xfull = np.arange(16384)

            # Read in the dispersion relationship here for the segments
            ### FUVA is simple
            disp_coeff_a, wavelength_a = self._lsf_get_disp_params(disptab, cenwave, "FUVA", x=xfull)
            ### FUVB isn't taken for cenwave 1105, nor 800:
            if (cenwave != 1105) & (cenwave != 800):
                disp_coeff_b, wavelength_b = self._lsf_get_disp_params(
                    disptab, cenwave, "FUVB", x=xfull)
            elif cenwave == 1105:
                # 1105 doesn't have an FUVB so set it to something arbitrary and clearly not real:
                wavelength_b = [-99.0, 0.0]

            # Get the step size info from the FUVA 1st order dispersion coefficient
            step = disp_coeff_a[1] * u.Angstrom

            # Read in the lsf file
            lsf, pix, w = self._lsf_read(lsf_file)

            # take median spacing between original LSF kernels
            deltaw = np.median(np.diff(w))

            lsf_array = [np.array(lsf[key]) for key in lsf.keys()]
            if (deltaw < len(pix) * step * 2):  # resamples if the spacing of the original LSF wvlns is too narrow
                # this is all a set up of the bins we want to use
                # The wvln difference between kernels of the new LSF should be about twice their width
                new_deltaw = np.round(len(pix) * step * 2.0)
                new_nw = (int(np.round((max(w) - min(w)) / new_deltaw)) + 1)  # nw = number of LSF wavelengths
                new_w = min(w) + np.arange(new_nw) * new_deltaw  # new version of lsf_wvlns

                # populating the lsf with the proper bins
                new_lsf = np.zeros((len(pix), new_nw))  # empty 2-D array to populate
                for i, current_w in enumerate(new_w):
                    dist = abs(current_w - w)  # Find closest original LSF wavelength to new LSF wavelength
                    lsf_index = np.argmin(dist)
                    orig_lsf_wvln_key = lsf.keys()[lsf_index]  # column name corresponding to closest orig LSF wvln
                    new_lsf[:, i] = np.array(lsf[orig_lsf_wvln_key])  # assign new LSF wvln the kernel of the closest original lsf wvln
            else:
                new_lsf = lsf
                new_w = w
            return new_lsf, new_w, step

        elif detector == "NUV":
            xfull = np.arange(1024)
            # Read in the dispersion relationship here for the segments
            disp_coeff_a, wavelength_a = get_disp_params(disptab, cenwave, "NUVA", x=xfull)
            disp_coeff_b, wavelength_b = get_disp_params(disptab, cenwave, "NUVB", x=xfull)
            disp_coeff_c, wavelength_c = get_disp_params(disptab, cenwave, "NUVC", x=xfull)

            # Get the step size info from the NUVB 1st order dispersion coefficient
            step = disp_coeff_b[1] * u.Angstrom

            # Read in the lsf file
            lsf, pix, w = read_lsf(lsf_file)

            # take median spacing between original LSF kernels
            deltaw = np.median(np.diff(w))

            lsf_array = [np.array(lsf[key]) for key in lsf.keys()]

            # this section is a set up of the new bins we want to use:
            new_deltaw = round(len(pix) * step * 2.0)  # The wvln difference between kernels of the new LSF should be about twice their width
            new_nw = (int(round((max(w) - min(w)) / new_deltaw)) + 1)  # nw = number of LSF wavelengths
            new_w = min(w) + np.arange(new_nw) * new_deltaw  # new version of lsf_wvlns

            # populating the lsf with the proper bins
            new_lsf = np.zeros((len(pix), new_nw))  # empty 2-D array to populate
            for i, current_w in enumerate(new_w):
                dist = abs(current_w - w)  # Find closest original LSF wavelength to new LSF wavelength
                lsf_index = np.argmin(dist)
                orig_lsf_wvln_key = lsf.keys()[lsf_index]  # column name corresponding to closest orig LSF wvln
                new_lsf[:, i] = np.array(lsf[orig_lsf_wvln_key])  # assign new LSF wvln the kernel of the closest original lsf wvln
            return new_lsf, new_w, step

    ######################################################################
    def plotspec(self,binfac = 1):

        newlen = self.totwave.size // binfac
        totwave = self.totwave[:binfac*newlen].reshape(-1,binfac).sum(axis=1)/binfac
        totflux = self.totflux[:binfac*newlen].reshape(-1,binfac).sum(axis=1)/binfac
        totivar = self.totivar[:binfac*newlen].reshape(-1,binfac).sum(axis=1)

        ptsize = 20
        plt.close('all')
        fig, ax = plt.subplots(self.pltwaves.size,2,sharex='col')
        plt.get_current_fig_manager().window.state('zoomed')
        for i in range(self.pltwaves.size):
            wr = totwave/self.pltwaves[i]
            beta = (wr*wr - (1+self.zqso)*(1+self.zqso))/(wr*wr + (1+self.zqso)*(1+self.zqso))
            velocity = const.c.to(u.km/u.s) * beta
            vdx = np.extract((velocity > np.min(self.vlims)) & (velocity < np.max(self.vlims)), range(velocity.size))
            if vdx.size > 0:
                ax[i,0].step(velocity[vdx],totflux[vdx].value/continuum(totwave[vdx]),label=self.pltlabs[i])
                ax[i,0].step(velocity[vdx],(1/np.sqrt(totivar[vdx].value))/continuum(totwave[vdx]))
                ax[i,0].plot(velocity[vdx],np.zeros(vdx.size),"k--")
                ax[i,0].plot(velocity[vdx],np.ones(vdx.size),"k--")
                for v in self.vm:
                    ax[i,0].plot(v * np.ones(2),
                                 np.array([0,1]),
                                 "r--")
                ax[i,0].tick_params(labelsize=ptsize)
                ax[i,0].set_ylim(-0.2,1.0 + 0.165*self.pltwaves.size)
                ax[i,0].legend(loc='upper left', fontsize=ptsize-7)

                nolyavdx = np.extract((totwave[vdx] < 1214.6 * u.Angstrom) | (totwave[vdx] > 1216.7 * u.Angstrom), vdx)
                ax[i,1].axis('off')
            else:
                ax[i,0].axis('off')
                ax[i,1].axis('off')
        ax[self.pltwaves.size-1,0].set_xlabel(r"Offset velocity [km s$^{-1}$]", fontsize=ptsize)
        ax[self.pltwaves.size//2,0].set_ylabel(r"Normalized Flux", fontsize=ptsize)
        ax[self.pltwaves.size//2,1].set_ylabel(r"Flux density [erg s$^{-1}$ cm$^{-2}$ $\mathrm{\AA}^{-1}$]", fontsize=ptsize)
        ax[0,0].set_title(self.titlestr, fontsize=ptsize)
        if self.saveit:
            plt.show(block=False)
            plt.pause(0.1)
            plt.savefig(self.plotfile, bbox_inches=Bbox([[1.5,0.],[9.5,9.5]]))
        else:
            plt.show(block=True)

    ######################################################################
    def remove_geolya(self):
        print("Removing Geochoronal Ly-alpha emission")
        geolya_reg  = (self.totwave > 1210.0 * u.Angstrom) & (self.totwave < 1218.0 * u.Angstrom)
        geolya_mask = (self.totflux == np.max(self.totflux[geolya_reg]))
        gdx_lo = np.extract(self.totflux == np.max(self.totflux[geolya_reg]), range(self.totflux.size))[0]
        while self.totflux[gdx_lo] * np.sqrt(self.totivar[gdx_lo]) > 1.0:
            gdx_lo -= 1
        gdx_hi = np.extract(self.totflux == np.max(self.totflux[geolya_reg]), range(self.totflux.size))[0]
        while self.totflux[gdx_hi] * np.sqrt(self.totivar[gdx_hi]) > 1.0:
            gdx_hi += 1
        geolya_mask[gdx_lo:gdx_hi] = True
        self.totflux[geolya_mask] = 0.0
        self.totivar[geolya_mask] = 0.5 * (self.totivar[gdx_lo-1] + self.totivar[gdx_hi+1])

    ######################################################################
    def splfit(self,
               binfac = 3,
               nump = 4,
               vm = np.array([])):

        newlen = self.totwave.size // binfac
        binwave = (self.totwave[:binfac*newlen].reshape(-1,binfac).sum(axis=1)/binfac).value
        binflux = (self.totflux[:binfac*newlen].reshape(-1,binfac).sum(axis=1)/binfac).value
        binivar = (self.totivar[:binfac*newlen].reshape(-1,binfac).sum(axis=1)).value


        gooddx = np.arange(newlen, dtype=np.int16)
        if self.absmask[:,0].size > 0:
            for adx in range(self.absmask[:,0].size):
                gooddx = np.delete(gooddx, np.extract((binwave[gooddx] > self.absmask[adx,0]) & (binwave[gooddx] < self.absmask[adx,1]), range(gooddx.size)))
        badness = np.zeros(gooddx.shape)

        if self.allx.size == 0:
            self.continuum = CubicSpline([0,1,2],[0,1,2])
        else:
            self.continuum = CubicSpline(self.allx,self.ally)

        corlyaarr = np.extract((binwave > 1210) & (binwave < 1220), binflux)
        if corlyaarr.size > 0:
            corlya = np.max(np.extract((binwave > 1210) & (binwave < 1220), binflux))
            corlyaldx = np.extract(binflux == corlya, range(newlen))[0]
            corlyaudx = corlyaldx
            while (binflux[corlyaldx] > binflux[0]) and (corlyaldx > 0):
                corlyaldx -= 1
            while (binflux[corlyaudx] > binflux[0]) and (corlyaudx < binflux.size-1):
                corlyaudx += 1
            print(binwave[corlyaldx],binwave[corlyaudx])
            if binwave[corlyaldx] > 1200.0 and binwave[corlyaudx] < 1250:
                binflux[corlyaldx:corlyaudx] = 0.0

        plt.close('all')
        (fig, ax) = plt.subplots(nump+1,1)
        if os.name == 'nt':
            plt.get_current_fig_manager().window.state('zoomed')
        plt.ion()
        selection = ''
        while not (selection == 'C' or selection == 'c'):
            print(f"selection: {selection}")
            if (self.allx.size > 1):
                sdx = np.argsort(self.allx)
                self.allx = self.allx[sdx]
                self.ally = self.ally[sdx]
                self.continuum = CubicSpline(self.allx,self.ally)
            window_size = 5
            smbinflux = self._boxcar_smooth(binflux, window_size)
            smbinivar = self._boxcar_smooth(binivar, window_size)
            badness = np.square(smbinflux[gooddx] - self.continuum(binwave[gooddx])) * smbinivar[gooddx]
            baddestdx = np.extract(badness == np.max(badness),range(gooddx.size))
            baddestwave = np.squeeze(np.extract(badness == np.max(badness),binwave[gooddx]))
            print(f"Baddest bin on the block at {baddestwave} with badness = {np.max(badness)}")
            self._splplt(nump,ax,binwave,binflux,binivar,baddestwave)
            print("A. Add points")
            print("B. Remove points")
            print("C. Mark complete")
            print("D. Delete a range of points")
            print("M. Mask current baddest pixel")
            print("W. Mask a range of wavelengths")
            selection = input("Make a selection: ")

            match selection:
                case 'A' | 'a':
                    points = fig.ginput(timeout=-1)
                    for (x,y) in points:
                        self.allx = np.append(self.allx, x)
                        self.ally = np.append(self.ally, y)
                case 'B' | 'b':
                    print("points")
                    for i in range(self.allx.size):
                        print(f"{i} {self.allx[i]} {self.ally[i]}")
                    rmdx = np.int16(input("Which point to remove? (-1 for nothing): "))
                    if not rmdx == -1:
                        self.allx = np.delete(self.allx, rmdx)
                        self.ally = np.delete(self.ally, rmdx)
                case 'C' | 'c':
                    print("Bye!")
                case 'D' | 'd':
                    print("Click on lower wavelength")
                    (w1,f1) = fig.ginput(timeout=-1)[0]
                    print(f"w1o = {w1}")
                    print("Click on upper wavelength")
                    (w2,f2) = fig.ginput(timeout=-1)[0]
                    print(f"whi = {w2}")
                    rmdx = np.extract((self.allx > w1) & (self.allx < w2), range(self.allx.size))
                    if rmdx.size > 0:
                        self.allx = np.delete(self.allx, rmdx)
                        self.ally = np.delete(self.ally, rmdx)
                case 'M' | 'm':
                    gooddx = np.delete(gooddx, np.extract(badness == np.max(badness),range(badness.size)))
                case 'W' | 'w':
                    print("Click on lower wavelength")
                    (w1,f1) = fig.ginput(timeout=-1)[0]
                    print(f"w1o = {w1}")
                    print("Click on upper wavelength")
                    (w2,f2) = fig.ginput(timeout=-1)[0]
                    print(f"whi = {w2}")
                    self.absmask = np.append(self.absmask, [[w1,w2]], axis=0)
                    print(f"Extracting bins {np.extract((binwave[gooddx] > w1) & (binwave[gooddx] < w2), range(gooddx.size))}")
                    gooddx = np.delete(gooddx, np.extract((binwave[gooddx] > w1) & (binwave[gooddx] < w2), range(gooddx.size)))
                case _:
                    print("Replotting")

    ######################################################################
    def _splplt(self,nump,ax,binwave,binflux,binivar,badwave):
        continuum = self.continuum(binwave)
        window_size = 5
        smbinflux = self._boxcar_smooth(binflux, window_size)
        smbinivar = self._boxcar_smooth(binivar, window_size)

        margin = 0.01
        for a in range(nump):
            wlo = binwave[0] +     a * (binwave[-1]-binwave[0])/nump
            whi = binwave[0] + (a+1) * (binwave[-1]-binwave[0])/nump
            ax[a].cla()
            for vv in range(self.vm.size//2):
                vlo = self.vm[2 * vv]
                vhi = self.vm[2 * vv+1]
                for ll in [1215.67,1025.7222,972.5367,1548.195,1550.771,1238.821,1242.804,1031.9261,1037.6167]:
                    llo = ll * (1.0 + self.zqso) * (1.0 + vlo/const.c)
                    lhi = ll * (1.0 + self.zqso) * (1.0 + vhi/const.c)
                    if wlo < lhi or lhi > wlo:
                        wdx = np.extract((binwave > wlo) & (binwave > llo) & (binwave < whi) & (binwave < lhi), range(binwave.size))
                        ax[a].fill_betweenx(continuum[wdx],llo * np.ones(wdx.size), lhi * np.ones(wdx.size),color='r',alpha=0.2)
            ax[a].fill_between(binwave,binflux+1/np.sqrt(binivar),binflux-1/np.sqrt(binivar),color='c',alpha=0.1)
            ax[a].step(binwave,binflux,"C0")
            ax[a].step(binwave,smbinflux,"C4")
            ax[a].plot(binwave,np.zeros(binwave.size),"k--")
            ax[a].plot(self.allx,self.ally,"C1o")
            if (self.allx.size > 2):
                ax[a].plot(binwave,continuum,"C2")
            if self.absmask.size > 0:
                adxs = np.extract((self.absmask[:,0] > wlo) & (self.absmask[:,1] < whi), range(self.absmask[:,0].size))
                if adxs.size > 0:
                    for adx in adxs:
                        ax[a].axvspan(self.absmask[adx,0],self.absmask[adx,1],color='y',alpha=0.2)
            pltflux = np.extract((binwave > wlo) & (binwave < whi), binflux)
            pdx = np.extract((binwave > wlo) & (binwave < whi), range(binflux.size))
            ax[a].set_xlim(left  = (1.0-margin) * wlo, right = (1.0+margin) * whi)
            ax[a].set_ylim(bottom = np.min(pltflux), top = np.max(pltflux))
            if (1.0-margin) * wlo < badwave and (1.0+margin) * whi > badwave:
                ax[a].plot(badwave*np.ones(2),np.array([np.min(pltflux), np.max(pltflux)]),"r--")
            
        bdx = np.extract((binwave > badwave - 5) & (binwave < badwave + 5), range(binwave.size))
        ax[nump].cla()
        if self.absmask.size > 0:
            adxs = np.extract((self.absmask[:,0] > badwave - 5) & (self.absmask[:,1] < badwave + 5), range(self.absmask[:,0].size))
            if adxs.size > 0:
                for adx in adxs:
                    ax[nump].axvspan(self.absmask[adx,0],self.absmask[adx,1],color='y')
        ax[nump].fill_between(binwave[bdx],binflux[bdx]+1/np.sqrt(binivar[bdx]),binflux[bdx]-1/np.sqrt(binivar[bdx]),color='c',alpha=0.1)
        ax[nump].step(binwave[bdx],binflux[bdx],"C0")
        ax[nump].step(binwave[bdx],smbinflux[bdx],"C4")
        adx = np.extract((self.allx > badwave - 5) & (self.allx < badwave + 5), range(binwave.size))
        ax[nump].plot(self.allx[adx],self.ally[adx],"C1o")
        if (self.allx.size > 2):
            ax[nump].plot(binwave[bdx],continuum[bdx],"C2")
        ax[nump].plot(badwave*np.ones(2),np.array([np.min(pltflux), np.max(pltflux)]),"r--")
        ax[nump].set_ylim(np.min(binflux[bdx]), np.max(binflux[bdx]))

        plt.show(block=False)
        plt.pause(0.001)

    ######################################################################
