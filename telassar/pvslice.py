from astropy.io import fits
import astropy.units as u
from astropy.stats import sigma_clip
from numpy import ma
import numpy as np
import scipy as sp
from lmfit import models
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import chebfit, chebval

from .domath import MathHandler
from .data import DataND
# from .world import Position, VelWave
from .spatial import SpatLine
from .spectral import SpecLine
from .plotter import (ImCoords, get_plot_norm, get_plot_extent,
                      get_background_rms, get_contour_levels,
                      configure_axes)
from .lines import lines
from .tools import parse_badlines

from datetime import datetime
from tqdm import trange


class PVSlice(MathHandler, DataND):
    '''
    This is to just manage the data shit, but it might
    end up being completely superfluous. We'll find out.
    '''

    def __getitem__(self, item):
        """
        Return an object:
        pvslice[i, j] = value
        pvslice[:, j] = spatial profile
        pvslice[i, :] = spectral profile
        pvslice[:, :] = sub-pvslice
        """
#        print("Item type: ", type(item))
#        print("Self type: ", type(self))
#        print(super(PVSlice, self))
        obj = super(PVSlice, self).__getitem__(item)
#        print(obj.shape)
        if isinstance(obj, DataND):
#            ndim = obj.ndim
#            print('object ndim = ', ndim)
            if obj.ndim == 2:
#                print('Check the shape')
                if obj.shape[1] == 1:
#                    print('Return spatline')
                    cls = SpatLine
                elif obj.shape[0] == 1:
#                    print('Return specline')
                    cls = SpecLine
                else:
#                    print('Return object')
                    return obj
            elif obj.ndim == 1 and obj._is_spatial:
                cls = SpatLine
            elif obj.ndim == 1 and obj._is_spectral:
                cls = SpecLine
            return cls.new_object(obj)
        else:
#            print('What happened')
            return obj

    def spectral_window(self, vmin, vmax=None, unit=None):
        '''
        Get a small view of the velocity/wavelength range.

        Parameters
        -----------
        vmin : float
            lower bound; if vmax is None, only a single pixel will be returned
        vmax : float or None
            upper bound of the view range
        unit : `astropy.units.Unit` or None
            if unit is None, vmin and vmax will be treated as pixels! otherwise,
            give it a velocity or wavelength value

        Returns
        --------
        out : `telassar.PVSlice` object
        '''

        if self.velwave.unit is None:
            raise ValueError("We need coordinates along the spectral direction")

        if vmax is None:
            vmax = vmin

        if unit is None:
            pmin = max(0, int(vmin + 0.5))
            pmax = min(self.shape[1], int(vmax + 0.5))

        else:
            pmin = max(0, self.velwave.wav2pix(vmin, nearest=True))
            pmax = min(self.shape[1], self.velwave.wav2pix(vmax, nearest=True) + 1)

        return self[:, pmin:pmax]

    def spatial_window(self, amin, amax=None, unit=None):
        '''
        Return a view of a spatial window

        Parameters
        ----------
        amin : float
            the lower bound of the view; assumed in arcseconds
        amax : float or None
            the upper bound of the view. if None, only a single pixel value will
            be returned (in arcseconds). NOTE: this pixel value will span the
            entire wavelength range!
        unit : `astropy.units.Unit` or None
            if no unit is specified, values will be treated like pixels. otherwise,
            give it a u.arcsec value or something.

        Returns
        ---------
        out : `telassar.PVSlice` obj
        '''

        if self.position.unit is None:
            raise ValueError("We need coordinates along the spatial direction")

        if amax is None:
            amax = amin

        if unit is None:
            pmin = max(0, int(amin + 0.5))
            pmax = min(self.shape[0], int(amax + 0.5))

        else:
            pmin = max(0, self.position.offset2pix(amin, nearest=True))
            pmax = min(self.shape[0], self.position.offset2pix(amax, nearest=True) + 1)

        return self[pmin:pmax, :]

    def spatial_profile(self, arc, wave, spat_unit=False, spec_unit=False):

        """
        Extract a 1D spatial profile from a position-velocity slice.

        Parameters
        -----------
        arc : array-like
            the region in distance from which to extract a profile
        wave : array-like
            the wavelength/velocity range over which to sum the profile
        spat_unit : bool
            toggles whether to use pixels or the native unit for distance;
            default is False (uses pixel)
        spec_unit : bool
            toggles whether to use pixels or native unit for wavelength/velocity;
            default is False (uses pixel)

        Returns
        -----------
        out : `telassar.SpatLine` object
        """

        if (isinstance(arc, int) or isinstance(wave, int) or len(arc) != 2 or
                len(wave) !=2):
            raise ValueError("Can't extract profile with only one point!")

        # get the spatial and spectral limits in pixel or arcseconds
        if spat_unit:
            pmin = max(0, self.position.offset2pix(arc[0], nearest=True))
            pmax = min(self.shape[0], self.position.offset2pix(arc[1], nearest=True))
        else:
            pmin = max(0, int(arc[0] + 0.5))
            pmax = min(self.shape[0], int(arc[1] + 0.5))

        if spec_unit:
            lmin = max(0, self.velwave.wav2pix(wave[0], nearest=True))
            lmax = min(self.shape[1], self.velwave.wav2pix(wave[1], nearest=True))
        else:
            lmin = max(0, int(wave[0] + 0.5))
            lmax = min(self.shape[1], int(wave[1] + 0.5))


        sx = slice(pmin, pmax+1)
        sy = slice(lmin, lmax+1)
        
        # let's try something wild here
#        res = self.data[sx, sy].sum(axis=1)
        new_wave = self.velwave.pix2wav(np.arange(lmin, lmax+1))
#        print(new_wave)
        res = np.trapz(self.data[sx, sy], x=new_wave, axis=1)
        wcs = self.position[sx]
        return SpatLine(data=res, wcs=wcs, unit=self.position.unit)

    def spectral_profile(self, wave, arc, spec_unit=False, spat_unit=False):

        """
        Extract a 1D spectral profile from a position-velocity slice

        Parameters
        ----------
        wave : array-like
            the spectral region over which to extract the profile
        arc : array-like
            the distance/offset range over which to sum the profile
        spec_unit : bool
            toggles whether to use pixels or native units for the spectral region;
            default is False (use pixels)
        spat_unit : bool
            toggles pixels/native units for distance region; default is False

        Returns
        ----------
        out : `telassar.SpecLine` object
        """

#        if (isinstance(wave, int) or isinstance(arc, int) or len(wave) != 2 or
#                len(arc) !=2):
        if (isinstance(wave, (int)) or len(wave) != 2):
            raise ValueError("Can't extract profile with only one point!")
        if isinstance(arc, (int, np.int64, float)):
            a1, a2 = arc, arc
        elif isinstance(arc, (list, np.ndarray)):
            a1, a2 = arc[0], arc[-1]

        # get the spectral and spatial limits in pixel or native unts
        if spec_unit:
            lmin = max(0, self.velwave.wav2pix(wave[0], nearest=True))
            lmax = min(self.shape[1], self.velwave.wav2pix(wave[1], nearest=True))
        else:
            lmin = max(0, int(wave[0] + 0.5))
            lmax = min(self.shape[1], int(wave[1] + 0.5))

        if spat_unit:
            pmin = max(0, int(self.position.offset2pix(a1, nearest=True)))
            pmax = min(self.shape[0], self.position.offset2pix(a2, nearest=True))
        else:
            pmin = max(0, int(a1 + 0.5))
            pmax = min(self.shape[0], int(a2 + 0.5))

        sx = slice(pmin, pmax + 1)
        sy = slice(lmin, lmax + 1)

        new_arc = self.position.pix2offset(np.arange(pmin, pmax+1))
        res = np.trapz(self.data[sx, sy], x=new_arc, axis=0)
        spec = self.velwave[sy]

        return SpecLine(data=res, spec=spec, unit=self.velwave.unit,
                        header=self.header)

    def plot(self, scale='linear', ax=None, fig_kws=None, imshow_kws=None,
             vmin=None, vmax=None, zscale=None, emline=None):
        '''
        This function generates an simple plot of the desired data.

        Parameters
        ----------
        scale : str
            the interpolation style desired. default is linear, but can accept
            others from the list:
                ['linear', 'log', 'asinh', 'arcsinh', 'sqrt']
        ax : None or `matplotlib.pyplot.axes` instance
            instanciate plotter with an axis?
        ax_kws : None or dict
            keywords to be passed to the `plt.subplots()` routine
        imshow_kws : None or dict
            keywords to be passed to the `plt.imshow()` routine
        vmin : None or float
            minimum value for plotting normalization
        vmax : None or float
            maximum value for plotting normalization
        zscale : None or str
            do you want a zscale normalization?
        emline : None or str
            can optionally pass an emission line name, and the list of lines
            from above is checked to create a pretty title

        '''

        if fig_kws is None:
            fig_kws = {'figsize' : (6, 9)}
        # TODO: set some defaults here
        if imshow_kws is None:
            imshow_kws = {}

        if emline is not None:
            if emline in lines.keys():
                emis = lines[emline][2]

        # set the data and plot parameters
        res = self.copy()
        data = self.data.copy()

        if ax is None:
            fig, ax = plt.subplots(**fig_kws)

        # get a norm
        norm = get_plot_norm(data, vmin=vmin, vmax=vmax, zscale=zscale,
                             scale=scale)
        # set the extent of the data
        extent = get_plot_extent(self.position, self.velwave)

        cax = ax.imshow(data, interpolation='nearest', origin='lower',
                        norm=norm, extent=extent, **imshow_kws)

        # format the axes and coordinates
        toggle_unit = True if extent is not None else False
        configure_axes(ax, self)
        ax.format_coord = ImCoords(res, data, toggle_unit)
        fig.subplots_adjust(left=0.15, right=0.85)

        return cax

    def plot_contours(self, figure=None, place=None, sigma=None, mask=None,
                      levels1=None, levels2=None, cmaps=None, fig_kws=None,
                      plt_kws=None, emline=None):
        '''
        Generate a contour plot of the data. Useful for jet visualization!

        Parameters
        -----------
        figure : None or `matplotlib.Figure` instance
            if you have a figure instance you want to send this to, specify it
        sigma : None or float
            the basis for generating levels. a 3sigma value indicates detection
            of a source, we abbreviate it here to just sigma
        place : int
            if you have multiple subplot axes, you can send the image to one
        mask : `np.ma.masked_array`
            if you want to specify a mask to send when computing the background
            levels, do it here
        levels1 : None or `np.ndarray` or list
            the contour levels for the jets
        levels2 : None or `np.ndarray` or list
            the contour levels for the background
        cmaps : None or `matplotlib.colors.Colormap`
            the list of colormaps to pass to `plt.contour`
        emline : str
            if your emission line of interest is in `lines.py` this will do
            a pretty formatting and set it as the figure title

        The `fig_kws` and `plt_kws` are just keyword dictionaries to pass to
        the plotter
        '''

        if fig_kws is None:
            fig_kws = {'figsize' : (5, 9)}

        emis = None
        if emline is not None:
            if emline in lines.keys():
                emis = lines[emline][2]
            else:
                emis = emline

        # default cmap colors
        colors = ['gist_gray', 'Oranges', 'gray']
        data = self.data.copy()
        # generate a sigma based on the data?
        if sigma is None:
            sigma = 3.
        sig = get_background_rms(data, sigma=sigma, N=10, mask=None)

        if (levels1 is None) or (levels2 is None):
            lvls1, lvls2 = get_contour_levels(data, sig)

        levels1 = levels1 if levels1 is not None else lvls1
        levels2 = levels2 if levels2 is not None else lvls2

        cmaps = colors if cmaps is None else cmaps

        # get an extent
        if plt_kws is None:    
            ext = get_plot_extent(self.position, self.velwave)
            plt_kws = {'extent': ext}

        # make the plot
        if figure is not None:
            fig = plt.gcf()
            if place < len(fig.axes):
                ax = fig.axes[place]
        else:
            fig, ax = plt.subplots(**fig_kws)
        if emis is not None:
            ax.set_title(rf'{emis}', fontsize=14)

        jet1 = ax.contour(data, levels=levels1, cmap=cmaps[0], **plt_kws)
        jet2 = ax.contourf(data, levels=levels1, cmap=cmaps[1], **plt_kws)
        bkgrd = ax.contourf(data, levels=levels2, cmap=cmaps[2], **plt_kws,
                            alpha = 0.8)

        # format the canvas and coordinates
        configure_axes(ax, self)
        toggle_unit = True
        ax.format_coord = ImCoords(self, data, toggle_unit)

        return ax

    def moments(self, units=False):
        '''
        Return [y_width, x_width] moments (order=1) of a 2D gaussian
        Essentially the same as the example from the SciPy Cookbook:
        https://scipy-cookbook.readthedocs.io/items/FittingData.html

        Parameters
        ----------
        units : bool
            if True, convert the widths to units; otherwise treat them as pixels

        Returns
        ----------
        out : `np.ndarray`
        '''

        # use absolute values to ensure no issues with sqrt
        total = np.abs(data).sum()
        Y, X = np.indices(data.shape)
        y = np.argmax((X * np.abs(data)).sum(axis = 1) / total)
        x = np.argmax((Y * np.abs(data)).sum(axis = 0) /total)
        col = data[int(y), :]
        row = data[:, int(x)]
        xwidth = np.sqrt(np.abs((np.arange(col.size) - y)*col).sum() /
                         np.abs(col).sum())
        ywidth = np.sqrt(np.abs((np.arange(row.size) - x)*row).sum() /
                         np.abs(row).sum())
        height = data.max()
        #mom = np.array([ywidth, xwidth])
        return height, y, x, ywidth, xwidth

    def update_header(self):
        '''Format a new header'''
        hdr = self.header.copy()
        hdr['date'] = (str(datetime.now()), 'Creation Date')
        hdr['author'] = ('TELASSAR', 'Origin of the file')

        # if the data array has been altered, update the header
        noff, nlbda = self.shape
        crval1 = self.velwave.get_start()
        crpix1 = 1.
        # let's keep the starting pixel where the world coord is 0
        crpix2 = self.position.offset2pix(0)
        crval2 = 0.
        cdelt1 = self.velwave.get_step()
        cdelt2 = self.position.get_step()

        hdr['NAXIS1'] = (nlbda, "Length of data axis 1")
        hdr['NAXIS2'] = (noff, "Length of data axis 2")
        hdr['CRPIX1'] = (crpix1, "Pixel coordinate at reference point")
        hdr['CRPIX2'] = (crpix2, "Pixel coordinate at reference point")
        hdr['CRVAL1'] = (crval1, "Coordinate value at reference point")
        hdr['CRVAL2'] = (crval2, "Coordinate value at reference point")
        hdr['CDELT1'] = (cdelt1, "Coordinate increment at reference point")
        hdr['CDELT2'] = (cdelt2, "Coordinate increment at reference point")
        hdr['CUNIT1'] = self.velwave.unit.to_string("fits")

        return hdr

    def to_fits(self, fname, overwrite=False):

        new_hdr = self.update_header()

        hdul = fits.PrimaryHDU(data=self.data.data, header=new_hdr)
        hdul.writeto(fname, overwrite=overwrite)

    def radial_velocity(self, ref, lbdas, vcorr=None, unit='angstrom',
                        nearest=False):
        '''
        Compute the radial velocity of an emission range based on some
        reference emission and wavelength array.

        Parameters
        -----------
        ref : float or str
            the emission line in vaccum or air. if float, just use the number; if
            str, then it needs to be something from the `lines` list at the top
        lbdas : list, tuple, or `np.ndarray`
            the wavelength range for which to compute the velocity.
        vcorr : float
            the velocity correction; None by default
        Returns
        -----------
        velocities : `np.ndarray`
            array of velocities (blue- and red-shifted) computed from arguments
        '''

        from astropy.constants import c

        if unit.lower() == 'angstrom':
            if nearest:
                px = self.velwave.wav2pix(lbdas, nearest=True)
                obs = self.velwave.pix2wav(px)
            else:
                obs = np.atleast_1d(lbdas)
        elif unit.lower() == 'pixel':
            obs = self.velwave.pix2wav(lbdas)
        else:
            raise Exception("unit must be pixel or angstrom")

        # Get the wavelength array

        # if vcorr is supplied, give it units; else set to 0
        if vcorr is None:
            vcorr = 0
        vcorr *= u.km / u.s

        # is the ref line in the list?
        try:
            for line, val in lines.items():
                if ref == line:
                    ref = val[0]

            # get the doppler shifted velocity
            vrad = c.to('km/s') * (obs - ref) / ref


            # If vcorr is supplied, note that it is multiplicative, not additive
            # so it should be expressed as:
            # v_t = v_m + v_b + (v_b * v_m) / c
            # see Wright & Eastman (2014)
            # https://ui.adsabs.harvard.edu/abs/2014PASP..126..838W/abstract
            vtrue = vrad + vcorr + (vcorr * vrad) / c

        except TypeError:
            return 'Line Not Found!'

        # # estimate the uncertainties
        # #   dv = c * dlbda/lbda
        # # where lbda is the rest wavelength of the reference
        # # and dlbda is HWHM
        # step = self.velwave.get_step()
        # dv = c.to('km/s') * (step / ref)

        return vtrue

    def get_flux(self, arc, wave):

        lpix = self.velwave.wav2pix(wave, nearest=True)
        apix = self.position.offset2pix(arc, nearest=True)

        return self.data[apix, lpix]

    def register_skylines(self, badlines=None):
        '''
        Emulating IRAF a bit here: register a skylines file to mask
        specified regions. if no file is specified, this will look
        in the working direcetory and try to find one with the name
        `badlines.dat`
        '''
        import pathlib

        # does a skylines attribute already exist?
        skylines = self.skylines if self.skylines else {}

        # look only in the current working directory
        workdir = pathlib.Path.cwd()

        # if no data file is given, look for it
        if badlines is None:
            look_for = "badlines.dat"
            path = workdir / look_for

            if path.exists():
                for emis, l1, l2 in parse_badlines(path.name):
                    skylines[emis] = [l1, l2]
            else:
                self._logger.warning("No badlines.dat file found!")
                self._logger.warning("Unable to register skylines.")
        else:
            # Does it look like a UNIX path?
            if badlines.startswith("~"):
                badlines = pathlib.Path(badlines).expanduser()

            # look for it in the working directory first
            if (workdir / badlines).exists():
                path = workdir / badlines
            elif badlines.exists():
                path = badlines
            else:
                self._logger.warning("No badlines.dat file found!")
                self._logger.warning("Unable to register skylines.")
            if path:
                for emis, l1, l2 in parse_badlines(path):
                    skylines[emis] = [l1, l2]

        self.skylines = skylines if skylines else None

    def unregister_skylines(self, key=None):
        '''Remove skyline info from the instance'''

        # if a key is passed, is it in the sky dict?
        if key is not None:
            try:
                self.skylines.pop(key)
                self._logger.info("Unregistering skyline %s" % key)
            except Exception:
                self._logger.warning("%s not in registry!" % key)
                self._logger.warning("No skyline info unregistered",
                                    exc_info=True)
        else:
            self._logger.info("Unregistering all skyline info!")
            self.skylines = None

    def _interp_skylines(self, pix, window):
        '''
        Sigma-clip the flux data and utilize `chebfit` and `chebval` 
        from`numpy.polynomial.chebyshev` to interpolate the data.
        By default, this uses a deg=3 fit; I'll customize it later

        Parameters:
        ----------
        pix : `int`
            The pixel index at which to perform the interpolation
        '''
        # get the pixel indices of the data to replace
        l1, l2 = window

        # get the spectrum and wavelength array at the given pixel
        data = self.data[pix, :].copy()
        lbda = self.velwave.pix2wav()

        # sigma clip the data
        clip_mask = sigma_clip(data, sigma=2.).mask

        # get the coefficients
        coeffs, full_fit = chebfit(lbda[~clip_mask], data[~clip_mask], deg=3,
                                   full=True)

        # now fit the data
        res = chebval(lbda[l1:l2], coeffs)

        return res

    def _get_skylims(self, wave=None, unit=u.angstrom):

        log = self._logger.warning
        if self.skylines:
            is_registered = True
        else:
            is_registered = False

        if not is_registered:
            if wave is None:
                log("No skylines are registered and no wavelength"
                    " is specified. Aborting procedure.")
                return
            else:
                wave = np.asarray(wave)

                if len(wave) != 2:
                    log("Only two wavelength values can be specified "
                        "at a time!")
                    return

                if unit:
                    l1, l2 = self.velwave.wav2pix(wave, nearest=True)
                else:
                    l1, l2 = wave.astype(int)

            return l1, l2

        elif is_registered:

            # get the wavelengths from the skylines dict
            for k, *v in self.skylines.items():
                l1, l2 = self.velwave.wav2pix(*v, nearest=True)
                return l1, l2

    def skymask(self, arcs=None, wave=None, spat_unit=u.arcsecond, spec_unit=u.angstrom, verbose=False):
        '''
        Mask skyline emission over the given range. Note that this
        assumes the skyline covers the entire spatial range, so use
        it with caution. If reading from the `skylines` attribute,
        it assumes the values are stored in wavelengths, not pixels.

        TODO: Add support for specifying spatial extents, and add an
        option for the `skylines` dict to toggle pixels/wavelength.

        NOTE: as of now this is an unused function for this iteration;
        call `skysub()` directly!

        Parameters:
        -----------
        wave : list or tuple, optional
            The wavelength range to mask; if None, try to read from
            the `self.skylines` dict
        spec_unit : astropy.unit.Unit, optional
            self-explanatory; if None, any given `wave` value is treated
            as pixels
        verbose : bool
            if True, print a logger line for each registered skyline masked
        '''

        # first see if any skylines are registered
        if self.skylines:
            is_registered = True
        else:
            is_registered = False

        # now sort out the spatial extent
        if arcs is not None:
            arcs = np.asarray(arcs)
            if spat_unit:
                arcs = self.position.offset2pix(arcs, nearest=True)
            else:
                arcs = arcs.astype(int)
        else:
            arcs = np.array([0, self.shape[0]])

        p1, p2 = arcs

        if not is_registered:
            if wave is None:
                self._logger.warning("No skylines are registered and no wavelength"
                                     "is specified. Aborting procedure.")
                return
            else:
                wave = np.asarray(wave)

                if len(wave) != 2:
                    self._logger.warning("Only two wavelength values can be "
                                        "specified at a time!")
                    return

                if spec_unit:
                    l1, l2 = self.velwave.wav2pix(wave, nearest=True)
                else:
                    l1, l2 = wave.astype(int)

            # mask it
            self.data[p1:p2, l1:l2] = ma.masked

        elif is_registered:
            # get the wavelengths from the skylines dict
            for k, *v in self.skylines.items():
                if verbose:
                    self._logger.info("Masking %s line..." % k)

                l1, l2 = self.velwave.wav2pix(*v, nearest=True)
                self.data[p1:p2, l1:l2] = ma.masked

    def skysub(self, arcs=None, unit=u.arcsec, inplace=False, progress=False):
        #, spline=False):
        '''
        Iterate over spatial pixels and perform a Chebyshev interpolation
        of the masked skyline regions. This calls `_interp_skylines()`.

        Parameters:
        -----------
        arcs : list or tuple, optional
            pixel or offset values over which to iterate; if None,
            operate on the entire spatial range
        unit : `astropy.units.Unit`, optional
            if None, assume `arcs` is in pixels
        verbose : bool
            if True, print a logger line for each sky emission 
        progress : bool
            if True, use `tqdm` to print a progress bar
        '''
        
        res = self if inplace else self.copy()
#        print("id of original:", id(self))
#        print('id of copy: ', id(res))
        if arcs is not None:
            arcs = np.asarray(arcs)
            if unit:
                arcs = self.position.offset2pix(arcs, nearest=True)
            else:
                arcs = arcs.asype(int)
        else:
            arcs = np.array([0, self.shape[0]])

        p1, p2 = arcs

        # get the pixel limits for the sky region
        l1, l2 = self._get_skylims()

        if progress:
            self._logger.info("Interpolating masked regions...")
            f = trange
        else:
            f = np.arange

        for p in f(p1, p2):
            # TODO: find a better way to do this
            data = res.data[p, :]
            try:
                data[l1:l2] = res._interp_skylines(pix=p, window=[l1, l2])
            except Exception as e:
                print(e)
                pass
            res.data[p, :] = data

        return res

    def mean(self, axis=None):

        data = ma.average(self.data, axis=axis)

        if axis is None:
            return data
        elif axis == 0:
            return SpecLine.new_object(self, data=data)
        elif axis == 1:
            return SpatLine.new_object(self, data=data)

    def sum(self, axis=None):

        data = ma.sum(self.data, axis=axis)

        if axis is None:
            return data
        elif axis == 0:
            return SpecLine.new_object(self, data=data)
        elif axis == 1:
            return SpatLine.new_object(self, data=data)

    def median(self, axis=None):

        data = ma.median(self.data, axis=axis)

        if axis is None:
            return data
        elif axis == 0:
            return SpecLine.new_object(self, data=data)
        elif axis == 1:
            return SpatLine.new_object(self, data=data)

    def std(self, axis=None):

        data = ma.std(self.data, axis=axis)

        if axis is None:
            return data
        elif axis == 0:
            return SpecLine.new_object(self, data=data)
        elif axis == 1:
            return SpatLine.new_object(self, data=data)

    def min(self, axis=None):

        data = np.ma.amin(self.data, axis=axis)
        if axis is None:
            return data
        elif axis == 0:
            return SpecLine.new_object(self, data=data)
        elif axis == 1:
            return SpatLine.new_object(self, data=data)

    def max(self, axis=None):
        
        data = np.ma.amax(self.data, axis=axis)

        if axis is None:
            return data
        elif axis == 0:
            return SpecLine.new_object(self, data=data)
        elif axis == 1:
            return SpatLine.new_object(self, data=data)
