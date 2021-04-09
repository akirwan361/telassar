from astropy.io import fits
import astropy.units as u
from numpy import ma
import numpy as np
from lmfit import models
import matplotlib.pyplot as plt

from .data import DataND
from .world import Position, VelWave
from .spatial import SpatLine
from .spectral import SpecLine
from .plotter import (ImCoords, get_plot_norm, get_plot_extent,
                      get_background_rms, get_contour_levels,
                      configure_axes)
from .lines import lines

from matplotlib.ticker import AutoMinorLocator, MaxNLocator
from datetime import datetime

class PVSlice(DataND):

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
        #print(item)
        obj = super(PVSlice, self).__getitem__(item)
        #print(type(obj))
        if isinstance(obj, DataND):
            if obj.ndim == 2:
                return obj
            elif obj.ndim == 1 and _is_spatial:
                cls = SpatLine
            elif obj.ndim == 1 and _is_spectral:
                cls = SpecLine
            return cls.new_object(obj)
        else:
            return obj



    def spectral_window(self, vmin, vmax=None, unit = None):
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
            pmax = min(self.shape[1], self.velwave.wav2pix(vmax, nearest = True) + 1)

        return self[:, pmin:pmax]

    def spatial_window(self, amin, amax = None, unit = None):
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
            #print(pmin)
            pmax = min(self.shape[0], self.position.offset2pix(amax, nearest = True) + 1)
            #print(pmax)

        return self[pmin:pmax, :]

    def spatial_profile(self, arc, wave, spat_unit = False, spec_unit = False):

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

        res = self.data[sx, sy].sum(axis=1)
        wcs = self.position[sx]
        return SpatLine(data = res, wcs = wcs, unit = self.position.unit)

    def spectral_profile(self, wave, arc, spec_unit = False, spat_unit = False):

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

        if (isinstance(wave, int) or isinstance(arc, int) or len(wave) != 2 or
                len(arc) !=2):
            raise ValueError("Can't extract profile with only one point!")

        # get the spectral and spatial limits in pixel or native unts
        if spec_unit:
            lmin = max(0, self.velwave.wav2pix(wave[0], nearest=True))
            lmax = min(self.shape[1], self.velwave.wav2pix(wave[1], nearest=True))
        else:
            lmin = max(0, int(wave[0] + 0.5))
            lmax = min(self.shape[1], int(wave[1] + 0.5))

        if spat_unit:
            pmin = max(0, self.position.offset2pix(arc[0], nearest=True))
            pmax = min(self.shape[0], self.position.offset2pix(arc[1], nearest=True))
        else:
            pmin = max(0, int(arc[0] + 0.5))
            pmax = min(self.shape[0], int(arc[1] + 0.5))

        sx = slice(pmin, pmax+1)
        sy = slice(lmin, lmax+1)

        res = self.data[sx, sy].sum(axis=0)
        spec = self.velwave[sy]

        return SpecLine(data = res, spec = spec, unit = self.velwave.unit)

    def plot(self, scale = 'linear', ax = None, fig_kws = None, imshow_kws = None,
             vmin = None, vmax = None, zscale = None, emline = None):
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
            #fig, ax = plt.subplots(subplot_kw = ax_kws)
            fig, ax = plt.subplots(**fig_kws)

        # get a norm
        norm = get_plot_norm(data, vmin = vmin, vmax = vmax, zscale = zscale,
                             scale = scale)
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
        crpix2 = round(self.position.offset2pix(0), 2)
        crval2 = 0.
        cdelt1 = round(self.velwave.get_step(), 2)
        cdelt2 = round(self.position.get_step(), 2)

        hdr['NAXIS1'] = (nlbda, "Length of data axis 1")
        hdr['NAXIS2'] = (noff, "Length of data axis 2")
        hdr['CRPIX1'] = (crpix1, "Pixel coordinate at reference point")
        hdr['CRPIX2'] = (crpix2, "Pixel coordinate at reference point")
        hdr['CRVAL1'] = (crval1, "Coordinate value at reference point")
        hdr['CRVAL1'] = (crval2, "Coordinate value at reference point")
        hdr['CDELT1'] = (cdelt1, "Coordinate increment at reference point")
        hdr['CDELT2'] = (cdelt2, "Coordinate increment at reference point")
        hdr.pop('CUNIT1')

        return hdr


    def to_fits(self, fname):

        new_hdr = self.update_header()

        hdul = fits.PrimaryHDU(data=self.data.data, header=new_hdr)
        hdul.write("fname.fits")

    def radial_velocity(self, ref, lbdas, vcorr = None, unit = 'angstrom',
                        nearest = False):
        '''
        Compute the uncorrected radial velocity of an emission range based on some
        reference emission and wavelength array. Test "obj" to see if it's just an
        `mpdaf.obj.Cube` object, and if so extract the wavelength data; if "obj" is
        a wavelength array, then just use that
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
            #pmin, pmax = self.velwave.wav2pix(lbdas, nearest = True)
            # if pmin < 0:
            #     pmin = 0
            # if pmax > self.shape[1]:
            #     pmax = self.shape[1]
            # lmin, lmax = self.velwave.pix2wav([pmin, pmax])
        elif unit.lower() == 'pixel':
            obs = self.velwave.pix2wav(lbdas)
        else:
            raise Exception("unit must be pixel or angstrom")

        # Get the wavelength array
        # step = self.velwave.get_step()
        # obs = np.arange(lmin, lmax+1, step)

        # if vcorr is supplied, give it units; else set to 0
        if vcorr is None:
            vcorr = 0
        vcorr *= u.km/u.s

        # is the ref line in the list?
        try:
            for line, val in lines.items():
                if ref == line:
                    ref = val[0]
                #    print('Our reference is ', ref)
                #else:
                #    print('our reference is,', ref)

            #print(ref - obs)
            # get the doppler shifted velocity

            vrad = c.to('km/s') * (obs - ref) / ref


            # If vcorr is supplied, note that it is multiplicative, not additive
            # so it should be expressed as:
            # v_t = v_m + v_b + (v_b * v_m) / c
            # see Wright & Eastman (2014)
            # https://ui.adsabs.harvard.edu/abs/2014PASP..126..838W/abstract
            vtrue = vrad + vcorr + (vcorr * vrad) / c

            #return vtrue

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

        lpix = self.velwave.wav2pix(wave, nearest= True)
        apix = self.position.offset2pix(arc, nearest=True)

        return self.data[apix, lpix]
