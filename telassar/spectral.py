from astropy.io import fits
import astropy.units as u
from numpy import ma
import numpy as np
from lmfit import models
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from .data import DataND
from .world import Position, VelWave
from .plotter import *
from .tools import is_notebook
from .lines import lines
from .fitter import Modeller, FitStats

class SpecLine(DataND):

    _is_spectral = True

    def mask_range(self, lmin=None, lmax=None, inside=True, unit=None):
        """
        Mask a region inside/outside a range [lmin, lmax]

        Note that this alters the view of the original array; copy the
        object if you want to act on a new one.

        Parameters
        -----------
        lmin : float
            Lower bound; if None, it starts at first pixel
        lmax : float
            Upper bound; if None, it picks the last pixel
        inside : bool
            toggle whether the region within or without the bounds is masked
        unit : `astropy.units.Unit`
            wave/velocity units. None by default, so pass a valid unit or
            the function will go by pixel values
        """

        if self.velwave is None:
            raise ValueError("Can't do this without spectral coordinates")

        else:
            if lmin is None:
                pmin = 0
            else:
                if unit is None:
                    pmin = max(0, int(lmin + 0.5))
                else:
                    pmin = max(0, self.velwave.wav2pix(lmin, nearest = True))
            if lmax is None:
                pmax = self.shape[0]
            else:
                if unit is None:
                    pmax = min(self.shape[0], int(lmax + 0.5))
                else:
                    pmax = min(self.shape[0], self.velwave.wav2pix(lmax, nearest = True))

            if inside:
                self.data[pmin:pmax] = ma.masked
            else:
                self.data[:pmin] = ma.masked
                self.data[pmax+1:] = ma.masked

    def integrate(self, lmin = None, lmax = None, unit = None):
        """
        This will integrate the (non-corrected) flux over a spectral range using
        a simple Simpson's Rule.

        I need to work out the units, but for now this will just return a
        result derived from the native units of the data. Note however that
        the flux over a range will be something like erg/cm^2/s and
        there are no corrections for ie reddening, extinction, etc.

        As of now, the data contains no flux units so the result will be
        simply `u.dimensionless_unscaled * u.unit`. Alternatively, if the units
        of flux are known, simply send them as a keyword when instanciating the
        object or assign the attribute manually, ie `obj.flux_unit = unit`

        Parameters
        ----------
        lmin : float
            lower bound of the velocity or wavelength; if None, the function
            begins at first pixel
        lmax : float
            upper bound; if None, this is the last pixel
        unit : `astropy.units.Unit` or None
            the spectral units of [lmin, lmax]; None by default if these are
            pixel indices

        Returns
        -----------
        out : we'll see
        """

        # get the indices of the specified ranges and all that
        # i1 is index; l1 is wave/velocity
        if lmin is None:
            i1 = 0
            lmin = self.velwave.pix2wav(-0.5)
            #print(f'lmin = {lmin}')
        else:
            if unit is None:
                l1 = lmin
                lmin = self.velwave.pix2wav(max(-0.5, l1))
            else:
                l1 = self.velwave.wav2pix(lmin, nearest=False)
            i1 = max(0, int(l1))

        if lmax is None:
            i2 = self.shape[0]
            lmax = self.velwave.pix2wav(i2 - 0.5)
        else:
            if unit is None:
                i2 = lmax
                lmax = self.velwave.pix2wav(min(self.shape[0] - 0.5, l2))
            else:
                l2 = self.velwave.wav2pix(lmax, nearest=False)
            i2 = min(self.shape[0], int(l2) + 1)

        # to work around the array limits, we'll take the lower wavelength or
        # velocity of each pixel + 1 pixel at the end
        d = self.velwave.pix2wav(-0.5 + np.arange(i1, i2+1))

        # truncate or extend the first and last pixels to the start/end of
        # the values in the spectrum
        d[0] = lmin
        d[-1] = lmax

        if unit is None:
            unit = self.velwave.unit

        # get the data over the range
        data = self.data[i1:i2]

        # do the units agree?
        if unit in self.flux_unit.bases:
            out_unit = self.flux_unit * unit
        else:
            try:
                # sort out flux density
                wunit = (set(self.flux_unit.bases) &
                         set([u.pm, u.angstrom, u.nm, u.um])).pop()

                # scale the wavelength axis
                d *= unit.to(wunit)

                # final units
                out_unit = self.flux_unit * wunit

            # if there's an error anywhere, just return unchanged units
            except Exception:
                out_unit = self.flux_unit * unit

        # standard integration: each pixel value is multiplied by the difference
        # in wavelength from the start of the pixel to the start of the next
        flux = (data * np.diff(d)).sum() * out_unit

        return flux


    def plot(self, **kwargs):
        """
        Placeholder for the moment: sort out plot interactivity etc

        NOTE: calling this function will clear the `_coords` attribute, so be
        careful there.
        """
        if is_notebook():
            self._logger.info('Click on data points to save coordinates.')
        else:
            self._logger.info("To enable point selection on the plot, press "
                          #"'ctrl+p'. Press 'ctrl+q' to disable.")
                              "'ctrl+p'. Press 'ctrl+q' to disable.")

        unit_fmt = u.Unit(self.velwave.unit).to_string('latex')

        coords = []
        # Set up some basic plot interactivity / messages and allow
        # the user to press a key to enter clickable plot mode and
        # save the clicked coordinates to a list
        def on_move(event):
            if event.inaxes is not None:
                xc, yc = event.xdata, event.ydata
                try:
                    #i = self.world.pix2val(xc)
                    i = self.velwave.wav2pix(xc, nearest = True)
                    x = self.velwave.pix2wav(i)
                    #event.canvas.toolbar.set_message(
                    event.canvas.toolbar.set_message(
                        # rf'xc = {xc:0.2f} yc = {yc:0.2f} = {x:0.1f} k = {i} '
                        # rf'data = {self._data[i]:0.2f}' )
                        'xc=%g yc=%g i=%d lbda/vel=%g data=%g' %
                        (xc, yc, i, x, self._data[i]))
                except Exception as e:
                    print(e)
                    pass

        def on_click(event):
            #global ix, iy
            xc, yc = event.xdata, event.ydata
            # ix, iy = event.xdata, event.ydata
            try:
                i = self.velwave.wav2pix(xc, nearest=True)
                x = self.velwave.pix2wav(i)
            #print(f'x={ix}, y={iy}')
                data = self._data[i]
                print(f'v={x}, flux={data}')
                coords.append((x, data))
            #coords.append((ix, iy))
            except Exception as e:
                print(e)
                pass

        def on_key(event):
            if event.key == 'a':#'ctrl+p':
                self._logger.info('Enabling point-selection mode...')
                cid = fig.canvas.mpl_connect('button_press_event', on_click)
            if event.key == 'q': #'ctrl+q':
                self._logger.info("Point-selection mode disabled.")
                cid = fig.canvas.mpl_connect('button_press_event', on_click)
                fig.canvas.mpl_disconnect(cid)
                #plt.disconnect(cid)

        fig, ax = plt.subplots(figsize = (9, 5))
        if self.velwave.unit == u.Unit("m/s"):
            wunit = u.Unit("km/s")
        else: 
            wunit = None
        xarr = self.velwave.pix2wav(unit = wunit)
        data = self.data.copy()
        kwargs.update({'drawstyle' : 'steps-mid', 'linewidth': 1})

        ax.plot(xarr, data, **kwargs)
        plt.connect('motion_notify_event', on_move)
        if is_notebook():
            cid = fig.canvas.mpl_connect('button_press_event', on_click)
        else:
            cid = fig.canvas.mpl_connect('key_press_event', on_key)

        self._coords = coords

    def fit_model(self, model_list, coords=None, plot=True, weight=False):
        '''
        A convenient wrapper around the `Modeller` class to prepare
        and fit a model.

        Parameters:
        -----------

        model_list : list
            a list of single-letter keys to pass to the modeller, corresponding
            to the type of model the user wishes to have fitted
        coords : list, optional
            coordinates to include as initial guesses for the fitter; these can
            be manually specified, or chosen interactively from the plots
        plot : bool
            if you want it plotted
        '''
        model = Modeller(self)
        model.fit_model(model_list, coords=coords, mode='components',
                                 plot=plot, densify=10, emline=None, fig_kws=None,
                                 ax_kws=None)
        return model
