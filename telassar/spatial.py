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
from .fitter import Modeller

class SpatLine(DataND):

    _is_spatial = True

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

        unit_fmt = u.Unit(self.position.unit).to_string('latex')
        xlab = f'Offset ({unit_fmt})'

        coords = []
        # Set up some basic plot interactivity / messages and allow
        # the user to press a key to enter clickable plot mode and
        # save the clicked coordinates to a list
        def on_move(event):
            if event.inaxes is not None:
                xc, yc = event.xdata, event.ydata
                try:
                    #i = self.world.pix2val(xc)
                    i = self.position.offset2pix(xc, nearest = True)
                    x = self.position.pix2offset(i)
                    #event.canvas.toolbar.set_message(
                    event.canvas.toolbar.set_message(
                        # rf'xc = {xc:0.2f} yc = {yc:0.2f} = {x:0.1f} k = {i} '
                        # rf'data = {self._data[i]:0.2f}' )
                        'xc=%g yc=%g i=%d dist=%g data=%g' %
                        (xc, yc, i, x, self._data[i]))
                except Exception as e:
                    print(e)
                    pass

        def on_click(event):
            #global ix, iy
            xc, yc = event.xdata, event.ydata
            # ix, iy = event.xdata, event.ydata
            try:
                i = self.position.offset2pix(xc, nearest = True)
                x = self.position.pix2offset(i)
            #print(f'x={ix}, y={iy}')
                data = self._data[i]
                print(f'arc={x}, flux={data}')
                coords.append((x, data))
            #coords.append((ix, iy))
            except Exception as e:
                print(e)
                pass

        def on_key(event):
            if event.key == 'a':#'f':#'ctrl+p':
                self._logger.info('Enabling point-selection mode...')
                cid = fig.canvas.mpl_connect('button_press_event', on_click)
            if event.key == 'q': #'ctrl+q':
                self._logger.info("Point-selection mode disabled.")
                cid = fig.canvas.mpl_connect('button_press_event', on_click)
                fig.canvas.mpl_disconnect(cid)
                #plt.disconnect(cid)

        fig, ax = plt.subplots(figsize = (9, 5))
        xarr = self.position.pix2offset()
        data = self.data.copy()
        kwargs.update({'drawstyle' : 'steps-mid', 'linewidth': 1})

        ax.plot(xarr, data, **kwargs)
        ax.set_xlabel(rf'{xlab}')
        ax.set_ylabel(r'Flux ($F_{\lambda}$)')
        plt.connect('motion_notify_event', on_move)
        if is_notebook():
            cid = fig.canvas.mpl_connect('button_press_event', on_click)
        else:
            cid = fig.canvas.mpl_connect('key_press_event', on_key)

        self._coords = coords

    def integrate(self, amin = None, amax = None, unit = None):
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
        if amin is None:
            i1 = 0
            amin = self.position.pix2offset(-0.5)
            #print(f'lmin = {lmin}')
        else:
            if unit is None:
                i1 = amin
                amin = self.position.offset2pix(max(-0.5, i1))
            else:
                i1 = self.position.offset2pix(amin, nearest=False)
            i1 = max(0, int(i1))

        if amax is None:
            i2 = self.shape[0]
            amax = self.position.pix2offset(i2 - 0.5)
        else:
            if unit is None:
                i2 = lmax
                lmax = self.position.pix2offset(min(self.shape[0] - 0.5, i2))
            else:
                i2 = self.position.offset2pix(amax, nearest=False)
            i2 = min(self.shape[0], int(i2) + 1)

        # to work around the array limits, we'll take the lower wavelength or
        # velocity of each pixel + 1 pixel at the end
        d = self.position.pix2offset(-0.5 + np.arange(i1, i2+1))

        # truncate or extend the first and last pixels to the start/end of
        # the values in the spectrum
        d[0] = amin
        d[-1] = amax

        if unit is None:
            unit = self.position.unit

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
        flux = (data * np.diff(d)).sum() #* out_unit

        return flux

    def test_fitter(self, model_list, coords = None, plot = True):

        print("We're running a test to send to `fitter.py`")

        my_model = ModelFit(self)
        result = my_model.fit_model(model_list, coords = coords, mode = 'components',
                      plot = plot, densify = 10, emline=None, fig_kws=None, ax_kws=None)

        print(my_model.fit_statistics())
