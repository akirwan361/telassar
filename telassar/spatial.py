from astropy.io import fits
import astropy.units as u
from numpy import ma
import numpy as np
from lmfit import models
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from .data import DataND
#from .pvslice import PVSlice
from .world import Position, VelWave
from .plotter import *
from .tools import is_notebook
from .lines import lines
from .fitter import Modeller
from .domath import MathHandler

class SpatLine(MathHandler, DataND):

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
                    i = self.position.offset2pix(xc, nearest=True)
                    x = self.position.pix2offset(i)
                    event.canvas.toolbar.set_message(
                        'xc=%g yc=%g i=%d dist=%g data=%g' %
                        (xc, yc, i, x, self._data[i]))
                except Exception as e:
                    print(e)
                    pass

        def on_click(event):
            
            xc, yc = event.xdata, event.ydata
            
            try:
                i = self.position.offset2pix(xc, nearest=True)
                x = self.position.pix2offset(i)
                data = self._data[i]
                print(f'arc={x}, flux={data}')
                coords.append((x, data))
            
            except Exception as e:
                print(e)
                pass

        def on_key(event):
            
            if event.key == 'a':
                self._logger.info('Enabling point-selection mode...')
                cid = fig.canvas.mpl_connect('button_press_event', on_click)
            if event.key == 'q':
                self._logger.info("Point-selection mode disabled.")
                cid = fig.canvas.mpl_connect('button_press_event', on_click)
                fig.canvas.mpl_disconnect(cid)

        fig, ax = plt.subplots(figsize=(9, 5))
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

    def integrate(self, arcs, unit=u.arcsec):
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
        arcs : list or tuple, optional
            the integration range; if none, integrate the whole profile
        unit : `astropy.units.Unit` or None
            the spatial units of `arcs`; if None, treat as pixels

        Returns
        -----------
        out : we'll see
        """
        
        if arcs is not None:
            arcs = np.asarray(arcs)
            if unit:
                arcs = self.position.offset2pix(arcs, nearest=True)
            else:
                arcs = arcs.astype(int)
        else:
            arcs = np.asarray([0, self.shape[0]])

        # we want the effective spatial range
        p1, p2 = arcs
        dist = self.position.pix2offset(np.arange(p1, p2))

        # get the data over the range
        data = self.data[p1:p2]

        # integrate over the spatial range using trapezoidal method
        # from `numpy.trapz`
        flux = np.trapz(y=data, x=dist)
        return flux

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
                        plot=plot, densify=10, weight=weight) #, emline=None, fig_kws=None,
#                        ax_kws=None)
        return model

    def mean(self, off_min=None, off_max=None, unit=u.arcsec):
        '''
        Simply return the mean flux over some wavelength range

        Parameters
        ----------
        off_min : float
            lower bound
        off_max : float 
            upper bound
        unit : `astropy.units.Unit`
            if None, assume pixels rather than coords 

        Returns 
        -------
        out : float 
            The mean flux
        '''

        if unit is not None:
            l1, l2 = self.position.offset2pix([off_min, off_max], nearest=True)
        else:
            l1, l2 = off_min+0.5, off_max+0.5

        sa = slice(l1, l2+1)

        flux = np.ma.average(self.data[sa])
        return flux


    def sum(self, off_min=None, off_max=None, unit=u.arcsec):

        if unit is not None:
            l1, l2 = self.velwave.wav2pix([off_min, off_max], nearest=True)
        else:
            l1, l2 = off_min+0.5, off_max + 0.5

        sa = slice(l1, l2+1)

        flux = np.ma.sum(self.data[sa])
        return flux

