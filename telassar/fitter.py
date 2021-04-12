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

import logging

class Modeller:

    def __init__(self, object, **kwargs):

        self._logger = logging.getLogger(__name__)

        self.data = object.data
        self.mask = object.mask

        if object._is_spectral:
            wcs = object.velwave

        if object._is_spatial:
            wcs = object.position

        self.wcs = wcs
        self.unit = wcs.unit

        # convert the flux?
        self.flux = u.Unit('erg/cm**2/s/Angstrom')#object.flux_unit

        if hasattr(object, "_coords") and object._coords is not None:
            self._coords = object._coords
        else:
            self._coords = None

        self.shape = object.shape

    def wcs2pix(self, val, nearest = False):
        """
        quick methods to prevent a lot of if/else try/except statements
        """
        x = np.atleast_1d(val)

        # tell world2pix to make 0-relative array coordinates
        pix = self.wcs.wcs.wcs_world2pix(x, 0)[0]

        if nearest:
            pix = (pix + 0.5).astype(int)
            np.maximum(pix, 0, out = pix)
            if self.shape is not None:
                np.minimum(pix, self.wcs.shape-1, out = pix)
        return pix[0] if np.isscalar(val) else pix

    def pix2wcs(self, pix=None, unit = None):

        if pix is None:
            pixarr = np.arange(self.wcs.shape, dtype = float)
        else:
            pixarr = np.atleast_1d(pix)

        res = self.wcs.wcs.wcs_pix2world(pixarr, 0)[0]
        
        if self.unit == u.Unit("m/s"):
            print("in m/s")
#            unit = u.Unit("km/s")
            res = (res * self.unit).to(u.Unit("km/s")).value
        
        if unit is not None:
            res = (res * self.unit).to(unit).value

        return res[0] if np.isscalar(pix) else res


    def _prep_data(self, interp='no'):
        """
        we may mask values in our view of the data, but `lmfit` will operate on
        the unmasked data. This function fills masked values with 0.

        TODO: include interpolation aspect in case we want to fill them with
        something more natural to the data

        """
        data = ma.filled(self.data, 0.)
        return data

    def _prep_model(self, model_list, unit = True):
        """
        Allows for a model or list of models to be sent to the model prepper

            'g' : Gaussian Model
            'l' : Lorentzian Model
            'v' : Voigt Model

        These allow quick entry of keywords for model types

        TODO: add option for unit conversion, but for now just make `unit` bool
        """
        if unit:
            xunit = self.unit
            yunit = self.flux
        else:
            xunit = u.dimensionless_unscaled
            yunit = u.dimensionless_unscaled

        pix = np.arange(self.wcs.shape, dtype=np.float64)

        xarr = self.pix2wcs(pix) if unit else pix
        res = self._prep_data()

        # Set the model keys
        model_keys = {
                'g' : 'GaussianModel',
                'l' : 'LorentzianModel',
                'v' : 'VoigtModel'
                }

        # set the model_data dict with the data we want to model
        model_data = {
                'x' : xarr,
                'y' : res,
                'unit' : xunit,#spec_unit,
                'model' : []
                }
        # Next, make list where 'type' is key and model_key vals are values
        mlist = []
        for m in model_list:
            m = m.lower()
            mdict = {}
            mdict['type'] = model_keys[m]
            mlist.append(mdict)
        # update the model with keyword data
        model_data.update(model=mlist)

        # make model_data an attribute?
        self.model_info = model_data
        return self.model_info

    def make_model(self, model_list, coords = None, prepped_model = None,
                  unit = True):
        """
        Generate a model using a model list and coord data. This function uses
        `lmfit` to generate the models, perform a least-squares fit, and evaluate
        the fit for plotting and analysis

        Parameters
        -----------
        model_list : list
            list of keywords or model names to pass to the fitter; the number of
            model keywords should equal the length of the `coords` array
        coords : array_like
            list of (x, y) coordinates containing the positions of the initial
            peak guesses
        prepped_model : dict
            a dictionary containing the data, model names, and units; if None,
            a model will be generated
        """

        # set up the model
        if prepped_model is None:
            prep = self._prep_model(model_list)
        else:
            prep = prepped_model.copy()

        composite_model = None
        params = None
        x = prep['x']
        y = prep['y']
        xmin = x.min()
        ymin = y.min()

        # if no coordinates are provided, make a guess.
        # TODO: update this for guessing when there are multiple peaks
        if coords is not None:
            coords = np.asarray(coords)
        elif coords is None and self._coords is not None:
            coords = np.asarray(self._coords)
        else:
            xpix = np.where(self.data == y.max())[0][0]
            xcoord = self.pix2wcs(xpix)
            coords = np.asarray([(xcoord, y.max())])

        # populate the model dictionary with initial parameters
        for i, func in enumerate(prep['model']):
            ctr, peak = coords[i]
            prefix = f'm{i}'
            model = getattr(models, func['type'])(prefix=prefix)

            if func['type'] in ['GaussianModel', 'LorentzianModel',
                                'VoigtModel']:
                model.set_param_hint('amplitude', value=1.1*peak,
                                      min=0.5 * peak)
                model.set_param_hint('center', value=ctr, min=ctr - 2,
                                      max=ctr + 2)
                model.set_param_hint('sigma', min=1e-6, max=10) #max=30
                default_params = {
                        prefix+'center': ctr,
                        prefix+'height': peak,
                        prefix+'sigma': 5
                    }
            else:
                raise NotImplementedError(f"Model {func['type']} not implemented yet")

            # make the parameters
            model_params = model.make_params(**default_params, **func.get('params', {}))

            if params is None:
                params = model_params
            else:
                params.update(model_params)
            if composite_model is None:
                composite_model = model
            else:
                composite_model = composite_model + model

        return composite_model, params

    def fit_model(self, model_list, coords=None, mode='components', plot=False,
                  densify=10, invert_x=False, emline=None, fig_kws=None,
                  ax_kws=None):

        """
        Fit a model or composite model based on user specified parameters.
        This function uses `lmfit` to handle the grunt work.

        Parameters
        -----------
        model_list : list, str
            An ordered list of model keywords used to fill the dict with
            the model types.
        coords : list, tuple
            A list of tuples containing (x, y) coordinates for the peaks
            and their locations
        plot : bool
            Do you want the data plotted?
        densify : int, optional
            optional parameter to make the data dense for plotting fitted curves;
            if not None, the data will be refined by the `densify` factor by:
                dense_array = np.arange(xarr[0], xarr[-1], xarr_step / densify)
            where the `xarr_step` is just the step value of the array

        Returns
        ------------
        out : `lmfit.models.results` or whatever it is
        """
        import matplotlib.pyplot as plt
        # if coords are given, format them and override the class attribute

        if fig_kws is None:
            fig_kws = {'figsize': (9, 5)}
        if ax_kws is None:
            ax_kws = {'drawstyle': 'steps-mid', 'linewidth': 1}

        # Get the emission line for the title?
        if emline is not None:
            if emline in lines.keys():
                emis = lines[emline][2]
            else:
                emis = None
        else:
            emis=None

        if coords is not None:
            try:
                coords = np.asarray(coords, dtype=np.float64)
            except ValueError:
                print("Coords must be numeric")
            self._coords = coords
        else:
            try:
                coords = self._coords
            except Exception:
                print("No initial parameters supplied! Estimating from data...")

        # make the model
        model_data, params = self.make_model(model_list=model_list, coords=coords)
        xarr = self.model_info['x']
        yarr = self.model_info['y']

        result = model_data.fit(yarr, params, x=xarr)
        self.fit_result = result

        # make a dense array for curve plotting
        x_dense = np.arange(xarr[0], xarr[-1], (xarr[1] - xarr[0])/densify)

        if plot:
            self.plot(
                mode=mode,
                densify=densify,
                invert_x=invert_x,
                emline=emline,
                fig_kws=fig_kws,
                ax_kws=ax_kws
            )

    def get_info(self, convert=True):

        """
        get the basic information about the fit, ie the centroid and HWHM. this
        doesn't worry about converting e.g. angstrom to km/s beccause the main
        PVSlice class will handle this in the `radial_velocity` function.
        """
        from astropy.constants import c

        center = []
        fwhm = []
        sigma = []

        # get the mimimum possible wavelength step of instrument
        minstep = self.wcs.get_step()
        for key, val in self.fit_result.params.items():
            if key.endswith('center'):
                center.append(val.value)
            if key.endswith('fwhm'):
                fwhm.append(val.value)
            if key.endswith('sigma'):
                sigma.append(val.value)

        return np.asarray((center, fwhm, sigma)).T

    def info(self):
        """
        Print the info if you want it
        """

        log = self._logger.info
        shape_str = (' x '.join(str(x) for x in self.shape)
                    if self.shape is not None else 'no shape')

        data = ('no data' if self.data is None else f'.data({shape_str})')
        xunit = str(self.unit)
        yunit = ('no unit' if self.flux is None else str(self.flux))

#        log('%s (%s, %s)', data, xunit, yunit)
        self.wcs.info()

        # has a fit been made?
        if hasattr(self, "fit_result"):
            center, fwhm, sigma = np.round(self.get_info().T, 2)
            log("Fit Info (in %s)" % self.unit)
            log("%8s %8s" % ('Centroid', 'FWHM'))
            for c, f in zip(center, fwhm):
                log("%8s %8s" % (c, f))

    def plot(self, mode='components', densify=10, invert_x=False, emline=None,
             ax_kws=None, fig_kws=None):
        '''convenience function'''

        if fig_kws is None:
            fig_kws = {'figsize': (9, 5)}
        if ax_kws is None:
            ax_kws = {'drawstyle': 'steps-mid', 'linewidth': 1}
        
        if emline is not None:
            if emline in lines.keys():
                emis = lines[emline][2]
            else:
                emis = None
        else:
            emis=None
        # get the x and y data
        xarr = self.model_info['x']
        yarr = self.model_info['y']

        # make a dense array for curve plotting
        x_dense = np.arange(xarr[0], xarr[-1], (xarr[1] - xarr[0])/densify)

        # Does a figure exist?
        if not plt.get_fignums():
            fig, ax = plt.subplots(**fig_kws)
        else:
            ax = plt.gca()
        
        # if it exists, is there data?
        if ax.lines:
            pass
        elif ax.collections:
            pass
        else:
            ax.plot(xarr, yarr, **ax_kws)

        xtype = self.wcs.wcs.wcs.ctype[0]
        xlab = f'{xtype} ({self.unit.to_string("latex")})'

        # do we want to invert the x-axis?
        if invert_x:
            text_offset = -0.5
            x_start = self.wcs.get_stop()
            x_stop = self.wcs.get_start()
            arm = str(' (Blue)')
        else:
            text_offset = 0.5
            x_start = self.wcs.get_start()
            x_stop = self.wcs.get_stop()
            arm = str(' (Red)')

        # Handle motion events?
        def on_move(event):
            if event.inaxes is not None:
                xc, yc = event.xdata, event.ydata
                try:
                    i = self.wcs2pix(xc, nearest=True)
                    x = self.pix2wcs(i)
                    event.canvas.toolbar.set_message(
                        'xc=%g yc=%g i=%d dist=%g data=%g' %
                        (xc, yc, i, x, self._data[i]))
                except Exception as e:
                    print(e)  # for debug
                    pass

        # NOTE: all `model_data` instances changed to `self.model_info`
        if mode.lower() == 'components':
            components = self.fit_result.eval_components(x=x_dense)
            for i, model in enumerate(self.model_info['model']):
                ax.plot(x_dense, components[f'm{i}'], **ax_kws)
            ax.set_xlabel(xlab)
            ax.set_ylabel(r'Flux ($F_{\lambda}$)')

            if emis:
                ax.set_title(emis + arm)

            # make centroid labels?
            for key, value in self.fit_result.params.items():
                val = value.value
                if key.endswith('center'):
                    lab = str(np.round(val, 2)) + self.unit.to_string('latex')
                    ax.axvline(val, ls=':')
                    plt.text(val + text_offset, y=0.8 * self.model_info['y'].max(),
                             s=r' %s'%lab, rotation=90)
            plt.connect('motion_notify_event', on_move)
            ax.set_xlim(x_start, x_stop)
            plt.tight_layout()

        if mode.lower() == 'residuals':
            print('Do something')
