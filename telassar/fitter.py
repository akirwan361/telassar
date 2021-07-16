# from astropy.io import fits
import astropy.units as u
from numpy import ma
import numpy as np
from lmfit import models
import matplotlib.pyplot as plt
# from matplotlib.ticker import AutoMinorLocator
import matplotlib.transforms as transforms
from itertools import chain

# from .data import DataND
# from .world import Position, VelWave
from .plotter import *
from .tools import get_noise1D
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
        self.flux = u.Unit('erg/cm**2/s')

        if hasattr(object, "_coords") and object._coords is not None:
            self._coords = object._coords
        else:
            self._coords = None

        self.shape = object.shape

    def wcs2pix(self, val, nearest=False):
        """
        quick methods to prevent a lot of if/else try/except statements
        """
        x = np.atleast_1d(val)

        # tell world2pix to make 0-relative array coordinates
        pix = self.wcs.wcs.wcs_world2pix(x, 0)[0]

        if nearest:
            pix = (pix + 0.5).astype(int)
            np.maximum(pix, 0, out=pix)
            if self.shape is not None:
                np.minimum(pix, self.wcs.shape-1, out=pix)
        return pix[0] if np.isscalar(val) else pix

    def pix2wcs(self, pix=None, unit=None):

        if pix is None:
            pixarr = np.arange(self.wcs.shape, dtype=float)
        else:
            pixarr = np.atleast_1d(pix)

        res = self.wcs.wcs.wcs_pix2world(pixarr, 0)[0]

        if self.unit == u.Unit("m/s"):
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

    def _prep_model(self, model_list, unit=True):
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
                'g': 'GaussianModel',
                'l': 'LorentzianModel',
                'v': 'VoigtModel'
                }

        # set the model_data dict with the data we want to model
        model_data = {
                'x': xarr,
                'y': res,
                'unit': xunit,#spec_unit,
                'model': []
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

    def make_model(self, model_list, coords=None, prepped_model=None,
                   unit=True):
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

        # I haven't figured out a good way to make general initial 
        # parameters for both WFM and NFM, so this is an ugly 
        # workaround
        if np.diff(x)[0] < 0.2:
            cstep = 0.05
            siginit = 0.1
        else:
            cstep = 1.
            siginit = 1

        # estimate the noise
        noise = get_noise1D(y, full=False)
        # conversion factor: 2 sqrt(ln(2))
        factor = 2 * np.sqrt(np.log(2))

        # if no coordinates are provided, make a guess.
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
                model.set_param_hint('amplitude', value=1.1 * peak, min=1e-6)
#                        max=1.1 * peak)
                model.set_param_hint('center', value=ctr, min=0.975*ctr,
                                     max=1.025*ctr)
#                        ctr - cstep,
#                                      max=ctr + cstep)
                model.set_param_hint('sigma', value=siginit, min=1e-6,
                        max=10 * np.diff(x)[0])
#                        max=10)
                default_params = {
                        prefix+'center': ctr,
#                        prefix+'height': peak
                        prefix+'amplitude': peak,
                        prefix+'sigma': siginit,
                    }
            else:
                raise NotImplementedError(f"Model {func['type']} not implemented yet")

            # make the parameters
            model_params = model.make_params(**default_params, **func.get('params', {}))

            # we have some custom parameters we want to send, as well
            model_params.add(f"{prefix}fwhm", expr=f"2.3584*{prefix}sigma")
            model_params.add(f"{prefix}snr", expr=f"{prefix}height/ {noise}")
            model_params.add(f"{prefix}cent_err", expr=f"{prefix}fwhm / ({factor}*{prefix}snr)")

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
                  ax_kws=None, weight=False):

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
        emline : str or int, optional
            basically just a plot title
        fig_kws : dict or None, optional
            a keyword dictionary to be passed to matplotlib to format the figure
        ax_kws : dict or None, optional
            a keyword dictionary to be passed to matplotlib to format the axes
        weight : array-like or None
            an array of weights to pass to `lmfit`; if None, the weight is 
            unity 

        Returns
        ------------
        out : `lmfit.models.results` or whatever it is
        """

        import matplotlib.pyplot as plt
        # if coords are given, format them and override the class attribute
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

        # if weight is passed, is it the same shape as the data?
#        if weight is not None:
#            if weight.shape == yarr.shape:
#                weights = weight
#            elif isinstance(weight, (float, int)):
#                weights = np.zeros(yarr.shape) 
#                weights[:] 1/weight
#        elif weight is 

#        weights = noise / np.ma.sqrt(abs(yarr)) if weight else None  
        noise = np.std(get_noise1D(yarr, full=True))
        weights = weight if weight is not None else None
#        weights = np.ma.sqrt(abs(yarr) / noise) if weight is None else weight
        result = model_data.fit(yarr, params, x=xarr, nan_policy='omit',
                weights=weights,
                method='least_squares'
                )
        self.fit_result = result

        # make a dense array for curve plotting
#        x_dense = np.arange(xarr[0], xarr[-1], (xarr[1] - xarr[0])/densify)

        if plot:
            ax = self.plot(
                mode=mode,
                densify=densify,
                invert_x=invert_x,
            )
            return ax

    def get_info(self, as_dataframe=False):
        if hasattr(self, "fit_result"):
            stats = FitStats(self.fit_result)
            res = stats.return_results(as_dataframe)
            return res

    def info(self):

        log = self._logger.info

        self.wcs.info()
        if hasattr(self, 'fit_result'):
            numComp = len(self.fit_result.model.components)
            log("Fit Info (%s components)" % numComp)

            stats = FitStats(self.fit_result)

            stats.print_results()


    def plot(self, mode='components', densify=10, invert_x=False, emline=None,
             ax_kws=None, fig_kws=None, ax=None):
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
        step = np.diff(xarr)[0]

        # let's try to fix the flux units?
#        yarr *= 1e-20

        # make a dense array for curve plotting
        x_dense = np.arange(xarr[0], xarr[-1], (xarr[1] - xarr[0])/densify)

        # Does a figure exist?
#        if not plt.get_fignums():
        if ax is None:
            fig, ax = plt.subplots(**fig_kws)
        else:
            ax = ax  # plt.gca()

        # if it exists, is there data?
        if ax.lines:
            pass
        elif ax.collections:
            pass
        else:
            ax.plot(xarr, yarr, c='k', **ax_kws)

        xtype = self.wcs.wcs.wcs.ctype[0]
        xlab = f'{xtype} ({self.unit.to_string("latex")})'
        funit = u.erg / u.s / u.cm**2
        trans = transforms.blended_transform_factory(
                ax.transData, ax.transAxes)
        # do we want to invert the x-axis?
        if invert_x:
            text_offset = -1.5 * step
            x_start = self.wcs.get_stop()
            x_stop = self.wcs.get_start()
            arm = str(' (Blue)')
        else:
            text_offset = 1.5 * step
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
#                    print(e)  # for debug
                    pass

        if mode.lower() == 'components':
            components = self.fit_result.eval_components(x=x_dense)
            full_fit = self.fit_result.eval(x=x_dense)

            ax.plot(x_dense, full_fit, 'r--', lw=1)
            for i in range(len(self.model_info['model'])):
                fitted = components[f'm{i}']
                ax.plot(x_dense, fitted, **ax_kws)
                ax.fill_between(x_dense, fitted.min(), fitted, alpha=0.5)
                print("Peak %s flux: %0.3e %s" % (i + 1,
                    np.trapz(fitted, x=x_dense)*1e-20,
                    funit))
            ax.set_xlabel(xlab)
            ax.set_ylabel(r'Flux ($F_{\lambda}$)')

            if emis:
                ax.set_title(emis + arm)

            # make centroid labels?
            for i, (key, value) in enumerate(self.fit_result.params.items()):
                val = value.value
                if key.endswith('center'):
                    lab = self.unit.to_string('latex')
                    ax.axvline(val, ls=':')
                    ax.text(val + text_offset, y=0.75,
#                            y=1.0 * self.model_info['y'].max(),
                             s=r' %0.3f %s' % (val, lab), rotation=90, fontsize=14,
                             transform=trans)
            plt.connect('motion_notify_event', on_move)
            ax.set_xlim(x_start, x_stop)
            ax.set_ylim(ax.get_ylim()[0], 1.3 * self.model_info['y'].max())
#            plt.tight_layout()

        if mode.lower() == 'residuals':
            full_model = self.fit_result.eval(x=x_dense)
            ax.plot(x_dense, full_model, **ax_kws)

        return ax


class FitStats:
    '''
    a convenient wrapper for sorting out some fit statistics
    Note the centroid error is calculated by the equation:

        sigma = FWHM / (2 sqrt(ln(2)) * SNR)

    see Porter et al, 2004, A&A, 428, 327
    '''
    def __init__(self, object):
        '''pass an `lmfit.model.ModelResult` instance'''
        self._logger = logging.getLogger(__name__)

        self._model = object.model
        self._params = object.params
        self._data = object.data
        self._numComp = len(object.model.components)

        self.parse_results()

    def parse_results(self):

        pars = self._params
        N = len(self._model.components)
        columns = ['value', 'stderr']
        params = ['amplitude', 'height', 'fwhm', 'sigma', 'center', 'cent_err']
#        noise = get_noise1D(self._data, full=False)

        for i in range(N):
            # we will dynamically add attributes to the instance
            # so stats can be accessed with `FitStats.modelN`
            name = f'model_{i}'

            filled_params = []

            for p in params:
                filled_params.append([getattr(pars[f'm{i}{p}'], k) for k in columns])

            # if any values are None, fix it
            for filled in filled_params:
                if filled[1] is None:
                    filled[1] = filled[0] * 0.1

            flux, peak, fwhm, sigma, center, centerr = filled_params

            # assign the true centroid error to the centroid array
            center[1] = centerr[0]

            # get the flux
#            flux_fac = np.sqrt(2 * np.pi * sigma[0]**2)
#            flux = list(map(lambda x: x * flux_fac, peak))

            _holder = {
                'flux': np.array(flux),
                'peak': np.array(peak),
                'fwhm': np.array(fwhm),
                'center': np.array(center),
                'sigma': np.array(sigma)
            }

            # now set the attributes
            setattr(self, name, _holder)

    def print_results(self):

        log = self._logger.info
        for i, model in enumerate(self._model.components):
            name = model._name.title()
            log(" MODEL %s :  %s" % (i, name.upper()))

            for k, (v, e) in getattr(self, f'model_{i}').items():
                v = np.round(v, 2)
                e = np.round(e, 2)
                log('%8s : %8s +/- %s' % (k.title(), v, e))

#    def return_results(self, as_dataframe=False):
#
#        param_values = []
#        cols = []
#        N = self._numComp
#        for i in range(N):
#            for k, (v, e) in getattr(self, f'model_{i}').items():
#                param_values.append(v)
#                cols.append(k)
#
#        try:
#            print(N)
#            print(cols)
#            param_values = np.asarray(param_values).reshape((N, 5))
#        except Exception:
##            print(param_values)
#            pass
#
#        if as_dataframe:
#            import pandas as pd
#            param_values = pd.DataFrame(
#                    data=param_values,
#                    columns=cols[:5]
#            )
#
#        return param_values

    def return_results(self, as_dataframe=False):

        param_values = []
        columns = []
        N = self._numComp
        for i in range(N):
            for k, (v, e) in getattr(self, f'model_{i}').items():
                param_values.append([v, e])
                columns.append([k, k + "_err"])

        # flatten the columns list and get the unique values
        # in the order they appear, i.e. NOT sorted!
        columns = list(chain(*columns))
        
        indexes = np.unique(columns, return_index=True)[1]
        columns = [columns[index] for index in sorted(indexes)]

        # make a dictionary
        ddict = dict.fromkeys(columns, [])
        shape = (N, len(ddict))
        param_values = np.asarray(param_values).reshape(shape).T

        for k, v in zip(ddict.keys(), param_values):
            ddict[k] = v

        if as_dataframe:
            import pandas as pd
            param_values = pd.DataFrame(
                    ddict
#                    data=param_values,
#                    columns=columns
            )
            return param_values
        else:
            return ddict
