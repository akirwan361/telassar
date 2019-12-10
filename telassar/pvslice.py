from astropy.io import fits
import astropy.units as u
from numpy import ma
import numpy as np
from lmfit import models

from .data import Data2D
from .world import World
from .plotter import ImPlotter, get_plot_norm, get_plot_extent

class PVSlice(Data2D):

    '''
    This is to just manage the 2d data shit, but it might
    end up being completely superfluous. We'll find out.
    '''

    def vel_range(self, vmin, vmax=None, unit = None):
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
        out : float or PVSlice object
        '''

        if self.world.spectral_unit is None:
            raise ValueError("We need coordinates along the spectral direction")

        if vmax is None:
            vmax = vmin

        if unit is None:
            pmin = max(0, int(vmin + 0.5))
            pmax = min(self.shape[1], int(vmax + 0.5))

        else:
            pmin = max(0, self.world.wav2pix(vmin, nearest=True))
            pmax = min(self.shape[1], self.world.wav2pix(vmax, nearest = True) + 1)

        return self[:, pmin:pmax]

    def plot(self, scale = 'linear', ax = None, ax_kws = None, imshow_kws = None,
             vmin = None, vmax = None, zscale = None):
        '''
        This function generates an interactive plot of the desired data,
        which allows the user to click points on the graph and populate
        a `coords` list. This list is accessible everywhere else in this
        script, which may then be sent to the model generator for the
        curve fitting.

        Parameters
        ----------
        title : str
            Want a title?
        show_xlab : bool
            Self explanatory
        show_ylab : bool
            again
        ax : `matplotlib.axes.Axes`
            send your own axis instance, else use `matplotlib.pyplot.gca()`
        unit : str, or `astropy.units.Unit`
            Do you want units? None by default
        kwargs : `matplotlib.artist.Artist`
            any other arguments to be passed to the `ax.plot()` function
        '''
        import matplotlib.pyplot as plt

        if ax_kws is None:
            ax_kws = {}
        if imshow_kws is None:
            imshow_kws = {}

        if ax is None:
            fig, ax = plt.subplots(subplot_kw = ax_kws)

        # set the data and plot parameters
        data = self.data.copy()
        spectral_unit = u.Unit(self.world.spectral_unit).to_string('latex')
        spatial_unit = u.Unit(self.world.spatial_unit).to_string('latex')
        if self.world.wcs.wcs.ctype[0] == 'OFFSET':
            y_type = rf'Offset'
        else:
            y_type = ''
        if self.world.wcs.wcs.ctype[1] == 'VELO':
            x_type = r'V$_{rad}$'
        elif self.world.wcs.wcs.ctype[1] in ['WAVE', 'AWAV']:
            x_type = r'$\lambda$'
        else:
            x_type = ''
        norm = get_plot_norm(data, vmin = vmin, vmax = vmax, zscale = zscale,
                             scale = scale)
        extent = get_plot_extent(self.world)
        cax = ax.imshow(data, interpolation = 'nearest', origin = 'lower', norm =
                        norm, extent = extent, **imshow_kws)

        #if title is not None:
        #    ax.set_title(title)
        ax.set_xlabel(rf'{x_type} ({spectral_unit})')
        ax.set_ylabel(rf'{y_type} ({spatial_unit})')

        # format the coordinates
        ax.format_coord = ImPlotter(self, data)

        return cax

    def _prep_data(self, interp = 'no'):
        '''
        any fitting routine will function best when extreme outliers are
        minimised; we may mask values in our view of the data, but `lmfit` will
        operate on the unmasked data. This function will fill the masked values
        with 0 and let `lmfit` work its magic from there

        for now, just leave interp as 'no' because I don't want to work out the
        interpolation, and it will likely not be reliable for extreme outliers
        at the endpoints of the data.

        Parameters
        ----------
        interp : str, 'no', 'linear', or 'spline'
            leaving this as 'no' for now

        Returns
        ---------
        out : np.ndarray
        '''
        #case 'no'
        data = np.ma.filled(self.data, 0.)
        return data

    def prep_model(self, model_list, unit = True):
        """
        Allows for a list of models to be sent to the prepper
        'g' - Guassian Model
        'l' - Lorentzian model
        'v' - Voigt Model

        Using the above, allow the user to quickly enter
        keywords for the model types to save time

        We can add unit support later as well
        """

        # TODO: Add a comprehensive-ish treatment for units
        pix = np.arange(self.shape[0], dtype = np.float64)
        '''if unit is not None:
            # try to convert units: if self.unit isn't angstrom it will fail
            try:
                res = (pix * self.unit).to(unit).value
            except UnitConversionError:
                print("Specified units are not convertible, using pixel values")
                res = pix.copy()
        else:
            unit = self.unit
            res = self.world.pix2val(pix, 0)
        '''

        if unit:
            cunit = u.Unit(self.unit)
        else:
            cunit = u.Unit('pixel')

        xarr = self.world.pix2val(pix) if unit else pix
        res = self._prep_data() #self._data.copy()

        #self.mod_list = model_list
        model_keys = {
                'g' : 'GaussianModel',
                'l' : 'LorentzianModel',
                'v' : 'VoigtModel'
                }

        # Now, set the model_data dict with the data we
        # want to model
        model_data = {
                'x' : xarr,
                'y' : res,
                'unit' : cunit,
                'model' : []
                }

        # Next, make a list where 'type' is key and
        # the model_key values are the values
        mlist = []
        for m in model_list:
            m = m.lower()
            mdict = {}
            mdict['type'] = model_keys[m]
            mlist.append(mdict)
        # update this with the model keyword data
        model_data.update(model = mlist)

        self.model_info = model_data
        return self.model_info


    def generate_model(self, model_list, prepped_model = None, unit = True):

        '''
        Generate a model using a model list and coord data. This function
        utilises `lmfit` to generate the function models, perform a lstsq fit
        to the data, and evaluate the fit for plotting and analysis.
        '''
        # set up the model
        if prepped_model is None:
            prep = self.prep_model(model_list)
        else:
            prep = prepped_model.copy()

        composite_model = None
        params = None
        x = prep['x']
        y = prep['y']
        x_min = x.min()
        x_max = x.max()
        # probably don't need range
        #x_range = x_max - x_min

        # format the coords list as an array, and check if it contains both
        # (x, y) components or just (x, ); if only (x, ) is present evaluate
        # the data at x.
        coords = np.asarray(self.coords)
        if len(coords.shape) == 1:
            if prep['unit'] == 'arcsec':
                k = self.world.val2pix(coords, nearest = True)
            elif prep['unit'] == 'pix':
                k = [int(c + 0.5) for c in coords]
            coords = np.vstack((k, y[k])).T

        #if prep['unit'].to_string() == 'pix':
        #    coords[:, 0] /= 0.2
        # For each basis function in the model dictionary, `lmfit` will need
        # some basic parameters to set the initial conditions. These are
        # estimated by evaluating the data at specified coordinates to set a
        # centroid/offset and peak value, and a basic sigma is provided
        for i, func in enumerate(prep['model']):
            ctr, peak = coords[i]
            prefix = f'm{i}'
            model = getattr(models, func['type'])(prefix=prefix)

            if func['type'] in ['GaussianModel', 'LorentzianModel',
                                'VoigtModel']:
                model.set_param_hint('amplitude', value = 1.1*peak,
                                      min = 0.01 * peak)
                model.set_param_hint('center', value = ctr, min = ctr-10,
                                      max = ctr + 10)
                model.set_param_hint('sigma', min = 1e-6, max = 30)
                default_params = {
                        prefix+'center' : ctr,
                        prefix+'height' : peak,
                        prefix+'sigma' : 5
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

    def fit_model(self, model_list, coords = None, mode = 'components', plot = False,
                  unit = True):

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
        mode : None, or str
            If plot is True, do you want residuals or components plotted?

        Returns
        ------------
        out : `lmfit.models.results` or whatever it is
        """
        import matplotlib.pyplot as plt
        # if coords are given, format them and override the class attribute
        if coords is not None:
            try:
                coords = np.asarray(coords, dtype = np.float64)
            except ValueError:
                print("Coords must be numeric")
            self.coords = coords

        # make the model
        model_data, params = self.generate_model(model_list)

        result = model_data.fit(self.model_info['y'], params, x = self.model_info['x'])
        self.fit_result = result

        if plot:
            ax = plt.gca()

            # figure out labels for the axes
            if self.unit == 'arcsec':
                xlab = 'Offset (arcsec)'
            if self.unit == 'angstrom':
                xlab = r'$\lambda$'
            else:
                xlab = 'Pixel'

            # Handle motion events?
            def _on_move(event):
                if event.inaxes is not None:
                    xc, yc = event.xdata, event.ydata
                    try:
                        #i = self.world.pix2val(xc)
                        i = self.world.val2pix(xc, nearest = True)
                        x = self.world.pix2val(i)
                        #event.canvas.toolbar.set_message(
                        event.canvas.toolbar.set_message(
                            f'xc = {xc:0.2f} yc = {yc:0.2f} {self.unit} = {x:0.1f} k = {i} data = {self._data[i]:0.2f}' )
                    except Exception:# as e:
                        #print(e) # for debug
                        pass

            # if no mode is given, default to 'components'?
            if mode is None:
                mode = 'components'

            if mode.lower() == 'components':
                ax.scatter(model_data['x'], model_data['y'], s = 4)
                components = result.eval_components(x = model_data['x'])
                for i, model in enumerate(model_data['model']):
                    ax.plot(model_data['x'], components[f'm{i}'], label = f'm{i}_{model["type"]}')
                ax.set_xlabel(xlab)
                # make centroid labels?
                for key, val in result.params.items():
                    if key.endswith('center'):
                        lab = str(np.round(val, 2)) + "''"
                        ax.axvline(val, ls = ':')
                        plt.text(val + 0.2, y = 0.8 * model_data['y'].max(), s = lab,
                                rotation = 90)

            if mode.lower() == 'residuals':
                print('Do something')

    def plot_components(self, **kwargs):
        """
        Make a pretty plot of the components of the fit
        """
        import matplotlib.pyplot as plt

        xarr = self.model_info['x']
        yarr = self.model_info['y']
        res = self.fit_result

        if self.unit is u.Unit('arcsec'):
            xlab = 'Offset (arcsec)'
            #print(xlab)
        elif self.unit is u.Unit('angstrom'):
            xlab = r'$\lambda$'
        else:
            xlab = 'Pixel'


        ax = plt.gca(**kwargs)
        ax.scatter(xarr, yarr, s = 4)
        components = res.eval_components(x = xarr)
        for i, model in enumerate(self.model_info['model']):
            ax.plot(xarr, components[f'm{i}'], label = f'm{i}_{model["type"]}')

        # set labels for the centroids?
        for key, val in res.params.items():
            if key.endswith('center'):
                lab = str(np.round(val, 2)) + "''"
                ax.axvline(val, ls = ':')
                plt.text(val + 0.2, y = 0.8 * self.max(), s = lab,
                        rotation = 90)

        # if ylabel is None:
        #     ylabel = 'ADU'
        # if xlabel is None:
        #     xlabel = xlab
        # ax.set_ylabel(ylabel)
        # ax.set_xlabel(xlabel)


'''
coords = []

# define a function for handling click events
# and saving the coordinates in the list
def _on_click(event):

    global ix, iy

    if event.button == 1:
        ix, iy = event.xdata, event.ydata
        print(f'x = {ix}, y = {iy}')
        coords.append((ix, iy))
    elif event.button == 3:
        fig.canvas.mpl_disconnect(cid)
    return
'''
