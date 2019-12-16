import astropy.units as u
import numpy as np
from numpy import ma
import matplotlib.pyplot as plt
from .tools import timeit


class ImPlotter:

    def __init__(self, image, data, toggle_unit=False):
        self.image = image
        self.data = data
        self.toggle_unit = toggle_unit


    def __call__(self, x, y):

        im = self.image

        # figure out if the image passed pixel units or data units
        if self.toggle_unit:
            # get the pixel values
            col = im.world.wav2pix(x, nearest=True)
            row = im.world.offset2pix(y, nearest= True)
            '''xc = x
            yc = y
            val = self.data[row, col]

            if np.isscalar(val):
                return 'y = %g x = %g p = %i q = %i data = %g' % (yc, xc, row, col, val)
            else:
                return 'y = %g x = %g p = %i q = %i data = %s' % (yc, xc, row, col, val)'''

        else:
            col = int(x + 0.5)
            row = int(y + 0.5)

        if (im.world is not None and row >=0 and row < im.shape[0] and
                col >= 0 and col < im.shape[1]):
            #print(f'{val} is scalar')

            xc = im.world.pix2wav(col, unit = im.world.spectral_unit)
            yc = im.world.pix2offset(row, unit = im.world.spatial_unit)
            val = self.data[row, col]

            if np.isscalar(val):
                return 'y=%g x=%g p=%i q=%i data=%g' % (yc, xc, row, col, val)
            else:
                return 'y=%g x=%g p=%i q=%i data=%s' % (yc, xc, row, col, val)
        else:
            return 'x=%1.4f, y=%1.4f' % (x, y)

def get_plot_norm(data, vmin = None, vmax = None, zscale = False, scale = 'linear'):
    from astropy import visualization as viz
    from astropy.visualization.mpl_normalize import ImageNormalize

    if zscale:
        interval = viz.ZScaleInterval()
        vmin, vmax = interval.get_limits(data.filled(0))

    if scale == 'linear':
        stretch = viz.LinearStretch
    elif scale == 'log':
        stretch = viz.LogStretch
    elif scale in ('asinh', 'arcsinh'):
        stretch = viz.AsinhStretch
    elif scale == 'sqrt':
        stretch = viz.SqrtStretch
    else:
        raise ValueError('Unknown scale: {}'.format(scale))

    norm = ImageNormalize(vmin = vmin, vmax = vmax, stretch = stretch(), clip = False)

    return norm

def get_plot_extent(wcs_obj):

    '''
    Assuming a `PVSlice.world` object is passed, get the extents for plotting
    '''
    xmin = wcs_obj.get_spectral_start()-0.5
    xmax = wcs_obj.get_spectral_end()+0.5
    ymin = wcs_obj.get_spatial_start()-0.5
    ymax = wcs_obj.get_spatial_end()+0.5

    return xmin, xmax, ymin, ymax

def get_background_rms(data, sigma = 3, N = 10, mask = None):
    '''
    Get the background rms/sigma value of the data. We consider a
    3sigma value as the detection limit, so it is useful to scale
    contour plots using this value to emphasize source detections.

    Parameters
    ----------
    data : `np.ndarray` or `np.ma.MaskedArray`
        the data to be plotted
    sigma : int
        the sigma value sent to `SigmaClip`. data above or below sigma stddevs
        will be ignored in computing the background RMS; default is 3.
    N : int
        the number of sampling boxes to fit in the data. Default is 10. if the
        box sizes are a non-integer value, then the `edge_method` keywork used
        in `Background2D` is set to "pad"
    mask : `np.ma.MaskedArray` or None
        if no mask is specified,

    '''

    from astropy.stats import SigmaClip
    from photutils import Background2D, MedianBackground, StdBackgroundRMS

    # get a boxsize. sides must be integer values, so if there's a
    # remainder then set an edge_method keyword
    ny, nx = data.shape
    sy = ny // N
    sx = nx // N

    if (ny % N != 0) or (nx % N != 0):
        edge_method = 'pad'
    else:
        edge_method = None

    # is there a mask?
    if mask is None:
        try:
            mask = data.mask
        except Exception:
            mask = ~(np.isfinite(data))
    else:
        mask = mask
    #mask = data.mask if mask is None else mask

    # do we need to unmask the data?
    # TODO: why does data[~data.mask] compress data to 1D?
    # reshape if necessary I guess
    try:
        ndata = data[~data.mask]
        #print(ndata.shape)
        if ndata.shape != data.shape:
            ndata = ndata.reshape(data.shape)
        #print(ndata.shape)
    except Exception as e:
        print('Exception: ', e)
        ndata = ma.getdata(data)

    # get the background RMS. This calls `photutils.Background2D`
    sigma_clip = SigmaClip(sigma=sigma)
    bkg_estimator = MedianBackground()
    rms_estimator = StdBackgroundRMS()
    bkg = Background2D(ndata, (sy, sx), filter_size = (3,3), sigma_clip = sigma_clip,
                    bkg_estimator = bkg_estimator, bkgrms_estimator = rms_estimator,
                    mask = mask, edge_method = edge_method)

    sigrms = bkg.background_rms_median

    return sigrms

def get_contour_levels(data, sigma):
    '''
    Generate contour levels?
    '''
    # quick upper/lower bound for exponents
    # Contours typically start between 3sigma and 5sigma and are based on a
    # sqrt(2) log scale, such that
    #   lvls = (4/3) * sigma * sqrt(2)^x
    # which begins the levels at 4sigma
    # By default, the lower and upper levels are 0.017 and 1, and the exponents
    # are found by:
    #   x = (2 * ln(3 * lvls / sigma) / ln(2) - 4
    expo = lambda lvl, sgma : (2 * np.log(3 * lvl/sgma) / np.log(2)) - 4

    # get min and max of data
    dmax = data.max()
    dmin = data.min()

    # get upper and lower contour levels
    l0 = 0.01697583 * dmax
    l1 = 1. * dmax

    # get upper and lower exponent bounds
    n0 = expo(l0, sigma)
    n1 = expo(l1, sigma)

    # get range of exponents. by default, ,there are 7 levels
    xp = np.linspace(n0, n1, 7)

    # get primary and background contour levels
    lvls1 = (4/3) * sigma * np.sqrt(2)**xp
    lvls2 = np.linspace(dmin, 0.8 * sigma, 9)

    #scale = np.array([0.01697583, 0.03395166, 0.06790333, 0.13580666,
    #                  0.27161332, 0.54322664, 1.08645327])
    #lvls1 = dmax * scale

    return lvls1, lvls2
