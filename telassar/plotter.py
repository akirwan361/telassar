import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt

class ImPlotter:

    def __init__(self, image, data):
    #def __init__(self, world, data):
        self.image = image
        #self.world = world
        self.data = data


    def __call__(self, x, y):

        #import pdb; pbd.set_trace()
        col = int(x + 0.5)
        row = int(y + 0.5)

        im = self.image
        #world = self.world
        xc = im.world.pix2wav(col, unit = im.world.spectral_unit)
        yc = im.world.pix2offset(row, unit = im.world.spatial_unit)
        val = self.data[row, col]

        return 'y = %g x = %g p = %i q = %i data = %g' % (yc, xc, row, col, val)

        '''if (im.world is not None and row >=0 and row < im.shape[0] and
                col >= 0 and col < im.shape[1]):
            #print(f'{val} is scalar')

            xc = im.world.pix2wav(col, unit = im.world.spectral_unit)
            yc = im.world.pix2offset(row, unit = im.world.spatial_unit)
            val = self.data[row, col]

            if np.isscalar(val):
                return 'y = %g x = %g p = %i q = %i data = %g' % (yc, xc, row, col, val)
            else:
                return 'y = %g x = %g p = %i q = %i data = %s' % (yc, xc, row, col, val)
        else:
            return 'x = %1.4f, y = %1.4f' % (x, y)'''

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
