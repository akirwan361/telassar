import numpy as np
from astropy.io import fits
from telassar import PVSlice
from telassar.tools import is_notebook
import astropy.units as u
import matplotlib.pyplot as plt


"""
This script loads two data cases for testing the `telassar` module: one file is
already in PV-format; the other file is a data cube which will automatically be
reduced to 2D for simplicity. Both show [SII] emissions from a young stellar jet
taken with the MUSE and X-Shooter instruments, respectively.

Further, this will show how a `PVSlice` object can be loaded from
either a numpy data array (provided a header is passed), or from a
FITS file directly (provided it is already in PV-format).

Basic usage:

If loading from a FITS file,

    from telassar import PVSlice
    obj = PVSlice(filename = fname)

Info about the object can be accessed by `obj.info()`

If loading from a numpy array,

    obj = PVSlice(data = my_data)

The `obj.info()` call will attempt to print both data info and world coordinates
info, so if a data array is passed with no `header` argument, no coordinates will
be installed. Otherwise, pass a FITS-format header with `header = my_header`.

Running the script will ask if you want to see the printed info, and if you want
to see the position-velocity plots of the data.

Other attributes you can check:
    obj.position -- view the spatial WCS info
    obj.velwave -- view the spectral/velocity WCS info

"""


# Load the files from the directory

fname1 = "SII_rotated_new.fits"
fname2 = "data/HD163296_SIIF_6730.fits"
f1 = fits.open(fname1)[0]
data1 = f1.data
hdr1 = f1.header

# the first data file is a cube, so it is reduced here
muse_pv = np.sum(data1[:, :, 16:29], axis = 2).T
# load from a data array
muse_sii = PVSlice(filename = fname1, data = muse_pv, header = hdr1)
# load from a FITS file
xshoo_sii = PVSlice(fname2)

show_data = input("Print data info? [y/n] ")

if show_data.lower() == 'y':
    print("\nPrinting info for `muse_sii`: \n")
    muse_sii.info()
    print("\nPrinting info for `xshoo_sii`: \n")
    xshoo_sii.info()
    print("")
else:
    pass

show_plot = input("Show contour plots? [y/n] ")

if show_plot.lower() == 'y':
    #ax1 = show_contours(muse_sii)
    #ax2 = show_contours(xshoo_sii)
    ax1 = muse_sii.plot_contours(emline = 'SII6731')
    ax2 = xshoo_sii.plot_contours(emline = 'SII6731')

    if is_notebook():
        plt.ion()
    else:
        plt.show()
else:
    pass
