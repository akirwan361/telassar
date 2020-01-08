import numpy as np
from astropy.io import fits
from telassar import PVSlice
import astropy.units as u
import matplotlib.pyplot as plt

def header():
    """Usage function"""
    print("\nA script to load two data types for testing the `telassar` module:")
    print("one file is already in PV-format; the other is a cube which this will")
    print("automatically reduce to 2D. Both show a [SII] line from a young stellar")
    print("jet taken with the MUSE and X-Shooter instruments, respectively.")
    print("\nFurther, this will show how a `PVSlice` object can be loaded from")
    print("either a numpy data array (provided a header is passed), or from a")
    print("FITS file directly (provided it is already in PV-format).")
    print("")

header()
#def main():
"""Load the files from the directory"""

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

print("Printing info for `muse_sii`: \n")
muse_sii.info()
print("\nPrinting info for `xshoo_sii`: \n")
xshoo_sii.info()
print("")

show_plot = input("Show contour plots? [y/n] ")

if show_plot.lower() == 'y':
    #ax1 = show_contours(muse_sii)
    #ax2 = show_contours(xshoo_sii)
    ax1 = muse_sii.plot_contours(emline = 'SII6731')
    ax2 = xshoo_sii.plot_contours(emline = 'SII6731')
    plt.show()
else:
    pass
