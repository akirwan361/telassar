import numpy as np
import astropy.units as u
import csv
from functools import wraps
from time import time
import collections

from astropy.wcs import WCS, WCSSUB_SPECTRAL

"""
This is just an assortment of possibly useful tools that can be called
from anywhere. Mostly to keep clean.
"""

def timeit(f, *args, **kwargs):
    @wraps(f)
    def timed(*args, **kwds):
        t0 = time()
        res = f(*args, **kwds)
        t1 = time()
        return res
    return timed

def solve_linear_set(matrix, vector):
    '''
    Like Cramer's rule but easier, and you get the solution
    immediately instead of handling determinants first
    '''
    matrix = np.array(matrix)
    inv_mat = np.linalg.inv(matrix)
    B = np.array(vector)
    x = np.dot(inv_mat, B)
    return x

def is_notebook():
    """
    Are we in a jupyter notebook?
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True
        elif shell == 'TerminalInteractiveShell':
            return False
        else:
            return False
    except NameError:
        return False

def progress_bar(*args, **kwargs):
    from tqdm import tqdm, tqdm_notebook
    func = tqdm_notebook if is_notebook() else tqdm
    return func(*args, **kwargs)

def get_simple_slice(center, slit_width, dshape, slit_length = None):

    '''
    Return a slice or rectangle or whatever based on the length and width
    you want

    Parameters
    ----------
    center : tuple
        the center in (y, x) format from which to make the cut
    slit_width : int, float
        the width of the cut in pixels
    dshape : tuple
        make sure the cut doesn't extend beyond the extent of the data
    slit_length : int or float
        the length of the slit; if None, the length is determined from the
        shape
    '''

    # convert width to two radii
    ny, nx = dshape

    # make sure the center is an array
    center = np.asarray(center)

    # the radius in x is half the slit width
    rx = width / 2.

    # make sure the width is in the limits of the cube
    xmin = max(0, np.floor((center[1] - rx) + 0.5).astype(int))
    xmax = min(nx, np.floor((center[1] + rx) + 0.5).astype(int))

    # if a length is specified, find where its lower and upper
    # bounds are in the data.
    # For a reference point (xc, yc) and slit of length L orientated
    # along the y-axis, the slit extends L/2 in either direction
    # from yc. The upper and lower bounds of the slit are a distance
    # (yc +/- L/2) from the origin
    if length is not None:
        lw = length / 2.
        ymin = max(0, np.floor((center[0] - lw) + 0.5).astype(int))
        ymax = min(ny, np.floor((center[0] + lw) + 0.5).astype(int))
    else:
        ymin = 0
        ymax = ny

    simple_slice = [ slice(ymin, ymax+1), slice(xmin, xmax+1) ]

    return simple_slice

def format_header(hdr, mode):

    hdu = hdr.copy()
    #print(hdu)
    # get some values and make some assumptions
    # `cunit` can be discovered from the mode
    # `cdelt` is, for now, either 0.2 (for arcsec) or 1.25 (for angstrom)
    # because this is how MUSE handles it. Setters exist to change these
    # values. If no mode is passed, `cdelt` is 1. and `cunit` is 'pixel'
    cunit = get_unit_from_mode(mode)
    #print('cunit = ', cunit)
    ctype = "LINEAR"

    if cunit is u.arcsec:
        cdelt = 0.2
    elif cunit is u.angstrom:
        cdelt = 1.25
        ctype = 'AWAV'
    else:
        cunit = u.Unit('pixel')
        cdelt = 1.

    # make essential keywords in case the header is minimal
    # we're assuming `crpix` and `crval` to be at the origin for simplicity
    hdr_keys = {
        'CRPIX' : 1.,
        'CRVAL' : 0.,
        'CDELT' : cdelt,
        'CUNIT' : cunit.to_string('fits'),
        'CTYPE' : ctype
    }

    # if the important keywords are not in the header, add them
    # again this is for a simple 1D case! we aren't handling n > 1.
    n = hdu['NAXIS']

    for key, val in hdr_keys.items():
        if f'{key}{n}' not in hdu:
            hdu.set(f'{key}{n}', val)

    return hdu

def read_modlist_from_csv(fname):
    '''
    If your model list is in a CSV file, here's a way to pick it out

    Parameters
    -----------
    fname : str
        Obviously a filename; may add more kwargs later for an option to use
        `pandas`

    Returns
    -----------
    out : `OrderedDict`
        A dictionary of the CSV, with headers being the keys and the columns
        being the entries
    '''
    modlist = collections.defaultdict(list)

    with open(fname) as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            # Are some columns longer than others? Skip the blank cells
            if any(row[key] in (None, '') for key in row):
                continue
            # Only need the first letter since that's how we roll
            for key in row:
                modlist[key].append(row[key][0])

    return modlist


def parse_badlines(fname):
    '''
    A convenience function to parse and install skyline information
    from a data file
    '''
    with open(fname) as f:
        for line in f.readlines():
            l1, l2, emis = line.strip().split()
            yield emis, l1, l2
