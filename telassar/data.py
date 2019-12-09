import astropy.units as u
from astropy.io import fits
import numpy as np
from numpy import ma

from .world import World

class Data2D:

    def __init__(self, filename = None, data = None, mask = False, dtype = None,
                 ext = None, header = None, unit = None, **kwargs):

        #hdul = None
        self.filename = filename
        self.ext = ext
        self._data = data
        self._dtype = dtype
        self.header = header or fits.Header()
        #self.unit = u.Unit(unit)
        self.world = None


        if (filename is not None) and (data is None):
            # read in a fits file
            hdul = fits.open(filename)
            if len(hdul) == 1:
                self.ext = 0
            elif isinstance(ext, int):
                self.ext = ext
            hdr = hdul[self.ext].header
            self.header = hdr #format_header(hdr=hdr, mode=mode)
            self._data = hdul[self.ext].data
            if 'BUNIT' in self.header:
                self.flux_unit = u.Unit(self.header['BUNIT'])
            else:
                self.flux_unit = u.dimensionless_unscaled
            self._mask = ~(np.isfinite(self._data))
            self.world = World(self.header)

        else:
            if mask is ma.nomask:
                self._mask = mask

            if data is not None:
                if self._dtype is None:
                    self._dtype = np.float64

                if isinstance(data, ma.MaskedArray):
                    self._data = np.array(data.data, dtype = np.float64,
                                          copy = True)
                    if data.mask is ma.nomask:
                        self._mask = data.mask
                    else:
                        self._mask = np.array(data.mask, dtype = bool,
                                              copy = True)
                else:
                    self._data = np.array(data, dtype = np.float64,
                                          copy = True)
                    if mask is None or mask is False:
                        self._mask = ~(np.isfinite(data))
                    elif mask is True:
                        self._mask = np.ones(shape = data.shape, dtype = bool)
                    elif mask is not ma.nomask:
                        self._mask = np.array(mask, dtype = bool, copy = True)

            self.flux_unit = u.Unit(u.dimensionless_unscaled)
            self.world = World(hdr = header)
            self.header = self.world.wcs.to_header()
            #self.world = World(self.header)

    @property
    def data(self):
        res = ma.MaskedArray(self._data, mask = self._mask, copy = False)
        return res

    @data.setter
    def data(self, val):
        if isinstance(val, ma.MaskedArray):
            self._data = val.data
            self._mask = val.mask
        else:
            self._data = val
            self._mask = ~(np.isfinite(val))

    @property
    def shape(self):
        return self._data.shape

    @property
    def ndim(self):
        try:
            return self.header['NAXIS']
        except KeyError:
            return None

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, val):
        if val is ma.nomask:
            self._mask = val
        else:
            self._mask = np.asarray(val, dtype = bool)

    @classmethod
    def new_object(cls, object, data = None, unit = None):
        '''
        Copy attributes from one object into a new instance
        Needs testing...

        Parameters
        ----------
        object : `SData`
            Template object
        data : array-like
            If you don't want to copy `object.data`, send a separate array

        '''
        data = object.data if data is None else data
        if unit is None:
            try:
                unit = object.unit
            except AttributeError:
                unit = None

        kwargs = dict(filename = object.filename, data = data, unit = unit,
                      ext = object.ext, header = object.header.copy())#,
                      #world = object.world)


        try:
            kwargs['world'] = object.world
            #kwargs['mode'] = get_mode_from_unit(object.unit)
            #print("I'm in the first try statement") #doctest
        except AttributeError:
            #mode = get_mode_from_unit(unit)
            kwargs['world'] = World(object.header)#, mode = mode)

        return cls(**kwargs)

    def copy(self):
        #res = deepcopy(self)
        #return res
        return self.__class__.new_object(self)

    def __repr__(self):
        '''
        For pretty printing
        '''
        fmt = """<{}(shape={}, unit = '{}', dtype = '{}')>"""
        return fmt.format(self.__class__.__name__, self.shape,
                          self.unit.to_string(), self.dtype)

    def min(self):
        '''
        return minimum unmasked value in the data
        '''

        res = ma.min(self.data)
        return res

    def max(self):
        '''
        return maximum unmasked value in data
        '''
        res = ma.max(self.data)
        return res
