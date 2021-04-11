import astropy.units as u
from astropy.io import fits
import numpy as np
from numpy import ma

import logging

from .world import Position, VelWave

class DataND:

    # maybe this will help with sorting the 1D conditions?
    _is_spectral = False
    _is_spatial = False

    def __init__(self, filename=None, data=None, mask=False, dtype=None,
                 ext=None, header=None, unit=None, wcs=None, spec=None,
                 **kwargs):

        self._logger = logging.getLogger(__name__)

        self.filename = filename
        self.ext = ext
        self._data = data
        self._dtype = dtype
        self.header = header or None
        self.position = None
        self.velwave = None
        self.skylines= None

        if (filename is not None) and (data is None):
            # read in a fits file
            hdul = fits.open(filename)

            if len(hdul) == 1:
                self.ext = 0
            elif isinstance(ext, int):
                self.ext = ext

            hdr = hdul[self.ext].header
            self.header = hdr
            self._data = hdul[self.ext].data

            if 'BUNIT' in self.header:
                self.flux_unit = u.Unit(self.header['BUNIT'])
            else:
                self.flux_unit = u.dimensionless_unscaled

            self._mask = ~(np.isfinite(self._data))
            self.position = Position(self.header)
            self.velwave = VelWave(self.header)

        else:
            if mask is ma.nomask:
                self._mask = mask

            if data is not None:
                if self._dtype is None:
                    self._dtype = np.float64

                if isinstance(data, ma.MaskedArray):
                    self._data = np.array(data.data, dtype=np.float64,
                                          copy=True)
                    if data.mask is ma.nomask:
                        self._mask = data.mask
                    else:
                        self._mask = np.array(data.mask, dtype=bool,
                                              copy=True)
                else:
                    self._data = np.array(data, dtype=np.float64,
                                          copy=True)
                    if mask is None or mask is False:
                        self._mask = ~(np.isfinite(data))
                    elif mask is True:
                        self._mask = np.ones(shape=data.shape, dtype=bool)
                    elif mask is not ma.nomask:
                        self._mask = np.array(mask, dtype=bool, copy=True)

            self.flux_unit = u.Unit(u.dimensionless_unscaled)

        # this didn't work like I thought, hopefully there are no issues
        self.set_coords(wcs=wcs, spec=spec)

    @property
    def data(self):
        res = ma.MaskedArray(self._data, mask=self._mask, copy=False)
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
        if self._data is not None:
            return self._data.ndim
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
            self._mask = np.asarray(val, dtype=bool)

    @classmethod
    def new_object(cls, object, data=None, unit=None, wcs=None, spec=None):
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

        kwargs = dict(filename=object.filename, data=data, unit=unit,
                      ext=object.ext, header=object.header.copy())

        try:
            kwargs['wcs'] = object.position
            kwargs['spec'] = object.velwave
        except AttributeError:
            kwargs['wcs'] = Position(object.header)
            kwargs['spec'] = VelWave(object.header)
        return cls(**kwargs)

    def copy(self):
        return self.__class__.new_object(self)

    def __repr__(self):
        '''
        For pretty printing
        '''
        if len(self.shape) == 2:
            fmt = """<{}(shape={}, spatial unit = '{}', spectral unit = '{}',
                     dtype = '{}')>"""
            return fmt.format(self.__class__.__name__, self.shape,
                              str(self.position.unit),
                              str(self.velwave.unit).replace(' ', ''),
                              self._dtype)

        elif len(self.shape) == 1 and self._is_spatial:
            fmt = """<{}(shape={}, spatial unit = '{}', dtype = '{}')>"""
            return fmt.format(self.__class__.__name__, self.shape,
                              str(self.position.unit), self._dtype)
        elif len(self.shape) == 1 and self._is_spectral:
            fmt = """<{}(shape={}, spectral unit = '{}', dtype = '{}')>"""
            return fmt.format(self.__class__.__name__, self.shape,
                              str(self.velwave.unit).replace(' ', ''),
                              self._dtype)

    def info(self):

        log = self._logger.info
        shape_str = (' x '.join(str(x) for x in self.shape)
                    if self.shape is not None else 'no shape')
        log('%s (%s)', self.__class__.__name__,
            self.filename or 'no name')

        data = ('no data' if self._data is None else f'Data shape: ({shape_str})')
        spat_unit = ('no unit' if self.position is None else str(self.position.unit))
        spec_unit = ('no unit' if self.velwave is None else str(self.velwave.unit) )

        log('%s (%s, %s)', data, spat_unit, spec_unit.replace(' ', ''))

        if self.position is None:
            log('No world coordinates installed')
        else:
            self.position.info()

        if self.velwave is None:
            log('No spectral coordinates installed')
        else:
            self.velwave.info()

    def __getitem__(self, item):

        '''
        Return a sliced object. 
        NOTE: This doesn't work the way I want
        '''
        data = self._data[item]
        mask = self._mask
        if mask is not ma.nomask:
            mask = mask[item]
        filename = self.filename
        reshape = None

        if self.ndim == 2:
            # handle a PVSlice[ii,jj] where ii, jj can be int or slice
            # objects
            if isinstance(item, (list, tuple)) and len(item) == 2:
                try:
                    wcs = self.position[item[0]]
                    spec = self.velwave[item[1]]
                    #print(wcs)
                except Exception:
                    print('No WCS information available')
                    wcs = None
                    spec = None
                #print(wcs)
                if isinstance(item[0], int) != isinstance(item[1], int):
                    if isinstance(item[0], int):
                        reshape = (1, data.shape[0])
                    else:
                        reshape = (data.shape[0], 1)

            elif isinstance(item, (int, slice)):

                try:
                    wcs = self.position[item[0], slice(None)]
                    spec = self.velwave[item[1], slice(None)]
                except Exception:
                    wcs = None
                    spec = None

                if isinstance(item, int):
                    reshape = (1, data.shape[0])

            elif item is not None or item == ():
                try:
                    wcs = self.position.copy()
                    spec = self.velwave.copy()
                except Exception:
                    wcs = None
                    spec = None

        elif self.ndim == 1:
            # handle a spatial or spectral profile
            if isinstance(item, slice):
                try:
                    wcs = self.position[item]
                except Exception:
                    wcs = None
                try:
                    spec = self.velwave[item]
                except Exception:
                    spec = None
            elif item is None or item == ():
                try:
                    wcs = self.position.copy()
                except Exception:
                    wcs = None
                try:
                    spec = self.velwave.copy()
                except Exception:
                    spec = None
        # do we need to reshape?
        if reshape is not None:
            data = data.reshape(reshape)
            if mask is not ma.nomask:
                mask = mask.reshape(reshape)

        return self.__class__(
            filename=filename, data=data, mask=mask, dtype=self._dtype,
            ext=self.ext, header=self.header, wcs=wcs, spec=spec)

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

    def set_coords(self, wcs=None, spec=None):
        """
        Set the wcs info for the object. Hopefully this sorts the issue
        of reducing 2D PV data to 1D spatial/spectral data?
        """

        if self.header is not None:
            hdr = self.header.copy()
        else:
            hdr = None

        # Install PV coordinates. This checks first if `wcs` or `velwave` args
        # are passed and installs them; if either is`None`, then it checks for
        # a header and attempts to install coords from that; otherwise, raise
        # an error
        if len(self.shape) == 2:
            try:
                if (wcs is not None) and (spec is not None):
                    self.position = wcs
                    self.velwave = spec
                elif (wcs is None or spec is None) and (hdr is not None):
                    self.position = Position(hdr)
                    self.velwave = VelWave(hdr)
            except Exception:
                    self._logger.warning("Unable to install coordinates",
                                        exc_info=True)
                    self.position = None
                    self.velwave = None

        # If the data is 1D, sort out which is which
        if len(self.shape) ==1:
            if self._is_spatial:
                try:
                    if wcs is not None:
                        self.position = wcs
                    elif wcs is None and hdr is not None:
                        self.position = Position(hdr)
                except Exception:
                    self._logger.warning("Unable to install spatial "
                                        "coordinates", exc_info=True)
                    self.position = None
            if self._is_spectral:
                try:
                    if spec is not None:
                        self.velwave = spec
                    elif spec is None and hdr is not None:
                        self.velwave = VelWave(hdr)
                except Exception:
                    self._logger.warning("Unable to install spatial "
                                        "coordinates", exc_info=True)
                    self.position = None

    def unmask(self):
        '''Unmask all but invalid data'''
        if self.mask is not ma.nomask:
            self._mask = ~np.isfinite(self._data)
