import astropy.units as u
import numpy as np
from numpy import ma

from .data import DataND
#from .pvslice import PVSlice
#from .spatial import SpatLine
#from .spectral import SpecLine


def _check_coords(a, b):

    if a.velwave is not None and b.velwave is not None:
        if not np.allclose(a.velwave.get_start(), b.velwave.get_start(), 
                           atol=1e-6, rtol=0):
            raise ValueError("Cannot perform operation on items with different\
                spectral coordinates!")

    if a.position is not None and b.position is not None:
        if not np.allclose(a.position.get_start(), b.position.get_start(),
                            atol=1e-6, rtol=0):
            raise ValueError("Cannot perform operation on items with different\
                    spatial coordinates!")


def _check_shape(a, b, dims=slice(None)):
    
    if not np.array_equal(a.shape[dims], b.shape[dims]):
        raise ValueError("Cannot perform operation on arrays with different\
                shapes!")


def _do_math(operation, a, b):

    _check_coords(a, b)
    
    if a._is_spatial:
#    if isinstance(a, SpatLine):
        unit = a.position.unit
    else:
        unit = a.velwave.unit
    funit = a.flux_unit

    if operation is ma.multiply:
        unit = unit ** 2
    elif operation is ma.divide:
        unit = u.dimensionless_unscaled
    else:
        unit = unit

    new_data = operation(a.data, b.data)
    return a.__class__.new_object(a, data=new_data, flux_unit=funit,
                                  unit=unit)


class MathHandler(DataND):

    def __add__(self, other):
        if not isinstance(other, DataND):
            return self.__class__.new_object(self, data=self._data + other)
        else:
            return _do_math(ma.add, self, other)

    def __sub__(self, other):
        print(type(other))
        if not isinstance(other, DataND):
            return self.__class__.new_object(self, data=self._data - other)
        else:
            return _do_math(ma.subtract, self, other)

    def __mul__(self, other):
        if not isinstance(other, DataND):
            return self.__class__.new_object(self, data=self._data * other)
        else:
            return _do_math(ma.multiply, self, other)

    def __div__(self, other):
        if not isinstance(other, DataND):
            return self.__class__.new_object(self, data=self._data / other)
        else:
            return _do_math(ma.divide, self, other)

    def __rsub__(self, other):
        if not isinstance(other, DataND):
            return self.__class__.new_object(self, data=other - self._data)

    def __rdiv__(self, other):
        if not isinstance(other, DataND):
            return self.__class__.new_object(self, other / self._data)
