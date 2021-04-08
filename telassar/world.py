from astropy.io import fits
import astropy.units as u
from astropy.wcs import WCS as pywcs
from astropy.wcs import WCSSUB_SPECTRAL
import numpy as np

import logging

def get_cdelt_from_cd(cd):
    '''Get the cdelt from the CD matrix'''


    dy = np.sqrt(cd[0,1]**2 + cd[1,1]**2)
    dx = np.sqrt(cd[1,0]**2 + cd[0,0]**2)
    if len(cd) == 3:
        dz = np.sqrt(cd[2,0]**2 + cd[2,2]**2)

        cdelt = np.array([dz, (dy*u.deg).to(u.arcsec).value])
    else:
        cdelt = np.array([dx, dy])
#    print(cdelt)
    return cdelt


def wcs_from_header(header):

    hdr = header.copy()

    cunit = None
    try:
        for i in range(1, hdr['NAXIS']+1):
            if 'angstrom' in hdr['CUNIT%d' %i].lower():
                cunit = u.Unit(hdr.pop('CUNIT%d'%i))
    except KeyError:
        pass
    # if 'CUNIT3' in hdr:
    #     cunit = u.Unit(hdr.pop('CUNIT3'))
    # elif 'CUNIT2' in hdr:
    #     cunit = u.Unit(hdr.pop('CUNIT2'))
    # else:
    #     cunit = u.Unit('angstrom')

    try:
        n = hdr['NAXIS']
    except KeyError:
        try:
            n = hdr['WCSAXES']
        except KeyError:
            print("Can't install coordinates!")
            return

    # generate WCS object
    mywcs = pywcs(hdr, fix=False)
    old_shape = mywcs.pixel_shape

    if mywcs.wcs.has_cd():
        #print('Has CD')
        #try:
        cdelt = get_cdelt_from_cd(mywcs.wcs.cd)
        #except IndexError:
        #    cdelt = mywcs.wcs.cd
#        print(cdelt)

    elif mywcs.wcs.has_pc():
        #print('Has PC')
        try:
            cd = np.dot(np.diag(mywcs.wcs.get_cdelt()), mywcs.wcs.get_pc())
            cdelt = get_cdelt_from_cd(cd)
        except IndexError:
            cdelt = mywcs.wcs.get_cdelt()

    if mywcs.wcs.has_cd():
        del mywcs.wcs.cd

    if n==3:
        nz, ny = old_shape[::-1][:2]
        new_wcs = mywcs.sub([WCSSUB_SPECTRAL, 0])
        new_wcs.pixel_shape = (nz, ny)

    if n==2:
        new_wcs = mywcs.copy()

    crpix2 = hdr['CRPIX2']

    ctype2 = 'OFFSET'

    # is this velocity or wavelength?
    if new_wcs.wcs.crval[0] < 0.:
        ctype1 = 'VELO'
    else:
        ctype1 = 'AWAV'

    new_wcs.wcs.crpix = [1., crpix2]
    new_wcs.wcs.cdelt = cdelt
    new_wcs.wcs.ctype = [ctype1, ctype2]

    # I don't know why it's necessary, but otherwise the step gets calculated
    # as the cd**2
    new_wcs.wcs.pc = np.diag([1., 1.])


    #new_wcs.wcs.cunit[1] = u.Unit('arcsec')
    return new_wcs

class Position:

    def __init__(self, hdr=None, crpix=1., crval=1., cdelt=0.2,
                 unit=u.arcsec, ctype='LINEAR', shape=None):

        self._logger = logging.getLogger(__name__)

        unit = u.Unit(unit).to_string('fits')

        self.shape = shape

        if (hdr is not None):
            h = hdr.copy()

            # the spatial axis should be axis=2
            axis = 2
            self.wcs = wcs_from_header(h).sub([axis])

            if self.wcs.wcs.cunit[0] != '':
                self.unit = self.wcs.wcs.cunit[0]
            elif self.wcs.wcs.ctype[0] == 'OFFSET':
                self.unit = u.Unit('arcsec')

        elif hdr is None:
            self.unit = u.Unit(unit)
            self.wcs = pywcs(naxis=1)

            # set the reference pixel
            self.wcs.wcs.crval[0] = crval
            self.wcs.wcs.ctype[0] = ctype
            self.wcs.wcs.cdelt[0] = cdelt
            self.wcs.wcs.crpix[0] = crpix
            self.wcs.pixel_shape = (shape,)

        try:
            self.shape = self.wcs.pixel_shape[0] if shape is None else shape
        except Exception:
            self._logger.warning('No shape is provided')

    def copy(self):
        """
        Copy the Position object
        """
        out = Position(shape=self.shape, unit=self.unit)
        out.wcs = self.wcs.deepcopy()
        return out

    def __repr__(self):
        return repr(self.wcs)

    def info(self, unit=None):
        try:
            unit = unit or self.unit
            start = self.get_start(unit=unit)
            step = self.get_step(unit=unit)
            type = self.wcs.wcs.ctype[0].capitalize()

            if self.shape is None:
                self._logger.info('Spatial %s: min: %0.2f" step: %0.3f"' %
                                 (type, start, step))
            else:
                end = self.get_stop(unit=unit)
                self._logger.info('Spatial %s: min: %0.1f" max: %0.1f" step: %0.3f"' %
                                 (type, start, end, step))
        except Exception as e:
            print(e)
            self._logger.info("something happened I can't fix yet")

    def __getitem__(self, item):

        if item is None:
            return self

        elif isinstance(item, int):
            if item >= 0:
                arc = self.pix2offset(pix=item)
            else:
                if self.shape is None:
                    raise ValueError("Can't return an index without a shape")
                else:
                    arc = self.pix2offset(pix=self.shape + item)
            return Position(crpix=1., crval=arc, cdelt=0., unit=self.unit,
                            ctype=self.wcs.wcs.ctype[0], shape=1)

        elif isinstance(item, slice):
            if item.start is None:
                start = 0
            elif item.start >= 0:
                start = item.start
            else:
                if self.shape is None:
                    raise ValueError("Can't return an index without a shape")
                else:
                    start = self.shape + item.start
            if item.stop is None:
                if self.shape is None:
                    raise ValueError("Can't return an index without a shape")
                else:
                    stop = self.shape
            elif item.stop >= 0:
                stop = item.stop
            else:
                if self.shape is None:
                    raise ValueError("Can't return an index without a shape")
                else:
                    stop = self.shape + item.stop
            newarc = self.pix2offset(pix=np.arange(start, stop, item.step))
            dimens = newarc.shape[0]

            if dimens < 2:
                raise ValueError("Offset with dimension < 2")
            cdelt = newarc[1] - newarc[0]
            return Position(crpix=1., crval=newarc[0], cdelt=cdelt,
                            unit=self.unit, ctype=self.wcs.wcs.ctype[0],
                            shape=dimens)
        else:
            raise ValueError("Can't do it!")

    def offset2pix(self, val, nearest=False):
        """
        Return a pixel value if given an offset value in
        arcseconds
        """
        x = np.atleast_1d(val)

        # tell world2pix to make 0-relative array coordinates
        pix = self.wcs.wcs_world2pix(x, 0)[0]

        if nearest:
            pix = (pix + 0.5).astype(int)
            np.maximum(pix, 0, out=pix)
            if self.shape is not None:
                np.minimum(pix, self.shape-1, out=pix)
        return pix[0] if np.isscalar(val) else pix

    def pix2offset(self, pix=None, unit=None):
        """
        Reverse of above: get spatial value of pixel
        """

        if pix is None:
            pixarr = np.arange(self.shape, dtype=float)
        else:
            pixarr = np.atleast_1d(pix)

        res = self.wcs.wcs_pix2world(pixarr, 0)[0]

        if unit is not None:
            res = (res * self.unit).to(unit).value

        return res[0] if np.isscalar(pix) else res


    def get_step(self, unit=None):
        '''
        get the step
        '''
        if self.wcs.wcs.has_cd():
            step = self.wcs.wcs.cd[0]
        else:
            cdelt = self.wcs.wcs.get_cdelt()[0]
            pc = self.wcs.wcs.get_pc()[0][0]
            step = cdelt #* pc

        if unit is not None:
            step = (step * self.unit).to(unit).value
        return step

    def set_step(self, x, unit=None):
        """
        If you want to change the step, do it here;
        useful if you've read in an array and now have minimal header data
        """

        if unit is not None:
            step = (x * unit).to(self.unit).value
        else:
            step = x

        # TODO: probably won't have any CD info for 1D files
        # so maybe just ignore this?
        if self.wcs.wcs.has_cd():
            self.wcs.wcs.cd[0][0] = step
        else:
            pc = self.wcs.wcs.get_pc()[0][0]
            self.wcs.wcs.cdelt[0] = step / pc

    def get_start(self, unit=None):
        """
        Get the starting pixel value
        """
        if unit is not None:
            return (self.pix2offset(0)* unit).to(self.unit).value
        else:
            return self.pix2offset(0)

    def get_stop(self, unit=None):

        if self.shape is None:
            raise IOError("Need a dimension!")
        else:
            if unit is not None:
                return (self.pix2offset(self.shape - 1) * unit).to(self.unit).value
            else:
                return self.pix2offset(self.shape - 1)

    def get_range(self, unit=None):
        '''
        Simply return the upper and lower bounds of the array
        '''
        try:
            return self.pix2offset([0, self.shape-1], unit)
        except AttributeError:
            print("no dimension provided for spatial array!")


class VelWave:

    def __init__(self, hdr=None, crpix=1., crval=1., cdelt=1.,
                 unit=u.angstrom, ctype='LINEAR', shape=None):

        self._logger = logging.getLogger(__name__)

        unit = u.Unit(unit)

        self.shape = shape

        if (hdr is not None):
            h = hdr.copy()

            # check if there's an instrument key in the header
            if 'INSTRUME' in hdr:
                if hdr['INSTRUME'] == 'XSHOOTER':
                    unit = u.Unit(u.nm)
            # the spectral axis should be 1
            axis = 1
            self.wcs = wcs_from_header(h).sub([axis])

            if self.wcs.wcs.ctype[0] == 'VELO':
                self.unit = u.Unit('km/s')
            elif self.wcs.wcs.ctype[0] == 'AWAV' or 'WAVE':
                self.unit = unit
            elif self.wcs.wcs.ctype[0] == 'LINEAR':
                self.unit = unit

        elif hdr is None:
            self.unit = u.Unit(unit)
            self.wcs = pywcs(naxis = 1)

            # set the reference pixel
            self.wcs.wcs.crval[0] = crval
            self.wcs.wcs.ctype[0] = ctype
            self.wcs.wcs.cdelt[0] = cdelt
            self.wcs.wcs.crpix[0] = crpix
            # this is weird but go with it
            self.wcs.pixel_shape = (shape,)

        try:
            self.shape = self.wcs.pixel_shape[0] if shape is None else shape
        except Exception:
            self._logger.warning('No shape is provided')

    def __repr__(self):
        return(repr(self.wcs))

    def info(self, unit=None):

        unit = unit or self.unit
        start = self.get_start(unit=unit)
        step = self.get_step(unit=unit)
        type = self.wcs.wcs.ctype[0].capitalize()

        if self.shape is None:
            unit = str(unit).replace(' ', '')
            self._logger.info('Spectral extent: min: %0.2f %s step: %0.3f %s' %
                             (start, unit, step,  unit))
        else:
            end = self.get_stop(unit=unit)
            unit = str(unit).replace(' ', '')
            self._logger.info('Spectral extent: min %0.2f %s max: %0.2f %s step: %0.3f %s' %
                             (start, unit, end, unit, step, unit))

    def __getitem__(self, item):

        if item is None:
            return self

        elif isinstance(item, int):
            if item >= 0:
                val = self.pix2wav(pix=item)
            else:
                if self.shape is None:
                    raise ValueError("Can't return an index without a shape")
                else:
                    val = self.pix2wav(pix=self.shape + item)
            return Position(crpix=1., crval=val, cdelt=0., unit=self.unit,
                            ctype=self.wcs.wcs.ctype[0], shape=1)

        elif isinstance(item, slice):
            if item.start is None:
                start = 0
            elif item.start >= 0:
                start = item.start
            else:
                if self.shape is None:
                    raise ValueError("Can't return an index without a shape")
                else:
                    start = self.shape + item.start
            if item.stop is None:
                if self.shape is None:
                    raise ValueError("Can't return an index without a shape")
                else:
                    stop = self.shape
            elif item.stop >= 0:
                stop = item.stop
            else:
                if self.shape is None:
                    raise ValueError("Can't return an index without a shape")
                else:
                    stop = self.shape + item.stop
            newval = self.pix2wav(pix=np.arange(start, stop, item.step))
            dimens = newval.shape[0]

            if dimens < 2:
                raise ValueError("Velocity/Wavelength with dimension < 2")
            cdelt = newval[1] - newval[0]
            return VelWave(crpix=1., crval=newval[0], cdelt=cdelt,
                            unit=self.unit, ctype=self.wcs.wcs.ctype[0],
                            shape=dimens)
        else:
            raise ValueError("Can't do it!")

    def wav2pix(self, val, nearest=False):
        """
        Return a pixel value if given an offset value in
        arcseconds
        """
        x = np.atleast_1d(val)

        # tell world2pix to make 0-relative array coordinates
        pix = self.wcs.wcs_world2pix(x, 0)[0]

        if nearest:
            pix = (pix + 0.5).astype(int)
            np.maximum(pix, 0, out=pix)
            if self.shape is not None:
                np.minimum(pix, self.shape-1, out=pix)
        return pix[0] if np.isscalar(val) else pix

    def pix2wav(self, pix=None, unit=None):

        if pix is None:
            pixarr = np.arange(self.shape, dtype=float)
        else:
            pixarr = np.atleast_1d(pix)

        res = self.wcs.wcs_pix2world(pixarr, 0)[0]

        if unit is not None:
            res = (res * self.unit).to(unit).value

        return res[0] if np.isscalar(pix) else res

    def get_start(self, unit=None):
        return self.pix2wav(0)

    def get_stop(self, unit=None):

        if self.shape is None:
            raise IOError("Need a dimension!")
        else:
            return self.pix2wav(self.shape - 1)

    def get_step(self, unit=None):
        '''
        get the step
        '''
        if self.wcs.wcs.has_cd():
            step = self.wcs.wcs.cd[0]
        else:
            cdelt = self.wcs.wcs.get_cdelt()[0]
            pc = self.wcs.wcs.get_pc()[0][0]
            step = cdelt * pc

        if unit is not None:
            step = (step * self.unit).to(unit).value
        return step

    def set_step(self, x, unit=None):
        """
        If you want to change the step, do it here;
        useful if you've read in an array and now have minimal header data
        """

        if unit is not None:
            step = (x * unit).to(self.unit).value
        else:
            step = x

        # TODO: probably won't have any CD info for 1D files
        # so maybe just ignore this?
        if self.wcs.wcs.has_cd():
            self.wcs.wcs.cd[0][0] = step
        else:
            pc = self.wcs.wcs.get_pc()[0][0]
            self.wcs.wcs.cdelt[0] = step / pc

    def get_range(self, unit=None):
        '''
        Simply return the upper and lower bounds of the array
        '''
        try:
            return self.pix2wav([0, self.shape-1], unit)
        except AttributeError:
            print("no dimension provided for spectral array!")
