from astropy.io import fits
import astropy.units as u
from astropy.wcs import WCS as pywcs
from astropy.wcs import WCSSUB_SPECTRAL
import numpy as np

import logging

def _get_cdelt_from_cd(mywcs):

    # if there's a cd, get it
    cd = mywcs.wcs.cd
    dy = np.sqrt(cd[0,1]**2 + cd[1,1]**2)
    # convert to arcseconds
    cdelt = np.round((dy * u.deg).to(u.arcsec).value, 1)

    return cdelt


def wcs_from_header(hdr):

    hdr = hdr.copy()

    cunit = None
    if 'CUNIT3' in hdr:
        cunit = u.Unit(hdr.pop('CUNIT3'))
    else:
        cunit = u.Unit('angstrom')

    try:
        n = hdr['NAXIS']
    except KeyError:
        n = hdr['WCSAXES']

    # generate WCS object
    mywcs = pywcs(hdr, fix=False)
    old_shape = mywcs.pixel_shape

    if mywcs.wcs.has_cd():
        cdelt = _get_cdelt_from_cd(mywcs)
        #cdelt2 = mywcs.wcs.cd[2,2]
    elif mywcs.wcs.cdelt[0] == 0.2:
        cdelt = 0.2
        #cdelt2 = mywcs.wcs.cdelt[1]
    elif mywcs.wcs.cdelt[1] == 0.2:
        cdelt = 0.2
        #cdelt2 = mywcs.wcs.cdelt[0]

    crval = None
    #if cunit is None:
    # is the header from a cube?
    if n==3:
        nx, ny = old_shape[:2]
        xc, yc = mywcs.wcs.crpix[:2]
        new_wcs = mywcs.sub([0, WCSSUB_SPECTRAL])
        new_wcs.pixel_shape = old_shape[1:]
        crval = 0. # this is arcsec, so we want center at 0

    # if this is a header from a pvslice check axes
    if n==2:
        if mywcs.wcs.cdelt[0] == 0.2:
            new_wcs = mywcs.copy()
        elif mywcs.wcs.cdelt[1] == 0.2:
            new_wcs = mywcs.swapaxes(0,1)
            # TODO: figure out why this is different
            # on laptop and desktop! temp workaround:
            if new_wcs.pixel_shape == old_shape:
                new_wcs.pixel_shape = new_wcs.pixel_shape[::-1]
        yc = hdr['CRPIX1']
        #crval = hdr['CRVAL1']

    if new_wcs.wcs.has_cd():
        del new_wcs.wcs.cd

    # set the basic wcs info
    new_wcs.wcs.crpix[0] = yc
    new_wcs.wcs.ctype[0] = 'OFFSET'
    new_wcs.wcs.cunit[0] = u.Unit('arcsec')

    # set the more specific info
    if cdelt is not None:
        new_wcs.wcs.cdelt[0] = cdelt

    if crval is not None:
        new_wcs.wcs.crval[0] = crval

    # Is this velocity or wavelength?
    if new_wcs.wcs.crval[1] < 0.:
        new_wcs.wcs.ctype[1] = 'VELO'
        new_wcs.wcs.cunit[1] = u.Unit('m/s')
        #cunit2 = u.Unit('km/s').to_string('fits')
    else:
        new_wcs.wcs.ctype[1] = 'AWAV'
        new_wcs.wcs.cunit[1] = cunit


    return new_wcs

class Position:

    def __init__(self, hdr = None, crpix = 1., crval = 1., cdelt = 0.2,
                 unit = u.arcsec, ctype = 'LINEAR', shape = None):

        self._logger = logging.getLogger(__name__)

        unit = u.Unit(unit).to_string('fits')

        self.shape = shape

        if (hdr is not None):
            h = hdr.copy()

            # the spatial axis should be axis=1
            axis = 1
            self.wcs = wcs_from_header(h).sub([axis])

            if self.wcs.wcs.cunit[0] != '':
                self.unit = self.wcs.wcs.cunit[0]
            elif self.wcs.wcs.ctype[0] == 'OFFSET':
                self.unit = u.Unit('arcsec')

        elif hdr is None:
            #if data is not None:
            self.unit = u.Unit(unit)
            self.wcs = pywcs(naxis = 1)
            #self.shape =

            # set the reference pixel
            self.wcs.wcs.crval[0] = crval
            self.wcs.wcs.ctype[0] = ctype
            self.wcs.wcs.cdelt[0] = cdelt
            self.wcs.wcs.crpix[0] = crpix
            self.wcs.pixel_shape = (shape,)
            #self.wcs.wcs.cunit = [unit1, unit2]

        #self.wcs.wcs.set()
        self.shape = self.wcs.pixel_shape[0] if shape is None else shape

    def copy(self):
        """
        Copy the Position object
        """
        out = Position(shape=self.shape, unit=self.unit)
        out.wcs = self.wcs.deepcopy()
        return out

    def __repr__(self):
        return repr(self.wcs)

    def info(self, unit = None):

        unit = unit or self.unit
        start = self.get_start(unit=unit)
        step = self.get_step(unit=unit)
        type = self.wcs.wcs.ctype[0].capitalize()

        if self.shape is None:
            self.__logger.info(f'Spatial {type}: min: {start:0.2f} step: \
                               {step:0.3f}"')
        else:
            end = self.get_stop(unit=unit)
            self.__logger.info(f'Spatial {type}: min {start:0.2f} max: {stop:0.2f} \
                               step: {step:0.3f}"')

    def __getitem__(self, item):

        if item is None:
            return self

        elif isinstance(item, int):
            if item >=0:
                arc = self.pix2offset(pix=item)
            else:
                if self.shape is None:
                    raise ValueError("Can't return an index without a shape")
                else:
                    arc = self.pix2offset(pix = self.shape + item)
            return Position(crpix=1., crval = arc, cdelt=0., unit=self.unit,
                            ctype = self.wcs.wcs.ctype[0], shape=1)

        elif isinstance(item, slice):
            if item.start is None:
                start = 0
            elif item.start >=0:
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
            elif item.stop >=0:
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
            return Position(crpix = 1., crval = newarc[0], cdelt = cdelt,
                            unit = self.unit, ctype = self.wcs.wcs.ctype[0],
                            shape = dimens)
        else:
            raise ValueError("Can't do it!")

    def offset2pix(self, val, nearest = False):
        """
        Return a pixel value if given an offset value in
        arcseconds
        """
        x = np.atleast_1d(val)

        # tell world2pix to make 0-relative array coordinates
        pix = self.wcs.wcs_world2pix(x, 0)[0]

        if nearest:
            pix = (pix + 0.5).astype(int)
            np.maximum(pix, 0, out = pix)
            if self.shape is not None:
                np.minimum(pix, self.shape-1, out = pix)
        return pix[0] if np.isscalar(val) else pix

    def pix2offset(self, pix = None, unit = None):
        """
        Reverse of above: get spatial value of pixel
        """

        if pix is None:
            pixarr = np.arange(self.shape, dtype = float)
        else:
            pixarr = np.atleast_1d(pix)

        res = self.wcs.wcs_pix2world(pixarr, 0)[0]

        if unit is not None:
            res = (res * self.unit).to(unit).value

        return res[0] if np.isscalar(pix) else res


    def get_step(self, unit = None):
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

    def set_step(self, x, unit = None):
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

    def get_start(self, unit = None):
        """
        Get the starting pixel value
        """

        return self.pix2offset(0)


    def get_stop(self, unit = None):

        if self.shape is None:
            raise IOError("Need a dimension!")
        else:
            return self.pix2offset(self.shape - 1)


class VelWave:

    def __init__(self, hdr = None, crpix = 1., crval = 1., cdelt = 1.,
                 unit = u.angstrom, ctype = 'LINEAR', shape = None):

        self._logger = logging.getLogger(__name__)

        unit = u.Unit(unit).to_string('fits')

        self.shape = shape

        if (hdr is not None):
            h = hdr.copy()

            # the spectral axis should be 2
            axis = 2
            self.wcs = wcs_from_header(h).sub([axis])

            #if self.wcs.wcs.cunit[0] != '':
            #    self.unit = self.wcs.wcs.cunit[0]
            if self.wcs.wcs.ctype[0] == 'VELO':
                self.unit = u.Unit('km/s')
            elif self.wcs.wcs.ctype[0] == 'AWAV' or 'WAVE':
                self.unit = unit

        elif hdr is None:
            #if data is not None:
            self.unit = u.Unit(unit)
            self.wcs = pywcs(naxis = 1)
            #self.shape =

            # set the reference pixel
            self.wcs.wcs.crval[0] = crval
            self.wcs.wcs.ctype[0] = ctype
            self.wcs.wcs.cdelt[0] = cdelt
            self.wcs.wcs.crpix[0] = crpix
            # this is weird but go with it
            self.wcs.pixel_shape = (shape,)

        #self.wcs.wcs.set()
        self.shape = self.wcs.pixel_shape[0] if shape is None else shape

    def __repr__(self):
        return(repr(self.wcs))

    def info(self, unit = None):

        unit = unit or self.unit
        start = self.get_start(unit=unit)
        step = self.get_step(unit=unit)
        type = self.wcs.wcs.ctype[0].capitalize()

        if self.shape is None:
            self.__logger.info(f'Spatial {type}: min: {start:0.2f} step: \
                               {step:0.3f}"')
        else:
            end = self.get_stop(unit=unit)
            self.__logger.info(f'Spatial {type}: min {start:0.2f} max: {stop:0.2f} \
                               step: {step:0.3f}"')

    def __getitem__(self, item):

        if item is None:
            return self

        elif isinstance(item, int):
            if item >=0:
                val = self.pix2wav(pix=item)
            else:
                if self.shape is None:
                    raise ValueError("Can't return an index without a shape")
                else:
                    val = self.pix2wav(pix = self.shape + item)
            return Position(crpix=1., crval = val, cdelt=0., unit=self.unit,
                            ctype = self.wcs.wcs.ctype[0], shape=1)

        elif isinstance(item, slice):
            if item.start is None:
                start = 0
            elif item.start >=0:
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
            elif item.stop >=0:
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
            return VelWave(crpix = 1., crval = newval[0], cdelt = cdelt,
                            unit = self.unit, ctype = self.wcs.wcs.ctype[0],
                            shape = dimens)
        else:
            raise ValueError("Can't do it!")

    def wav2pix(self, val, nearest = False):
        """
        Return a pixel value if given an offset value in
        arcseconds
        """
        x = np.atleast_1d(val)

        # tell world2pix to make 0-relative array coordinates
        pix = self.wcs.wcs_world2pix(x, 0)[0]

        if nearest:
            pix = (pix + 0.5).astype(int)
            np.maximum(pix, 0, out = pix)
            if self.shape is not None:
                np.minimum(pix, self.shape-1, out = pix)
        return pix[0] if np.isscalar(val) else pix

    def pix2wav(self, pix=None, unit = None):

        if pix is None:
            pixarr = np.arange(self.shape, dtype = float)
        else:
            pixarr = np.atleast_1d(pix)

        res = self.wcs.wcs_pix2world(pixarr, 0)[0]

        if unit is not None:
            res = (res * self.unit).to(unit).value

        return res[0] if np.isscalar(pix) else res

"""

    @property
    def naxis(self):
        return self.wcs.naxis

    @property
    def naxis1(self):

        if self.wcs.pixel_shape is not None:
            return self.wcs.pixel_shape[0]
        else:
            return 0

    @property
    def naxis2(self):

        if self.wcs.pixel_shape is not None:
            return self.wcs.pixel_shape[1]
        else:
            return 0

    def __repr__(self):
        return repr(self.wcs)

    def __getitem__(self, item):
        '''
        Get a bit of the data, mm mmm...
        '''
        if isinstance(item[0], slice):
            if item[0].start is None:
                imin = 0
            else:
                imin = int(item[0].start)
                if imin < 0:
                    imin = self.naxis1 + imin
                if imin > self.naxis1:
                    imin = self.naxis1

            if item[0].stop is None:
                imax = self.naxis1
            else:
                imax = int(item[0].stop)
                if imax < 0:
                    imax = self.naxis1 + imax
                if imax > self.naxis1:
                    imax = self.naxis1

            if item[0].step is not None and item[0].step != 1:
                raise ValueError('Can only handle integer steps')
        else:
            imin = int(item[0])
            imax = int(item[0] + 1)
        #print(imin, imax)
        if isinstance(item[1], slice):
            if item[1].start is None:
                jmin = 0
            else:
                jmin = int(item[1].start)
                #print(jmin)
                if jmin < 0:
                    jmin = self.naxis2 + jmin
                if jmin > self.naxis2:
                    jmin = self.naxis2
            if item[1].stop is None:
                jmax = self.naxis2
            else:
                jmax = int(item[1].stop + 1)
                if jmax < 0:
                    jmax = self.naxis2 + jmax
                if jmax > self.naxis2:
                    jmax = self.naxis2

            if item[1].step is not None and item[1].step != 1:
                raise ValueError('Can only handle integer steps')
        else:
            jmin = int(item[1])
            jmax = int(item[1] + 1)

        # get the new array  indices
        new_crpix = [1., 1.]#(self.wcs.wcs.crpix[0] - imin, self.wcs.wcs.crpix[1] - jmin)
        new_spec = self.pix2wav([jmin, jmax])
        new_spat = self.pix2offset([imin, imax])
        new_crval = np.array([new_spat[0], new_spec[0]])
        new_dim = (imax - imin, jmax - jmin)
        ctype1 = str(self.wcs.wcs.ctype[0])
        ctype2 = str(self.wcs.wcs.ctype[1])

        return World(crpix = new_crpix, cdelt = self.wcs.wcs.cdelt, crval = new_crval,
                     unit1 = u.Unit(self.spatial_unit), unit2 = u.Unit(self.spectral_unit),
                     ctype = [ctype1, ctype2], shape = new_dim)

    def info(self):
        try:
            #print('we are workings')
            #self._logger.info('We are working')
            spec_unit = str(self.spectral_unit).replace(' ', '')
            spat_unit = str(self.spatial_unit).replace(' ', '')
            dy = self.get_spatial_step(unit = u.Unit(self.spatial_unit))
            dx = self.get_spectral_step(unit = u.Unit(self.spectral_unit))
            extentx = np.array([self.get_spectral_start(),
                                self.get_spectral_end()])
            extenty = np.array([self.get_spatial_start(),
                                self.get_spatial_end()])

            self._logger.info(
                'spatial extent:(%s", %s") step:(%0.3f") ',
                extenty[0], extenty[1], dy)
            self._logger.info(
                'spectral extent:(%0.3f, %0.3f) %s step:(%0.3f %s) ',
                extentx[0], extentx[1], spec_unit, dx, spec_unit)

        except Exception:
            print('Exception')
            self._logger.info("something happened I can't fix yet")


    def wav2pix(self, val, nearest = False):
        '''
        Return a pixel value given a wavelength or velocity
        value
        '''
        x = np.atleast_1d(val)

        # get 0-relative array coords
        pix = self.wcs.wcs_world2pix(0, x, 0)[1]

        if nearest:
            pix = (pix + 0.5).astype(int)
            np.maximum(pix, 0, out = pix)
            if self.shape is not None:
                np.minimum(pix, self.shape[1]-1, out=pix)
        return pix[0] if np.isscalar(val) else pix

    def pix2wav(self, pix = None, unit = None):
        '''
        Get spectral value of pixel
        '''
        if pix is None:
            pixarr = np.arange(self.shape[1], dtype = float)
        else:
            pixarr = np.atleast_1d(pix)

        res = self.wcs.wcs_pix2world(0, pixarr, 0)[1]

        if unit is not None:
            res = (res * self.spectral_unit).to(unit).value

        return res[0] if np.isscalar(pix) else res


    def get_spectral_step(self, unit = None):
        if self.wcs.wcs.has_cd():
            step = self.wcs.wcs.cd[0]
        else:
            cdelt = self.wcs.wcs.get_cdelt()[1]
            pc = self.wcs.wcs.get_pc()[1][1]
            step = cdelt * pc

        if unit is not None:
            step = (step * self.spectral_unit).to(unit).value

        return step


    def set_spatial_step(self, x, unit = None):
        '''
        If you want to change the step, do it here;
        useful if you've read in an array and now have minimal header data
        '''

        if unit is not None:
            step = (x * unit).to(self.spatial_unit).value
        else:
            step = x

        # TODO: probably won't have any CD info for 1D files
        # so maybe just ignore this?
        if self.wcs.wcs.has_cd():
            self.wcs.wcs.cd[0][0] = step
        else:
            pc = self.wcs.wcs.get_pc()[0][0]
            self.wcs.wcs.cdelt[0] = step / pc

        self.wcs.wcs.set()

    def set_spectral_step(self, x, unit = None):
        '''
        Same as above but for spectral coords
        '''
        if unit is not None:
            step = (x * unit).to(self.spectral_unit).value
        else:
            step = x

        # TODO: probably won't have any CD info for 1D files
        # so maybe just ignore this?
        if self.wcs.wcs.has_cd():
            self.wcs.wcs.cd[1][1] = step
        else:
            pc = self.wcs.wcs.get_pc()[1][1]
            self.wcs.wcs.cdelt[1] = step / pc

        self.wcs.wcs.set()

    def get_spectral_start(self, unit = None):
        return self.pix2wav(0)

    def get_spectral_end(self, unit = None):

        if self.shape is None:
            raise IOError("Need a dimension!")
        else:
            return self.pix2wav(self.shape[1]-1)
"""
