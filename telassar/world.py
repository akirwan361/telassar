from astropy.io import fits
import astropy.units as u
from astropy.wcs import WCS as pywcs
from astropy.wcs import WCSSUB_SPECTRAL
import numpy as np

import logging

def wcs_from_cube_header(hdr):

    '''
    Install coordinates for the data. This will have one spatial axis and
    one spectral axis
    '''
    # Don't convert units
    if 'CUNIT3' in hdr:
        unit = u.Unit(hdr.pop('CUNIT3'))

    try:
        naxis = hdr['NAXIS']
    except KeyError:
        naxis = hdr['WCSAXES']


    # generate a WCS object
    mywcs = pywcs(hdr, fix = False)

    # Figure out the spatial limits
    # get the full extent of the data
    nx, ny = mywcs.pixel_shape[:2]
    #print(nx, ny)

    # ideally, crpix should be our centers
    # TODO: add option to pass a center
    xc, yc = mywcs.wcs.crpix[:2]

    # make new WCS object with Spatial and Wavelength axes
    # `wcs.sub()` sets naxis1 to None, so make sure it has
    # the same pixel shape as the parent data
    new_wcs = mywcs.sub([0, WCSSUB_SPECTRAL])
    new_wcs.pixel_shape = mywcs.pixel_shape[1:]

    # get cdelt from CD
    cd = mywcs.wcs.cd
    dy = np.sqrt(cd[0, 1]**2 + cd[1, 1]**2)

    # get dy in arcseconds
    dy = np.round((dy * u.deg).to(u.arcsec).value, 1)
    cdelt1 = dy
    cdelt2 = cd[2, 2]

    # if the WCS header is taken from a cube or a 2D spatial image,
    # get rid of the CD parameter so astropy won't ignore a cdelt value
    if new_wcs.wcs.has_cd():
        del new_wcs.wcs.cd

    # set the WCS info
    new_wcs.wcs.crpix[0] = yc
    new_wcs.wcs.cdelt[0] = cdelt1
    new_wcs.wcs.cdelt[1] = cdelt2
    new_wcs.wcs.crval[0] = 0. # this is arcsec, so we want the center at 0
    new_wcs.wcs.ctype[0] = 'OFFSET'
    new_wcs.wcs.cunit[0] = u.Unit('arcsec')#.to_string('fits')
    #new_wcs.wcs.cunit[1] = u.Unit(unit)#.to_string('fits')
    #new_wcs.wcs.set()
    try:
        new_wcs.wcs.pc[1,0] = new_wcs.wcs.pc[0,1] = 0
    except AttributeError:
        pass

    return new_wcs

def wcs_from_pv_header(hdr):
    '''
    This is mostly for dealing with the Ellerbroek data
    '''
    # Create a WCS object
    mywcs = pywcs(hdr, fix = False)

    # Make sure the WCS info is in [spatial, spectral]
    # order
    if mywcs.wcs.cdelt[0] == 0.2:
        pass
    elif mywcs.wcs.cdelt[1] == 0.2:
        mywcs = mywcs.swapaxes(0,1)
        #mywcs.pixel_shape = mywcs.pixel_shape[::-1]

    # Is this velocity or wavelength?
    if mywcs.wcs.crval[1] < 1:
        ctype2 = 'VELO'
        cunit2 = u.Unit('km/s').to_string('fits')
    else:
        ctype2 = 'AWAV'
        cunit2 = u.Unit('nm').to_string('fits')

    ctype1 = 'OFFSET'
    cunit1 = u.Unit('arcsec').to_string('fits')

    mywcs.wcs.crpix = np.array([1., 1.])
    mywcs.wcs.ctype = [ctype1, ctype2]
    #mywcs.wcs.cunit = [cunit1, cunit2]

    return mywcs

class World:

    def __init__(self, hdr = None, crpix = [1., 1.], crval = [1.,1.], cdelt = [1., 1.],
                 unit1 = u.arcsec, unit2 = u.dimensionless_unscaled,
                 ctype = ['LINEAR', 'AWAV'], shape = None):

        #self._logger = logging.getLogger(__name__)
        #print(__name__)

        unit1 = u.Unit(unit1).to_string('fits')
        unit2 = u.Unit(unit2)

        self.shape = shape

        #if mode is not None:
        #    print(self.unit)
        if (hdr is not None):
            h = hdr.copy()
            #print(h)
            n = h['NAXIS']
            self.shape = h['NAXIS%d' % n]
            if n == 3:
                self.wcs = wcs_from_cube_header(h)
            elif n == 2:
                self.wcs = wcs_from_pv_header(h)

            if self.wcs.wcs.cunit[0] != '':
                self.spatial_unit = self.wcs.wcs.cunit[0]
            elif self.wcs.wcs.ctype[0] == 'OFFSET':
                self.spatial_unit = u.Unit('arcsec')
            #print(self.wcs.wcs.cunit[1])
            if self.wcs.wcs.cunit[1] == 'm':
                self.spectral_unit = u.Unit('angstrom')
            elif (self.wcs.wcs.cunit[1] == u.Unit('km/s')) or (self.wcs.wcs.ctype[1] == 'VELO'):
                self.spectral_unit = u.Unit('km/s')

        elif hdr is None:
            #if data is not None:
            self.spatial_unit = u.Unit(unit1)
            self.spectral_unit = u.Unit(unit2)
            self.wcs = pywcs(naxis = 2)
            #self.shape =

            # set the reference pixel
            self.wcs.wcs.crval = np.array([crval[0], crval[1]])
            self.wcs.wcs.ctype = np.array([ctype[0], ctype[1]])
            self.wcs.wcs.cdelt = np.array([cdelt[0], cdelt[1]])
            self.wcs.wcs.crpix = np.array([crpix[0], crpix[1]])

        #self.wcs.wcs.set()
        self.shape = self.wcs.pixel_shape

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

    def info(self):
        try:
            #print('we are workings')
            #self._logger.info('We are working')
            spec_unit = self.spectral_unit
            spat_unit = self.spatial_unit
            dy = self.get_spatial_step(unit = u.Unit(spat_unit))
            dx = self.get_spectral_step(unit = u.Unit(spec_unit))
            sizex = dx * self.naxis2
            sizey = dy * self.naxis1
            xc = (self.naxis2 - 1) / 2.
            yc = (self.naxis1 - 1) / 2.
            pixoff = self.pix2offset(yc, unit = u.Unit(spat_unit))
            pixwav = self.pix2wav(xc, unit = u.Unit(spec_unit))
            '''self._logger.info(
                'center:(%s, %s) '
                'size:(%0.3f", %0.3f %s) '
                'step:(%0.3f", %0.3f %s) ',
                xc, yc, sizex, sizey, spec_unit.to_string(),
                dx, dx, spec_unit.to_string())'''
            print(
                'center:(%s, %s) \n'
                'size:(%0.3f", %0.3f %s) \n'
                'step:(%0.3f", %0.3f %s) ' % (yc, xc,
                sizey, sizex, spec_unit.to_string(),
                dy, dx, spec_unit.to_string()))
        except Exception:
            print('Exception')
            self._logger.info("something happened I can't fix yet")

    def offset2pix(self, val, nearest = False):
        """
        Return a pixel value if given an offset value in
        arcseconds
        """
        x = np.atleast_1d(val)

        # tell world2pix to make 0-relative array coordinates
        pix = self.wcs.wcs_world2pix(x, 0, 0)[0]

        if nearest:
            pix = (pix + 0.5).astype(int)
            np.maximum(pix, 0, out = pix)
            if self.shape is not None:
                np.minimum(pix, self.shape[0]-1, out = pix)
        return pix[0] if np.isscalar(val) else pix

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

    def pix2offset(self, pix = None, unit = None):
        """
        Reverse of above: get spatial value of pixel
        """

        if pix is None:
            pixarr = np.arange(self.shape[0], dtype = float)
        else:
            pixarr = np.atleast_1d(pix)

        res = self.wcs.wcs_pix2world(pixarr, 0, 0)[0]

        if unit is not None:
            res = (res * self.spatial_unit).to(unit).value

        return res[0] if np.isscalar(pix) else res

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

    def get_spatial_step(self, unit = None):
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
            step = (step * self.spatial_unit).to(unit).value
        return step

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
        """
        If you want to change the step, do it here;
        useful if you've read in an array and now have minimal header data
        """

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

    def get_spatial_start(self, unit = None):
        """
        Get the starting pixel value
        """

        return self.pix2offset(0)

    def get_spectral_start(self, unit = None):
        return self.pix2wav(0)

    def get_spatial_end(self, unit = None):

        if self.shape is None:
            raise IOError("Need a dimension!")
        else:
            return self.pix2offset(self.shape[0] - 1)

    def get_spectral_end(self, unit = None):

        if self.shape is None:
            raise IOError("Need a dimension!")
        else:
            return self.pix2wav(self.shape[1]-1)

    def __getitem__(self, item):

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
                if imax < 0:
                    imax = self.naxis1 + imax
                if imax > self.naxis1:
                    imax = self.naxis1

            if item[0].step is not None and item[0].step != 1:
                raise ValueError('Can only handle integer steps')
        else:
            imin = int(item[0])
            imax = int(item[0] + 1)

        if isinstance(item[1], slice):
            if item[1].start is None:
                jmin = 0
            else:
                jmin = int(item[1].start)
                if jmin < 0:
                    jmin = self.naxis2 + jmin
                if jmin > self.naxis2:
                    jmin = self.naxis2
            if item[1].stop is None:
                jmax = self.naxis2
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
        crpix = (self.wcs.wcs.crpix[0] - imin, self.wcs.wcs.crpix[1] - jmin)

        # copy the object and get the new ref pix and all that
        res = self.copy()
        res.wcs.wcs.crpix = np.array(crpix)
        res.naxis1 = int(imax - imin)
        res.naxis2 = int(jmax - jmin)
        res.wcs.wcs.set()

        return res
