TELASSAR : Two dimEnsionaL spectrAl analysiS for muSe And xshooteR
<<<<<<< HEAD
==================================================================
=======
-----
>>>>>>> a6f29410a6b111d3a84027877c63102f4e4075e6

A clunky, hideous package for doing what I want and upsetting every real programmer in our galactic quadrant. Brute force and ignorance are the tools of my trade. Nothing much beyond visualisation for (pseudo)long-slit spectra, with new functionalities to be added when I figure out how!

Basic Bits
-----
<<<<<<< HEAD
Can generate WCS information for a 2D slice either from a FITS file directly, or from a custom data array + header. No promises, though. At the very least it makes an attempt!
=======
Can generate WCS information for a 2D slice either from a FITS file directly, or from a custom data array + header. No promises, though. If the header is from a MUSE cube, it tries to sort out what's needed. If it's from, e.g. XShooter, then it makes an attempt. 
>>>>>>> a6f29410a6b111d3a84027877c63102f4e4075e6

Usage
-----
```
<<<<<<< HEAD
from telassar import PVSlice
my_file = PVSlice(filename='some_file.fits')
``` 
or
```
my_file = PVSlice(data=my_data, header=my_header)
=======
$ from telassar import PVSlice
$ my_file = PVSlice(filename = 'some_file.fits')
``` 
or
```
$ my_file = PVSlice(data = my_data, header = my_header)
>>>>>>> a6f29410a6b111d3a84027877c63102f4e4075e6
```

A 2D slice can also be reduced to a 1D spatial or spectral profile given a wavelength range and offset range:
```
<<<<<<< HEAD
new_file = my_file.spatial_profile(arc=[0, 30], wave=[6725, 6730], spat_unit=True, spec_unit=True)
=======
$ new_file = my_file.spatial_profile(arc = [0, 30], wave = [6725, 6730], spat_unit = True, spec_unit = True)
>>>>>>> a6f29410a6b111d3a84027877c63102f4e4075e6
```

For quick data viewing, a contour plot is easily generated:
```
<<<<<<< HEAD
my_file.plot_contours()
```

Some further customization can be done with the plots; check out the notebook for some of those examples.

=======
$ my_file.plot_contours()
```

>>>>>>> a6f29410a6b111d3a84027877c63102f4e4075e6
TODO
-----
- Add [LMFit-py](https://lmfit.github.io/lmfit-py/) modeling options for fitting profiles
- Expand spatial/spectral profile class functionality (ie plotting, fitting, and integration)
- Learn how to do everything properly.
