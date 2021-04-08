TELASSAR : Two dimEnsionaL spectrAl analysiS for muSe And xshooteR
==================================================================

A clunky, hideous package for doing what I want and upsetting every real programmer in our galactic quadrant. Brute force and ignorance are the tools of my trade. Nothing much beyond visualisation for (pseudo)long-slit spectra, with new functionalities to be added when I figure out how!

Basic Bits
-----
Can generate WCS information for a 2D slice either from a FITS file directly, or from a custom data array + header. No promises, though. At the very least it makes an attempt!

Usage
-----
```
from telassar import PVSlice
my_file = PVSlice(filename='some_file.fits')
``` 
or
```
my_file = PVSlice(data=my_data, header=my_header)
``` 

A 2D slice can also be reduced to a 1D spatial or spectral profile given a wavelength range and offset range:
```
new_file = my_file.spatial_profile(arc=[0, 30], wave=[6725, 6730], spat_unit=True, spec_unit=True)
```

For quick data viewing, a contour plot is easily generated:
```
my_file.plot_contours()
```

Some further customization can be done with the plots; check out the notebook for some of those examples.

=======
TODO
-----
- Add [LMFit-py](https://lmfit.github.io/lmfit-py/) modeling options for fitting profiles
- Expand spatial/spectral profile class functionality (ie plotting, fitting, and integration)
- Learn how to do everything properly.
