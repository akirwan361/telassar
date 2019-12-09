TELASSAR : Two dimEnsionaL spectrAl analysiS for muSe And xshooteR
-----

*"Have the gods of the nations delivered those whom my fathers have destroyed, Gozan and Haran and Rezeph, and the people of Eden who were in Telassar?" - Isaiah 37:12*

A clunky, hideous package for doing what I want and upsetting every real programmer in our galactic quadrant. Brute force and ignorance are the tools of my trade. Nothing much beyond visualisation for (pseudo)long-slit spectra, with new functionalities to be added when I figure out how!

Basic Bits
-----
Can generate WCS information for a 2D slice either from a FITS file directly, or from a custom data array + header. No promises, though. If the header is from a MUSE cube, it tries to sort out what's needed. If it's from, e.g. XShooter, then it makes an attempt. 

Usage
-----
```
$ from telassar import PVSlice
$ my_file = PVSlice(filename = 'some_file.fits')
``` 
or
```
$ my_file = PVSlice(data = my_data, header = my_header)
```

TODO
-----
- Add capability to extract spatial/spectral profiles from the data
- Add [LMFit-py](https://lmfit.github.io/lmfit-py/) modeling options for fitting profiles
- Learn how to do everything properly.
