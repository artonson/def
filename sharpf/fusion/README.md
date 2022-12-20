# `def/fusion/`

The following is the brief explanation of the code in this directory. 
If you would like to know more about something, please feel free to read 
the sources.
If you find yourself really lost, you can try to contact the authors 
via [artonson at yandex ru], or [open an issue](https://github.com/artonson/def/issues/new).

This folder contains the main implementation blocks of the DEF approach. 

 * `combiners.py` routines for merging multiple predictions to a single 
resulting value.
 * `io.py` I/O specifications for reading files emerging during fusion.
 * `smoothers.py` functions for postprocessing predictions via smoothing.
 * `images/interpolate.py` functions for per-pixel bilinear interpolation.
 * `images/interpolators.py` classes for running the interpolation between N views.
 * `images/pairwise.py` functions for computing interpolations between a pair of views. 
