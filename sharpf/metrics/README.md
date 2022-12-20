# `def/metrics/`

The following is the brief explanation of the code in this directory. 
If you would like to know more about something, please feel free to read 
the sources.
If you find yourself really lost, you can try to contact the authors 
via [artonson at yandex ru], or [open an issue](https://github.com/artonson/def/issues/new).

This folder contains the main implementation blocks of the DEF approach. 

 * `numpy_metrics.py` a set of functions for statistical estimation of quality *(these are now
obsolete as Recall/FPR/RMSE have been adopted in the paper, but are left as they give more stats).*
 * `torch_metrics/` a set of metrics implemented in separate files (`mfpr.py` implements FPR,
`mrecall.py` implements Recall, `rmse.py` implements RMSE).
