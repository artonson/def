.. image:: https://travis-ci.org/emulbreh/bridson.svg?branch=master
    :target: https://travis-ci.org/emulbreh/bridson


bridson
=======

Two Dimensional Poisson Disc Sampling using `Robert Bridson's algorithm <https://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph07-poissondisk.pdf>`_

.. image:: https://cdn.rawgit.com/emulbreh/bridson/master/sample.svg


Usage
-----

.. code-block:: python

    >>> from bridson import poisson_disc_samples
    >>> poisson_disc_samples(width=100, height=100, r=10)
    [(18.368737154138397, 0.6537095218417459),
     (31.037620677039477, 0.11127035812202124),
     (42.36176894248073, 7.038053455899708),
     ...
     (39.42578238367568, 99.18831048188478),
     (73.33459914827051, 99.50928386778354),
     (98.09958160385061, 99.1575330533914)]
