# `def/data/`

The following is the brief explanation of the code in this directory. 
If you would like to know more about something, please feel free to read 
the sources.
If you find yourself really lost, you can try to contact the authors 
via [artonson at yandex ru], or [open an issue](https://github.com/artonson/def/issues/new).

This folder contains the main implementation blocks of the DEF approach. 

 * `annotation.py` routines for computing distance-to-feature and 
direction-to-feature signals. 
 * `camera_pose_manager.py` rountines for camera positioning for making 3D scans. 
 * `data_smells.py` functions for detecting data imperfections. 
 * `imaging.py` functions for raytracing (*warning! this code has bugs!
avoid using it!*).
 * `mesh_nbhoods.py` routines for randomly extracting parts of meshes.
 * `noisers.py` functions for modelling simulated noise.
 * `patch_cropping.py` methods for extracting point patches from whole point clouds.
 * `point_samplers.py` methods for sampling points over the mesh surface.
