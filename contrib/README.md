# `contrib/`

The following is the brief explanation of the code in this directory. 
If you would like to know more about something, please feel free to read 
the sources.
If you find yourself really lost, you can try to contact the authors 
via [artonson at yandex ru], or [open an issue](https://github.com/artonson/def/issues/new).

This folder contains code used "as is" or wrapped into appropriate
docker images. 
 * `CGAL`: dockerfiles and implementations of the [Edge-aware point set resampling](https://dl.acm.org/doi/abs/10.1145/2421636.2421645) 
and [Voronoi-based curvature and feature estimation from point clouds](https://ieeexplore.ieee.org/abstract/document/5669298) using CGAL library.
 * `ec_net`: dockerfiles and implementation of [EC-Net: an Edge-aware Point set Consolidation Network](http://openaccess.thecvf.com/content_ECCV_2018/html/Lequan_Yu_EC-Net_an_Edge-aware_ECCV_2018_paper.html).
 * `hdf5_utils`: helper scripts for working with files in HDF5 format.
 * `pointweb`: operations used in the work [PointWeb: Enhancing Local Neighborhood Features for Point Cloud Processing](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhao_PointWeb_Enhancing_Local_Neighborhood_Features_for_Point_Cloud_Processing_CVPR_2019_paper.html).
 * `pythonaabb`: C++ implementation and Python wrapper of a axis-aligned 
bounding box container class (*warning! this code had bugs last time we updated 
our repository! for queries to sharp edges, use `point_mesh_squared_distance` 
from `libIGL`*).
 * `sharpness_fields`: dockerfiles and implementations of the paper 
[Sharpness fields in point clouds using deep learning](https://www.sciencedirect.com/science/article/pii/S009784931830181X).
