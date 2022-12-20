# `scripts/data_scripts/`

The following is the brief explanation of the code in this directory. 
If you would like to know more about something, please feel free to read 
the sources.
If you find yourself really lost, you can try to contact the authors 
via [artonson at yandex ru], or [open an issue](https://github.com/artonson/def/issues/new).

In this directory, we present an application of our approach 
to extraction of a shape wireframe representation using 
distance-to-feature fields reconstructed by our method.

This part of our method requires installing a few custom 
packages providing dependencies and a runnable environment 
for a curve reconstruction method.
To install them, simply run
```bash
python3 -m pip install papermill
```

Description of individual scripts: 
 * `metric_calculation_pc2w.ipynb`, `metric_calculation_pienet.ipynb` 
notebooks for computing metrics for shape wireframes reconstructed using 
PC2WF and PIE-Net approaches.
 * `parametric_curve_move_method.ipynb` main runnable ipython notebook.
 * `run_curve.sh` shell script for running curve extraction for multiple shapes.
 * `topological_graph.py` functions for graph construction and optimization.
 * `utils.py` helper functions. 
