# Obtaining predictions made by DEF

To enable easy comparisons with our data, we release 
the files containing predictions obtained using our 
approach on our test set of synthetic and real-world 
object instances. 
We release both per-point sharpness estimation results
and the parametric vectorization results. 
All results are accompanied by the ground-truth versions. 


## Sharp feature estimation (DEF-Sim)

The folder [def_predictions/def_sim_complete_feature_estimation/](https://www.dropbox.com/scl/fo/8ad1y2r1938urwipti89r/h?dl=0&rlkey=msvik3wdwl0gukm4jsgpoxgzf)
contains all the files.

| **Link**                                  | **Modality** | **Resolution** | **Sampling distance** | **Noise level**  | **Num. views** | **Num. shapes** |
|-------------------------------------------|--------------|----------------|-----------------------|------------------|----------------|-----------------|
| `def_sim-images-high-0-18views.tar.gz`    | depth images | 1024 x 1024    | 0.02                  | 0                | 18             | 85              | 
| `def_sim-images-high-0-128views.tar.gz`   | depth images | 1024 x 1024    | 0.02                  | 0                | 128            | 95              |
| `def_sim-images-high-0.005.tar.gz`        | depth images | 1024 x 1024    | 0.02                  | 0.005 (SNR=4:1)  | 18             | 87              |
| `def_sim-images-high-0.02.tar.gz`         | depth images | 1024 x 1024    | 0.02                  | 0.02 (SNR=1:1)   | 18             | 87              |
| `def_sim-images-high-0.08.tar.gz`         | depth images | 1024 x 1024    | 0.02                  | 0.08 (SNR=1:4)   | 18             | 87              |
| `def_sim-images-med.tar.gz`               | depth images | 1024 x 1024    | 0.05 (2.5x)           | 0                | 18             | 100             |
| `def_sim-images-low.tar.gz`               | depth images | 1024 x 1024    | 0.125 (2.5^2x)        | 0                | 18             | 104             |
| `def_sim-points-high-0.tar.gz`            | point clouds | various        | 0.02                  | 0                | point patches  | 84              |
| `def_sim-points-high-0.005.tar.gz`        | point clouds | various        | 0.02                  | 0.005 (SNR=4:1)  | point patches  | 83              |
| `def_sim-points-high-0.02.tar.gz`         | point clouds | various        | 0.02                  | 0.02 (SNR=1:1)   | point patches  | 84              |
| `def_sim-points-high-0.08.tar.gz`         | point clouds | various        | 0.02                  | 0.08 (SNR=1:4)   | point patches  | 83              |
| `def_sim-points-med-0.tar.gz`             | point clouds | various        | 0.05 (2.5x)           | 0                | point patches  | 103             |
| `def_sim-points-low-0.tar.gz`             | point clouds | various        | 0.125 (2.5^2x)        | 0                | point patches  | 75              |


## Sharp feature estimation (DEF-Scan)

As the results of our investigation of fusion on complex, 
complete real-world 3D models, we release 
 * Ground-truth annotated shapes,
 * Per-view predictions obtained using DEF networks,
 * Final fused predictions obtained `min` aggregation,
 * HTML 3D snapshots of our predictions overlayed with the ground-truth,
 * PNG renders of the reconstructed field and its respective ground-truth,
 * Per-shape metric values.

The folder [def_predictions/def_sim_complete_feature_estimation/](https://www.dropbox.com/scl/fo/c9um6hwqflqtsm4jturbk/h?dl=0&rlkey=imukpyryogacwl05m0f5nbbmk)
contains all the files.

| **Link**                 | **Modality** | **Resolution** | **Sampling distance** | **Noise level** | **Num. views** | **Num. shapes** |
|--------------------------|--------------|----------------|-----------------------|-----------------|----------------|-----------------|
| `def_scan-images.tar.gz` | depth images | 2048 x 1536    | 0.5 mm                | <unknown>       | 12             | 39              | 


## Parametric vectorization (DEF-Sim)

We provide:
 * Ground-truth vectorizations exported from ABC data: [def_sim-images-points-high-0-vectorization-gt.zip (5Mb)](https://www.dropbox.com/s/cnc2g28fmbtrsu9/def_sim-images-points-high-0-vectorization-gt.zip?dl=0)
 * Predicted vectorization results by our method [def_sim-images-high-0-vectorizations.tar.gz (400Kb)](https://www.dropbox.com/s/50nr8l8d41qp1pm/def_sim-images-high-0-vectorizations.tar.gz?dl=0)
 * Predicted vectorization results with lots of variants and visuals [def_sim-images-high-0-vectorizations_with_visuals.tar.gz (2Gb)](https://www.dropbox.com/s/0c2in9bzyeqdug3/def_sim-images-high-0-vectorizations_with_visuals.tar.gz?dl=0)
