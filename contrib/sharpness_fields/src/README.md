# sharpness-field
Code for the paper [Sharpness Fields in Point Clouds Using Deep Learning](https://www.sciencedirect.com/science/article/pii/S009784931830181X)

## Usage

````bash
python compute_sharpness.py blade_mls.xyz sharpness_values.txt -m sharpness_model.pt -r 5.0
````

Tested using: Linux (Manjaro), Python 3.7, PyTorch 1.2, numpy 1.16, CUDA 10.1, CUDNN 7.6

The input point cloud should be smoothed using MLS before passing to the script, even if it is not noisy.  
The recommended smoothing kernel radius is 3 * (median separation of a point and its nearest neighbor).  
This is required because all sharpness field compuations are done on the proxy surface, so the neural network is only trained for points on MLS surfaces.

## Cite
```
@article{raina2019sharpness,
  title={Sharpness fields in point clouds using deep learning},
  author={Raina, Prashant and Mudur, Sudhir and Popa, Tiberiu},
  journal={Computers \& Graphics},
  volume={78},
  pages={37--53},
  year={2019},
  publisher={Elsevier}
}
``` 
