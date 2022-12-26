# Generating real-world training datasets

## Provided files *(DEF-Scan)*

We provide the following two sets of files:
 * The intermediate files used within the processing pipeline, including: 
   * the original STL files from the ABC used for 3D printing,
   * the original OBJ and YML files from the ABC dataset used
   for 3D feature annotation,
   * the scanned `.x` files,
   * converted `.hdf5` and `.ply` files,
   * `.mlp` files containing alignment information
     obtained in MeshLab,
   * image-based renderings of each model in PNG format,
   * 3D snapshots of each model in HTML format,
   * Additional diagnostic information e.g. alignment
     histograms, 3D alignment snapshots.
 * All intermediate files are available in separate zips:
   * [SharpF_200_01_21.tar.gz (586 Mb)](https://www.dropbox.com/s/6ffru67tq6ldy5p/SharpF_200_01_21.tar.gz?dl=0)
   * [SharpF_200_22_48.tar.gz (565 Mb)](https://www.dropbox.com/s/polm0nlvc6ucqbw/SharpF_200_22_48.tar.gz?dl=0)
   * [SharpF_200_49_57.tar.gz (155 Mb)](https://www.dropbox.com/s/w110sjgs9zcu7z0/SharpF_200_49_57.tar.gz?dl=0)
   * [SharpF_200_58_61.tar.gz (40 Mb)](https://www.dropbox.com/s/nio0vymnoezfwqg/SharpF_200_58_61.tar.gz?dl=0)
   * [SharpF_200_62_67.tar.gz (336 Mb)](https://www.dropbox.com/s/kd3nw1yiggv16u2/SharpF_200_62_67.tar.gz?dl=0)
   * [SharpF_200_68_96.tar.gz (952 Mb)](https://www.dropbox.com/s/eromr6i9h4d3cih/SharpF_200_68_96.tar.gz?dl=0)
   * [SharpF_200_97_105.tar.gz (327 Mb)](https://www.dropbox.com/s/v0yn7y07swfarrd/SharpF_200_97_105.tar.gz?dl=0)
   * [new.tar.gz (573 Mb)](https://www.dropbox.com/s/3pnvi3tgtk2ta9c/new.tar.gz?dl=0)
 * The final aligned, annotated training, testing, and evaluation datasets **(DEF-Scan)**
   * In image-based format: [images_align4mm_fullmesh_whole.tar.gz (753 Mb)](https://www.dropbox.com/s/5k2swrpb0vhqv15/images_align4mm_fullmesh_whole.tar.gz?dl=0)
      * 981 training, 479 validation, 468 testing instances (depth images)
   * In point-based format: [points_align4mm_partmesh_whole.tar.gz (6.2 Gb)](https://www.dropbox.com/s/ej7qzmh2153birb/points_align4mm_partmesh_whole.tar.gz?dl=0)
      * 15574 training, 4119 validation, 9770 testing instances (point patches) 

## Obtaining the datasets

**Warning:** we strongly recommend using the final aligned,
annotated shapes as the final training/evaluation dataset. 
We found that producing this real-world dataset is quite
cumbersome and may result in failing to reproduce the results
of our work. 

The list below outlines the construction sequence for the dataset
of real-world scans that we followed during our work.

 1. Select shapes for printing. The original STL models
using for 3D printing are available within the zipped data. 
 2. Manufacture the shape using a 3D printer. The original 
manufactured plastic models are stored at Skoltech. 
A photo of these models you can find in the paper.
 3. Scan the shape using the structured-light scanning device. 
For each shape, we performed 24 scans, positioning the shape
first in a normal orientation, then changing orientation once
by 90 degrees for a 360 degree scan. 
The raw results of this scanning for all STL models are 
available within the zipped data. 
 4. We convert DirectX `.x` files using an utility script 
`convert_x_to_hdf5.py` in the following fashion: 
```bash
python3 convert_x_to_hdf5.py -i file.x -o output_dir/ --hdf5 --verbose 
```
resulting in a single `.hdf5` output file in `RangeVisionIO` 
schema. 
We additionally export `.ply` files for semi-automatic 
alignment in MeshLab by running:
```bash
python3 convert_x_to_hdf5.py -i file.x -o output_dir/ --ply --verbose 
```
All `.hdf5` and `.ply` conversion results are parts 
of the raw scanned data are available within the zipped data. 
 5. We semi-automatically examine the raw scans to identify
flawed scans or prints, and align the PLY scans to their respective STL 
shapes. Some fail to align tightly and are located in `bad_scan/`
subdirectory in respective archives. Shapes with printing flaws
are located in `bad_print/` subfolder. Shapes that were correctly 
aligned are located in `aligned/` subfolder.  
We additionally 
 6. We prepare the real scans by exporting them to a format 
similar to our real-world data (`ViewIO` schema) 
```bash
python3 prepare_real_scans.py -i input-dir -o output-dir --verbose --debug 
```
 7. We annotate the scans into image-based format using 
```bash
python3 prepare_real_images_dataset.py \
  -i input_dir/ -o output.hdf5 \ 
  --verbose --debug \
  --max_point_mesh_distance 4.0 \  # 4 mm
  --max_distance_to_feature 10.0 \  # 10 mm
  --full_mesh  # use all features of the mesh in annotation
```
 8. We annotate the scans into point-based format using 
```bash
python3 prepare_real_points_dataset.py \
  -i input_dir/ -o output.hdf5 \ 
  --verbose --debug \
  --max_point_mesh_distance 4.0 \  # 4 mm
  --max_distance_to_feature 10.0   # 10 mm
```
