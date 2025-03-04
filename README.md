# Traversability Mapping

This is the code for creating traversability maps from fusing semantic and geometric information! The project originally uses image and point cloud rosbag data to calculate traversability costs conditioned on semantics and geometry. 

## Running the pipeline
The full pipeline from input images & point clouds to output traversability map can be run using `run_pipeline.sh`. Before executing the pipeline, ensure that all necessary data are correctly named and saved and that the correct filepaths for accessing and saving data/outputs are set in `accumulate_pointcloud.cpp` and `config.yaml`. The required data inputs are point clouds, rgb images, and any necessary transformations between camera, lidar, and world frames. If the detection model under the `semantic_detection` library from this project is to be used for generating semantics, then model weights must be loaded and the filepaths within the config files should be changed to match your local machine. 

Assuming the traversability_mapping project is built under the `/src` directory of a catin workspace, running the following commands after building will run the pipeline:
```
cd /path/to/catkin_ws/src/traversability_mapping
chmod +x run_pipeline.sh
./run_pipeline.sh
```
If the executables are built somewhere else, make sure to change the filepaths in the script. Since the semantic data only needs to be generated once, lines 5-7 in the script dealing with `demo_inference.py` can be commented out if the semantic bitmaps already exist. 

## The Pipeline
- [Semantic Processing](#semantic-processing)
- [Point Cloud Accumulation](#point-cloud-accumulation)
- [Geometric Processing](#geometric-processing)
- [Semantic Geometric Fusion](#semantic-geometric-fusion)
- [Traversability Mapping](#traversability-mapping)

### Semantic Processing
The `semantic_detection` directory contains the code used for training and testing different feature extraction backbones. `demo_inference.py` is used to generate the required semantic bitmaps from the output masks of the chosen network model.

### Point Cloud Accumulation
After the Hesai lidar point clouds are extracted from the rosbags and semantic masks from the rgb images are converted into bitmaps, the original lidar point clouds are converted into custom point clouds with the point type in `pointXYZCustom.hpp`. Then point clouds from every frame and consecutive rosbags are accumulated to form a semantically labeled global map in `accumulate_pointcloud.cpp`. This file also contains options for point cloud visualization and for creating point clouds for evaluation, all which can be set in `config.yaml`

### Geometric Processing
After obtaining the full point cloud map of the environment, the relative height change of each point, used as the geometric cost, is calculated and stored in the point cloud in `geometric_processing.cpp`.

### Semantic Geometric Fusion
Only a small subset of points in the point cloud map have a semantic label due to the camera's limited frame of view. In `sem_geo_fusion.cpp`,  unlabeled semantic points are assigned to classes through a kmeans partition of the point cloud using the relative height change and relative height from the ground of each point. For each k-means cluster, all unlabeled points are assigned with the most frequently occuring label in its respective cluster. The output would ideally be a fully annotated point cloud that incorporates both semantic and geometric information. 

### Traversability Mapping
In `map_representation.cpp`, class traversability scores are calculated by averaging the geometry costs from all the points within each class. The traversability scores are then stored in a 2D map representation of the point cloud in the x-y plane, and the final map is saved as a numpy array. Options to normalize the stored traversability costs and to calculate the costs from semantics and geometry or geometry only can be set in `config.yaml`.

## Additional Files
- [Data Extraction](#data-extraction)
- [Evaluation](#evaluation)

### Data Extraction
The `data_extraction` directory contains files used for matching and extracting the necessary RGB image, point cloud, and transformation files from the rosbags. The data must be extracted and saved before running the pipeline and the correct file paths corresponding to the data should be set in `demo_inference.py` and `accumulate_pointcloud.cpp`.

### Evaluation
The `evaluation` directory contains files to quantitatively evaluate the pipeline's performance in predicting missing semantic labels and achieving cost consistency between semantics and geometry. In addition to assessing traversability costs calcuated from semantic and geometric fusion vs. just geometry, `visualize_map.py` can visualize a saved traversability map with or without the projected robot trajectory. 