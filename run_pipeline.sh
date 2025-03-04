#!/bin/bash

# This is the script to run the full pipeline from images + point clouds -> traversability map. 

# cd semantic_detection
# python3 demo_inference.py
# cd ..

cd ../../devel/lib/traversability_mapping
./accumulate_pointcloud
./geometric_processing
./sem_geo_fusion
./map_representation



