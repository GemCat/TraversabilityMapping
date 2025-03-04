cmake_minimum_required(VERSION 3.4)
set(CMAKE_CXX_STANDARD 20)

project(traversability_mapping)

# Find required ROS packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  pcl_ros
  sensor_msgs
  tf
  tf_conversions
)

find_package(PCL 1.2 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)
find_package(yaml-cpp REQUIRED)
#find_package(pybind11 REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  /home/rsl/cnpy
  ${catkin_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIR}
)

link_directories(${PCL_LIBRARY_DIRS} /home/rsl/cnpy/build)
add_definitions(${PCL_DEFINITIONS})

# Find Open3D
set(Open3D_DIR "/home/rsl/Documents/Open3D_install/lib/cmake/Open3D")
find_package(Open3D REQUIRED)
include_directories(${Open3D_INCLUDE_DIRS})
link_directories(${Open3D_LIBRARY_DIRS})

# Specify python3 path
# set(PYTHON_INCLUDE_DIRS "/usr/include/python3.8")
# set(PYTHON_LIBRARIES "/usr/lib/x86_64-linux-gnu/libpython3.8.so")
# find_package(PythonLibs 3 REQUIRED)


catkin_package(
  LIBRARIES pointcloud_fusion
  CATKIN_DEPENDS roscpp pcl_ros sensor_msgs tf tf_conversions
  DEPENDS PCL OpenCV
)

# Accumulating semantic point clouds
add_executable(accumulate_pointcloud accumulate_pointcloud.cpp)
target_link_libraries(accumulate_pointcloud ${PCL_LIBRARIES} ${OpenCV_LIBS} cnpy z fmt::fmt ${catkin_LIBRARIES} ${YAML_CPP_LIBRARIES})
#set_target_properties(pointcloud_fusion_exe PROPERTIES OUTPUT_NAME pointcloud_fusion PREFIX "")

# New ROS Node
add_executable(point_cloud_processor_node data_extraction/point_cloud_processor_node.cpp)
target_link_libraries(point_cloud_processor_node ${catkin_LIBRARIES})

# Geometry cost calculation
add_executable(geometric_processing geometric_processing.cpp)
target_link_libraries(geometric_processing ${Open3D_LIBRARIES} ${PCL_LIBRARIES} OpenMP::OpenMP_CXX ${catkin_LIBRARIES} ${YAML_CPP_LIBRARIES})

# Fusion of geo amd sem costs
add_executable(sem_geo_fusion sem_geo_fusion.cpp)
target_link_libraries(sem_geo_fusion yaml-cpp ${PCL_LIBRARIES} ${OpenCV_LIBS} ${catkin_LIBRARIES})

# Evaluate geo+sem fusion method
add_executable(evaluate_fusion evaluation/evaluate_fusion.cpp)
target_link_libraries(evaluate_fusion ${PCL_LIBRARIES} ${catkin_LIBRARIES} ${YAML_CPP_LIBRARIES}) 

# Create final map representation
add_executable(map_representation map_representation.cpp)
target_include_directories(map_representation PRIVATE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(map_representation ${PCL_LIBRARIES} ${catkin_LIBRARIES} ${YAML_CPP_LIBRARIES} cnpy) 