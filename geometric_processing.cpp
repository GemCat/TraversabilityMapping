#define PCL_NO_PRECOMPILE
#include <iostream>
#include <yaml-cpp/yaml.h>
#include <open3d/Open3D.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/impl/extract_indices.hpp>
#include <omp.h>
#include "pointXYZCustom.hpp"


// Function to create a color map based on intensity values
Eigen::Vector3d create_color_map(float intensity, float min_intensity, float max_intensity)
{
  float normalized_intensity = (intensity - min_intensity) / (max_intensity - min_intensity);

  // Define the two colors for the color map (blue and red)
  Eigen::Vector3d color1(0.0, 0.0, 1.0); // Blue
  Eigen::Vector3d color2(1.0, 1.0, 0.0); // Red

  // Linear interpolation between the two colors based on the normalized intensity
  Eigen::Vector3d color = color1 * (1.0 - normalized_intensity) + color2 * normalized_intensity;
  
  return color;
}

void extract_clusters(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud,
                      pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud_filtered,
                      double cluster_tolerance,
                      int min_cluster_size)
{
    // Create a KdTree object for searching
    pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>);
    tree->setInputCloud(cloud);
    cloud_filtered->clear();
    cloud_filtered->points.reserve(cloud->points.size());

    // Perform Euclidean Cluster Extraction
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
    ec.setClusterTolerance(cluster_tolerance); // Tolerance for distance between points (e.g., 0.1)
    ec.setMinClusterSize(min_cluster_size);    // Minimum number of points in a cluster (e.g., 100)
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);

    // Iterate through the clusters
    for (const auto &indices : cluster_indices)
    {
        // Filter out small clusters
        if (indices.indices.size() >= min_cluster_size)
        {
            // Add the points of the remaining clusters to cloud_filtered
            for (const auto &index : indices.indices)
            {
                cloud_filtered->points.push_back(cloud->points[index]);
            }
        }
    }
    cloud_filtered->width = cloud_filtered->points.size();
    cloud_filtered->height = 1;
    cloud_filtered->is_dense = true;
}

// Function to convert a PCL point cloud to an Open3D point cloud and intensity values
std::pair<std::shared_ptr<open3d::geometry::PointCloud>, std::vector<float>> pcl_to_open3d(const pcl::PointCloud<PointXYZCustom>::ConstPtr &pcl_cloud)
{
  auto open3d_cloud = std::make_shared<open3d::geometry::PointCloud>();
  std::vector<float> intensities;

  for (const auto &point : pcl_cloud->points)
  {
    open3d_cloud->points_.emplace_back(point.x, point.y, point.z);
    intensities.push_back(point.intensity);
  }

  return std::make_pair(open3d_cloud, intensities);
}

int main(int argc, char** argv)
{
  // Check the number of arguments
  // if (argc != 7)
  // {
  //   std::cerr << "Usage: " << argv[0] << " intensity_thred num_threads radius Mean MulThresh data_folder " << std::endl;
  //   return -1;
  // }
  YAML::Node config = YAML::LoadFile("/home/rsl/catkin_ws/src/traversability_mapping/config.yaml");
  char path_separator = '/';

  // Read the intensity threshold from the command line
  float intensity_thred = 0.15;

  // Read the number of threads from the command line
  int num_threads = 16;
  omp_set_num_threads(num_threads);

  // Read the search radius from the command line
  float radius = 2.50;
  int min_cluster_size = 100;


  std::string cloud_dir = config["geometric"]["cloud_dir"].as<std::string>();

  // Load the point cloud from file
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::io::loadPCDFile<pcl::PointXYZ>(cloud_dir, *cloud_xyz);

  std::cout << "Loaded " << cloud_xyz->size() << " points" << std::endl;

  // Apply the Statistical Outlier Removal filter
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor (true);
  sor.setInputCloud(cloud_xyz);
  sor.setMeanK(min_cluster_size); // Adjust this value depending on the density of your point cloud
  sor.setStddevMulThresh(0.1); // Adjust this value to control the threshold for filtering
  sor.filter(*cloud_filtered);
  std::vector<int> indices_rem = *sor.getRemovedIndices();

  // Convert the point cloud from pcl::PointXYZ to PointXYZCustom
  pcl::PointCloud<PointXYZCustom>::Ptr og_cloud(new pcl::PointCloud<PointXYZCustom>);
  pcl::PointCloud<PointXYZCustom>::Ptr free_space_cloud(new pcl::PointCloud<PointXYZCustom>);
  pcl::PointCloud<PointXYZCustom>::Ptr obstacle_cloud(new pcl::PointCloud<PointXYZCustom>);

  pcl::io::loadPCDFile<PointXYZCustom>(cloud_dir, *og_cloud);
  pcl::PointCloud<PointXYZCustom>::Ptr cloud(new pcl::PointCloud<PointXYZCustom>);

  // Create the ExtractIndices filter
  pcl::ExtractIndices<PointXYZCustom> extract;
  extract.setInputCloud(og_cloud);

  pcl::PointIndices::Ptr indices_to_remove(new pcl::PointIndices());
  indices_to_remove->indices = indices_rem;

  extract.setIndices(indices_to_remove);
  extract.setNegative(true);  // This means we want to REMOVE the points at the specified indices, not extract them
  extract.filter(*cloud);

  // cloud->resize(cloud_filtered->size());
  // #pragma omp parallel for
  // for (int i = 0; i < cloud_filtered->size(); i++)
  // {
  //   cloud->points[i].x = cloud_filtered->points[i].x;
  //   cloud->points[i].y = cloud_filtered->points[i].y;
  //   cloud->points[i].z = cloud_filtered->points[i].z;
  //   cloud->points[i].intensity = 0.0;
  // }

  std::cout << "Filtered " << cloud->size() << " points" << std::endl;

  // Create a KdTree to search for neighbors
  pcl::KdTreeFLANN<PointXYZCustom> kdtree;
  kdtree.setInputCloud(cloud);

  // Process each point in the cloud
  #pragma omp parallel for
  for (int i = 0; i < cloud->size(); i++)
  {
    PointXYZCustom& point = cloud->points[i];

    // Search for neighbors around the current point
    std::vector<int> indices;
    std::vector<float> distances;
    int neighbors = kdtree.radiusSearch(point, radius, indices, distances);
    if (neighbors < min_cluster_size)
    {
      point.intensity = -1.0f;
      continue;
    }
    // Calculate the relative height change
    float relative_height = 0.0;
    for (int j = 0; j < neighbors; j++)
    {
        relative_height += std::abs(cloud->points[indices[j]].z - point.z);
    }
    relative_height /= neighbors;

    // Store the relative height change in the intensity value
    point.intensity = relative_height;
  }

  // Process each point in the cloud and separate points into free space and obstacle clouds
  free_space_cloud->reserve(cloud->size());
  obstacle_cloud->reserve(cloud->size());

  for (int i = 0; i < cloud->size(); i++)
  {
    if (cloud->points[i].intensity < 0.0f)
    {
      continue;
    }
    PointXYZCustom& point = cloud->points[i];
    if (point.intensity <= intensity_thred)
    {
      free_space_cloud->points.push_back(point);
    }
    else
    {
      obstacle_cloud->points.push_back(point);
    }
  }

  // Resize the free_space_cloud and obstacle_cloud to their actual sizes
  free_space_cloud->width = free_space_cloud->points.size();
  free_space_cloud->height = 1;
  free_space_cloud->is_dense = true;

  obstacle_cloud->width = obstacle_cloud->points.size();
  obstacle_cloud->height = 1;
  obstacle_cloud->is_dense = true;

  *cloud = *free_space_cloud + *obstacle_cloud;
  std::cout << "Final cloud size: " << cloud->size() << std::endl;

  //  Set parameters for clustering
  double cluster_tolerance = 0.1;

  // Extract clusters
  // pcl::PointCloud<pcl::PointXYZI>::Ptr free_space_cloud_filtered(new pcl::PointCloud<pcl::PointXYZI>);
  // extract_clusters(free_space_cloud, free_space_cloud_filtered, cluster_tolerance, min_cluster_size*10);

  // Save the processed point cloud to file
  std::string save_dir = config["geometric"]["save_dir"].as<std::string>();
  pcl::io::savePCDFileASCII(save_dir, *cloud);
  //pcl::io::savePCDFileASCII(data_folder + path_separator + "free_space.pcd", *free_space_cloud_filtered);
  // pcl::io::savePCDFileASCII(data_folder + path_separator + "free_space.pcd", *free_space_cloud);
  // pcl::io::savePCDFileASCII(data_folder + path_separator + "obstacle.pcd", *obstacle_cloud);

  // Convert the PCL point cloud to an Open3D point cloud and intensity values
  std::shared_ptr<open3d::geometry::PointCloud> open3d_cloud;
  std::vector<float> intensities;
  std::tie(open3d_cloud, intensities) = pcl_to_open3d(cloud);

  // Find the minimum and maximum intensity values in the 'intensities' vector
  float min_intensity = std::numeric_limits<float>::max();
  float max_intensity = std::numeric_limits<float>::min();

  for (const float intensity : intensities) {
      min_intensity = std::min(min_intensity, intensity);
      max_intensity = std::max(max_intensity, intensity);
  }
  std::cout << "Min intensity: " << min_intensity << std::endl;
  std::cout << "Max intensity: " << max_intensity << std::endl;

  open3d_cloud->colors_.resize(intensities.size());
  for (size_t i = 0; i < intensities.size(); ++i)
  {
    Eigen::Vector3d color = create_color_map(intensities[i], min_intensity, max_intensity);

    // Convert Eigen::Vector3d to open3d::geometry::PointCloud::Color and assign to colors_[i]
    open3d_cloud->colors_[i] = color;
  }

  // Visualize the point cloud using Open3D
  open3d::visualization::DrawGeometries({open3d_cloud}, "Terrain Cloud", 1600, 900);

  return 0;
}
