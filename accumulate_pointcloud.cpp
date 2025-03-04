#include <iostream>
#include <yaml-cpp/yaml.h>
#include <vector>
#include <tuple>
#include <string>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <vector>
#include <optional>
#include <fmt/core.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/impl/point_cloud_geometry_handlers.hpp>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/kdtree/kdtree_flann.h>
#include <cnpy.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Dense>
#include <fstream>
#include "pointXYZCustom.hpp"

using namespace std;

// Structure to hold index and distance
struct IndexDistance {
    int index;
    float distance;
};

// Function to compute distance from the origin
float computeDistance(const pcl::PointXYZ point) {
    return std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
}

// Comparison function for sorting
bool compare(const IndexDistance& a, const IndexDistance& b) {
    return a.distance < b.distance;
}

// Project the 3D point to 2D image plane: 
// point cloud frame (ROS): +x forward, +y left, +z up
// image frame: +x right, +y down, +z forward
std::pair<int, int> project3DToPixel(const PointXYZCustom &point, float fx, float fy, float cx, float cy) {
    int u = round(-point.y * fx / point.x + cx);
    int v = round(-point.z * fy / point.x + cy);
    return std::make_pair(u, v);
}


// Change original xyz point clouds from rosbag into custom point clouds with additional fields for class scores and colors
pcl::PointCloud<PointXYZCustom>::Ptr createCustomCloud (pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz, int id, std::string dataset, std::string base, std::vector<float> K) {

    // Resize custom cloud to match original cloud
    // cloud_custom->resize(cloud_xyz->size());

    std::string bitmap_dir = fmt::format("/home/rsl/harveri_inference/{}/{}_bitmaps/image_{}_bitmap.npy", base, dataset, id);
    cnpy::NpyArray arr = cnpy::npy_load(bitmap_dir);
    double* data = arr.data<double>();

    std::string camLidar_transform_matx = fmt::format("/home/rsl/harveri_transforms/{}/cam_lidar/transform_{}.txt", dataset, id);

    Eigen::Matrix4f T_cam_lidar = Eigen::Matrix4f::Identity();
    std::ifstream infile(camLidar_transform_matx);
    if(!infile.is_open()) {
        std::cerr << "Error opening transformation matrix file!" << std::endl;
        return nullptr;
    }

    for(int i = 0; i < 4; ++i) {
        for(int j = 0; j < 4; ++j) {
            infile >> T_cam_lidar(i,j);
        }
    }
    infile.close();

    // Sort cloud points from closest to furthest
    std::vector<IndexDistance> indexDistanceVec(cloud_xyz->points.size());

    // Compute distances
    for (size_t i = 0; i < cloud_xyz->points.size(); ++i) {
        indexDistanceVec[i].index = i;
        indexDistanceVec[i].distance = computeDistance(cloud_xyz->points[i]);
    }

    // Sort based on distance
    std::sort(indexDistanceVec.begin(), indexDistanceVec.end(), compare);

    // Copy data from the original to the custom cloud
    pcl::PointCloud<PointXYZCustom>::Ptr cloud_custom(new pcl::PointCloud<PointXYZCustom>);
    std::vector<std::vector<int>> pixels(360, std::vector<int>(640, 0)); // tracks which pixels of image are matched with 3D point
   
    // Access points from closest to furthest 
    for (const auto& item : indexDistanceVec) {
        const pcl::PointXYZ& point = cloud_xyz->points[item.index];
        PointXYZCustom newPoint;

        Eigen::Vector4f point_in_lidar(point.x, point.y, point.z, 1.0);
        Eigen::Vector4f point_in_camera = T_cam_lidar * point_in_lidar;

        newPoint.x = point_in_camera(0); // set in camera frame
        newPoint.y = point_in_camera(1);
        newPoint.z = point_in_camera(2);

        if (newPoint.x < 0){ // Points with negative depth are not projected onto image 
            if (id == 0 && item.distance > 4.0){
                cloud_custom->points.push_back(newPoint);
            }
            continue;
        }

        auto [u, v] = project3DToPixel(newPoint, K[0], K[1], K[2], K[3]);

        if (u >= 0 && u < 640 && v >= 0 && v < 360) {
            int idx = (v * 640 + u) * 8;

            if (pixels[v][u] == 0){ // assign semantic label to current point if it is the first (aka closest) point to be matched to the pixel
                pixels[v][u] == 1;
                // Set class scores
                newPoint.background = data[idx];
                newPoint.smooth = data[idx + 1];
                newPoint.grass = data[idx + 2];
                newPoint.rough = data[idx + 3];
                newPoint.lowVeg = data[idx + 4];
                newPoint.highVeg = data[idx + 5];
                newPoint.sky = data[idx + 6];
                newPoint.obstacle = data[idx + 7];
            }
        }
       
        // Remove stationary points 
        if (item.distance <= 4.0) {
            continue;
        }
        cloud_custom->points.push_back(newPoint);
    }
    return cloud_custom;
}

// Create validation cloud that contains no semantic info
pcl::PointCloud<PointXYZCustom>::Ptr createValCloud (pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz, int id, std::string dataset) {
    std::string camLidar_transform_matx = fmt::format("/home/rsl/harveri_transforms/{}/cam_lidar/transform_{}.txt", dataset, id);

    Eigen::Matrix4f T_cam_lidar = Eigen::Matrix4f::Identity();
    std::ifstream infile(camLidar_transform_matx);
    if(!infile.is_open()) {
        std::cerr << "Error opening transformation matrix file!" << std::endl;
        return nullptr;
    }

    for(int i = 0; i < 4; ++i) {
        for(int j = 0; j < 4; ++j) {
            infile >> T_cam_lidar(i,j);
        }
    }
    infile.close();

    pcl::PointCloud<PointXYZCustom>::Ptr cloud_custom(new pcl::PointCloud<PointXYZCustom>);
    for (const auto& point : cloud_xyz->points){
        float dist = computeDistance(point);
        if (dist <= 4){
            continue;
        }

        PointXYZCustom newPoint;

        Eigen::Vector4f point_in_lidar(point.x, point.y, point.z, 1.0);
        Eigen::Vector4f point_in_camera = T_cam_lidar * point_in_lidar;

        newPoint.x = point_in_camera(0); // set in camera frame
        newPoint.y = point_in_camera(1);
        newPoint.z = point_in_camera(2);

        if (newPoint.x < 0){
            if (id == 0){
                cloud_custom->points.push_back(newPoint);
            }
            continue;
        }
        cloud_custom->points.push_back(newPoint);
    }
    return cloud_custom;
}

// Transforms point cloud into world(odom) frame 
pcl::PointCloud<PointXYZCustom>::Ptr transformPointCloud(const pcl::PointCloud<PointXYZCustom>::Ptr& cloud, const std::string& matrix_file) {
    // Read transformation matrix from file
    Eigen::Matrix4f transformation = Eigen::Matrix4f::Identity();
    std::ifstream infile(matrix_file);
    if(!infile.is_open()) {
        std::cerr << "Error opening transformation matrix file!" << std::endl;
        return nullptr;
    }

    for(int i = 0; i < 4; ++i) {
        for(int j = 0; j < 4; ++j) {
            infile >> transformation(i,j);
        }
    }
    infile.close();

    // Transform the point cloud
    pcl::PointCloud<PointXYZCustom>::Ptr transformed_cloud(new pcl::PointCloud<PointXYZCustom>);
    pcl::transformPointCloud(*cloud, *transformed_cloud, transformation);

    return transformed_cloud;
}


/* Custom Voxel downsampling functionality:
- Voxel key represents one voxel in the voxel map that contains a vector of points
- Voxel map is an unordered map that contains all voxels 
- Each voxel of size leafSize^3
- Compute centroid averages all fields of all points in a voxel */

struct VoxelKey {
    int x, y, z;
    bool operator==(const VoxelKey& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

namespace std {
    template <> struct hash<VoxelKey> {
        std::size_t operator()(const VoxelKey& k) const {
            return (std::hash<int>()(k.x) ^
                    (std::hash<int>()(k.y) << 1) ^ 
                    (std::hash<int>()(k.z) << 2));
        }
    };
}

PointXYZCustom computeCentroid(const std::vector<PointXYZCustom>& points) {
    PointXYZCustom centroid;
    memset(&centroid, 0, sizeof(PointXYZCustom));

    if (points.empty()) return centroid;

    for (const auto& p : points)
    {
        centroid.x += p.x;
        centroid.y += p.y;
        centroid.z += p.z;
        centroid.background += p.background;
        centroid.smooth += p.smooth;
        centroid.grass += p.grass;
        centroid.rough += p.rough;
        centroid.lowVeg += p.lowVeg;
        centroid.highVeg += p.highVeg;
        centroid.sky += p.sky;
        centroid.obstacle += p.obstacle;
        centroid.rgb += p.rgb;
    }

    float invN = 1.0f / points.size();
    centroid.x *= invN;
    centroid.y *= invN;
    centroid.z *= invN;
    centroid.background *= invN;
    centroid.smooth *= invN;
    centroid.grass *= invN;
    centroid.rough *= invN;
    centroid.lowVeg *= invN;
    centroid.highVeg *= invN;
    centroid.sky *= invN;
    centroid.obstacle *= invN;
    centroid.rgb *= invN;

    return centroid;
}

bool isCentroidZeroedOut(const PointXYZCustom& centroid) {
    return centroid.x == 0.0f && centroid.y == 0.0f && centroid.z == 0.0f &&
           centroid.background == 0.0f && centroid.smooth == 0.0f && 
           centroid.grass == 0.0f && centroid.rough == 0.0f && 
           centroid.lowVeg == 0.0f && centroid.highVeg == 0.0f && 
           centroid.sky == 0.0f && centroid.obstacle == 0.0f && 
           centroid.rgb == 0.0f ;
}

pcl::PointCloud<PointXYZCustom>::Ptr downsamplePointCloud(const pcl::PointCloud<PointXYZCustom>::Ptr& inputCloud, float leafSize) {
    std::unordered_map<VoxelKey, std::vector<PointXYZCustom>> voxelMap;

    for (const auto& point : inputCloud->points) {
        VoxelKey key{
            static_cast<int>(point.x / leafSize),
            static_cast<int>(point.y / leafSize),
            static_cast<int>(point.z / leafSize)
        };

        voxelMap[key].push_back(point);
    }

    pcl::PointCloud<PointXYZCustom>::Ptr outputCloud(new pcl::PointCloud<PointXYZCustom>);
    outputCloud->points.reserve(voxelMap.size());

    for (const auto& voxel : voxelMap) {
        PointXYZCustom centroid = computeCentroid(voxel.second);
        if (!isCentroidZeroedOut(centroid)) {
            outputCloud->points.push_back(centroid);
        }
    }
    outputCloud->width = outputCloud->points.size();
    outputCloud->height = 1;
    outputCloud->is_dense = true;

    return outputCloud;
}

int main() {

    YAML::Node config = YAML::LoadFile("/home/rsl/catkin_ws/src/traversability_mapping/config.yaml");
    std::vector<std::string> datasets = config["accumulate"]["datasets"].as<std::vector<std::string>>();
    std::vector<int> numClouds = config["accumulate"]["num_clouds"].as<std::vector<int>>();
    bool visOnly = config["accumulate"]["vis_only"].as<bool>();
    bool create_eval = config["accumulate"]["create_eval"].as<bool>();
    float distThreshold = 0.01;
    

    if (visOnly){

        pcl::PointCloud<PointXYZCustom>::Ptr mergedCloud(new pcl::PointCloud<PointXYZCustom>);
        std::string cloud_dir = config["accumulate"]["vis_cloud_dir"].as<std::string>();
        bool filter = config["accumulate"]["filter"].as<bool>();
        pcl::io::loadPCDFile(cloud_dir, *mergedCloud);

        // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>);
        // std::string cloud_dir = "/home/rsl/harveri_pc/trail/cloud_0.pcd";
        // pcl::io::loadPCDFile(cloud_dir, *cloud_xyz);
       
        // Convert to XYZRGB
        // pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_xyzrgb(new pcl::PointCloud<pcl::PointXYZRGB>);
        // cloud_xyzrgb->points.resize(cloud_xyz->points.size());

        // for (size_t i = 0; i < cloud_xyz->points.size(); i++) {
        //     cloud_xyzrgb->points[i].x = cloud_xyz->points[i].x;
        //     cloud_xyzrgb->points[i].y = cloud_xyz->points[i].y;
        //     cloud_xyzrgb->points[i].z = cloud_xyz->points[i].z;

        //     // Assign a color (for example, red)
        //     cloud_xyzrgb->points[i].r = 255;
        //     cloud_xyzrgb->points[i].g = 0;
        //     cloud_xyzrgb->points[i].b = 0;
        // }

        pcl::PointCloud<PointXYZCustom>::Ptr filteredCloud(new pcl::PointCloud<PointXYZCustom>);
        pcl::PointCloud<PointXYZCustom>::Ptr visCloud = mergedCloud;

        if(filter){
            for (const auto& point : mergedCloud->points){
                if (point.r != 201 || point.g != 187 || point.b != 202){
                    filteredCloud->points.push_back(point);
                }
            }
            visCloud = filteredCloud;
        }
       

        boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
        viewer->setBackgroundColor(255, 255, 255);

        pcl::visualization::PointCloudColorHandlerRGBField<PointXYZCustom> rgb(visCloud);
        viewer->addPointCloud<PointXYZCustom>(visCloud, rgb, "trailHesaiMergedCloud");
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "trailHesaiMergedCloud");

        // viewer->addPointCloud<pcl::PointXYZRGB>(cloud_xyzrgb, "singleCloud");
        // viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "singleCloud");
        
        viewer->addCoordinateSystem(1.0);
        viewer->initCameraParameters();

    
        while (!viewer->wasStopped()) {
            viewer->spinOnce(100);
        }

        return 0;
    }


    /* Colors for each class (BGR):
    0 background: (0, 0, 0) Black
    1 smooth trail: (91, 123, 166) Cafe au lait
    2 traversible grass: (97, 182, 123) Light Green
    3 rough trail: (0, 63, 123) Chocolate
    4 non_traversable_low_vegetation: (107, 134, 120) Cameoflage
    5 high vegetation: (62, 77, 27) Brunswick Green
    6 sky: (234, 204, 147) Cornflower
    7 obstacle: (161, 0, 244) Fuschia
    8 puddle: (64, 224, 208) Yellow */

    std::vector<std::tuple<int, int, int>> thing_colors = {
        {0, 0, 0},
        {91, 123, 166},
        {97, 182, 123},
        {0, 63, 123},
        {107, 134, 120},
        {62, 77, 27},
        {234, 204, 147},
        {161, 0, 244},
        {64, 224, 208}
    };

    // camera instrinsics {fx, fy, cx, cy}
    std::vector<float> K = {487.44329833984375, 487.44329833984375, 325.61627197265625, 189.09432983398438};

    // Cloud ptr to hold the accumulated point cloud
    pcl::PointCloud<PointXYZCustom>::Ptr mergedCloud(new pcl::PointCloud<PointXYZCustom>);

    std::cout<< "datasets size: "<<datasets.size()<<std::endl;
    std::string base = config["accumulate"]["base"].as<std::string>();

    for (int idx = 0; idx < datasets.size(); ++idx){
        std::string dataset = datasets[idx];
        int totalClouds = numClouds[idx];

        std::cout<<dataset<<std::endl;

        // For each dataset, loop through all point clouds, merge points with distance < threshold or add new points to the merged cloud
        for (int id = 0; id < totalClouds; ++id) {

            std::cout<<id<<std::endl;

            std::string cloud_dir = fmt::format("/home/rsl/harveri_pc/{}/cloud_{}.pcd", dataset, id);
            std::string transform_matx = fmt::format("/home/rsl/harveri_transforms/{}/odom_cam/transform_{}.txt", dataset, id);
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::io::loadPCDFile(cloud_dir, *cloud_xyz);

            pcl::PointCloud<PointXYZCustom>::Ptr new_cloud(new pcl::PointCloud<PointXYZCustom>);
            std::string val_dataset = config["accumulate"]["val_dataset"].as<std::string>();

            if (create_eval && dataset == val_dataset){
                new_cloud = createValCloud(cloud_xyz, id, dataset); // If generated cloud is being used for evaluation
            }
            else{
                new_cloud = createCustomCloud(cloud_xyz, id, dataset, base, K); 
            }
            new_cloud = transformPointCloud(new_cloud, transform_matx); 

            // KD-Tree for merged cloud for efficient point iteration 
            pcl::KdTreeFLANN<PointXYZCustom> kdtree;
            if(mergedCloud->points.size() > 0){
                kdtree.setInputCloud(mergedCloud);
            }

            for (const auto& newPoint : new_cloud->points) {

                if (std::isinf(newPoint.x) || std::isinf(newPoint.y) || std::isinf(newPoint.z) ||
                    std::isnan(newPoint.x) || std::isnan(newPoint.y) || std::isnan(newPoint.z)) {
                    continue;  
                }

                if (mergedCloud->points.empty()){
                    // No points in merged cloud
                    mergedCloud->points.push_back(newPoint);
                    kdtree.setInputCloud(mergedCloud);
                    continue;
                }

                std::vector<int> pointIdxNKNSearch(1);
                std::vector<float> pointNKNSquaredDistance(1);
                if (kdtree.nearestKSearch(newPoint, 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0) {
                    if (pointNKNSquaredDistance[0] < distThreshold) {
                        // Points are close enough to be considered overlapping

                        int idx = pointIdxNKNSearch[0];
                        PointXYZCustom &matchPoint = (*mergedCloud)[idx];
                        matchPoint.background = (matchPoint.background + newPoint.background)/2;
                        matchPoint.smooth = (matchPoint.smooth + newPoint.smooth)/2;
                        matchPoint.grass = (matchPoint.grass + newPoint.grass)/2;
                        matchPoint.rough = (matchPoint.rough + newPoint.rough)/2;
                        matchPoint.lowVeg = (matchPoint.lowVeg + newPoint.lowVeg)/2;
                        matchPoint.highVeg = (matchPoint.highVeg + newPoint.highVeg)/2;
                        matchPoint.sky = (matchPoint.sky + newPoint.sky)/2;
                        matchPoint.obstacle = (matchPoint.obstacle + newPoint.obstacle)/2;
                        
                    } else {
                        // No overlapping point found in merged cloud
                        mergedCloud->points.push_back(newPoint);
                    }
                } 
            }
        }
        // Downsample accumulated point cloud
        mergedCloud = downsamplePointCloud(mergedCloud, 0.1);
    }
    

    // Downsample accumulated point cloud
    //mergedCloud = downsamplePointCloud(mergedCloud, 0.1);

    // Colourize every point in the merged cloud based on max class score
    for (auto& point : mergedCloud->points) {

        std::vector<float> class_scores = {point.background, point.smooth, point.grass, point.rough, point.lowVeg, point.highVeg, point.sky, point.obstacle};

        // Get iterator to the maximum element in the vector
        auto max_it = std::max_element(class_scores.begin(), class_scores.end());

        if (*max_it != 0.0) {
            // point has semantic label(s)
            // Calculate the index of the class with the max score
            // std::cout << u << ", " << v << std::endl;
            // for (const auto &value : class_scores) {
            //     std::cout << value << " ";
            // }
            // cout<<"\n";
            int max_idx = std::distance(class_scores.begin(), max_it);

            // Assign color to point based on its max class score 
            point.r = std::get<2>(thing_colors[max_idx]);
            point.g = std::get<1>(thing_colors[max_idx]);
            point.b = std::get<0>(thing_colors[max_idx]);
            
        }
        
    }

    // Visualize the point cloud
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(255, 255, 255);

    pcl::visualization::PointCloudColorHandlerRGBField<PointXYZCustom> rgb(mergedCloud);
    viewer->addPointCloud<PointXYZCustom>(mergedCloud, rgb, "trailHesaiMergedCloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "trailHesaiMergedCloud");
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();

   
    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
    }

    // Save the custom point cloud if needed
    std::string save_dir = config["accumulate"]["save_cloud_dir"].as<std::string>();
    pcl::io::savePCDFileASCII(save_dir, *mergedCloud);

    return 0;
}
