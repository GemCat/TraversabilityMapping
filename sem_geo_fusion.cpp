#include <iostream>
#include <yaml-cpp/yaml.h>
#include <vector>
#include <unordered_map>
#include <map>
#include <tuple>
#include <algorithm>
#include <boost/functional/hash.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/impl/point_cloud_geometry_handlers.hpp>
#include "pointXYZCustom.hpp"

using namespace std;


// Custom tuple key for colour hashmap
struct TupleHash {
    template <typename T, typename U, typename V>
    std::size_t operator()(const std::tuple<T, U, V>& tuple) const {
        std::size_t seed = 0;
        boost::hash_combine(seed, std::get<0>(tuple));
        boost::hash_combine(seed, std::get<1>(tuple));
        boost::hash_combine(seed, std::get<2>(tuple));
        return seed;
    }
};

// Retrieve most common semantic class (color) from a group of points
std::tuple<uint8_t, uint8_t, uint8_t> getMostCommonColor(const std::vector<PointXYZCustom*>& points, const std::vector<float> pointsPerClass, const std::vector<std::tuple<int, int, int>> thing_colors, int totalPoints) {
    std::unordered_map<std::tuple<int, int, int>, float, TupleHash> color_histogram;

    for (const auto& point : points) {
        if (point->r == 201) {
            continue;
        }
        color_histogram[std::make_tuple(point->b, point->g, point->r)]++;
    }

    float maxRatio = 0.0;
    std::tuple<int, int, int> most_common_color;
    for (const auto& pair : color_histogram) {
        float classRatio = 0.0;
        auto it = std::find(thing_colors.begin(), thing_colors.end(), pair.first);
        int idx = std::distance(thing_colors.begin(), it);

        if (pointsPerClass[idx] > 0.01*totalPoints){ // only consider classes that have more than 1 percent of points out of all points
            classRatio= pair.second/pointsPerClass[idx];
        }
        
        std::cout<<"Color: " << std::get<0>(pair.first) << ", "<< std::get<1>(pair.first) << ", "<< std::get<2>(pair.first) << ": " <<pair.second << ", " <<classRatio<<std::endl;
        if (classRatio > maxRatio) {
            maxRatio = classRatio;
            most_common_color = pair.first;
        }
    }
    if (maxRatio == 0.0){
        most_common_color = {202, 187, 201}; // if cluster has no semantically labeled points, keep color as unlabeled 
    }
    return most_common_color;
}

int main() {
    YAML::Node config = YAML::LoadFile("/home/rsl/catkin_ws/src/traversability_mapping/config.yaml");
    std::string cloud_dir = config["fusion"]["cloud_dir"].as<std::string>();
    std::string save_dir = config["fusion"]["save_cloud_dir"].as<std::string>();

    pcl::PointCloud<PointXYZCustom>::Ptr cloud(new pcl::PointCloud<PointXYZCustom>());
    pcl::io::loadPCDFile(cloud_dir, *cloud);

    int num_points = cloud->points.size();

    // Find height of ground at each 1x1m partition of the ground
    // Use a map to represent the grid. Each cell in the grid is a vector of points.
    std::map<int, std::map<int, std::vector<PointXYZCustom>>> grid;
    std::map<int, std::map<int, float>> groundHeights;

    for (const auto& point : cloud->points) {
        int xIndex = static_cast<int>(std::floor(point.x));
        int yIndex = static_cast<int>(std::floor(point.y));

        grid[xIndex][yIndex].push_back(point);
    }

    // Find lowest 10% of point in each grid cell
    std::map<int, std::map<int, std::vector<PointXYZCustom>>> lowestPointsGrid;

    for (auto& xPair : grid) {
        for (auto& yPair : xPair.second) {
            // Sort the points based on the z value
            std::sort(yPair.second.begin(), yPair.second.end(), [](const PointXYZCustom& a, const PointXYZCustom& b) {
                return a.z < b.z;
            });

            int count;
            // Compute the number of points for 10%
            if (yPair.second.size() < 10) {
                count = static_cast<long int>(1);
            } 
            else {
                count = static_cast<long int>(0.1 * yPair.second.size());
            }
            // Extract the lowest 10% of points
            auto startIter = yPair.second.begin();
            auto endIter = startIter + count;
            lowestPointsGrid[xPair.first][yPair.first] = std::vector<PointXYZCustom>(startIter, endIter);
        }
    }

    // Iterate through lowest 10% and k-means into two clusters, cluster with lower average z position is the ground 
    for (auto& xPair : lowestPointsGrid) {
        for (auto& yPair : xPair.second) {
            std::vector<PointXYZCustom>& points = yPair.second;
            cv::Mat z_data(points.size(), 1, CV_32F);

            if (points.size() == 1){
                groundHeights[xPair.first][yPair.first] = points[0].z;
                continue;
            }

            for(int i = 0; i < points.size(); ++i){
                z_data.at<float>(i, 0) = points[i].z;
            }
            int k = 2;
            cv::Mat clusters;
            cv::Mat centers;
            cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0);
            cv::kmeans(z_data, k, clusters, criteria, 3, cv::KMEANS_PP_CENTERS, centers);

            std::vector<float> heights;
            for (int cls = 0; cls < 2; ++cls){
                float avgHeight = 0;
                int numPoints = 0;
                for (int i = 0; i < points.size(); ++i) {
                    int current_label = clusters.at<int>(i, 0);
                    if (current_label == cls) {
                        avgHeight += points[i].z;
                        numPoints ++;
                    }
                }
                avgHeight /= numPoints;
                heights.push_back(avgHeight);
            }
            groundHeights[xPair.first][yPair.first] = *min_element(heights.begin(), heights.end());
        }
    }


    // Run kmeans on relative height change and relative height from ground to cluster points
    cv::Mat data_points(num_points, 2, CV_32F);  // 2D matrix for 2D partitioning
    for (int i = 0; i < num_points; ++i) {
        int xIndex = static_cast<int>(std::floor(cloud->points[i].x));
        int yIndex = static_cast<int>(std::floor(cloud->points[i].y));
        float groundHeight = groundHeights[xIndex][yIndex];

        data_points.at<float>(i, 0) = cloud->points[i].intensity; 
        data_points.at<float>(i, 1) = (cloud->points[i].z - groundHeight);
    }
    // cv::Mat data_points(num_points, 1, CV_32F);  // 2D matrix for 2D partitioning
    // for (int i = 0; i < num_points; ++i) {
    //     data_points.at<float>(i, 0) = cloud->points[i].intensity; 
    // }

    // Normalize data to account for different ranges
    cv::Mat col1 = data_points.col(0);
    cv::Mat col2 = data_points.col(1);

    cv::Scalar mean1, stddev1;
    cv::Scalar mean2, stddev2;

    cv::meanStdDev(col1, mean1, stddev1);
    cv::meanStdDev(col2, mean2, stddev2);

    cv::Mat normalized_col1 = (col1 - mean1[0]) / stddev1[0];
    cv::Mat normalized_col2 = (col2 - mean2[0]) / stddev2[0];

    std::vector<cv::Mat> normalized_cols = {normalized_col1, normalized_col2};
    cv::Mat normalized_data;
    cv::hconcat(normalized_cols, normalized_data);

    int k = 8; // number of clusters = number of valid semantic classes
    cv::Mat clusters;
    cv::Mat centers;
    cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0);
    cv::kmeans(normalized_data, k, clusters, criteria, 3, cv::KMEANS_PP_CENTERS, centers);

    bool vis_kmeans = config["fusion"]["vis_kmeans"].as<bool>();
    if (vis_kmeans){
        // Create a blank image to draw on
        cv::Mat image(500, 500, CV_8UC3, cv::Scalar(255,255,255));  // white background

        // Define some colors for the clusters
        cv::Scalar colors[] = { cv::Scalar(255,0,0), cv::Scalar(0,255,0), cv::Scalar(0,0,255), cv::Scalar(255,255,0), cv::Scalar(255,0,255), cv::Scalar(0,255,255), cv::Scalar(127,127,127), cv::Scalar(127,0,127) };

        // Loop through each point and plot it on the image
        for (int i = 0; i < num_points; ++i) {
            int cluster_idx = clusters.at<int>(i, 0);
            cv::Point2f pt(normalized_data.at<float>(i, 0), normalized_data.at<float>(i, 1));
            pt *= 10;  // scale up to fit in the image
            pt += cv::Point2f(250, 250);  // shift to center in the image
            cv::circle(image, pt, 1, colors[cluster_idx], -1);
        }

        // Display the image
        cv::namedWindow("Clusters", cv::WINDOW_AUTOSIZE);
        cv::imshow("Clusters", image);
        cv::waitKey(0);

        return 0;
    }

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

    // Colors for visualizing kmeans clusters
    // std::vector<std::tuple<int, int, int>> thing_colors = {
    //     {255, 0, 0},
    //     {0, 255, 0},
    //     {0, 0, 255},
    //     {255,255,0},
    //     {255,0,255},
    //     {0,255,255},
    //     {127,127,127},
    //     {127,127,127}
    // };

    // Find the total number of points each semantic class currently contains
    std::vector<float> pointsPerClass = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    for (const auto& point : cloud->points){
        if (point.r != 201 || point.g != 187 || point.b != 202){
            std::tuple<int, int, int> pColor = {point.b, point.g, point.r};
            auto it = std::find(thing_colors.begin(), thing_colors.end(), pColor);
            int idx = std::distance(thing_colors.begin(), it);
            pointsPerClass[idx]++;
        }
    }
    std::cout<<"Points per class: ";
    for (int i = 0; i < 8; i++){
        std::cout<<pointsPerClass[i] << " ";
    }
    std::cout << std::endl;

    int cloudSize = cloud->points.size();

    // Loop through each cluster and assign all points without semantic classes with the most frequently occuring semantic class in its respective cluster
    for (int cls = 0; cls < 8; ++cls) {

        std::vector<PointXYZCustom*> clusterPoints;

        // Find all points belonging to one cluster
        for (int i = 0; i < num_points; ++i) {
            int current_label = clusters.at<int>(i, 0);
            if (current_label == cls) {
                clusterPoints.push_back(&cloud->points[i]);
            }
        }

        std::cout<<"Cluster "<<cls<<std::endl;
        auto [b, g, r] = getMostCommonColor(clusterPoints, pointsPerClass, thing_colors, cloudSize); // Semantic label with highest ratio in cluster

        std::cout << "Most common color: R=" << static_cast<int>(r) 
            << " G=" << static_cast<int>(g) 
            << " B=" << static_cast<int>(b) << std::endl;

        // Assign each point without semantic label to the semantic label with highest ratio
        for (auto& point : clusterPoints){
            if(point->r == 201 && point->g == 187 && point->b == 202){
                point->r = static_cast<int>(r);
                point->g = static_cast<int>(g);
                point->b = static_cast<int>(b);
            }
            // point->r = std::get<2>(thing_colors[cls]);
            // point->g = std::get<1>(thing_colors[cls]);
            // point->b = std::get<0>(thing_colors[cls]);
        }
    }

    // Visualize the point cloud
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(255, 255, 255);

    pcl::visualization::PointCloudColorHandlerRGBField<PointXYZCustom> rgb(cloud);
    viewer->addPointCloud<PointXYZCustom>(cloud, rgb, "map");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "map");
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();

   
    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
    }

    // Save the custom point cloud if needed
    pcl::io::savePCDFileASCII(save_dir, *cloud);
  
    return 0;
}

