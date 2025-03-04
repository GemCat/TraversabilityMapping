#include <iostream>
#include <yaml-cpp/yaml.h>
#include <vector>
#include <tuple>
#include <algorithm>
#include <iomanip>  // Include for std::setw
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/kdtree/kdtree_flann.h>
#include "../pointXYZCustom.hpp"

using namespace std;

int main(){
    YAML::Node config = YAML::LoadFile("/home/rsl/catkin_ws/src/traversability_mapping/config.yaml");
    string eval_cloud_dir = config["evaluate"]["cloud_dir"].as<std::string>();
    string gt_cloud_dir = config["evaluate"]["gt_dir"].as<std::string>();
    string geo_cloud_dir = config["evaluate"]["geo_dir"].as<std::string>();

    pcl::PointCloud<PointXYZCustom>::Ptr eval_cloud(new pcl::PointCloud<PointXYZCustom>);
    pcl::PointCloud<PointXYZCustom>::Ptr gt_cloud(new pcl::PointCloud<PointXYZCustom>);
    pcl::PointCloud<PointXYZCustom>::Ptr geo_cloud(new pcl::PointCloud<PointXYZCustom>);

    pcl::io::loadPCDFile(eval_cloud_dir, *eval_cloud);
    pcl::io::loadPCDFile(gt_cloud_dir, *gt_cloud);
    pcl::io::loadPCDFile(geo_cloud_dir, *geo_cloud);


    /* Colors for each class (BGR):
        0 background: (0, 0, 0) Black
        1 smooth trail: (91, 123, 166) Cafe au lait
        2 traversable grass: (97, 182, 123) Light Green
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

    // Find the average geometric cost of each class using the geo cloud -> only uses points with semantic labels
    std::vector<float> pointsPerClass = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    std::vector<float> geoCosts = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    for (const auto& point : geo_cloud->points){
        if (point.r != 201 || point.g != 187 || point.b != 202){
            std::tuple<int, int, int> pColor = {point.b, point.g, point.r};
            auto it = std::find(thing_colors.begin(), thing_colors.end(), pColor);
            int idx = std::distance(thing_colors.begin(), it);
            pointsPerClass[idx]++;
            geoCosts[idx] += point.intensity;
        }
    }

    std::vector<string> classes = {"background", "smooth", "grass", "rough", "lowVeg", "highVeg", "sky", "obs"};
    std::cout<<"Average geometry cost by class: "<<std::endl;
    for (int i = 0; i < 8; i++){
        geoCosts[i] /= pointsPerClass[i];
        std::cout<<classes[i]<<": "<<geoCosts[i]<<std::endl;
    }
    geoCosts[0] = 1;
    geoCosts[6] = 1;
    geoCosts[7] = 1;
    auto start = geoCosts.begin() + 1;
    auto end = geoCosts.begin() + 6;
    auto max_it = std::max_element(start, end);
    for (int i = 1; i < 6; ++i){
        geoCosts[i] /= *max_it;
    }
    std::cout<<"Normalzed average geometry cost by class: "<<std::endl;
    for (int i = 0; i < 8; i++){
        std::cout<<classes[i]<<": "<<geoCosts[i]<<std::endl;
    }

    pcl::KdTreeFLANN<PointXYZCustom> kdtree;
    kdtree.setInputCloud(eval_cloud);

    float totalSemanticPoints = 0;
    int correctPoints = 0;
    int classMatrix[8][8] = {0};
    float traversabilityDiff = 0;
    bool print = true;

    for (const auto& point : gt_cloud->points){
        // Don't compare GT point if it has Nan position values or if it has no semantic label
        if (std::isinf(point.x) || std::isinf(point.y) || std::isinf(point.z) ||
            std::isnan(point.x) || std::isnan(point.y) || std::isnan(point.z) || point.r == 201) {
            continue;  
        }
        
        // Iterate through eval cloud and find corresponding point 
        std::vector<int> pointIdxNKNSearch(1);
        std::vector<float> pointNKNSquaredDistance(1);
        if (kdtree.nearestKSearch(point, 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0) {
            if (pointNKNSquaredDistance[0] == 0) {
                int idx = pointIdxNKNSearch[0];
                PointXYZCustom &valPoint = (*eval_cloud)[idx];

                std::tuple<int, int, int> gtColor = {point.b, point.g, point.r};
                std::tuple<int, int, int> valColor = {valPoint.b, valPoint.g, valPoint.r};
                auto gtIt = std::find(thing_colors.begin(), thing_colors.end(), gtColor);
                auto valIt = std::find(thing_colors.begin(), thing_colors.end(), valColor);
                int gtIdx = std::distance(thing_colors.begin(), gtIt);
                int valIdx = std::distance(thing_colors.begin(), valIt);

                if (valIdx >= 8 || gtIdx >= 8){
                    continue;
                }

                classMatrix[gtIdx][valIdx]++;
                traversabilityDiff += abs(geoCosts[gtIdx] - geoCosts[valIdx]);
                totalSemanticPoints++;

                // if (traversabilityDiff > totalSemanticPoints && print){
                //     cout<< "gtIdx, valIdx: "<< gtIdx<<", "<<valIdx<<std::endl;
                //     cout <<" val color: " <<static_cast<int>(valPoint.b)<<", "<<static_cast<int>(valPoint.g)<<", "<<static_cast<int>(valPoint.r)<<std::endl;
                //     cout << "cost diff: "<< abs(geoCosts[gtIdx] - geoCosts[valIdx]) << std::endl;
                //     cout<< "traversability diff: " << traversabilityDiff << std::endl;
                //     cout << "total points: "<<totalSemanticPoints<<std::endl;
                //     print = false;
                // }
                
                
                if (gtColor == valColor){
                    correctPoints++;  
                }
            }
        }
    }

    traversabilityDiff /= totalSemanticPoints;

    for (int i = -1; i < 8; ++i) {
        if (i == -1){
            std::cout << std::setw(10) << " ";
        }
        else{
            std::cout << std::setw(10) << classes[i] << " ";
        }
        // Loop through each column
        for (int j = 0; j < 8; ++j) {
            if (i == -1){
                std::cout << std::setw(10) << classes[j] << " ";
            }
            else{
                std::cout << std::setw(10) << classMatrix[i][j] << " "; 
            }  
        }
        std::cout << std::endl;  // Print a newline at the end of each row
    }

    std::cout<<"Average traversability difference from semantic mismatch: "<<traversabilityDiff<<std::endl;

    // cout<<"Correctly labeled points in total: "<<correctPoints<<"/"<<totalSemanticPoints<<", "<<correctPoints/totalSemanticPoints << "%"<<endl;
    // cout<<"Background: " << correctByClass[0]<<"/"<<pointsByClass[0]<<endl;
    // cout<<"Smooth trail: " << correctByClass[1]<<"/"<<pointsByClass[1]<<endl;
    // cout<<"Traversable grass: " << correctByClass[2]<<"/"<<pointsByClass[2]<<", "<<correctByClass[2]/pointsByClass[2] << "%"<<endl;
    // cout<<"Rough trail: " << correctByClass[3]<<"/"<<pointsByClass[3]<<endl;
    // cout<<"Low vegetation: " << correctByClass[4]<<"/"<<pointsByClass[4]<<endl;
    // cout<<"High vegetation: " << correctByClass[5]<<"/"<<pointsByClass[5]<<", "<<correctByClass[5]/pointsByClass[5] << "%"<<endl;
    // cout<<"Sky: " << correctByClass[6]<<"/"<<pointsByClass[6]<<endl;
    // cout<<"Obstacle: " << correctByClass[7]<<"/"<<pointsByClass[6]<<endl;


    return 0;
}
