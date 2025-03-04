#include <iostream>
#include <yaml-cpp/yaml.h>
#include <vector>
#include <tuple>
#include <algorithm>
#include <unordered_map>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/impl/point_cloud_geometry_handlers.hpp>
#include "pointXYZCustom.hpp"
#include <cnpy.h>

using namespace std;

// Finds the most common semantic class given a vector of points and returns the avg geometry cost of that class
float compute_geometry_cost(const std::vector<PointXYZCustom>& points, std::vector<float> geoCosts, std::vector<std::tuple<int, int, int>> thing_colors, bool usingGeoOnly){

    if (points.size() == 0){
        return 1.0;
    }

    std::vector<int> class_counter = {0,0,0,0,0,0,0,0};
    std::vector<float> geoScoreOnly = {0,0,0,0,0,0,0,0};

    // Loop through and categorize all points
    for(const auto& point : points){
        if (point.r != 201 || point.g != 187 || point.b != 202){
            std::tuple<int, int, int> pColor = {point.b, point.g, point.r};
            auto it = std::find(thing_colors.begin(), thing_colors.end(), pColor);
            int idx = std::distance(thing_colors.begin(), it);
            class_counter[idx]++;
            if (point.intensity > geoScoreOnly[idx]){
                geoScoreOnly[idx] = point.intensity;
            }
        }
    }

    // Find most frequently occuring class and retrieve precaluclated geometry score
    auto max_it = std::max_element(class_counter.begin(), class_counter.end());
    int max_idx = std::distance(class_counter.begin(), max_it);

    if (usingGeoOnly){
        return geoScoreOnly[max_idx];
    }
    return geoCosts[max_idx];
}

// define hashing function for unordered map key 
struct pair_hash {
    template <class T1, class T2>
    std::size_t operator () (std::pair<T1,T2> const &pair) const {
        std::size_t seed = 0;
        hash_combine(seed, pair.first);
        hash_combine(seed, pair.second);
        return seed;
    }
    
    template <class T>
    void hash_combine(std::size_t& seed, const T& val) const {
        std::hash<T> hasher;
        seed ^= hasher(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
};

typedef std::pair<int, int> cellIdx;
typedef std::vector<PointXYZCustom> pointVector;

int main() {

    YAML::Node config = YAML::LoadFile("/home/rsl/catkin_ws/src/traversability_mapping/config.yaml");
    std::string cloud_dir = config["map"]["cloud_dir"].as<std::string>();
    bool normalize = config["map"]["normalize"].as<bool>();
    pcl::PointCloud<PointXYZCustom>::Ptr cloud(new pcl::PointCloud<PointXYZCustom>());
    pcl::io::loadPCDFile(cloud_dir, *cloud);

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

    // Find the average geometric cost of each class
    std::vector<float> pointsPerClass = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    std::vector<float> geoCosts = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    for (const auto& point : cloud->points){
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
    if(normalize){
        geoCosts[0] = 1;
        geoCosts[6] = 1;
        geoCosts[7] = 1;
        auto start = geoCosts.begin() + 1;
        auto end = geoCosts.begin() + 6;
        auto max_it = std::max_element(start, end);
        for (int i = 1; i < 6; ++i){
            geoCosts[i] /= *max_it;
        }
        std::cout<<"Normalized average geometry cost by class: "<<std::endl;
        for (int i = 0; i < 8; i++){
            std::cout<<classes[i]<<": "<<geoCosts[i]<<std::endl;
        }
    }
    
    // Turn point cloud xy plane into a 0.25x0.25 grid and find most common semantic class at each cell
    std::unordered_map<cellIdx, pointVector, pair_hash> grid;

    float cell_dim = 0.25;
    for (const auto& point : cloud->points) {
        if (point.r != 201 || point.g != 187 || point.b != 202){
            int i = static_cast<int>(std::floor(point.x / cell_dim));
            int j = static_cast<int>(std::floor(point.y / cell_dim));
            cellIdx idx = std::make_pair(i, j);
            grid[idx].push_back(point);
        }
    }

    int min_i = std::numeric_limits<int>::max();
    int max_i = std::numeric_limits<int>::min();
    int min_j = std::numeric_limits<int>::max();
    int max_j = std::numeric_limits<int>::min();

    // Find x and y ranges of the grid 
    for (const auto& [idx, _] : grid) {
        min_i = std::min(min_i, idx.first);
        max_i = std::max(max_i, idx.first);
        min_j = std::min(min_j, idx.second);
        max_j = std::max(max_j, idx.second);
    }
    std::cout<<"Min i: "<<min_i<<std::endl;
    std::cout<<"Max i: "<<max_i<<std::endl;
    std::cout<<"Min j: "<<min_j<<std::endl;
    std::cout<<"Max j: "<<max_j<<std::endl;

    const std::size_t rows = max_i - min_i + 1;
    const std::size_t cols = max_j - min_j + 1;

    std::cout<<"rows: "<<rows<<std::endl;
    std::cout<<"cols: "<<cols<<std::endl;
    
    bool save_map = config["map"]["save_map"].as<bool>();
    if (save_map){
        std::vector<float> trav_map(rows * cols);
        bool usingGeoOnly = config["map"]["usingGeoOnly"].as<bool>();

        // For each i,j index of the grid, find the corresponding points from the point cloud and retrieve the geometry cost
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                cellIdx idx = std::make_pair(i + min_i, j + min_j);
                auto iter = grid.find(idx);
                if (iter != grid.end()) {
                    const pointVector& points = iter->second;
                    float cost = compute_geometry_cost(points, geoCosts, thing_colors, usingGeoOnly);  // Get the avg geometry cost of the most common semantic class in this cell
                    trav_map[i * cols + j] = cost;
                } else {
                    trav_map[i * cols + j] = 1.0; // If the cell contains no points, set cost as highest value
                }
            }
        }

        // Save the traversability map as a numpy array
        std::string save_dir = config["map"]["save_dir"].as<std::string>();

        std::vector<std::size_t> shape = {rows, cols};
        cnpy::npy_save(save_dir, &trav_map[0], shape, "w");

        std::cout<<"Successfully saved traversability map to "<<save_dir<<std::endl;
    }

    return 0;
}