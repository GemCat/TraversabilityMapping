#include <iostream>
#include <ros/ros.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <tf_conversions/tf_eigen.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Dense>
#include <sstream>
#include <fstream>

class PointCloudProcessor {
private:
    ros::NodeHandle nh_;
    ros::Subscriber point_cloud_sub_;
    tf::TransformListener listener_;
    int file_index_;

public:
    PointCloudProcessor() : file_index_(0) {
        //point_cloud_sub_ = nh_.subscribe("/zed2i/zed_node/point_cloud/cloud_registered", 10, &PointCloudProcessor::pointCloudCallback, this);
        point_cloud_sub_ = nh_.subscribe("/hesai/pandar", 10, &PointCloudProcessor::pointCloudCallback, this);
    }

    void pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg) {
        std::string target_frame1 = "zed2i_left_camera_frame";
        std::string source_frame1 = cloud_msg->header.frame_id;
        std::string target_frame2 = "/odom";
        std::string source_frame2 = "zed2i_left_camera_frame";
        std::cout<<file_index_<<std::endl;

        tf::StampedTransform transform1;
        tf::StampedTransform transform2;
        try {
            listener_.lookupTransform(target_frame1, source_frame1, ros::Time(0), transform1);
            listener_.lookupTransform(target_frame2, source_frame2, ros::Time(0), transform2);
        } catch (tf::TransformException& ex) {
            ROS_ERROR("%s", ex.what());
            return;
        }
        saveTransformToEigenFile(transform1, file_index_, "cam_lidar");
        saveTransformToEigenFile(transform2, file_index_++, "odom_cam");
    }

    void saveTransformToEigenFile(const tf::Transform& transform, int index, std::string frames) {
        // Convert tf transform to Eigen matrix
        Eigen::Affine3d eigen_transform;
        tf::transformTFToEigen(transform, eigen_transform);

        std::cout<<index<<std::endl;

        // Create filename based on index
        std::ostringstream filename;
        filename << "/home/rsl/harveri_transforms/road_val/" << frames << "/transform_" << index << ".txt";

        // Save to file
        std::ofstream file(filename.str());
        file << eigen_transform.matrix() << std::endl;
        file.close();
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "point_cloud_processor");
    PointCloudProcessor processor;
    ros::spin();
    return 0;
}
