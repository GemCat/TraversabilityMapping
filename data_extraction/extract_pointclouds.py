#!/usr/bin/env python

# import rosbag
# import os
# import sys
# from sensor_msgs.msg import PointCloud2
# import sensor_msgs.point_cloud2 as pc2

# def extract_pointclouds_from_rosbag(bag_path, topic, output_folder):
#     bag = rosbag.Bag(bag_path, "r")
#     count = 0
    
#     for topic, msg, t in bag.read_messages(topics=[topic]):
#         if msg._type == 'sensor_msgs/PointCloud2':
#             points_list = []
#             for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
#                 points_list.append(p)
            
#             # Save to a file or process in some way
#             # For now, I'll just print the first 10 points as an example
#             #print(points_list[:10])
            
#             count += 1
#             output_file = os.path.join(output_folder, "pointcloud_" + str(count) + ".pcd")
#             with open(output_file, 'w') as f:
#                 for point in points_list:
#                     f.write(f"{point[0]} {point[1]} {point[2]}\n")

#     bag.close()
#     print(f"Extracted {count} point clouds from the rosbag.")

# if __name__ == '__main__':
#     if len(sys.argv) < 4:
#         print("Usage: {} BAGFILE TOPIC OUTPUT_DIRECTORY".format(sys.argv[0]))
#         sys.exit(-1)
    
#     bagfile = sys.argv[1]
#     topic = sys.argv[2]
#     output_directory = sys.argv[3]

#     extract_pointclouds_from_rosbag(bagfile, topic, output_directory)

#!/usr/bin/env python

import rosbag
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import pcl  # Ensure you've installed python-pcl
import sys
import os

def extract_pointclouds_from_bag(bagfile, topic, output_directory):
    bag = rosbag.Bag(bagfile, 'r')
    count = 0

    try:
        for topic, msg, t in bag.read_messages(topics=[topic]):
            if msg._type == 'sensor_msgs/PointCloud2':
                # Convert ROS PointCloud2 to PCL PointXYZ
                points_list = []
                for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
                    points_list.append(p)
                cloud_pcl = pcl.PointCloud()
                cloud_pcl.from_list(points_list)

                # Save the point cloud to PCD format
                pcl.save(cloud_pcl, '{}/cloud_{}.pcd'.format(output_directory, count))
                count += 1

    finally:
        bag.close()

if __name__ == '__main__':
    # if len(sys.argv) < 4:
    #     print("Usage: {} BAGFILE TOPIC OUTPUT_DIRECTORY".format(sys.argv[0]))
    #     sys.exit(-1)
    
    bagfile = '/home/rsl/trail4_harveri-hpk_2023-08-22-20-05-45_6.bag'
    #topic = '/zed2i/zed_node/point_cloud/cloud_registered'
    topic = '/hesai/pandar'
    output_directory = '/home/rsl/harveri_pc/trail4'

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    extract_pointclouds_from_bag(bagfile, topic, output_directory)
