import rosbag
from sensor_msgs.msg import Image, PointCloud2
import os
import shutil
import cv2
from cv_bridge import CvBridge
import rosbag

def extract_timestamps_from_bag(bag_file, topic_name, msg_type, image_data=None):
    timestamps = []
    with rosbag.Bag(bag_file, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=[topic_name]):
            if msg._type == msg_type:
                timestamps.append(t.to_sec())
            if image_data != None:
                image = bridge.compressed_imgmsg_to_cv2(msg, "rgb8")
                #image = bridge.imgmsg_to_cv2(msg, "32FC1")
                image_data.append(image)
    return timestamps

def find_closest_image(pointcloud_timestamps, image_timestamps):
    closest_indices = []
    for pc_ts in pointcloud_timestamps:
        # Find the index of the closest image timestamp for the current point cloud timestamp
        closest_idx = min(range(len(image_timestamps)), key=lambda i: abs(image_timestamps[i] - pc_ts))
        closest_indices.append(closest_idx)
    return closest_indices


bag_file = '/home/rsl/trail4_harveri-hpk_2023-08-22-20-05-45_6.bag'
#image_topic = '/zed2i/zed_node/confidence/confidence_map'
image_topic = '/zed2i/zed_node/rgb/image_rect_color/compressed'
#pointcloud_topic = '/zed2i/zed_node/point_cloud/cloud_registered'
pointcloud_topic = '/hesai/pandar'

image_data = []
bridge = CvBridge()

# Extract timestamps
image_timestamps = extract_timestamps_from_bag(bag_file, image_topic, 'sensor_msgs/CompressedImage', image_data=image_data)
pointcloud_timestamps = extract_timestamps_from_bag(bag_file, pointcloud_topic, 'sensor_msgs/PointCloud2')


image_timestamps.sort()
pointcloud_timestamps.sort()

matches = find_closest_image(pointcloud_timestamps, image_timestamps)

# Directorz for saved images
output_directory = '/home/rsl/harveri_imgs/trail4'

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Save images whose indices match 
for idx, image_index in enumerate(matches):
    output_path = os.path.join(output_directory, f"image_{idx}.png")
    cv2.imwrite(output_path, image_data[image_index])

