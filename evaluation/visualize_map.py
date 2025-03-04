import open3d as o3d
import math
import os
import numpy as np
import matplotlib.pyplot as plt

# trail: x_trans = -353, y_trans = -697
# road: x_trans = -2166, y_trans = -1829

map = np.load("/home/rsl/semantic_pointclouds/road/elevation_map_geo_only.npy")
overlay = False
visualize = False
with_robot_poses = False
plot_colorbar = False

transformation_matx = np.array([[1, 0, 0, -2166*0.25],  # x-axis translation # trail: -353
                                [0, 1, 0, -1829*0.25],  # y-axis translation # trail: - 697
                                [0, 0, 1, 42],  # z-axis translation # trail: 0
                                [0, 0, 0, 1]]) 
scaling_matx = np.array([[0.25, 0, 0, 0],  
                        [0, 0.25, 0, 0],  
                        [0, 0, 0.25, 0], 
                        [0, 0, 0, 1]]) 
combined = np.dot(transformation_matx, scaling_matx)

# read and transform robot poses
poses = []
x_trans = -2166
y_trans = -1829

with open('/home/rsl/harveri_transforms/road_xy_poses.txt', 'r') as f:
    for line in f:
        # Split the line at spaces
        parts = line.split()
        
        # Convert the parts to floats and store in the list
        x, y = math.floor(float(parts[0])/0.25), math.floor(float(parts[1])/0.25)
        x -= x_trans
        y -= y_trans

        poses.append((x, y))

# for each robot pose, find the traversability cost and calculate the std dev between all costs
costs = []
for pose in poses:
    costs.append(map[pose[0], pose[1]])
stddev = np.std(costs)
mean = np.mean(costs)
print("Standard deviation of costs along robot trajectory: " + str(stddev))
print("Mean cost along robot trajectory: " + str(mean))

if plot_colorbar:
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.subplots_adjust(bottom=0.5, top=0.95)

    # Create a colormap
    cmap = plt.cm.rainbow

    # Display the colormap
    norm = plt.Normalize(0, 1)
    cb1 = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), orientation='vertical')
    cb1.set_label('Traversability Cost')

    plt.show()

elif visualize:
    # Make each grid a vertex and assign a color to each vertex
    rows, cols = map.shape
    vertices = []
    colors = []
    for i in range(rows):
        for j in range(cols):
            vertices.append([i, j, map[i, j]*20])
            # Convert the z-value to a color using a colormap
            color = plt.cm.rainbow(map[i, j])[0:3]  # Using the rainbow colormap
            colors.append(color)

    vertices = np.array(vertices)
    colors = np.array(colors)

    # Create faces by connecting adjacent vertices
    faces = []
    for i in range(rows - 1):
        for j in range(cols - 1):
            idx = i * cols + j
            faces.append([idx, idx + 1, idx + cols])
            faces.append([idx + 1, idx + cols, idx + cols + 1])
    faces = np.array(faces)

    # Create mesh to represent the terrain
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors) 
    #mesh = mesh.filter_smooth_simple(number_of_iterations=1) 
    #mesh.compute_vertex_normals()

    # Overlay the traversability map over the original point cloud
    if overlay:
        mesh.transform(combined)

        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultLitTransparency"
        mat.base_color = [1, 1, 1, 0.6]

        pcd = o3d.io.read_point_cloud("/home/rsl/semantic_pointclouds/road/map_fusion.pcd")
        colors = np.asarray(pcd.colors)

        target_color_rgb = [201, 187, 202]
        target_color = [c/255.0 for c in target_color_rgb]

        # Find all points that do not have the target color
        indices = np.where(~np.all(colors == target_color, axis=1))[0]

        # Extract points and colors without the target color
        new_points = np.asarray(pcd.points)[indices]
        new_colors = colors[indices]

        # Create a new point cloud without points of the target color
        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(new_points)
        filtered_pcd.colors = o3d.utility.Vector3dVector(new_colors)

        # Visualize the mesh
        o3d.visualization.draw([{'name': 'elevation', 'geometry': mesh, 'material': mat}, filtered_pcd])
        #o3d.visualization.draw([mesh, filtered_pcd])

    # Visualize the robot trajectory over the traversability map
    elif with_robot_poses:
        points = [[x, y, 20] for x, y in poses]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        colors = [[0, 0, 0] for _ in points]  # black color for each point
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Visualize the mesh and point cloud together
        o3d.visualization.draw([mesh, pcd], show_skybox=False)

    else:
        o3d.visualization.draw([mesh], show_skybox=False)

