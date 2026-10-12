import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from PIL import Image
import torch
from transformers import GLPNImageProcessor, GLPNForDepthEstimation
import numpy as np
import open3d as o3d

# Load the model and feature extractor
feature_extractor = GLPNImageProcessor.from_pretrained("vinvino02/glpn-nyu")
model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")

# Load and preprocess the image
image = Image.open(r"E:\Python\Office\2.jpg")
new_height = 480 if image.height > 480 else image.height
new_height -= (new_height % 32)
new_width = int(new_height * image.width / image.height)
diff = new_width % 32
new_width = new_width - diff if diff < 16 else new_width + 32 - diff
new_size = (new_width, new_height)
image = image.resize(new_size)

# Get image dimensions
width, height = image.size

# Prepare input for the model
inputs = feature_extractor(images=image, return_tensors="pt")

# Perform depth estimation
with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth

# Post-process the depth output
pad = 16
output = predicted_depth.squeeze().cpu().numpy() * 1000.0  # Convert to millimeters
output = output[pad:-pad, pad:-pad]
image = image.crop((pad, pad, image.width - pad, image.height - pad))

# Print depth output range for debugging
print(f"Depth output min: {output.min()}, max: {output.max()}")

# Visualize the depth map
fig, ax = plt.subplots(1, 2)
ax[0].imshow(image)
ax[0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
ax[1].imshow(output, cmap='plasma')
ax[1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
plt.tight_layout()
plt.pause(5)

# Scale the depth data to meters from mm (by dividing by 1000)
depth_image = np.clip(output, 0, 65535).astype('uint16')  # Ensure it's within the valid range
depth_image = depth_image / 1000.0  # Convert to meters

# Create Open3D images
depth_o3d = o3d.geometry.Image(depth_image.astype(np.float32))  # Ensure it is float for Open3D
image = np.array(image)
image_o3d = o3d.geometry.Image(image)

# Check shapes and data types
print(f"Depth image shape: {depth_image.shape}, type: {depth_image.dtype}")
print(f"Color image shape: {image.shape}")

# Check depth image values
print(f"Depth image min: {np.min(depth_image)}, max: {np.max(depth_image)}")

# Create RGBD image
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    image_o3d, depth_o3d, convert_rgb_to_intensity=False, depth_scale=1.0, depth_trunc=4.0
)

# Confirm RGBD image type
print(f"RGBD image type: {type(rgbd_image)}")
print(f"RGBD Depth min: {np.min(np.asarray(rgbd_image.depth))}, max: {np.max(np.asarray(rgbd_image.depth))}")

# Set camera intrinsics
camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
camera_intrinsic.set_intrinsics(width, height, 500, 500, width / 2, height / 2)

# Create point cloud from RGBD image
pcd_raw = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)

# Check the number of points in the point cloud
num_points = len(pcd_raw.points)
print(f"Number of points in raw point cloud: {num_points}")

# If the point cloud contains points, visualize it
if num_points > 0:
    # Visualize the raw point cloud
    o3d.visualization.draw_geometries([pcd_raw], point_show_normal=True)

    # Remove statistical outliers
    cl, ind = pcd_raw.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
    pcd = pcd_raw.select_by_index(ind)

    # Estimate and orient normals
    pcd.estimate_normals()
    pcd.orient_normals_to_align_with_direction()

    # Visualize the cleaned point cloud
    o3d.visualization.draw_geometries([pcd], point_show_normal=True)

    # Create a mesh using Poisson surface reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=10, n_threads=1)

    # Rotate the mesh to correct orientation
    rotation = mesh.get_rotation_matrix_from_xyz((np.pi, 0, 0))
    mesh.rotate(rotation, center=(0, 0, 0))

    # Visualize the mesh
    print(f"Mesh created with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

    # Paint the mesh with a uniform color and visualize
    mesh_uniform = mesh.paint_uniform_color([0.9, 0.8, 0.9])
    mesh_uniform.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh_uniform], mesh_show_back_face=True)

    # Save the mesh as a PLY file
    o3d.io.write_triangle_mesh('output_mesh.ply', mesh)

else:
    print("No points in the point cloud! Exiting...")
