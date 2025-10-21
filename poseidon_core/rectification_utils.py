import os
import re
import zarr
import cv2

import numpy as np
import pandas as pd

from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator as reg_interp


def gen_grid(xMin, xMax, yMin, yMax, resolution, z=None, dir='generated_grids'):

    if not os.path.exists(dir):
        os.makedirs(dir)
        print(f'Directory to store grids created: {dir}')
    else:
        print(f'Directory to store grids already exists: {dir}')

    grid_x, grid_y = np.mgrid[xMin:xMax:resolution, yMin:yMax:resolution]

    if z is None:
        z = 0
    
    if isinstance(z, int):
        grid_z = np.full_like(grid_x, z)

    if isinstance(z, np.ndarray):
        x = z[0]
        y = z[1]
        z = z[2]
        grid_z = griddata((x, y), z, (grid_x, grid_y), method='linear')
    
    # Save the grid arrays to a compressed Zarr files
    zarr.save(os.path.join(dir, 'grid_x.zarr'), grid_x)
    zarr.save(os.path.join(dir, 'grid_y.zarr'), grid_y)
    zarr.save(os.path.join(dir, 'grid_z.zarr'), grid_z)

    return grid_x, grid_y, grid_z

def reshape_grids(grid_x, grid_y, grid_z):
    x_vec = grid_x.T.reshape(-1, 1)
    y_vec = grid_y.T.reshape(-1, 1)
    z_vec = grid_z.T.reshape(-1, 1)

    xyz = np.concatenate([x_vec, y_vec, z_vec], axis=1)

    return xyz

def CIRNangles2R(azimuth, tilt, swing):
    R = np.empty((3,3))

    R[0,0] = -np.cos(azimuth) * np.cos(swing) - np.sin(azimuth) * np.cos(tilt) * np.sin(swing)
    R[0,1] = np.cos(swing) * np.sin(azimuth) - np.sin(swing) * np.cos(tilt) * np.cos(azimuth)
    R[0,2] = -np.sin(swing) * np.sin(tilt)
    R[1,0] = -np.sin(swing) * np.cos(azimuth) + np.cos(swing) * np.cos(tilt) * np.sin(azimuth)
    R[1,1] = np.sin(swing) * np.sin(azimuth) + np.cos(swing) * np.cos(tilt) * np.cos(azimuth)
    R[1,2] = np.cos(swing) * np.sin(tilt);
    R[2,0] = np.sin(tilt) * np.sin(azimuth)
    R[2,1] = np.sin(tilt) * np.cos(azimuth)
    R[2,2] = -np.cos(tilt)

    return R

def intrinsicsExtrinsics2P(intrinsics, extrinsics):
    K = np.zeros((3,3))
    K[0,0] = -intrinsics[4]
    K[1,1] = -intrinsics[5]
    K[0,2] = intrinsics[2]
    K[1,2] = intrinsics[3]
    K[2,2] = 1

    azimuth = extrinsics[3]
    tilt = extrinsics[4]
    swing = extrinsics[5]
    R = CIRNangles2R(azimuth, tilt, swing)

    x = extrinsics[0]
    y = extrinsics[1]
    z = extrinsics[2]
    column_vec = np.array([-x, -y, -z]).reshape(-1, 1)
    IC = np.concatenate([np.eye(3), column_vec], axis=1)

    P = np.dot(K, np.dot(R, IC))
    P /= P[2, 3]

    return P, K, R, IC

def distortUV(UV, intrinsics):
    NU = intrinsics[0]
    NV = intrinsics[1]
    c0U = intrinsics[2]
    c0V = intrinsics[3]
    fx = intrinsics[4]
    fy = intrinsics[5]
    d1 = intrinsics[6]
    d2 = intrinsics[7]
    d3 = intrinsics[8]
    t1 = intrinsics[9]
    t2 = intrinsics[10]

    U = UV[0, :]
    V = UV[1, :]

    x = (U - c0U) / fx
    y = (V - c0V) / fy

    # Radial distortion
    r2 = x**2 + y**2
    fr = 1 + d1 * r2 + d2 * r2**2 + d3 * r2**3

    # Tangential distortion
    dx = 2 * t1 * x * y + t2 * (r2 + 2 * x**2)
    dy = t1 * (r2 + 2 * y**2) + 2 * t2 * x * y

    # Apply correction
    xd = x * fr + dx
    yd = y * fr + dy
    Ud = xd * fx + c0U
    Vd = yd * fy + c0V

    # Find negative UV coordinates
    flag_mask = (Ud < 0) | (Ud > NU) | (Vd < 0) | (Vd > NV)
    Ud[flag_mask] = 0
    Vd[flag_mask] = 0

    # Define corners of the image
    Um = np.array([0, 0, NU, NU])
    Vm = np.array([0, NV, NV, 0])

    # Normalization
    xm = (Um - c0U) / fx
    ym = (Vm - c0V) / fy
    r2m = xm**2 + ym**2

    # Tangential Distortion at corners
    dxm = 2 * t1 * xm * ym + t2 * (r2m + 2 * xm**2)
    dym = t1 * (r2m + 2 * ym**2) + 2 * t2 * xm * ym

    # Find values larger than those at corners
    max_dym = np.max(np.abs(dym))
    max_dxm = np.max(np.abs(dxm))

    # Indices where distortion values are larger than those at corners
    exceeds_dy = np.where(np.abs(dy) > max_dym)
    exceeds_dx = np.where(np.abs(dx) > max_dxm)

    # Initialize flag array (assuming it’s previously defined)
    flag = np.ones_like(Ud)
    flag[exceeds_dy] = 0.0
    flag[exceeds_dx] = 0.0

    return Ud, Vd, flag

def xyz2DistUV(intrinsics, extrinsics, grid_x, grid_y, grid_z):
    P, K, R, IC = intrinsicsExtrinsics2P(intrinsics, extrinsics)

    xyz = reshape_grids(grid_x, grid_y, grid_z)
    xyz_homogeneous = np.vstack((xyz.T, np.ones(xyz.shape[0])))
    
    UV_homogeneous = np.dot(P, xyz_homogeneous)
    UV = UV_homogeneous[:2, :] / UV_homogeneous[2, :]

    Ud, Vd, flag = distortUV(UV, intrinsics)

    DU = Ud.reshape(grid_x.shape, order="F")
    DV = Vd.reshape(grid_y.shape, order="F")
    
    # Compute camera coordinates
    xyzC = np.dot(np.dot(R, IC), xyz_homogeneous)

    # Find negative Zc coordinates (Z <= 0) and update the flag
    negative_z_indices = np.where(xyzC[2, :] <= 0.0)
    flag[negative_z_indices] = 0.0
    flag = flag.reshape(grid_x.shape, order="F")

    return DU * flag, DV * flag

def getPixels(image, Ud, Vd, s):

    """
    Pulls rgb or gray pixel intensities from image at specified
    pixel locations corresponding to X,Y coordinates calculated in either
    xyz2DistUV or dlt2UV.

    Args:
        image (ndarray): image where pixels will be taken from
        Ud: Nx1 vector of distorted U coordinates for N points
        Vd: Nx1 vector of distorted V coordinates for N points
        s: shape of output image

    Returns:
        ir (ndarray): pixel intensities

    """

    # Use regular grid interpolator to grab points
    im_s = image.shape
    if len(im_s) > 2:
        ir = np.full((s[0], s[1], im_s[2]), np.nan)
        for i in range(im_s[2]):
            rgi = reg_interp(
                (np.arange(0, image.shape[0]), np.arange(0, image.shape[1])),
                image[:, :, i],
                bounds_error=False,
                fill_value=np.nan,
            )
            ir[:, :, i] = rgi((Vd, Ud))
    else:
        ir = np.full((s[0], s[1], 1), np.nan)
        rgi = reg_interp(
            (np.arange(0, image.shape[0]), np.arange(0, image.shape[1])),
            image,
            bounds_error=False,
            fill_value=np.nan,
        )
        ir[:, :, 0] = rgi((Vd, Ud))

    # Mask out values out of range
    with np.errstate(invalid="ignore"):
        mask_u = np.logical_or(Ud <= 1, Ud >= image.shape[1])
        mask_v = np.logical_or(Vd <= 1, Vd >= image.shape[0])
    mask = np.logical_or(mask_u, mask_v)
    if len(im_s) > 2:
        ir[mask, :] = np.nan
    else:
        ir[mask] = np.nan

    return ir

def mergeRectify(image_path, intrinsics, extrinsics, grid_x, grid_y, grid_z):
    
    s = grid_x.shape

    I = cv2.imread(image_path)
    I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)

    Ud, Vd = xyz2DistUV(intrinsics, extrinsics, grid_x, grid_y, grid_z)

    Ud = np.round(Ud)
    Vd = np.round(Vd)
    Ud = Ud.astype(int)
    Vd = Vd.astype(int)
    
    ir = getPixels(I, Ud, Vd, s)
    ir = np.array(ir, dtype=np.uint8)

    return ir

def mergeRectifyFolder(folder_path, intrinsics, extrinsics, grid_x, grid_y, grid_z):
    
    s = grid_x.shape

    # Calculate Ud, Vd once since they are the same for all images
    Ud, Vd = xyz2DistUV(intrinsics, extrinsics, grid_x, grid_y, grid_z)
    Ud = np.round(Ud).astype(int)
    Vd = np.round(Vd).astype(int)
    
    data = []
    
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        I = cv2.imread(image_path)
        I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
        
        ir = getPixels(I, Ud, Vd, s)
        ir = np.array(ir, dtype=np.uint8)
        
        # Append the image name and ir array to the list
        data.append({'image_name': image_name, 'ir': ir})

    results = pd.DataFrame(data)
    
    return results

def mergeRectifyLabelsFolder(folder_path, intrinsics, extrinsics, grid_x, grid_y, grid_z):
    
    s = grid_x.shape

    # Calculate Ud, Vd once since they are the same for all images
    Ud, Vd = xyz2DistUV(intrinsics, extrinsics, grid_x, grid_y, grid_z)
    Ud = np.round(Ud).astype(int)
    Vd = np.round(Vd).astype(int)
    
    data = []
    
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        I = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        ir = getPixels(I, Ud, Vd, s)
        ir = np.array(ir, dtype=np.uint8)
        
        # Append the image name and ir array to the list
        data.append({'image_name': image_name, 'ir': ir})

    results = pd.DataFrame(data)
    
    return results

def save_df_to_zarr(df, zarr_store_path, depth_map=False):
    # Create or open a Zarr store
    store = zarr.open_group(zarr_store_path, mode='a')
    
    for index, row in df.iterrows():
        image_name = row['image_name']

        if depth_map is False:
            array = row['ir']
        
        else:
            array = row['depth_map']
        
        # Create a dataset name by appending 'rectified' to the original image name
        dataset_name = f"{os.path.splitext(image_name)[0]}_rectified"
        
        # Save the ir array to the Zarr store
        store.create_dataset(dataset_name, data=array, compression='zlib')
        
    return store


def mergeRectifyLabels(image_path, intrinsics, extrinsics, grid_x, grid_y, grid_z):
    
    s = grid_x.shape

    I = cv2.imread(image_path)
    I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

    Ud, Vd = xyz2DistUV(intrinsics, extrinsics, grid_x, grid_y, grid_z)

    Ud = np.round(Ud)
    Vd = np.round(Vd)
    Ud = Ud.astype(int)
    Vd = Vd.astype(int)
    
    ir = getPixels(I, Ud, Vd, s)
    ir = np.array(ir, dtype=np.uint8)

    return ir

def plot_depth_maps(zarr_store_path, grid_z, plotting_folder, depth_min=None, depth_max=None):
    
    if not os.path.exists(plotting_folder):
        os.makedirs(plotting_folder)
        print(f'Directory to store grids created: {plotting_folder}')
    else:
        print(f'Directory to store grids already exists: {plotting_folder}')

    # Open the Zarr store in read mode
    store = zarr.open_group(zarr_store_path, mode='r')

    pattern = re.compile(r'(?P<cam_id>CAM_[A-Z]{2}_\d{2})_(?P<timestamp>\d{14})_rectified')

    for array_name in store.keys():
        match = pattern.match(array_name)
        if match:
            cam_id = match.group('cam_id')
            timestamp = match.group('timestamp')
            
            # Load the rectified image array
            rectified_image = store[f"{cam_id}_{timestamp}_rectified"]
            grayscale_image = np.dot(rectified_image[...,:3], [0.2989, 0.5870, 0.1140])
            
            # Load the depth map array
            depth_map_name = f"{cam_id}_{timestamp}_predseg_labels_rectified_depth_map_95th_rectified"
            depth_map = store[depth_map_name]

            max_elev_point_indices = np.unravel_index(np.nanargmin(depth_map), depth_map.shape)

            # Plot the image
            plt.imshow(grayscale_image, cmap='gray')  # Assuming ir_array is your grayscale image

            # Overlay the depth map
            im = plt.imshow(depth_map, cmap=cmocean.cm.deep, vmin=depth_min, vmax=depth_max)  # Adjust alpha for transparency
            # print(f"Vmin:{depth_min} Vmax: {depth_max} Max depth: {np.nanmax(depth_map)} Min depth: {np.nanmin(depth_map)}")

            plt.scatter(max_elev_point_indices[1], max_elev_point_indices[0], c='red', s=10, marker='o')

            # Add a colorbar for the depth map
            cbar = plt.colorbar(im, label='Depth')
            cbar.set_label('Depth (meters)')

            plt.gca().invert_yaxis()
            plt.xlabel('X (cm)')
            plt.ylabel('Y (cm)')

            # Save the figure
            plt.savefig(os.path.join(plotting_folder, f'{cam_id}_{timestamp}_depth_map_rectification.png'), 
                        bbox_inches='tight', pad_inches=0.1, dpi=300)
            
            plt.close()

    return None
