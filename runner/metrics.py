"""
Stitcing metrics.
"""
import json
import numpy as np
import vedo

def get_bndbox(src_volume_size, affine_matrix):
    a = affine_matrix[:3,:3]
    b = affine_matrix[:3,3]
    #xyz
    z, y, x = src_volume_size

    points = np.array([[0, 0, 0],[0, 0, z],
                        [0, y, 0],[0, y, z],
                        [x, 0, 0],[x, 0, z],
                        [x, y, 0],[x, y, z]])

    new_points = np.zeros_like(points)
    for i, point in enumerate(points):
        new_point = np.floor(np.dot(a,point) + b.T)
        new_points[i]= new_point

    x_max = max(new_points[:,0])
    y_max = max(new_points[:,1])
    z_max = max(new_points[:,2])

    return [x_max, y_max , z_max]

def HaarPSI3D(volume1, volume2):
    volume1[volume1 < 0] = 0
    volume2[volume2 < 0] = 0
    max_value = np.max([volume1, volume2])
    #print(f'max_value = {max_value}')
    volume1 = 255 * volume1 / max_value
    volume2 = 255 * volume2 / max_value
    results_by_slices = []
    for i in range(volume1.shape[0]):
        res = haarPsi.haar_psi(volume1[i], volume2[i], False)
        #print(f'Similarity score = {res[0]}')
        if np.isnan(res[0]):
            pass
        else:
            results_by_slices.append(res[0])
    return np.mean(np.array(results_by_slices))

def distance(point1, point2):
    """
    Returns distances between point1 and point2.
    """
    return np.linalg.norm(np.array(point1) - np.array(point2))

def mse(markup_volume1, volume2, ax = None):
    """
    Mean squared error.
    """
    return np.sum(np.square(markup_volume1 - volume2))/(2 * np.sum(np.square(markup_volume1)))

def geometry_mse(markup_volume, gt_transform, error_transform):
    """
    Mean squared error - transform estimation.
    """
    gt_a = gt_transform[:3,:3]
    gt_b = gt_transform[:3,3]

    err_a = error_transform[:3,:3]
    err_b = error_transform[:3,3]

    z, y, x = markup_volume.shape
    points = np.array([[0, 0, 0],[0, 0, z],
                        [0, y, 0],[0, y, z],
                        [x, 0, 0],[x, 0, z],
                        [x, y, 0],[x, y, z]])

    gt_transfrom_points = np.zeros_like(points)
    founded_transfrom_points = np.zeros_like(points)
    distances = []

    print("Corners (gt, founded), Distances:")
    for i, point in enumerate(points):
        gt_transfrom_point = point
        founded_transfrom_point = np.dot(err_a,point) + err_b.T
        gt_transfrom_points[i]= gt_transfrom_point
        founded_transfrom_points[i]= founded_transfrom_point
        distances.append(distance(gt_transfrom_point, founded_transfrom_point))
        print(gt_transfrom_point)
        print(founded_transfrom_point)
        print(distances[i])
        print('---------------------------------')

    # gt_transfrom_points_vedo = vedo.Points(gt_transfrom_points, r=5)
    # founded_transfrom_points_vedo = vedo.Points(founded_transfrom_points, r=5)
    # plotter = vedo.Plotter()
    # plotter.add(gt_transfrom_points_vedo.color('red'))
    # plotter.add(founded_transfrom_points_vedo.color('blue'))
    # #plotter.show(f'{float((np.square(distances)).mean(axis = None) / np.linalg.norm([z, y, x])**2), float(((np.square(distances)).mean(axis = None) / (np.linalg.norm([z, y, x])**2))**0.5), float(max(distances))}', axes = 1)
    # plotter.remove(gt_transfrom_points_vedo.color('red'),
    #                founded_transfrom_points_vedo.color('blue'))
    
    return (np.square(distances)).mean(axis = None) / np.linalg.norm([z, y, x])**2, ((np.square(distances)).mean(axis = None) / (np.linalg.norm([z, y, x])**2))**0.5, ((np.square(distances)).mean(axis = None))**0.5, max(distances)/np.linalg.norm([z, y, x]) , max(distances)

def geometry_mse_calibration(markup_volume, gt_transform):
    """
    Mean squared error - transform estimation.
    """
    gt_a = gt_transform[:3,:3]
    gt_b = gt_transform[:3,3]
    print(gt_a)
    print(gt_b)
    z, y, x = markup_volume.shape
    points = np.array([[0, 0, 0],[0, 0, z],
                        [0, y, 0],[0, y, z],
                        [x, 0, 0],[x, 0, z],
                        [x, y, 0],[x, y, z]])
    
    gt_transfrom_points = np.zeros_like(points)

    distances = []

    print("Corners |gt - founded| = Distances:")
    for i, point in enumerate(points):
        gt_transfrom_point = np.dot(gt_a,point) + gt_b.T
        print(np.dot(gt_a,point) + gt_b.T)
        gt_transfrom_points[i]= gt_transfrom_point
        distances.append(distance(gt_transfrom_point, point))
        print(f'|{point} - {gt_transfrom_point}| = {distances[i]}')
        print('---------------------------------')

    return (np.square(distances)).mean(axis = None) / np.linalg.norm([z, y, x])**2, ((np.square(distances)).mean(axis = None) / (np.linalg.norm([z, y, x])**2))**0.5, max(distances)

def final_rotation_angle(gt_transform, founded_transform):
    """
    Returns final rotation angle between 
    the column vectors of gt transformation matrix
    and founded transformation matrix. ???
    """
def pad_to_shape(array, t_shape):
    """
    Returns the expanded volume to the specified target shape.
    """
    padding = [(0, t - s) for s,t in zip(array.shape, t_shape)]
    return np.pad(array, padding, mode = 'constant', constant_values = 0)

def show_volumes(v1, v2):
    plotter = vedo.Plotter()
    vedo_v1 = vedo.Volume(v1)
    vedo_v1.permute_axes(2,1,0)
    plotter.add(vedo_v1)
    vedo_v2 = vedo.Volume(v2)
    vedo_v2.permute_axes(2,1,0)
    plotter.add(vedo_v2)
    plotter.show(vedo_v1.alpha([0,0, 0.05, 0]).color('red'), vedo_v2.alpha([0,0, 0.05, 0]).color('blue'), axes = 1)
    plotter.remove(vedo_v1, vedo_v2)

def apply_transform(volume, matrix):
    linear_transform = vedo.LinearTransform()
    linear_transform.matrix = matrix
    #print(f'transform matrix = {matrix}')
    new_volume_shape = np.array(get_bndbox(volume.shape, matrix))
    #print(f'new volume shape (x, y, z) = {new_volume_shape}')
    
    vedo_volume = vedo.Volume(volume)
    vedo_volume.permute_axes(2,1,0)
    try:
        vedo_volume.apply_transform(linear_transform, fit = True, interpolation = 'linear')
    except:
        pass

    empty_arr = np.zeros(new_volume_shape)
    vedo_volume_new = vedo.Volume(empty_arr, origin = [0,0,0], dims = empty_arr.shape)
    vedo_volume_new.resample_data_from(vedo_volume)
    vedo_volume = vedo_volume_new
    vedo_volume.permute_axes(2,1,0)
    return vedo_volume.tonumpy().astype(np.float32)

def estimation_interpolation_error(volume, matrix):
    volume = volume.astype(np.float32)
    volume_transformed = apply_transform(volume, matrix)

    #show_volumes(volume_transformed, volume)

    linear_transform = vedo.LinearTransform()
    linear_transform.matrix = matrix
    volume_backtransformed = apply_transform(volume_transformed, linear_transform.invert().matrix)

    #print(volume.shape)
    #print(volume_backtransformed.shape)

    target_shape = np.maximum(volume.shape, volume_backtransformed.shape)
    #print(target_shape)
    padded_volume = pad_to_shape(volume, target_shape)
    padded_volume_backtransformed = pad_to_shape(volume_backtransformed, target_shape)
    #calculated_mse = mse(markup_volume, transformed_volume)
    
    #show_volumes(volume_backtransformed, volume)
    
    interp_mse = mse(padded_volume_backtransformed, padded_volume)
    interp_HaarPSI = 0 #HaarPSI3D(padded_volume_backtransformed, padded_volume)
    print(f'interp_mse = {interp_mse}')
    print(f'interp_HaarPSI = {interp_HaarPSI} ')
    #show_volumes(padded_volume_backtransformed, padded_volume)
    return interp_mse, interp_HaarPSI
def calculate_metrics(sample_result_folder_path,
                      markup_volume, transformed_volume,
                      gt_transform, error_transform, metric_volume, is_fail = False):
    """
    Calculates and writes metrics in json: intensity mse, geometry mse, geometry rmse, maximum deviation.
    """
    metrics_store = {}
    # try:
    #     interpolation_mse, interpolation_HaarPSI = estimation_interpolation_error(metric_volume, gt_transform)
    # except ValueError:
    #     interpolation_mse, interpolation_HaarPSI = 0.5, 0
    interpolation_mse, interpolation_HaarPSI = 0.5, 0

    if is_fail:
        metrics_store["interpolation_mse"] = interpolation_mse
        metrics_store["interpolation_HaarPSI"] = interpolation_HaarPSI
        metrics_store["MSE"] =  0.5
        metrics_store["norm_geometry_MSE"] =  0.2
        metrics_store["norm_geometry_rmse"] =  0.2
        metrics_store["geometry_rmse"] =  - 10
        metrics_store["maximum deviation of distances (from geometry MSE)"] = - 10
        metrics_store["normalized maximum deviation of distances (from geometry MSE)"] = 0.2
        metrics_store["HaarPSI"] = 0
        with open(f'{sample_result_folder_path}metrics.json', 'w', encoding='UTF-8') as f:
            json.dump(metrics_store, f)
        return 0

    #interpolation_mse, interpolation_HaarPSI = estimation_interpolation_error(metric_volume, gt_transform)
    metrics_store["interpolation_mse"] = float(interpolation_mse)
    metrics_store["interpolation_HaarPSI"] = float(interpolation_HaarPSI)

    try:
        calculated_mse = mse(markup_volume, transformed_volume)
    except:
        calculated_mse = 0.5
    
    calculated_norm_geometry_mse, norm_geometry_rmse, geometry_rmse, norm_max_dev, max_dev = geometry_mse(metric_volume, gt_transform, error_transform)

    if calculated_mse > 0.5:
        calculated_mse = 0.5

    if calculated_norm_geometry_mse > 0.2:
        calculated_norm_geometry_mse = 0.2
    
    if norm_geometry_rmse > 0.2:
        norm_geometry_rmse = 0.2

    
    print(f'MSE = {calculated_mse}')
    print(f'normalized_geometry_MSE = {calculated_norm_geometry_mse}')
    print(f'normalized_geometry_RMSE = {norm_geometry_rmse}')
    print(f'geometry_RMSE = {geometry_rmse}')
    print(f'normalized_max_deviation = {norm_max_dev}')
    print(f'max_deviation = {max_dev}')
    metrics_store["HaarPSI"] = 0 #HaarPSI3D(markup_volume, transformed_volume)
    metrics_store["MSE"] = float(calculated_mse)
    metrics_store["norm_geometry_MSE"] = float(calculated_norm_geometry_mse)
    metrics_store["norm_geometry_rmse"] = float(norm_geometry_rmse)
    metrics_store["geometry_rmse"] = float(geometry_rmse)
    metrics_store["normalized maximum deviation of distances (from geometry MSE)"] = float(norm_max_dev)
    metrics_store["maximum deviation of distances (from geometry MSE)"] = float(max_dev)
    print(f'HaarPSI = {metrics_store["HaarPSI"]}')

    with open(f'{sample_result_folder_path}metrics.json', 'w', encoding='UTF-8') as f:
        json.dump(metrics_store, f)
