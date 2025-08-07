import numpy as np
import os, os.path
import SimpleITK as sitk
import sys
import json
from PIL import Image
import registration_gui as rgui
from collections import Counter
import re

def inverse_affine_4x4(matrix_4x4):
    """
    Computes the inverse of a 4x4 affine transformation matrix.
    
    Args:
        matrix_4x4 (np.ndarray): Input 4x4 affine matrix
    
    Returns:
        np.ndarray: 4x4 inverse matrix
    
    Raises:
        ValueError: If matrix is singular or not affine
    """
    # Validate input
    if matrix_4x4.shape != (4, 4):
        raise ValueError("Input must be a 4x4 matrix")
    if not np.allclose(matrix_4x4[3, :], [0, 0, 0, 1]):
        raise ValueError("Last row must be [0, 0, 0, 1]")
    
    # Extract components
    A = matrix_4x4[:3, :3]
    t = matrix_4x4[:3, 3]
    
    try:
        # Compute inverse of linear transform
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        raise ValueError("Linear transform component is singular (cannot be inverted)")
    
    # Compute inverse translation
    t_inv = -A_inv @ t
    
    # Build inverse matrix
    inv_matrix = np.eye(4)
    inv_matrix[:3, :3] = A_inv
    inv_matrix[:3, 3] = t_inv
    return inv_matrix

def extract_uniform_scale_and_rotation(matrix_3x3):
    """
    Extracts uniform scale from a 3x3 matrix and returns the rotation matrix.
    
    Args:
        matrix_3x3 (np.ndarray): Input 3x3 transformation matrix (assumed: R * s).
        
    Returns:
        tuple: (scale, rotation_matrix), where:
            - scale (float): Uniform scale factor.
            - rotation_matrix (np.ndarray): 3x3 pure rotation matrix (no scale).
    """
    # Ensure input is a NumPy array
    if not isinstance(matrix_3x3, np.ndarray):
        matrix_3x3 = np.array(matrix_3x3, dtype=np.float64)
    
    # Compute scale as the average of column norms (since scale is uniform)
    scale = np.mean([np.linalg.norm(matrix_3x3[:, i]) for i in range(3)])
    
    # Remove scaling to get pure rotation matrix
    rotation_matrix = matrix_3x3 / scale
    
    return scale, rotation_matrix

def read_from_imagej(file_path):
    """
    Reads a text file from the given path and returns two lists of flattened landmark coords.
    
    Args:
        file_path (str): Path to the text file to be read.
        
    Returns:
        flattened list with markup point coords, flattened list with corresponding test point coords.
    """
    markup = []
    test = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                # Remove any leading/trailing whitespace (including newline characters)
                stripped_line = line.strip()
                # Split the line by tab characters
                if stripped_line:  # Only process non-empty lines
                    parts = stripped_line.split('\t')
                    for i in range(3):
                        markup.append(float(parts[i]))
                        test.append(float(parts[i+8]))
    except FileNotFoundError:
        print(f"Error: The file at path '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
    return markup, test

def rotation_matrix_to_quaternion(R):
    """
    Convert a 3x3 rotation matrix to a quaternion.
    
    Parameters:
    R (numpy.ndarray): 3x3 rotation matrix
    
    Returns:
    numpy.ndarray: (qx, qy, qz, qw) quaternion
    """
    q = np.empty(4)
    tr = np.trace(R)
    
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        q[3] = 0.25 * S
        q[0] = (R[2,1] - R[1,2]) / S
        q[1] = (R[0,2] - R[2,0]) / S
        q[2] = (R[1,0] - R[0,1]) / S
    elif (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
        S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
        q[3] = (R[2,1] - R[1,2]) / S
        q[0] = 0.25 * S
        q[1] = (R[0,1] + R[1,0]) / S
        q[2] = (R[0,2] + R[2,0]) / S
    elif R[1,1] > R[2,2]:
        S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
        q[3] = (R[0,2] - R[2,0]) / S
        q[0] = (R[0,1] + R[1,0]) / S
        q[1] = 0.25 * S
        q[2] = (R[1,2] + R[2,1]) / S
    else:
        S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
        q[3] = (R[1,0] - R[0,1]) / S
        q[0] = (R[0,2] + R[2,0]) / S
        q[1] = (R[1,2] + R[2,1]) / S
        q[2] = 0.25 * S
    
    return q

def rotations_to_versor(theta_x, theta_y, theta_z, order='XYZ'):
    """
    Convert three principal axis rotations to a versor (quaternion) for SimpleITK.
    
    Parameters:
        theta_x, theta_y, theta_z: Rotation angles in radians
        order: Rotation order string (e.g., 'XYZ', 'ZYX', 'XZY', etc.)
    
    Returns:
        List [x, y, z, w] for SimpleITK's SetRotation()
    """
    # Compute half-angles in radians
    hx, hy, hz = theta_x*np.pi / 360.0, theta_y*np.pi / 360.0, theta_z*np.pi / 360.0
    
    # Create individual quaternions
    qx = np.array([np.cos(hx), np.sin(hx), 0.0, 0.0])  # X-axis
    qy = np.array([np.cos(hy), 0.0, np.sin(hy), 0.0])  # Y-axis
    qz = np.array([np.cos(hz), 0.0, 0.0, np.sin(hz)])  # Z-axis
    
    # Dictionary of quaternions for each axis
    quats = {'X': qx, 'Y': qy, 'Z': qz}
    
    # Validate order parameter
    order = order.upper()
    if len(order) != 3 or any(c not in 'XYZ' for c in order):
        raise ValueError("Order must be a 3-character string containing only X, Y, Z")
    
    # Multiply quaternions in specified order (right-to-left)
    q_total = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
    for axis in reversed(order):
        q = quats[axis]
        # Quaternion multiplication (Hamilton product)
        w = q_total[0]*q[0] - q_total[1]*q[1] - q_total[2]*q[2] - q_total[3]*q[3]
        x = q_total[0]*q[1] + q_total[1]*q[0] + q_total[2]*q[3] - q_total[3]*q[2]
        y = q_total[0]*q[2] - q_total[1]*q[3] + q_total[2]*q[0] + q_total[3]*q[1]
        z = q_total[0]*q[3] + q_total[1]*q[2] - q_total[2]*q[1] + q_total[3]*q[0]
        q_total = np.array([w, x, y, z])
    
    # Return in SimpleITK order (x, y, z, w)
    return [q_total[1], q_total[2], q_total[3], q_total[0]]

def SITK3DReg(markup_volume:np.ndarray,test_volume:np.ndarray, metrics_folder_path, params, initial_matrix = np.array([[0]])):
    #parameters

    TrDict = {
        "Affine3D": sitk.AffineTransform(3),
        "Similarity": sitk.Similarity3DTransform()
    }

    # print("Fixed image size = "+str(np.shape(markup_volume)))
    # print("Moving image size = "+str(np.shape(test_volume)))

    moving_image = sitk.GetImageFromArray(test_volume)
    fixed_image = sitk.GetImageFromArray(markup_volume)
    HistMatching = params["HistMatching"]
    if HistMatching == True:
        hist_match = sitk.HistogramMatchingImageFilter()
        hist_match.SetNumberOfHistogramLevels(256)
        hist_match.SetNumberOfMatchPoints(10)
        hist_match.SetThresholdAtMeanIntensity(True)
        moving_image = hist_match.Execute(moving_image, fixed_image)
    del markup_volume, test_volume

    print("Fixed shape: ",str(fixed_image.GetSize()))
    print("Moving shape: ",str(moving_image.GetSize()))
    print("Fixed spacing, origin, direction: ", fixed_image.GetSpacing(), fixed_image.GetOrigin(), fixed_image.GetDirection())
    print("Moving spacing, origin, direction: ", moving_image.GetSpacing(), moving_image.GetOrigin(), moving_image.GetDirection())

    #setting initial transform
    InitialTransformType = params["InitalTransform"]
    initTrDict = {
        "GEOMETRY": sitk.CenteredTransformInitializerFilter.GEOMETRY,
        "MOMENTS": sitk.CenteredTransformInitializerFilter.MOMENTS
    }
    
    if InitialTransformType=="MANUAL":
        InitialTransformParams = params["InitialTransformParams"]
        angles = InitialTransformParams["rotation"]
        versor = rotations_to_versor(angles[0], angles[1], angles[2])
        initial_transform = sitk.CenteredTransformInitializer(
                fixed_image,
                moving_image,
                TrDict[params["TransformType"]],
                sitk.CenteredTransformInitializerFilter.GEOMETRY
        ) 
        initial_transform.SetScale(InitialTransformParams["scale"])
        initial_transform.SetRotation(versor)
        print(initial_transform.GetTranslation())
        print(initial_transform.GetVersor())
        if InitialTransformParams["InitialTranslationOption"] == "MOMENTS":
            moments_initializer = sitk.CenteredTransformInitializer(
                fixed_image,
                moving_image,
                TrDict[params["TransformType"]],
                sitk.CenteredTransformInitializerFilter.MOMENTS
            )
            init_translation = moments_initializer.GetTranslation()
            initial_transform.SetTranslation(init_translation)
        elif InitialTransformParams["InitialTranslationOption"]=="MANUAL":
            init_translation = InitialTransformParams["translation"]
            initial_transform.SetTranslation(init_translation)
        print(initial_transform.GetTranslation())
    elif InitialTransformType == "MATRIX":
        initial_transform = TrDict[params["TransformType"]]
        try:
            scale, matrix3x3 = extract_uniform_scale_and_rotation(initial_matrix[:3,:3])
            versor = rotation_matrix_to_quaternion(matrix3x3)
            translation  = initial_matrix[:3,3]
            initial_transform.SetScale(scale)
            initial_transform.SetRotation(versor)
            initial_transform.SetTranslation(translation.tolist())
        except:
            raise ValueError("Wrong initial matrix input")
    elif InitialTransformType == "POINTS":
        # fixed_landmarks = [153.0    ,268.0, 188.0   ,139.0, 279.0   ,218.0  ,251.0  ,317.0  ,246.0  ,137.0, 246.0,  272.0]    
        # moving_landmarks = [261.0   ,208.0, 115.0,250.0,    190.0,  147.0, 204.0,   300.0,  184.0 ,292.0,   188.0,  201.0]
        fixed_landmarks,moving_landmarks = read_from_imagej(params["imagej_landmark_coords_file_path"])
        initial_transform = sitk.LandmarkBasedTransformInitializer(
            TrDict[params["TransformType"]],
            fixed_landmarks,
            moving_landmarks
        )
        print(initial_transform.GetMatrix())
    else:
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_image,
            moving_image,
            TrDict[params["TransformType"]],
            initTrDict[InitialTransformType],
        )

    registration_method = sitk.ImageRegistrationMethod()

    if params["InitialTransformViewer"] == True:
        #sitk.ImageViewer.SetGlobalDefaultApplication(params["imagej_exe_path"])
        image_viewer = sitk.ImageViewer()
        image_viewer.SetApplication(params["imagej_exe_path"])
        resampled_image = sitk.Resample(
        moving_image, 
        fixed_image,  # Reference image (defines output space)
        initial_transform, 
        sitk.sitkLinear,  # Interpolation method (Linear for most cases)
        0.0,  # Default pixel value for out-of-range areas
        moving_image.GetPixelID()  # Preserve pixel type (e.g., sitk.sitkFloat32)
        )
        image_viewer.Execute(sitk.CheckerBoard(fixed_image,resampled_image))


    SamplingMethodDict = {
        "Random": registration_method.RANDOM,
        "RegularGrid": registration_method.REGULAR
    }
    LearningRateDict = {
        "Once" : sitk.ImageRegistrationMethod.Once,
        "EachIteration" : sitk.ImageRegistrationMethod.EachIteration,
        "Never" : sitk.ImageRegistrationMethod.Never
    }

    # Similarity metric settings.
    if (params["Metric"]=="MattesMutualInformation"):
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=params["MMI_MetricNumberOfHistogramBins"])
        registration_method.SetMetricSamplingStrategy(SamplingMethodDict[params["MetricSamplingStrategy"]])
        registration_method.SetMetricSamplingPercentage(params["MetricSamplingPercentage"])

    if (params["Interpolator"]=="Linear"):
        registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    if (params["Optimizer"]=="RegularStepGradientDescent"):
        registration_method.SetOptimizerAsRegularStepGradientDescent(
            learningRate=params["LearningRate"],
            numberOfIterations=params["NumberOfIterations"],
            minStep = params["minStep"],
            relaxationFactor=params["relaxationFactor"],
            gradientMagnitudeTolerance=params["gradientMagnitudeTolerance"],
            estimateLearningRate = LearningRateDict[params["Estimate_learning_rate_option"]]
        )
    elif params["Optimizer"]=="GradientDescent":
        registration_method.SetOptimizerAsGradientDescent(
            learningRate=params["LearningRate"],
            numberOfIterations=params["NumberOfIterations"],
            estimateLearningRate = LearningRateDict[params["Estimate_learning_rate_option"]],
            convergenceMinimumValue=params["ConvergenceMinimumValue"],
            convergenceWindowSize=params["ConvergenceWindowSize"]
        )
    registration_method.SetOptimizerScalesFromPhysicalShift()

    #registration_method.SetOptimizerWeights([1.0,1.0,1.0,1.0,1.0,0.0,10])

    # Setup for the multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = params["ShrinkFactors"])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas = params["SmoothingSigmas"])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    registration_method.AddCommand(sitk.sitkStartEvent, rgui.start_plot(metrics_folder_path))
    registration_method.AddCommand(sitk.sitkEndEvent, rgui.end_plot)
    registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, rgui.update_multires_iterations)
    registration_method.AddCommand(sitk.sitkIterationEvent, lambda: rgui.plot_values(registration_method))

    try:
        final_transform = registration_method.Execute(fixed_image, moving_image)
    except RuntimeError as e:
         print(f"Error: {e}")

    #reason optimization terminated.
    print("Final metric value: {0}".format(registration_method.GetMetricValue()))
    print(
        "Optimizer's stopping condition, {0}".format(
            registration_method.GetOptimizerStopConditionDescription()
        )
    )

    # #moving image
    # moving_resampled = sitk.Resample(
    #     moving_image,
    #     fixed_image,
    #     final_transform,
    #     sitk.sitkLinear,
    #     0.0,
    #     moving_image.GetPixelID(),
    # )
    # markup_volume_moved=sitk.GetArrayFromImage(moving_resampled)
    # del moving_resampled

    del moving_image, fixed_image

    #Markup to test matrix calculation
    inv_rotation_matrix = np.array(list(final_transform.GetNthTransform(0).GetInverse().GetMatrix())).reshape(3, 3)
    inv_translation = np.array(final_transform.GetNthTransform(0).GetInverse().GetTranslation())
    inv_center = np.array(final_transform.GetNthTransform(0).GetInverse().GetCenter())

    # Adjust the translation for the origin-based matrix
    inv_adjusted_translation = inv_translation + inv_center - inv_rotation_matrix @ inv_center
    inv_transform = np.column_stack((inv_rotation_matrix, inv_adjusted_translation)).astype(np.float32)
    inv_transform = inv_transform.tolist()
    inv_transform.append([0,0,0,1])

     #Test to markup matrix calculation
    rotation_matrix = np.array(list(final_transform.GetNthTransform(0).GetMatrix())).reshape(3, 3)
    translation = np.array(final_transform.GetNthTransform(0).GetTranslation())
    center = np.array(final_transform.GetNthTransform(0).GetCenter())

    # Adjust the translation for the origin-based matrix
    adjusted_translation = translation + center - rotation_matrix @ center
    transform = np.column_stack((rotation_matrix, adjusted_translation)).astype(np.float32)
    transform = transform.tolist()
    transform.append([0,0,0,1])
    
    #saving transformation matrices
    with open(f'{os.path.join(metrics_folder_path,"matrices.json")}', 'w', encoding='UTF-8') as f:
        json.dump({"matrix":inv_transform, "inv_matrix":transform}, f)
    np.save(f'{os.path.join(metrics_folder_path,"transformation_estimated.npy")}', np.array(inv_transform))
    np.save(f'{os.path.join(metrics_folder_path,"inv_transformation_estimated.npy")}', np.array(transform))
    return(1)

def load_volume(files, size_z, size_y, size_x):#, sigma = None):
    """
    Returns npy array with volume.
    """
    volume = np.zeros((size_z, size_y, size_x))
    for slice_number, file in enumerate(files):
        if slice_number%100 == 0:
            print(f'Load slice number {slice_number}/{len(files)}', end = "\r")
        volume[slice_number] = np.array(Image.open(file))
    print(f'Load slice number {len(files)}/{len(files)}')
    return volume

def natural_sort_key(s):
    """
    Key function for natural sorting of strings containing numbers.
    """
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split('([0-9]+)', s)]

def get_most_common_extension(files):
    """
    Returns the most common file extension in the list of files.
    """
    extensions = [os.path.splitext(f)[1].lower() for f in files]
    if not extensions:
        return None
    return Counter(extensions).most_common(1)[0][0]

def load_volume_from_dir(in_path):
    """
    Returns npy array with volume by given directory path.
    Automatically detects and uses the most common file extension in the directory.
    Handles different naming conventions (0.ext, 000.ext, etc.)
    """
    # Get all files in directory
    all_files = [f for f in os.listdir(in_path) if os.path.isfile(os.path.join(in_path, f))]
    
    if not all_files:
        raise FileNotFoundError("No files found in the directory")
    
    # Find most common extension
    common_ext = get_most_common_extension(all_files)
    if not common_ext:
        raise ValueError("No files with extensions found in the directory (maybe DICOM).")
    
    print(f"Using files with extension: {common_ext}")
    
    # Filter files by most common extension
    files = [f for f in all_files if os.path.splitext(f)[1].lower() == common_ext]
    
    if not files:
        raise ValueError(f"No files with extension {common_ext} found")
    
    # Natural sort files
    files.sort(key=natural_sort_key)
    
    # Get full paths
    file_paths = [os.path.join(in_path, x) for x in files]
    print(file_paths)
    # Verify first file is a valid image
    try:
        with Image.open(file_paths[0]) as img:
            size_y, size_x = np.array(img).shape
            
    except Exception as e:
        print(f"Error opening image file '{file_paths[0]}': {e}")
        print(f"Note: Files with extension {common_ext} must be valid image files")
        sys.exit(1)
    print(f'size_y, size_x = {size_y, size_x}')
    size_z = len(files)
    return load_volume(file_paths, size_z, size_y, size_x)

def numpy_parser(num):
    return np.float64(num)

if __name__ == "__main__":
    markup_volume_path  = sys.argv[1]
    test_volume_path = sys.argv[2]
    processing_folder_path = sys.argv[3]
    metrics_folder_path = sys.argv[4]
    initial_transform_matrix_path = sys.argv[5]
    alg_params_json = sys.argv[6]

    with open(alg_params_json, 'r', encoding='UTF-8') as json_file:
        alg_params = json.load(json_file)

    print(f"Loading markup from path: {markup_volume_path}")
    markup_volume = load_volume_from_dir(markup_volume_path)
    print(f"Loading test from path: {test_volume_path}")
    test_volume = load_volume_from_dir(test_volume_path)
    print("run_SITK")
    if initial_transform_matrix_path[-10:] != "none_given":
        with open(initial_transform_matrix_path, 'r', encoding='UTF-8') as json_file:
            matrix = np.array(json.load(json_file, parse_float= numpy_parser, parse_int= numpy_parser )["matrix"],dtype=np.float64)
        print(f'Initial matrix reading from given path: \n{matrix}')
        #inv_matrix = inverse_affine_4x4(np.array(matrix))
        #print(inv_matrix)
        is_good_result = SITK3DReg(markup_volume, test_volume, metrics_folder_path, alg_params, initial_matrix=matrix)
    else:
        is_good_result = SITK3DReg(markup_volume, test_volume, metrics_folder_path, alg_params)
    alg_out_json = f'{processing_folder_path}/alg_out_json.json'
    alg_result = {'output' : is_good_result}

    with open(alg_out_json, "w", encoding="utf-8") as file:
        json.dump(alg_result, file)

    print(f'is_good_result = {is_good_result}')

