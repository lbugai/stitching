"""
Main pipeline of testing stitching volumes methods.
"""
import json
import sys
import os
import shutil
from datetime import datetime
import numpy as np
from cv2 import imwrite
import metrics
import subprocess
from timer import timer
import tracemalloc
import psutil
import time
from joblib import Parallel, delayed
from data_loader import load_volume_from_dir, get_volume_shape

def make_folder(folder_path, delete_existing = True):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    elif delete_existing: 
        shutil.rmtree(folder_path)
        os.mkdir(folder_path)

def timestamped_config_safe_copy(source, alg,  alg_source, target_dir):
    base_name = os.path.basename(source)
    alg_base_name = os.path.basename(alg_source)
    target_path = os.path.join(target_dir,f'{alg}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_configs')
    if os.path.exists(target_path):
        counter = 1
        while True:
            new_path = target_path + f'{counter:03d}'
            if not os.path.exists(new_path):
                break
            counter += 1
        target_path = new_path
    make_folder(target_path)
    shutil.copy(source, os.path.join(target_path,base_name))
    shutil.copy(alg_source, os.path.join(target_path,alg_base_name))
    pass

def crop_and_pad_to_shape(array, t_shape):
    """
    Returns croped and expanded volume, matching the specified target shape.
    """
    slices = tuple(slice(0, min(a, t)) for a, t in zip(array.shape, t_shape))
    cropped = array[slices]
    padding = [(0, max(0, t - s)) for s, t in zip(cropped.shape, t_shape)]
    return np.pad(cropped, padding, mode='constant', constant_values = 0)

if __name__ == '__main__':
    config_path = sys.argv[1]
    with open(config_path, 'r', encoding='UTF-8') as json_file:
        config = json.load(json_file)
    python_venv = sys.executable
    alg_interpreter_path = config["alg_interpreter_path"]
    main_folder = config["path_to_main_folder"]
    exp_path = os.path.join(main_folder,f'exp{config["exp"]}/')
    transformed_volumes_path = os.path.join(main_folder, f'exp{config["exp"]}', "transformed_volumes/")
    VolumeLoadingMode = config["VolumeLoadingMode"]
    minimize_padding = config["minimize_padding"]
    algorithm = config["algorithm_name"]
    algorithm_config_path = config["algorithm_execution_parameters_path"]
    algorithm_path = config["algorithm_executable_path"]
    registered_volumes_writing = config["registered_volumes_writing"]
    calculate_metrics = config["calculate_metrics"]
    timestamped_config_safe_copy(config_path, algorithm, algorithm_config_path,  exp_path)
    lists_of_paths_to_test_volumes = []
    markup_path_list = []
    markup_package_list = []
    path_to_gt_matrix_list = []
    path_to_init_transform_matrix_list = []

    if VolumeLoadingMode == "TwoVolumes":
        markup_path_list.append(config["path_to_markup"])
        path_to_moving = config["path_to_moving"]
        markup_package_list.append("custom_markup")
        lists_of_paths_to_test_volumes.append([path_to_moving])
        path_to_init_transform_matrix_list.append(config["path_to_inital_transform_matrix_json"])
        if calculate_metrics:
            path_to_gt_matrix_list.append(config["path_to_gt_matrix_json"])
    else:
        print(f'Wrong VolumeLoadingMode. Select TwoVolumes in {config_path}')
        exit()
    
    print(f'Selected algorithm is {algorithm}.')
    processing_folder_path = os.path.join(exp_path, 'processing/')
    make_folder(processing_folder_path)
    alg_processing_folder_path = os.path.join(exp_path, 'processing', algorithm + '/')
    make_folder(alg_processing_folder_path)
    stitching_results_folder_path = os.path.join(exp_path, 'stitching_results/')
    make_folder(stitching_results_folder_path, delete_existing=False) 
    alg_stitching_results_folder_path = os.path.join(exp_path, 'stitching_results', algorithm + '/')
    make_folder(alg_stitching_results_folder_path) 

    for index,(markup_package,current_markup_path, paths_to_test_volumes) in enumerate(zip(markup_package_list, markup_path_list,lists_of_paths_to_test_volumes)):
        package_processing_folder_path = os.path.join(exp_path, 'processing', algorithm,  markup_package + '/')
        make_folder(package_processing_folder_path)
        package_stitching_results_folder_path = os.path.join(exp_path, 'stitching_results', algorithm, markup_package + '/')
        make_folder(package_stitching_results_folder_path)
        number_of_samples = len(paths_to_test_volumes)
        for i in range(number_of_samples):
            print("--"*40)
            sample_processing_folder_path = package_processing_folder_path
            print('sample_processing_folder_path = ', sample_processing_folder_path)
            os.makedirs(sample_processing_folder_path, exist_ok=True)
            sample_result_folder_path = package_stitching_results_folder_path
            print('sample_result_folder_path = ', sample_result_folder_path)
            os.makedirs(sample_result_folder_path, exist_ok=True)
            markup_path = current_markup_path
            repeat = 1
            if calculate_metrics:
                with open(path_to_gt_matrix_list[index], 'r', encoding='UTF-8') as json_file:
                    metric_matrix = json.load(json_file)["gt_matrix"]
                metric_matrix = np.array(metric_matrix)

            make_folder(f'{sample_result_folder_path}/metrics/') 
            for j in range(repeat):
                os.mkdir(f'{sample_result_folder_path}/metrics/{j}/')
                sample_metrics_folder_path = f'{sample_result_folder_path}/metrics/{j}/'
                info = {
                    "alg_path" : algorithm_path,
                    "markup_path" : markup_path,
                    "test_volume_path" : paths_to_test_volumes[i],
                    "sample_processing_folder_path" : sample_processing_folder_path,
                    "initial_tr_matrix_path" : path_to_init_transform_matrix_list[index],
                    "alg_config_path" : algorithm_config_path
                }
                tracemalloc.start()
                with timer(unit = "ms") as t:
                    if alg_interpreter_path == "":
                        subpr_arg_seqence = [algorithm_path,markup_path, paths_to_test_volumes[i],sample_processing_folder_path, sample_metrics_folder_path, path_to_init_transform_matrix_list[index], algorithm_config_path]
                    else:
                        subpr_arg_seqence = [alg_interpreter_path, algorithm_path,markup_path, paths_to_test_volumes[i], sample_processing_folder_path, sample_metrics_folder_path, path_to_init_transform_matrix_list[index], algorithm_config_path]
                        info.update({"alg_interpreter_path" : alg_interpreter_path})
                    proc = subprocess.Popen(subpr_arg_seqence, stderr=subprocess.PIPE, text=True)
                    p=psutil.Process(proc.pid)
                    children = p.children(recursive=True)
                    if children:
                        p = children[0]
                    peak_mem = 0
                    while proc.poll() is None:
                        try:
                            mem = p.memory_info().rss/(10**6)
                            peak_mem = max(peak_mem, mem)
                            time.sleep(0.01)
                        except psutil.NoSuchProcess:
                            break
                    stdout, stderr = proc.communicate()
                alg_time = t.elapse/1000
                current, peak = tracemalloc.get_traced_memory()
                print(f"{algorithm} timer = {alg_time:.6f} seconds")
                print(f"peak {algorithm} memory = {peak_mem:.4f} MB")
                time_mem_params = {"Time sec":alg_time,"Overall_peak_mem_MB":peak_mem,"Peak_memory_of_pipeline_MB":peak/10**6,"markup":markup_path,"test":paths_to_test_volumes[i]}
                time_mem_params.update(info)
                try:
                    with open(algorithm_config_path, 'r', encoding='UTF-8') as json_file:
                        alg_params = json.load(json_file)
                    time_mem_params.update(alg_params)
                except:
                    print(f"Unable to read {algorithm} json configuration file from:\n{algorithm_config_path}")
                    pass
                time_json = f'{sample_result_folder_path}/metrics/{j}/time_mem.json'
                with open(time_json, "w", encoding="utf-8") as file:
                    json.dump(time_mem_params, file)
                print(f'result_stdout = {stdout}')
                print(f'result_stderr = {stderr}')

                try:
                    alg_out_json = f'{sample_processing_folder_path}/alg_out_json.json'
                    with open(alg_out_json, 'r', encoding='UTF-8') as json_file:
                        alg_out = json.load(json_file)
                    print(f'alg_out = {alg_out}')
                    if alg_out["output"] == 0:
                        print(f'Failed to estimate transform with {algorithm}')
                        if calculate_metrics :
                            metric_volume = load_volume_from_dir(markup_path)
                            metrics.calculate_metrics(f'{sample_result_folder_path}/metrics/{j}/',
                                                0, 0,
                                                metric_matrix, 0, metric_volume, is_fail=True)
                            del metric_volume
                        continue
                except:
                    if calculate_metrics :
                        metric_volume = load_volume_from_dir(markup_path)
                        metrics.calculate_metrics(f'{sample_result_folder_path}/metrics/{j}/',
                                            0, 0,
                                            metric_matrix, 0, metric_volume, is_fail=True)
                        del metric_volume
                    print(f'Stackoverflow while calculating {algorithm} matrix')
                    continue

                with open(f'{os.path.join(sample_metrics_folder_path,"matrices.json")}', 'r', encoding='UTF-8') as json_file:
                    transformation_estimated = json.load(json_file)["matrix"]
                transformation_estimated = np.array(transformation_estimated)
                np.set_printoptions(precision = 8, suppress=True)
                if calculate_metrics :
                    print("GT matrix")
                    print(metric_matrix)
                    print("Deviation matrix")
                    deviation_matrix = transformation_estimated @ metric_matrix
                    print(deviation_matrix)
                print(f"Estimated by {algorithm} matrix")
                print(transformation_estimated)
                x_shape, y_shape, z_shape = get_volume_shape(markup_path)[::-1]
                if registered_volumes_writing:
                    make_folder(f'{sample_result_folder_path}/test_transformed_{j}/')
                    result = subprocess.run([python_venv, "sample_creator.py",
                                                paths_to_test_volumes[i], f'{sample_metrics_folder_path}/matrices.json' ,
                                                f'{sample_result_folder_path}/test_transformed_{j}/',
                                                str(minimize_padding),
                                                str(x_shape), str(y_shape), str(z_shape)],
                                                stderr=subprocess.PIPE, text = True)
                    print(f'result.stderr = {result.stderr}') 
                    if result.stderr:
                        if calculate_metrics :
                            metric_volume = load_volume_from_dir(markup_path)
                            metrics.calculate_metrics(f'{sample_result_folder_path}/metrics/{j}/',
                                                0, 0,
                                                metric_matrix, deviation_matrix, metric_volume, is_fail=True)
                            del metric_volume
                        print('Stackoverflow while creating markup transformed volume')
                        continue
                    try:
                        test_transformed_volume = load_volume_from_dir(f'{sample_result_folder_path}/test_transformed_{j}')
                        if minimize_padding == True:
                            padded_test = test_transformed_volume.astype(np.float32)
                        else:
                            with open(f'{os.path.join(sample_metrics_folder_path,"origin_negative_offset.json")}', 'r', encoding='UTF-8') as json_file:
                                origin_offset = json.load(json_file)["origin_offset"]
                            markup_shape = get_volume_shape(markup_path)
                            target_shape = np.maximum(get_volume_shape(f'{sample_result_folder_path}/test_transformed_{j}'), np.array([markup_shape[0]+origin_offset[2],markup_shape[1]+origin_offset[1],markup_shape[2]+origin_offset[0]]))
                            padded_test = crop_and_pad_to_shape(test_transformed_volume, target_shape).astype(np.float32)
                        del test_transformed_volume
                        print('Saving padded_test...')
                        make_folder(f'{sample_result_folder_path}/padded_test_{j}/')
                        for k, Slice in enumerate(padded_test):
                            save_path = os.path.join(f'{sample_result_folder_path}/padded_test_{j}/{k:04d}.tif')
                            imwrite(save_path, Slice) # from Re
                    except:
                        print('Stackoverflow while creating padded test volume')
                        continue

                    try:
                        markup_volume = load_volume_from_dir(markup_path)
                        if minimize_padding == True:
                            padded_markup = markup_volume.astype(np.float32)
                        else:
                            padded_markup = np.zeros(list(target_shape)).astype(np.float32)
                            padded_markup[origin_offset[2]:origin_offset[2]+markup_volume.shape[0],origin_offset[1]:origin_offset[1]+markup_volume.shape[1],origin_offset[0]:origin_offset[0]+markup_volume.shape[2]] = markup_volume
                        del markup_volume
                        print('Saving padded_markup...')
                        make_folder(f'{sample_result_folder_path}/padded_markup_{j}/')
                        for k, Slice in enumerate(padded_markup):
                            save_path = os.path.join(f'{sample_result_folder_path}/padded_markup_{j}/{k:04d}.tif')
                            imwrite(save_path, Slice) # from cv2 
                    except:
                        print('Stackoverflow while creating padded markup volume')
                        continue

                    try:
                        if calculate_metrics :
                            metric_volume = load_volume_from_dir(markup_path)
                            metrics.calculate_metrics(f'{sample_result_folder_path}/metrics/{j}/',
                                                    padded_markup, padded_test,
                                                    metric_matrix, deviation_matrix, metric_volume)
                            del metric_volume
                    except Exception as e:
                            print(f'Error while calculating metrics:\n{e}')
                    try:
                        print('Saving join...')
                        np.clip(padded_markup, a_min=0, a_max=None, out=padded_markup)
                        np.clip(padded_test, a_min=0, a_max=None, out=padded_test)

                        # max_value = percentile_across_arrays([volume1,volume2], 99)
                        # print('99-percentile = ', max_value)
                        max_value1 = np.percentile(padded_markup, 99)
                        max_value2 = np.percentile(padded_test, 99)
                        print('99-percentile1 = ', max_value1, '99-percentile2 = ', max_value2 )

                        uint_volume1 =  np.floor(255 * padded_markup / max_value1)
                        del padded_markup
                        uint_volume2 = np.floor( 255 * padded_test/ max_value2)
                        del padded_test

                        np.clip(uint_volume1, a_min=None, a_max=255, out=uint_volume1)
                        np.clip(uint_volume2, a_min=None, a_max=255, out=uint_volume2)

                        def make_color_slice(slice_number, uint_volume1, uint_volume2, save_path):
                            color_slice = np.stack([uint_volume2[slice_number],uint_volume2[slice_number],uint_volume1[slice_number]],axis=-1).astype(np.uint8)
                            save_path_slice = os.path.join(f'{save_path}/{slice_number:04d}.tif')
                            imwrite(save_path_slice, color_slice)
                        save_path = os.path.join(f'{sample_result_folder_path}', 'colored_joining_volume')
                        make_folder(save_path)
                        Parallel(n_jobs=-2, prefer="threads")(delayed(make_color_slice)(slice_number, uint_volume1, uint_volume2, save_path) for slice_number in range(uint_volume2.shape[0]))
                    except:
                        print('Stackoverflow while creating joined volume')
