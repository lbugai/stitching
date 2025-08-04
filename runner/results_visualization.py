"""
Metric graphs. 
"""
import json
import sys
import os
import shutil
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

def sortkey(folder_name):
    """
    Sorting key function.
    """
    return int(folder_name.rsplit("_")[-1])

def GetMetric(path_to_cases, metric):
    cases = sorted([x for x in os.listdir(path_to_cases)if '.json' not in x],
                                                key = sortkey)
    metric_list = []
    for case in cases:
        case_path = os.path.join(path_to_cases, case)
        metrics_path = os.path.join(case_path, 'metrics/')
        sum_metric_value = 0
        repeat_number = len(os.listdir(metrics_path))
        for num in range(repeat_number):
            metrics_json_path = os.path.join(metrics_path, str(num), 'metrics.json')
            with open(metrics_json_path, 'r', encoding='UTF-8') as json_file:
                metric_value = json.load(json_file)[metric]
            sum_metric_value += metric_value
        avg_metric_value = sum_metric_value / repeat_number
        metric_list.append(avg_metric_value)
    return metric_list

def make_folder(folder_path, keep_existing = False):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    elif keep_existing == False:
        shutil.rmtree(folder_path)
        os.mkdir(folder_path)

def SaveGraph2D(save_path, group_path, gr_params, MetricsList):
    QualityMetricsList = [x for x in MetricsList if 'interpolation' not in MetricsList]
    v1_params = gr_params["variation1"]
    v1_values = np.linspace(v1_params["left_border"], v1_params["right_border"], v1_params["points_amount"])
    MetricStore = {}
    for metric in QualityMetricsList:
        MetricStore[metric] = []
        MetricStore[metric] = GetMetric(group_path, metric)
        if metric == 'MSE':
            # plt.ylim(0,0.5)
            MetricStore['interpolation_mse'] = GetMetric(group_path, 'interpolation_mse')
            plt.plot(v1_values, MetricStore['interpolation_mse'], color = 'green')
        elif metric == 'HaarPSI':
            MetricStore['interpolation_HaarPSI'] = GetMetric(group_path, 'interpolation_HaarPSI')
            plt.plot(v1_values, MetricStore['interpolation_HaarPSI'], color = 'green')
        # if metric == 'norm_geometry_rmse' or metric == 'norm_geometry_MSE':
        #                     plt.ylim(0,0.2)
        plt.grid()
        plt.xlabel(gr_params["variation1"]["param_name"])
        plt.ylabel(metric)
        plt.title(f'{metric}. {algorithm}. {object_name}')
        plt.scatter(v1_values, MetricStore[metric])
        plt.plot(v1_values, MetricStore[metric])
        plt.savefig(f'{save_path}/{metric}.png')
        plt.show()

def SaveData2D(save_path, group_path, gr_params, MetricsList):
    QualityMetricsList = [x for x in MetricsList if 'interpolation' not in MetricsList]
    v1_params = gr_params["variation1"]
    v1_values = np.linspace(v1_params["left_border"], v1_params["right_border"], v1_params["points_amount"])
    MetricStore = {}
    group_metrics = pd.DataFrame(index=v1_values)
    for metric in QualityMetricsList:
        MetricStore[metric] = []
        MetricStore[metric] = GetMetric(group_path, metric)
        group_metrics[metric] = MetricStore[metric]
        if metric == 'MSE':
            group_metrics['interpolation_mse'] = GetMetric(group_path, 'interpolation_mse')
        elif metric == 'HaarPSI':
            group_metrics['interpolation_HaarPSI'] = GetMetric(group_path, 'interpolation_HaarPSI')
        
    group_metrics.to_csv(save_path+'/group.csv', encoding='UTF-8', index=True)
    print(group_metrics)

if __name__ == '__main__':
    config_path = sys.argv[1]
    generator_config_path = sys.argv[2]

    with open(config_path, 'r', encoding='UTF-8') as json_file:
        config = json.load(json_file)
    main_folder = config["path_to_main_folder"]
    exp_path = os.path.join(main_folder,f'exp{config["exp"]}/')
    stitching_results_path = os.path.join(main_folder, f'exp{config["exp"]}',"stitching_results/")
    Testing_group_index_list = config["Testing_group_list"]
    MetricsList = config["SelectedVisualizedMetricsList"]

    with open(generator_config_path, 'r', encoding='UTF-8') as json_file:
        generator_config = json.load(json_file)

    graphs_save_path = os.path.join(exp_path, 'metrics_graphs')
    make_folder(graphs_save_path, keep_existing=True)

    algorithms = os.listdir(stitching_results_path)
    for algorithm in algorithms:
        algorithm_path = os.path.join(stitching_results_path, algorithm)
        Testing_group_list = []
        if len(Testing_group_index_list) == 0:
            for name in os.listdir(algorithm_path):
                if name[0:2]=='gr':
                    Testing_group_list.append(name)
        else:
            for name in os.listdir(algorithm_path):
                if int(name[2]) in Testing_group_index_list:
                    Testing_group_list.append(name)
        Testing_group_list = sorted(Testing_group_list)
        alg_save_path = os.path.join(graphs_save_path, algorithm)
        make_folder(alg_save_path)
        for gr_name in Testing_group_list:
            gr_params = generator_config["testing_group_list"][int(gr_name.lstrip('gr').split('_',1)[0])]
            object_name = gr_params['testing_package']
            group_path = os.path.join(algorithm_path, gr_name)
            
            case_save_path = os.path.join(alg_save_path, gr_name)
            make_folder(case_save_path)
            if int(gr_name[-1]) == 2:
                MetricStore = {}
                InterpMetricsList =  [x for x in MetricsList if 'interpolation' in MetricsList]
                QualityMetricsList = [x for x in MetricsList if 'interpolation' not in MetricsList]
                MetricStore['interpolation_mse'] = []
                MetricStore['interpolation_HaarPSI'] = []
                for metric in QualityMetricsList: #QualityMetricsList:
                    MetricStore[metric] = []
                    external_cases = sorted([x for x in os.listdir(group_path) if '.json' not in x],
                                            key = sortkey)
                    for e_case in external_cases:
                        e_case_path = os.path.join(group_path, e_case)
                        MetricStore[metric].append(GetMetric(e_case_path, metric))
                        if metric == 'MSE':
                            MetricStore['interpolation_mse'].append(GetMetric(e_case_path, 'interpolation_mse'))
                        elif metric == 'HaarPSI':
                            MetricStore['interpolation_HaarPSI'].append(GetMetric(e_case_path, 'interpolation_HaarPSI'))

                    
                    v1_params = gr_params["variation1"]
                    v2_params = gr_params["variation2"]
                    v1_values = np.linspace(v1_params["left_border"],v1_params["right_border"],v1_params["points_amount"])
                    v2_values = np.linspace(v2_params["left_border"],v2_params["right_border"],v2_params["points_amount"])

                    metric_df = pd.DataFrame(data = MetricStore[metric], columns = v2_values, index= v1_values).T
                    if metric == 'MSE':
                        interp_df = pd.DataFrame(data = MetricStore['interpolation_mse'], columns = v2_values, index= v1_values).T
                    elif metric == 'HaarPSI':
                        interp_df = pd.DataFrame(data = MetricStore['interpolation_HaarPSI'], columns = v2_values, index= v1_values).T
                    
                    X,Y = np.meshgrid(v2_values, v1_values)
                    fig = plt.figure()
                    ax = plt.axes(projection='3d')
                    
                    if metric == 'MSE':
                        ax.plot_surface(X, Y, np.array(MetricStore['interpolation_mse']), cmap='Greys', edgecolor='none')
                    elif metric == 'HaarPSI':
                        ax.plot_surface(X, Y, np.array(MetricStore['interpolation_HaarPSI']), cmap='Greys', edgecolor='none')
                    ax.plot_surface(X, Y, np.array(MetricStore[metric]), cmap='Reds', edgecolor='none')
                    ax.set_title(f'{metric}. {algorithm}. {object_name}')
                    ax.set_xlabel(gr_params["variation2"]["param_name"])
                    ax.set_ylabel(gr_params["variation1"]["param_name"])
                    ax.set_zlabel(metric)
                    plt.show()
                    metric_df.to_csv(case_save_path+f'/{metric}.csv', encoding='UTF-8', index=True)
                    if metric == 'MSE' or 'HaarPSI':
                        interp_df.to_csv(case_save_path+f'/interpolation_{metric}.csv', encoding='UTF-8', index=True)
                

                param_save_path = os.path.join(case_save_path, f'{gr_params["variation2"]["param_name"]}/')
                make_folder(param_save_path)
                for line_n, line in enumerate(np.array(MetricStore[metric])):
                        axis_save_path = os.path.join(param_save_path, f'{gr_params["variation1"]["param_name"]}_{line_n}/')
                        make_folder(axis_save_path)
                for metric in QualityMetricsList: #QualityMetricsList:
                    for line_n, line in enumerate(np.array(MetricStore[metric])):
                        axis_save_path = os.path.join(param_save_path, f'{gr_params["variation1"]["param_name"]}_{line_n}/')
                        fig = plt.figure()
                        plt.grid()
                        plt.xlabel(gr_params["variation2"]["param_name"])
                        plt.ylabel(metric)
                        plt.title(f'{metric}. {algorithm}. {object_name}')
                        if metric == 'MSE':
                            # plt.ylim(0,0.5)
                            plt.plot(v2_values, MetricStore['interpolation_mse'][line_n], color = 'green')
                        elif metric == 'HaarPSI':
                            plt.plot(v2_values, MetricStore['interpolation_HaarPSI'][line_n], color = 'green')
                        
                        # if metric == 'norm_geometry_rmse' or metric == 'norm_geometry_MSE':
                        #     plt.ylim(0,0.2)

                        plt.plot(v2_values, line)

                        plt.savefig(f'{axis_save_path}/{metric}_{line_n}.png')
                        plt.close()


                param_save_path = os.path.join(case_save_path, f'{gr_params["variation1"]["param_name"]}/')
                make_folder(param_save_path)
                for column_n, column in enumerate(np.array(MetricStore[metric]).T):
                        axis_save_path = os.path.join(param_save_path, f'{gr_params["variation2"]["param_name"]}_{column_n}/')
                        make_folder(axis_save_path)
                for metric in QualityMetricsList:
                    for column_n, column in enumerate(np.array(MetricStore[metric]).T):
                        axis_save_path = os.path.join(param_save_path, f'{gr_params["variation2"]["param_name"]}_{column_n}/')
                        fig = plt.figure()
                        plt.grid()
                        plt.xlabel(gr_params["variation1"]["param_name"])
                        plt.ylabel(metric)
                        plt.title(f'{metric}. {algorithm}. {object_name}')
                        if metric == 'MSE':
                            # plt.ylim(0,0.5)
                            plt.plot(v1_values, np.array(MetricStore['interpolation_mse'])[:,column_n], color = 'green')
                        elif metric == 'HaarPSI':
                            plt.plot(v1_values, np.array(MetricStore['interpolation_HaarPSI'])[:,column_n], color = 'green')
                        
                        # if metric == 'norm_geometry_rmse' or metric == 'norm_geometry_MSE':
                        #     plt.ylim(0,0.2)

                        plt.plot(v1_values, column)
                        plt.savefig(f'{axis_save_path}/{metric}_{column_n}.png')
                        plt.close()

            if int(gr_name[-1]) == 1:
                SaveGraph2D(case_save_path, group_path, gr_params, MetricsList)
                SaveData2D(case_save_path, group_path, gr_params, MetricsList)
