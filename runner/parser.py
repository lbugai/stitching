import os
import pandas as pd
import numpy as np
import json

stitching_results_path = 'C:\Leonid\stitching\exp3\stitching_results'
column_names = ["alg_name",
                "package_name",
                "initial_transform",
                "interpolation_mse",
                "interpolation_HaarPSI",
                "HaarPSI",
                "MSE",
                "norm_geometry_MSE",
                "norm_geometry_rmse",
                "geometry_rmse",
                "normalized maximum deviation of distances (from geometry MSE)",
                "maximum deviation of distances (from geometry MSE)"]
df = pd.DataFrame(columns=column_names)

for alg_name in os.listdir(stitching_results_path):
    alg_results_path = os.path.join(stitching_results_path, alg_name)
    for package_name in os.listdir(alg_results_path):
        package_metrics_json_path = os.path.join(alg_results_path, package_name,"metrics","0", "metrics.json")
        with open(package_metrics_json_path, 'r', encoding='UTF-8') as json_file:
            metrics = json.load(json_file)
        package_info_json_path = os.path.join(alg_results_path, package_name,"metrics","0", "time_mem.json")
        with open(package_info_json_path, 'r', encoding='UTF-8') as json_file:
            info = json.load(json_file)
        if info["initial_tr_matrix_path"][-10:] == "none_given":
            initial_transform = "none_given"
        else:
            initial_transform = info["initial_tr_matrix_path"]
        #initial_transform = "none_given"
        row = [alg_name,
               package_name,
               initial_transform,
               metrics["interpolation_mse"],
               metrics["MSE"],
               metrics["norm_geometry_MSE"],
               metrics["norm_geometry_rmse"],
               metrics["geometry_rmse"],
               metrics["normalized maximum deviation of distances (from geometry MSE)"],
               metrics["maximum deviation of distances (from geometry MSE)"]]
        df.loc[len(df)] = row

df.to_csv(os.path.join(stitching_results_path, "stitching_results.csv"), index=False)
        