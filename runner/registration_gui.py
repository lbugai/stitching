import os
import shutil
import matplotlib.pyplot as plt
import pandas as pd

def start_plot(result_folder_path):
    """
    Callback invoked when the StartEvent happens, sets up our new data.
    """
    global metric_values, multires_iterations, ax, fig, df, res_path
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    df = pd.DataFrame({"r1": [], "r2": [], "r3": [],"trX": [], "trY": [], "trZ": [],"scale": []})
    res_path = result_folder_path
    metric_values = []
    multires_iterations = []

def end_plot():
    """
    Callback invoked when the EndEvent happens, do cleanup of data and figure.
    """
    global metric_values, multires_iterations, ax, fig, df, res_path
    #plt.show()
    save_path = os.path.join(res_path, "graphs/")
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    else:
        shutil.rmtree(save_path)
        os.mkdir(save_path)
    fig.savefig(save_path + '/metric.png')
    df.to_csv(save_path + '/results.csv', index=True, encoding='UTF-8')
    for column in df.columns:
        plt.figure(figsize=(10,6))
        plt.plot(df.index, df[column], 
                marker='o',
                linestyle='-', 
                linewidth=2,
                markersize=8)
        plt.title(f'{column} vs Index')
        plt.xlabel('Row Index')
        plt.ylabel(column)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = os.path.join(save_path, f'{column}_vs_index.png')
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()
    del metric_values
    del multires_iterations
    del ax
    del fig
    del df
    del res_path



def plot_values(registration_method):
    """
    Callback invoked when the IterationEvent happens, update our data and display new figure.
    """
    global metric_values, multires_iterations, ax, fig, df

    metric_values.append(registration_method.GetMetricValue())
    
    if len(metric_values)%20 == 0:
        print(f"SITK interation number = {len(metric_values)}", end='\r')
    #print(registration_method.GetOptimizerPosition())
    df.loc[len(df)] = registration_method.GetOptimizerPosition()
    # Plot the similarity metric values
    ax.plot(metric_values, "r")
    ax.plot(
        multires_iterations,
        [metric_values[index] for index in multires_iterations],
        "b*",
    )
    ax.set_xlabel("Iteration Number", fontsize=12)
    ax.set_ylabel("Metric Value", fontsize=12)
    fig.canvas.draw() 

def update_multires_iterations():
    """
    Callback invoked when the sitkMultiResolutionIterationEvent happens,
    update the index into the metric_values list.
    """
    global metric_values, multires_iterations
    multires_iterations.append(len(metric_values))
