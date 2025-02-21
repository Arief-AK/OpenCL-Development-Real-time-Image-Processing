import os
import pandas as pd
import matplotlib.pyplot as plt

# 0 - Linux
# 1 - Windows
USING_OS = 0

# 0 - Debug
# 1- Release
BUILD_TYPE = 1

SHOW_FIGURES = False
SHOW_TERMINAL_OUTPUT = False
SAVE_DATAFRAME = True

def CleanData(data: pd.DataFrame) -> pd.DataFrame:
    # Strip whitespace from column names
    data.columns = data.columns.str.strip()

    # Extract the image name from the 'Image' column
    data['Image_Name'] = data['Image'].str.extract(r'images/(\w+)_')

    # Extract the resolution as separate width and height columns for scaling
    data[['Width', 'Height']] = data['Resolution'].str.split('x', expand=True).astype(int)
    data['Pixel Count'] = data['Width'] * data['Height']

    # Sort the data by Pixel Count (smallest to largest)
    data = data.sort_values(by='Pixel Count')

    # Make sure the timestamps are recognised as timestamps
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])

    return data

# CPU vs OpenCL (end-to-end)
def CPUvsOpenCLEndtoEnd(ax, data: pd.DataFrame):
    ax.plot(data['Pixel Count'], data['avg_CPU_Time_ms'], label='avg_CPU Time (ms)', marker='o')
    ax.plot(data['Pixel Count'], data['avg_OpenCL_Time_ms'], label='avg_OpenCL Time (ms)', marker='o')
    ax.set_xlabel('Pixel Count (Resolution)')
    ax.set_ylabel('Average Execution Time (ms)')
    ax.set_title('CPU vs OpenCL Execution Time (End-to-End)')
    ax.legend()
    ax.grid(True)

# OpenCL Kernel Operations vs OpenCL Total Time
def KernelvsOpenCLTotal(ax, data: pd.DataFrame):
    ax.plot(data['Pixel Count'], data['avg_CPU_Time_ms'], label='CPU Operation Time (ms)', marker='o')
    ax.plot(data['Pixel Count'], data['avg_OpenCL_kernel_ms'], label='OpenCL Kernel Execution Time (ms)', marker='o')
    ax.plot(data['Pixel Count'], data['avg_OpenCL_kernel_write_ms'], label='OpenCL Kernel Write Time (ms)', marker='o')
    ax.plot(data['Pixel Count'], data['avg_OpenCL_kernel_read_ms'], label='OpenCL Kernel Read Time (ms)', marker='o')
    ax.plot(data['Pixel Count'], data['avg_OpenCL_kernel_operation_ms'], label='OpenCL Kernel operation Time (ms)', marker='o')
    ax.set_title('Average CPU vs Average Kernel Operation Time')
    ax.set_xlabel('Pixel Count (Resolution)')
    ax.set_ylabel('Time (ms)')
    ax.legend()
    ax.grid(True)

# OpenCL vs CPU speedup factor (end-to-end)
def SpeedFactorEndtoEnd(ax, data: pd.DataFrame):
    # Calculate the speedup factor
    data['Speedup'] = data['avg_CPU_Time_ms'] / data['avg_OpenCL_Time_ms']

    ax.plot(data['Pixel Count'], data['Speedup'], label='Speedup', marker='o')
    ax.axhline(1, color='red', linestyle='--', label='No Speedup')
    ax.set_title('Speedup vs Resolution (End-to-End)')
    ax.set_xlabel('Pixel Count (Resolution)')
    ax.set_ylabel('Speedup Factor')
    ax.legend()
    ax.grid(True)

def SpeedFactorOperation(ax, data: pd.DataFrame):
    data['operation_speedup'] = data['avg_CPU_Time_ms'] / data['avg_OpenCL_kernel_operation_ms']

    ax.plot(data['Pixel Count'], data['operation_speedup'], label='Operation speedup', marker='o')
    ax.axhline(1, color='red', linestyle='--', label='No Speedup')
    ax.set_title('Speedup vs Resolution (Operation)')
    ax.set_xlabel('Pixel Count (Resolution)')
    ax.set_ylabel('Speedup Factor')
    ax.legend()
    ax.grid(True)

# MAE
def MAE(ax, data: pd.DataFrame):
    ax.plot(data['Pixel Count'], data['Error_MAE'], label='MAE', color='red', marker='o')
    ax.set_title('MAE vs Resolution')
    ax.set_xlabel('Pixel Count (Resolution)')
    ax.set_ylabel('Error (MAE)')
    ax.grid(True)

if __name__ == "__main__":
    # Get the current script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    os_name = ""
    build_type = ""

    # Initialise build type
    if BUILD_TYPE == 0:
        build_type = "Debug"
    elif BUILD_TYPE == 1:
        build_type = "Release"
    else:
        print("Unrecognised Build Type")

    # Setup correct OS
    if USING_OS == 0:
        os_name = "Linux"
    elif USING_OS == 1:
        os_name = "Windows"
    else:
        os_name = "UNDEFINED"
        print("Unrecognised OS")

    # Build the absolute path
    csv_path = os.path.abspath(os.path.join(script_dir, f"../../../build/src/Grayscale/{build_type}/results.csv"))
    print(f"Looking for the csv in: {csv_path}")

    # Load the csv file
    data = pd.read_csv(csv_path, delimiter=',')

    # Data exists
    if not data.empty:
        data = CleanData(data)

        # Group the data by 'Image_Name' and create separate sorted DataFrames
        image_groups = {name: group for name, group in data.groupby('Image_Name')}

        # Create a directory for saving the figures
        figures_dir = os.path.join(script_dir, "figures")
        os.makedirs(figures_dir, exist_ok=True)

        # Iterate through each group and create a separate plot
        for image_name, group_data in image_groups.items():
            print(f"\nGenerating plots for {image_name}...")

            # Get number of iterations
            num_iterations = group_data['Num_Iterations'].iloc[0]

            # Create subplots for this image group
            fig, ax = plt.subplots(3, 2, figsize=(12, 10))
            
            CPUvsOpenCLEndtoEnd(ax[0, 0], group_data)
            KernelvsOpenCLTotal(ax[0, 1], group_data)
            SpeedFactorEndtoEnd(ax[1, 0], group_data)
            SpeedFactorOperation(ax[1, 1], group_data)
            MAE(ax[2, 0], group_data)

            # Set the overall title for the group
            fig.suptitle(f'Performance Metrics for {image_name} on {os_name} with {num_iterations} iterations', fontsize=16)

            # Set the figure window title
            plt.gcf().canvas.manager.set_window_title(f"Performance Metrics - {image_name}")

            # Adjust layout and show the plot
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            if SHOW_FIGURES:
                plt.show()

            if SHOW_TERMINAL_OUTPUT:
                print(f"{group_data}")

            if SAVE_DATAFRAME:
                # Save the figure
                figure_path = os.path.join(figures_dir, f"{os_name}_{num_iterations}_{image_name}_performance_metrics.png")
                fig.savefig(figure_path, dpi=300)
                print(f"Saved figures in: {figure_path}")

                # Save dataframe to csv
                csv_result_path = os.path.abspath(os.path.join(script_dir, f"{os_name}_{num_iterations}_{image_name}_sorted_results.csv"))
                group_data.to_csv(csv_result_path, index=False)
                print(f"Saved csv in: {csv_result_path}")