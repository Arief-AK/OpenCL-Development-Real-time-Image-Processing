import os
import re
import matplotlib.pyplot as plt

build_type = "Release"

# Define file path (you can change this to the actual file location)
script_dir = os.path.dirname(os.path.abspath(__file__))
log_file_path = os.path.abspath(os.path.join(script_dir, f"../../../build/src/RealtimeImageProcessing/{build_type}/RealtimeImageProcessing.log"))

# Define regex patterns
method_pattern = r"Performing OpenCL (.*?)\.\.\."
time_pattern = r"execution time: (\d+\.\d+) ms"

# Initialize lists to hold execution times for each method
grayscale_times = []
edge_times = []
gaussian_times = []

current_method = None

# Initialize lists to hold execution times for each method
grayscale_times = []
edge_times = []
gaussian_times = []

current_method = None

# Read the log file and parse entries
with open(log_file_path, 'r') as file:
    for line in file:
        # Check for method names
        method_match = re.search(method_pattern, line)
        if method_match:
            current_method = method_match.group(1)

        # Check for execution times
        time_match = re.search(time_pattern, line)
        if time_match:
            exec_time = float(time_match.group(1))
            if current_method == 'Grayscaling':
                grayscale_times.append(exec_time)
            elif current_method == 'Edge Detection':
                edge_times.append(exec_time)
            elif current_method == 'Gaussian Blur':
                gaussian_times.append(exec_time)

# Calculate averages per method
def average_time(times):
    return sum(times) / len(times) if times else 0

# Determine the minimum frame count across all methods
min_frame_count = min(len(grayscale_times), len(edge_times), len(gaussian_times))

# Function to truncate the times list to the min_frame_count
def truncate_to_min_length(times, min_len):
    return times[:min_len]

# Truncate each method's times list to match the minimum frame count
grayscale_times = truncate_to_min_length(grayscale_times, min_frame_count)
edge_times = truncate_to_min_length(edge_times, min_frame_count)
gaussian_times = truncate_to_min_length(gaussian_times, min_frame_count)

# Generate consistent frame count for all methods (up to the minimum frame count)
frame_count = list(range(1, min_frame_count + 1))

# Function to plot and add data for each method
def plot_method(ax, method_name, frame_count, times, color):
    ax.plot(frame_count, times, label=method_name, color=color)
    ax.set_xlabel('Frame Count')
    ax.set_ylabel('Execution Time (ms)')
    ax.set_title(f'Execution Time vs. Frame Count for {method_name}')
    ax.legend()
    ax.grid(True)

# Create a figure with subplots for each method
fig, axs = plt.subplots(3, 1, figsize=(10, 18))  # 3 rows, 1 column

# Plot Grayscaling
plot_method(axs[0], 'Grayscaling', frame_count, grayscale_times, 'blue')

# Plot Gaussian Blur
plot_method(axs[1], 'Gaussian Blur', frame_count, gaussian_times, 'red')

# Plot Edge Detection
plot_method(axs[2], 'Edge Detection', frame_count, edge_times, 'green')

# Adjust layout to prevent overlap
plt.tight_layout()

# Save plot to a file (e.g., PNG or PDF)
figures_dir = os.path.join(script_dir, "figures")
os.makedirs(figures_dir, exist_ok=True)

figure_path = os.path.join(figures_dir, f"execution_times_plot.png")
plt.savefig(figure_path)
