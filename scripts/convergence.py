import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import numpy as np

# Read all the files in the 'data/space' directory. 
# For each file read, compute the average of the numerical values contained within it and return a dictionary where the keys are the filenames and the values are the computed averages.
# Give the values, compute the L2 norm
def compute_averages_in_directory(directory_path):
    averages = {}
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            try:
                data = np.loadtxt(file_path)                
                norm = np.linalg.norm(data)
                
                # Remove the '.txt' extension from the filename for the key
                filename_key = os.path.splitext(filename)[0]
                averages[filename_key] = norm
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
                averages[filename_key] = None

    # Order by filename keys
    averages = dict(sorted(averages.items()))
    return averages



directory_path = 'data/space/velocity/'
data_height = compute_averages_in_directory(directory_path+'height')
data_velocity = compute_averages_in_directory(directory_path+'velocity')

# Plot the results on a log-log scale
x_height = sorted(float(k) for k in data_height.keys())
y_height = [data_height[str(int(k))] for k in x_height]
x_velocity = sorted(float(k) for k in data_velocity.keys())
y_velocity = [data_velocity[str(int(k))] for k in x_velocity]

plt.figure()
plt.title("Convergence analysis")
plt.loglog(x_height, y_height, marker='o', linestyle='-', label='Water height error')
plt.loglog(x_velocity, y_velocity, marker='s', linestyle='-', label='Velocity error')
plt.xlabel("Mesh size")
plt.ylabel("$error_H$ and $error_U$")
plt.grid(True, which="both")

ax = plt.gca()

# Add reference linear decrease (slope -1 in log-log)
ref_x = np.array([min(x_height), max(x_height)])
ref_y = ref_x**-1 * y_height[0] * x_height[0]
plt.loglog(ref_x, ref_y, 'k--', label='$\Delta x$', color='red')
# Add reference quadratic decrease (slope -2 in log-log)
ref_y2 = ref_x**-2 * y_height[0] * x_height[0]**2
plt.loglog(ref_x, ref_y2, 'k-.', label='$\Delta x^2$', color='green')
# Set x-ticks explicitly
ax.set_xticks(x_height)
ax.get_xaxis().set_major_formatter(ticker.FormatStrFormatter('%g'))
ax.get_yaxis().set_major_formatter(ticker.FormatStrFormatter('%g'))

plt.legend()
plt.show()

plt.close()