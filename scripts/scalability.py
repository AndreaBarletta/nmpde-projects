import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import numpy as np

# Read all the files in the 'data/time' directory. 
# For each file read, compute the average of the numerical values contained within it and return a dictionary where the keys are the filenames and the values are the computed averages.
# Note that you should remove outliers before computing the averages using the interquartile range (IQR) method.
def compute_averages_in_directory(directory_path):
    averages = {}
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            try:
                data = np.loadtxt(file_path)
                
                # Remove outliers using IQR method
                Q1 = np.percentile(data, 25)
                Q3 = np.percentile(data, 75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
                
                if len(filtered_data) > 0:
                    average_value = np.mean(filtered_data)
                else:
                    average_value = None  # No data left after removing outliers
                
                # Remove the '.txt' extension from the filename for the key
                filename_key = os.path.splitext(filename)[0]
                averages[filename_key] = average_value
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
                averages[filename_key] = None

    # Order by filename keys
    averages = dict(sorted(averages.items()))
    return averages



directory_path = 'data/time/'
data_amg = compute_averages_in_directory(directory_path+'amg')
data_ilu = compute_averages_in_directory(directory_path+'ilu')
data_sor = compute_averages_in_directory(directory_path+'sor')

# Plot the results on a log-log scale
x = sorted(float(k) for k in data_amg.keys())
y_amg = [data_amg[str(int(k))]/1e9 for k in x]
y_ilu = [data_ilu[str(int(k))]/1e9 for k in x]
y_sor = [data_sor[str(int(k))]/1e9 for k in x]

plt.figure()
plt.title("Scalability analysis")
plt.loglog(x, y_amg, marker='o', linestyle='-', label='Measured average time (AMG preconditioner)')
plt.loglog(x, y_ilu, marker='o', linestyle='-', label='Measured average time (ILU preconditioner)')
plt.loglog(x, y_sor, marker='o', linestyle='-', label='Measured average time (SOR preconditioner)')
plt.xlabel("Number of processes")
plt.ylabel("Average time per timestep (s)")
plt.grid(True, which="both")

ax = plt.gca()

# Add reference linear decrease (slope -1 in log-log)
# y_ref = k / x
k = y_sor[0] * x[0]  # pass through the first data point
y_ref = k / x
plt.loglog(x, y_ref, linestyle='--', color='red', label='Reference linear decrease')

# Set x-ticks explicitly
ax.set_xticks(x)
ax.get_xaxis().set_major_formatter(ticker.FormatStrFormatter('%g'))
ax.get_yaxis().set_major_formatter(ticker.FormatStrFormatter('%g'))
ax.get_yaxis().set_minor_formatter(ticker.FormatStrFormatter('%g'))

plt.legend()
plt.show()