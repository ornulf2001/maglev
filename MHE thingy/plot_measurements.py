import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load CSV data
data = pd.read_csv(r'C:\Users\ornul\Desktop\Kyb master\Maglev\MHE thingy\measured_rotation.csv')
print(data)

plt.rcParams.update({
    'font.size': 14,            # General font size
    'axes.titlesize': 14,       # Title font size
    'axes.labelsize': 14,       # Axes label font size
    'legend.fontsize': 13,      # Legend font size
    'xtick.labelsize': 14,      # X-tick label font size
    'ytick.labelsize': 14       # Y-tick label font size
})

plt.plot(data[data.columns[1]], label="Measured angle",zorder=10)
plt.xlabel("Measurements")
plt.ylabel("Theta")
plt.legend(loc="upper right")

plt.show()