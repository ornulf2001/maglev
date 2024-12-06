import numpy as np
import matplotlib.pyplot as plt

# Example errors (replace with your error data)
errors = [1, 2, 1, 3, 2, 2, 1, 3, 3, 1, 2, 3, 3, 1, 1, 2, 1]

# Create the histogram
plt.hist(errors, bins=range(1, 5), align='left', edgecolor='black')  # bins specify error ranges

# Customize the plot
plt.xlabel('Error Value')
plt.ylabel('Frequency')
plt.title('Histogram of Errors')
plt.xticks(range(1, 4))  # Adjust ticks to match error values
plt.show()
