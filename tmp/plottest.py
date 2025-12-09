import matplotlib.pyplot as plt
import numpy as np

s = np.linspace(0, 1, 11) 
print(s)

# First data set
x1 = [1, 2, 3, 4, 5]
y1 = [2, 4, 6, 8, 10]

# Second data set
x2 = [1, 2, 3.5, 4, 14]
y2 = [1, 3, 6, 7, 9]

# Plot both lines
plt.plot(x1, y1, label='Dataset 1', color='blue', marker='o')
plt.plot(x2, y2, label='Dataset 2', color='red', linestyle='--', marker='x')

# Add labels and legend
plt.title('Two Sets of Data')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid(True)
plt.show()
