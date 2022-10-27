import numpy as np
import matplotlib.pyplot as plt
from preprocessing import *

set_of_features = 0
y, X, id = load_data(set_of_features)

plt.hist(X[:, 1], density=True, bins=50, range=[0, 200])
plt.hist(X[:, 2], density=True, bins=50, range=(0, 200))
plt.legend(['Feature 1', 'Feature 2'])
plt.show()