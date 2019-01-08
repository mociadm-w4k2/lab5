from imblearn.over_sampling import SMOTE
from utils import load_prepare_data
import numpy as np

# Load data
X, y, classes = load_prepare_data("data/yeast1.csv")

# Resample data
sm = SMOTE()
X_res, y_res = sm.fit_sample(X, y)

# Count classes
name, count = np.unique(y, return_counts=True)
res_name, res_count = np.unique(y_res, return_counts=True)

print("Before resampling")
print(name[0], count[0])
print(name[1], count[1])
print("After resampling")
print(res_name[0], res_count[0])
print(res_name[1], res_count[1])
