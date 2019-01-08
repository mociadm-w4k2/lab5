# Laboratory 5

This laboratory allows you to familiarize yourself with a phenomenon called imbalance data. Imbalance data is such data, where one of the classes is much less than the other classes. Such problems are most often considered in the case of two classes, where we have positive classes (minority data) and a negative class (majority data).

Before you go to solving this lab, make sure that the imbalanced-stream library is installed. To check, you must run the `check.py` script. This script has an example of using this library.

```python
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
```

If the above script (from the check.py file) did not execute without the error message, then the library imbalanced-learn is probably missing. To install it, use the following command.

```bash
pip install imbalanced-learn>=2.3.0
```

[`Imbalance-learn library website`](https://imbalanced-learn.org/en/stable/index.html)

Please put solutions in the [`solution.py`](solution.py) file.

## Exercises

### Exercise 1 (2 pts)

- Load imbalance data from a file[`yeast1.csv`](data/yeast1.csv)
- Divide the set into the feature set (`X`) and the set of labels (`y`). Use the `prepare_data` function from the `utils.py` file
- Make a k-fold cross validation for `k = 10` for the loaded data [`k-fold cross validation`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html)
- For each fold:
  - Learn the classifier on the training set
  - Calculate the prediction for the learned classifier on the test set
  - Put the [`recall_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html) on the list for the classification of the training set on the given fold
- Display on the screen the average recall score of the classification
- Select and repeat this process for three different classifiers
- Answer in comment which classifier achieve best score

### Exercise 2 (3 pts)

- Load imbalance data from a file[`yeast1.csv`](data/yeast1.csv)
- Divide the set into the feature set (`X`) and the set of labels (`y`). Use the `prepare_data` function from the `utils.py` file
- Re-sample data using the SMOTE method. (`res_X`, `res_y`)
- Make a k-fold cross validation for `k = 10` for the **re-sampled** data [`k-fold cross validation`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html)
- For each fold:
  - Learn the classifier on the training set
  - Calculate the prediction for the learned classifier on the test set
  - Put the [`recall_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html) on the list for the classification of the training set on the given fold
- Display on the screen the average recall score of the classification
- Repeat this process for selected classifiers from Exercise 1.
- Answer in comment which achieve best score.
- Answer in the comment how re-sampling affects the quality of the classification. Compare the results obtained in exercises 1 and 2.

### Exercise 3 (5pts)

- Extend "Exercise 2" to use different methods of preprocessing imbalanced data. At least two from each category:
  - [Under-sampling methods](https://imbalanced-learn.org/en/stable/api.html#module-imblearn.under_sampling)
  - [Over-sampling methods](https://imbalanced-learn.org/en/stable/api.html#module-imblearn.over_sampling)
  - [Combine methods](https://imbalanced-learn.org/en/stable/api.html#module-imblearn.combine)
  - [Ensemble methods](https://imbalanced-learn.org/en/stable/api.html#module-imblearn.ensemble)
- Answer in the comment which of the tested methods achieves the best recall score.
