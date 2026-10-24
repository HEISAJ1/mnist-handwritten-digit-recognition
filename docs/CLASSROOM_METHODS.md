# Classroom-method source inventory

The original lesson notebooks are private and intentionally absent. This project-authored inventory reproduces the evidence table already in the technical report; it does not distribute classroom code or completed MNIST solutions.

Cell numbers below are one-based positions in the notebook's cell list, including Markdown and empty cells, rather than execution counters. The relevant original code was reinspected for this report.

**Table A1. Teaching patterns used and project-specific adaptations.**

| Pattern | Original classroom notebook and cells | Reuse or adaptation in this project |
|---|---|---|
| Imports, aliases and plotting setup | `Bank_Logistic_Regression.ipynb`, 1; `linear regression.ipynb`, 1 | Familiar `pd`, `np`, `plt`, `seaborn` and `seaborn.set()`; required libraries added directly |
| Raw/working data and features/targets | `Bank_Logistic_Regression.ipynb`, 2-6 | `raw_data`, `data`, `x`, `y`, `reg_log`; omit binary label mapping and constant column for Scikit-learn |
| Descriptive statistics | `linear regression.ipynb`, 4 | `.describe()` applied to meaningful image summaries; pixel reductions and counts are explicit additions |
| Labels, titles and plots | `linear regression.ipynb`, 6, 8; `Untitled10.ipynb`, 12, 14 | Labelled plots and loops; bars, histograms, grids and heatmaps extend the style |
| Array inspection and reshape | `Untitled13.ipynb`, 8-17, 46 | Shape/type inspection and 784-feature reshaping; dtype and memory checks are additional |
| Shuffle and paired slicing | `Untitled20.ipynb`, 6-7 | Shuffle indices once; change the split to 54,000/6,000 while preserving the official test set |
| Save/load arrays | `Untitled20.ipynb`, 8-9 | Save indices and prediction records; hashing and frozen-cache checks are additional |
| Float32 and divide-by-255 scaling | `Untitled19.ipynb`, 3 | Same pixel rule, expressed directly on NumPy arrays without a custom function |
| Logistic fitting and prediction | `Bank_Logistic_Regression.ipynb`, 5-6; `Bank_Data_Logistic_Regression (1).ipynb`, 6-8 | Required Scikit-learn multinomial implementation replaces Statsmodels binary `Logit` |
| Confusion tables and diagonal accuracy | `Bank_Data_Logistic_Regression (1).ipynb`, 10-12, 17-21 | Extend binary held-out reasoning to ten classes; library macro metrics are additions |
| Weights, biases and updates | `Untitled14.ipynb`, 5-7; `Untitled18.ipynb`, 4-8 | Explain learned parameters; use taught Keras training rather than copying a manual loop |
| Sequential architecture | `Untitled19.ipynb`, 7; `Untitled20.ipynb`, 10 | Flatten and two ReLU layers; correct output width to ten and use appropriate Input shape |
| Compile, batches, epochs and validation | `Untitled19.ipynb`, 5-9; `Untitled20.ipynb`, 12 | Adam, sparse loss, batch 100 and five epochs; avoid the lesson's hardcoded batch mismatch |
| Evaluation and learning-curve style | `Untitled19.ipynb`, 10-11; `Untitled18.ipynb`, 10 | Held-out evaluation and line plots; History curves, argmax and metrics added explicitly |

Not all useful project techniques were established in the classroom evidence. Fixed-seed utilities, integrity checks, timing, warning capture, Joblib/native Keras model persistence, macro metrics, experiment records and prediction caching were explained necessary additions. Neither `.save()` for neural-network models nor Scikit-learn's Logistic Regression API is misrepresented as having been taught in the cited binary lesson.

