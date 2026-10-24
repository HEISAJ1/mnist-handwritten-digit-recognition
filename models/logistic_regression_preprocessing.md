# Logistic Regression preprocessing and reuse

Model: logistic_regression.joblib (Scikit-learn LogisticRegression).
Use the recorded mnist_project environment and versions in
results/stage2/logistic_regression_run.json.

1. Start with MNIST-style 28 by 28 grayscale images, dark background and light strokes.
2. Raw pixels use the original 0-255 range. Arbitrary photographs need additional work.
3. Convert to NumPy float32 and divide by 255 exactly once.
4. Reshape to (number_of_images, 784), using C/row-major pixel order.
5. Call the loaded classifier's predict method; outputs are digits 0-9.

Example using already prepared validation_x from the notebook:
    loaded_reg_log = joblib.load(model_path)
    predictions = loaded_reg_log.predict(validation_x)

The model does not contain normalization or resizing. Do not divide normalized
pixels by 255 again. Integer labels are not scaled. Exact train/validation indices
are in results/stage2/split_indices.npz. The official test set remains reserved.
