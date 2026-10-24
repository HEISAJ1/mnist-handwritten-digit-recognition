# Neural Network preprocessing

The selected model is neural_network.keras. Both trial models are retained.
Use the environment and versions in results/stage3/neural_network_run.json.
Input: MNIST-style grayscale images, dark background and light strokes, raw range 0-255.
Convert pixels to NumPy float32 and divide by 255 exactly once. Retain shape (N,28,28).
Do not flatten externally: the model includes Flatten. Do not normalize twice.
Output: ten softmax probabilities in digit order 0-9. Use argmax along axis 1 for labels.
These scores are not guarantees of correctness; arbitrary photos need additional work.
Use the exact saved Stage 2 indices for validation comparison. Test data remains reserved.
