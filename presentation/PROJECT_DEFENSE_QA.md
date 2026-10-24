# Project Defense Questions and Answers

Project: Comparative Analysis of Machine Learning and Deep Learning Techniques for Handwritten Digit Recognition

The answers below describe the saved implementation and results. Proposed improvements are future work, not completed experiments. Cell numbers mean one-based positions including Markdown and empty cells.

## Purpose and classroom methods

### 1. What problem did you solve?
I compared two classifiers that assign one digit label, zero through nine, to a 28 × 28 grayscale handwritten image. I used the same fitting and validation memberships for both model families and evaluated the frozen choices on the same official test examples.

### 2. Why is this supervised classification?
Every example has a supplied correct label. Training learns a relationship between numerical pixels and those labels. The ten labels are categories rather than numerical quantities to predict by regression.

### 3. How are Data Science, AI, Machine Learning and Deep Learning connected?
Data Science combines programming, data analysis and statistical reasoning to investigate a question. AI is the broader field. Machine Learning learns relationships from data. Multilayer neural networks are a family of Machine Learning models studied in Deep Learning. This project uses plots and tables for analysis and does not deploy a business dashboard.

### 4. How does your work follow classroom teaching?
I used sequential notebook cells and familiar np, pd, plt, seaborn, tf, x, y and reg_log names. Bank_Logistic_Regression.ipynb cells 5–6 support feature/target selection and logistic fitting. Untitled20.ipynb cells 6–7 support shuffling and slicing. Untitled19.ipynb cells 3 and 7–11 support scaling and Keras build/compile/fit/evaluate. Untitled18.ipynb cells 5–8 support weights, biases and prediction. The detailed inventory is ../docs/CLASSROOM_METHODS.md.

### 5. What required adaptations were not directly taught?
The brief required Scikit-learn rather than the binary Statsmodels lesson. I used multiclass LogisticRegression, macro metrics, fixed-seed checks, fit timing, saving/loading models, hash verification and saved prediction records. These were explained additions, not falsely attributed to class. The Keras loader was required by the PDF.

### 6. Did you copy a classroom mistake?
No. Untitled19.ipynb cell 7 used the hidden width in its final layer despite defining ten output classes. I corrected the output to exactly ten Softmax units. I matched Input to (28,28) and explicitly passed batch size 100 rather than copying the mismatched hardcoded batch in Untitled20.ipynb cell 12.

## Dataset, statistics and preparation

### 7. What is MNIST and how did you load it?
It contains 70,000 labelled grayscale images, each 28 × 28 pixels: 60,000 official training images and 10,000 official test images. The required call is tf.keras.datasets.mnist.load_data(). KERAS_HOME is set before importing TensorFlow. Existing local caching avoids another download. The first uncached run normally needs internet access.

### 8. How is an image represented numerically?
It is a 28-by-28 matrix of pixel intensities. Zero is black and 255 is white, with intermediate gray levels. There are 784 numerical pixel positions. The supplied digit label is separate from those pixel values.

### 9. What did your exploratory analysis find?
All ten classes occur and no missing pixels were reported. Digit 1 has 6,742 examples, while digit 5 has 5,421, so the original training classes are approximately balanced but not identical. Across 47,040,000 raw training pixels the mean is 33.318421 and population standard deviation 78.567490. These do not describe a mean digit label.

### 10. How do mean and standard deviation help?
The mean summarises average intensity, and standard deviation summarises spread around it. Dark backgrounds are consistent with the low global pixel mean. The global calculation uses population standard deviation. Pandas describe() on per-image means uses sample standard deviation, which describes different observations.

### 11. Why normalise by 255?
It puts pixel features on a common zero-to-one scale while preserving relative intensities. Float32 avoids unnecessary image memory. The same fixed rule applies to every partition and to the demonstration. It does not estimate a mean or variance from the test set, and it is not standardisation to zero mean and unit variance.

### 12. Why keep integer labels?
The labels are already the ten correct category indices. Sparse categorical cross-entropy accepts these integer labels, so one-hot encoding is unnecessary.

### 13. Why does Logistic Regression need 784 inputs?
Scikit-learn expects a two-dimensional feature array with one row per example. Reshape turns each 28 × 28 image into a vector of 784 pixels in consistent row-major order. It does not change pixel values. The neural network receives (28,28) images and performs the rearrangement in Flatten.

### 14. How did you split the data?
I set NumPy seed 42, created indices 0 through 59,999, shuffled once, and took the first 54,000 for fitting and last 6,000 for validation. The split was random and unstratified. The same saved indices from results/stage2/split_indices.npz select images and labels for both model families. Checks established no overlap and complete coverage.

### 15. Why keep training, validation and test data separate?
Training updates weights. Validation supports development choices such as hidden width. Testing assesses frozen choices on examples that did not influence those choices. Using test results to tune settings would make the final assessment optimistic. The final models were not refitted on the combined 60,000 images after validation.

### 16. Does shuffling within neural-network fit change the partitions?
No. Shuffling the fitting order during epochs changes the order of examples already in the fitting partition. It does not move examples between the fixed fitting and validation memberships.

## Logistic Regression

### 17. Why use Logistic Regression for digits?
It is the classifier required by the brief and gives a useful simple baseline using numerical pixel features. Its name contains regression, but its output here is a digit class.

### 18. How does it predict?
For each class k it forms z_k = sum_j(w_kj x_j) + b_k. The x values are normalised pixels, w values are learned coefficients and b is an intercept. Multinomial Softmax converts the ten scores to probabilities. The most probable digit becomes the prediction.

### 19. What were its exact important settings?
LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000), with recorded tolerance 0.0001, fitted intercept and default L2 regularisation. It converged after 235 solver iterations with no convergence warnings. C is inverse regularisation strength and was not tuned. Solver iterations differ from neural-network epochs.

### 20. How does this differ from the lecturer's Statsmodels model?
The lesson used binary sm.Logit and explicitly added a constant. The project needs ten classes and Scikit-learn. Scikit-learn fits its intercept internally and supports the multinomial problem. I preserved the feature/target, fit, predict and compare sequence without claiming that the two APIs or statistical specifications are identical.

## Neural-network mathematics and training

### 21. What are the two architectures?
Both have Input(28,28), Flatten(784), two Dense/ReLU hidden layers and Dense(10,Softmax). The hidden widths are 50/50 and 100/100. No convolution, transfer learning or custom training loop was used.

### 22. What are weights and biases?
Weights determine how strongly inputs contribute to a neuron's score. A bias is a learned offset. The layer calculates a weighted sum plus bias and applies an activation. Training adjusts these parameters to reduce the loss.

### 23. What does ReLU do and why use it?
ReLU(z) = max(0,z). It passes positive scores and returns zero for negative ones. It introduces nonlinearity so successive layers can represent more flexible pixel combinations. Stacking purely linear transformations would still produce a linear transformation.

### 24. What does Softmax do?
It turns ten class scores into nonnegative probabilities summing to one: p_k = exp(z_k)/sum_l exp(z_l). Entry zero corresponds to digit zero, and so on. Argmax gives the index of the largest value. A large probability is not guaranteed correctness and calibration was not assessed.

### 25. What is sparse categorical cross-entropy?
For one image with correct label y, loss = -log(p_y). The loss grows when the model assigns little probability to the correct class. Sparse means the target is supplied as an integer category rather than a one-hot vector. Loss and accuracy measure different things.

### 26. What do Adam and backpropagation do?
Backpropagation calculates how the loss changes with each parameter. Adam uses those gradients and running estimates of their moments to update weights and biases. The recorded learning rate was approximately 0.001. Keras handles these operations inside fit.

### 27. What are an epoch and batch size?
An epoch is one pass over all 54,000 fitting images. Batch size 100 means up to 100 images contribute to a fitting update. Both trials ran five epochs. Validation at the end of each epoch measured progress without changing weights.

### 28. What setting did you experiment with?
Only the two hidden-layer widths changed from 50/50 to 100/100. The split, preprocessing, optimizer, loss, epochs, batch size and initialization seed 42 stayed the same. The trials ran sequentially to conserve RAM.

### 29. Why did you select the 100/100 model?
Its final validation accuracy was 97.2833%, compared with 96.4000% for 50/50. Its validation loss was 0.093159 versus 0.118696, although accuracy was the selection criterion. An exact accuracy tie would have selected the smaller model. I did not select using test results.

### 30. How many parameters did the networks have?
The 50/50 model has 42,310 and 100/100 has 89,610. A Dense layer with m inputs and h outputs has mh+h weights and biases. Flatten has no learned parameters. For 100/100, (784×100+100)+(100×100+100)+(100×10+10)=89,610.

### 31. What is overfitting, and did it occur?
Overfitting means learning patterns that help fitting data more than unseen data. Both final training accuracies exceeded validation accuracy. The curves show increasing accuracy and decreasing loss over five epochs, but one run and five epochs cannot prove that overfitting is absent. No early stopping or additional epoch search was performed.

### 32. Does seed 42 guarantee identical results everywhere?
No. It records the partition and initialization procedure and improves traceability. Floating-point libraries, platform differences and computation order can change outcomes and timing. One seed does not estimate experimental variability.

## Evaluation and interpretation

### 33. What is accuracy?
It is correct predictions divided by all predictions. Logistic Regression has 9,254/10,000 = 92.54%. The selected network has 9,753/10,000 = 97.53%.

### 34. What are precision and recall?
For a chosen digit, TP means correctly predicted examples, FP means other digits incorrectly assigned that label, and FN means actual examples missed. Precision = TP/(TP+FP), while recall = TP/(TP+FN). Precision focuses on predicted members and recall on actual members.

### 35. What is F1-score?
It is the harmonic mean of precision and recall: F1 = 2PR/(P+R). It is low when either component is poor. For class counts it is also 2TP/(2TP+FP+FN).

### 36. What is macro averaging?
Calculate a metric separately for each digit, then take the unweighted mean across ten digits. Every digit contributes equally despite different supports. Macro F1 is the average of per-class F1 values, not generally the F1 of macro precision and macro recall.

### 37. How do you read a confusion matrix?
Rows are actual labels and columns predictions. A diagonal entry is correct. Cell (5,3), for example, counts true fives called three. Both final matrices sum to 10,000, with diagonal sums 9,254 and 9,753.

### 38. Can you work through a real metric example?
For digit 8, Logistic Regression has TP=859, FP=119 and FN=115. Precision is 859/978 = 0.878323, recall is 859/974 = 0.881930, and F1 is 1718/1952 = 0.880123. The selected network has TP=939, FP=30, FN=35, giving precision 0.969040, recall 0.964066 and F1 0.966547.

### 39. What were the final macro results?
Logistic Regression: precision 0.924432, recall 0.924348, F1 0.924303. Selected network: precision 0.975181, recall 0.974954, F1 0.975045. These rounded values come from saved Stage 4 records, not external benchmark numbers.

### 40. By how much did the network improve accuracy?
97.53% minus 92.54% is 4.99 percentage points. It made 247 errors instead of 746, which is 499 fewer errors. Percentage points and a relative percentage improvement are different quantities.

### 41. What were the frequent digit confusions?
Logistic Regression: 5 as 3 occurred 38 times, 2 as 8 occurred 36, and 7 as 9 occurred 34. Network: 4 as 9 occurred 15 times, 5 as 3 occurred 9, and 8 as 5 occurred 9. These counts are directional. Similar strokes are a possible explanation, not a demonstrated causal mechanism.

### 42. Why might the network perform better here?
Its ReLU hidden layers can represent nonlinear combinations of pixels beyond the linear class scores of Logistic Regression. This is a plausible explanation. Other differences include optimisation, regularisation and parameter count, so the experiment does not isolate a cause or prove universal superiority.

### 43. What is Logistic Regression good at?
It is a simple useful baseline with a direct relationship between fixed pixel inputs and class scores. It converged without warnings and correctly classified 9,254 test images. Structural simplicity does not by itself prove human interpretability or general practical superiority.

### 44. Was the network definitely faster to train?
The measured selected fit was 23.330180 seconds, versus 52.845251 seconds for Logistic Regression. Both network candidates together took 47.077246 seconds. Network fit included validation each epoch, while the Logistic Regression timer excluded later validation prediction. One run, scheduling and different stopping rules prevent a definitive speed ranking.

### 45. Were the differences statistically significant?
No significance test, repeated-run uncertainty estimate or confidence interval was produced. I report a descriptive difference on this official test set. It would be inaccurate to invent a p-value or claim a proven causal effect.

### 46. Did the higher test score mean the network kept learning?
No. Test accuracy 97.53% was slightly above validation accuracy 97.2833%, but the model did not learn during prediction. The partitions contain different examples, so their accuracies can differ.

### 47. Did you train again on all 60,000 images?
No. The final frozen models retained their original fits on the same 54,000 examples. Validation selected the network width. The rejected 50/50 network was not evaluated on the official test set.

## Demonstration, reproducibility and limitations

### 48. How does the demonstration work?
MNIST_Demonstration.ipynb loads trusted hash-checked saved models. sample_index selects an original training image. It converts that same image to float32/255 and prepares (1,784) for Logistic Regression and (1,28,28) for Keras. Predict returns model output and argmax converts the network's probabilities to a digit. Actual and predicted labels are displayed.

### 49. Which real example can you show?
The saved example is sample_index=0, actual label 5. Both models predict 5. Predetermined indices 0,1,2,10,100,1000,10000,59999 had actual labels 5,0,4,3,5,0,3,8 and both models matched each. They were not chosen by filtering for correctness.

### 50. What is the limitation of using training images for a demonstration?
Some examples may have helped fit the models, so correct demonstration predictions are not evidence of performance on previously unseen images. The demonstration illustrates loading, preprocessing and prediction. The separately held-out official test comparison provides independent evaluation. I do not report the eight demonstration cases as a new accuracy experiment.

### 51. Is there a user-drawn interface or a deployed web application?
No. The actual deliverable is a simple Jupyter Notebook selecting MNIST images by index. There is no drawing interface, web application or deployment.

### 52. How do you know the saved models stayed unchanged?
SHA-256 identifiers matched the frozen records before and after approved evaluation and demonstration. Model files, source notebooks and results were preserved. SHA-256 identifies file bytes and does not prove that a model is scientifically correct or make an untrusted model safe.

### 53. What are the main limitations?
Only MNIST, one fixed split and one run per candidate were studied. External handwriting, photographs, shifts and multi-digit strings were not assessed. There is no calibration assessment, peak-memory profile or inference benchmark. The comparison does not imply universal superiority.

### 54. What improvements would you recommend?
Repeat runs with multiple seeds, measure identical timing scopes and assess new handwriting using suitable preprocessing and fresh independent evaluation data. These are recommendations only. Since official test outcomes are already known, tuning in response to them would require an honest new protocol rather than calling the same test set untouched.

### 55. How did the 8 GB RAM constraint affect implementation?
Normalised images use float32, raw exploration keeps uint8, and reshapes avoid unnecessary image copies where possible. Pandas tables contain small summaries rather than 784-feature datasets. Neural networks trained sequentially. Peak memory was not measured, so I do not claim a quantified memory improvement.

### 56. How can someone review or reproduce the work?
The repository contains notebooks, pinned direct dependencies, trained artifacts, saved splits, histories, labels, predictions, metrics and relative figure paths. Reviewing saved outputs does not require retraining. The documented verifier checks saved artifacts without model inference. A fresh training reproduction must use a disposable copy so it does not replace the historical artifacts. Cross-machine identical results are not guaranteed.

### 57. What challenges were resolved?
An earlier dataset download failed with WinError 10013. After the student downloaded MNIST normally, the project used its local cache offline. Classroom output-width and batch inconsistencies were corrected explicitly. Logistic Regression converged without warnings.

### 58. What is your main conclusion?
Under the recorded MNIST protocol, the validation-selected 100/100 network outperformed Logistic Regression on final accuracy and macro metrics. The project demonstrates an evidence-based, classroom-influenced workflow with preserved artifacts, transparent limitations and a working index-based prediction demonstration.

## Evidence and background references

Experimental evidence: ../results/stage2/logistic_regression_run.json, ../results/stage3/neural_network_trials.csv, ../results/stage3/selected_settings.json, ../results/stage4/final_model_comparison.csv, ../results/stage4/confusion_pairs.csv, ../results/stage4/digit_8_metric_example.csv, ../docs/demonstration_verification.json and ../report/TECHNICAL_REPORT.md.

Classroom evidence: ../docs/CLASSROOM_METHODS.md. Original lesson files remain private and unchanged.

External references already checked for the technical report:

- Keras, MNIST digits classification dataset: https://keras.io/api/datasets/mnist/
- Scikit-learn, LogisticRegression API: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- Scikit-learn, Logistic regression user guide: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
- Keras, Layer activation functions: https://keras.io/api/layers/activations/
- Keras, SparseCategoricalCrossentropy: https://keras.io/api/losses/probabilistic_losses/#sparsecategoricalcrossentropy-class
- Keras, Adam: https://keras.io/api/optimizers/adam/
- Scikit-learn, model evaluation: https://scikit-learn.org/stable/modules/model_evaluation.html
- Scikit-learn, confusion_matrix: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
- Scikit-learn, data leakage guidance: https://scikit-learn.org/stable/common_pitfalls.html#data-leakage
- LeCun, Bottou, Bengio and Haffner (1998), Gradient Based Learning Applied to Document Recognition. Proceedings of IEEE 86(11), 2278–2324. Primary publication record: https://bottou.org/papers/lecun-98h
- Kingma and Ba, Adam: A Method for Stochastic Optimization. First arXiv submission 2014, conference publication 2015: https://arxiv.org/abs/1412.6980

These references explain concepts and historical context. They are not the source of this project's experimental accuracy values.

## Personal preparation checklist

- Replace the cover placeholders with confirmed personal details.
- Practise the 14 slides for approximately 9 minutes 30 seconds, excluding questions.
- Open the demonstration with the correct kernel and existing cache before speaking.
- Practise one metric calculation and explain the actual-versus-predicted matrix axes.
- Distinguish recorded observations, possible explanations and future recommendations.
- Use MNIST_Presentation_Compatible.pptx or MNIST_Presentation_Compatible.pdf. The student confirmed Windows opening, and Stage 7 rendered and inspected all 14 PDF pages. Rehearse on the presentation computer before submission.
