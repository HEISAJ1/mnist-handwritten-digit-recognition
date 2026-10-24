# Presentation Speaker Notes

Project: Comparative Analysis of Machine Learning and Deep Learning Techniques for Handwritten Digit Recognition

Use [the repaired PowerPoint](MNIST_Presentation_Compatible.pptx) or [its PDF](MNIST_Presentation_Compatible.pdf). These are the current submission versions. The superseded deck is excluded from the public inventory.

Fourteen slides. Suggested speaking time: approximately 9 minutes 30 seconds, excluding questions. Timings are rehearsal targets rather than measured delivery. Each numbered section corresponds to the same PowerPoint slide, and the explanations are also embedded in its speaker notes. Short “What to say” paragraphs form the spoken narrative. The additional explanation is for preparation and questions.

All experimental values come from existing saved records. The figures are the original project outputs. The original report and notebooks remain historical records, including their earlier stage-boundary statements.

## Presentation setup

- The cover identifies AJIBADE DARASIMI MICHAEL, matriculation number 2023/46544, Osun State University, Department of Computer Science. The report additionally identifies the Faculty of Computing and Information Technology. Optional unconfirmed supervisor and official submission-date fields are omitted.
- The student previously confirmed that the compatible PowerPoint opens correctly on Windows. Stage 9 integrated the downloaded personalised version, checked its editable slide content and speaker notes, and rendered and inspected all 14 matching PDF pages. Rehearse with the repaired deck on the presentation computer before submission.
- Keep the repository folder structure intact for the source figures and demonstration.
- Open MNIST_Demonstration.ipynb in Jupyter with the documented Python 3.11 kernel before speaking. Prepare the existing cache without running training or final test evaluation.
- Use the saved results for discussion. Do not alter models in response to demonstration or test errors.


## Slide 1. Comparative Analysis of Machine Learning and Deep Learning Techniques for Handwritten Digit Recognition

Suggested time: 20 seconds.

**What to say.** My project compares Scikit-learn Logistic Regression with a small TensorFlow/Keras neural network for handwritten digit recognition. I will explain the data preparation, the validation experiment, the final independent test results and the prediction demonstration.

**Before presenting.** Check the personalised cover: AJIBADE DARASIMI MICHAEL; 2023/46544; Osun State University; Department of Computer Science. Introduce yourself with these confirmed details. The full official project title remains unchanged. Supervisor and official submission date are omitted because they are unconfirmed and not required by the supplied brief.

**Source.** Original assignment PDF, pages 1 and 4. The technical report supplies the completed methodology and findings.

## Slide 2. Problem and project objectives

Suggested time: 35 seconds.

**What to say.** Handwriting varies in stroke shape, thickness and slant. This is supervised multiclass classification: each image has a known digit label from zero to nine. My aim was to compare a linear model with a simple nonlinear network, using the same data memberships and appropriate evaluation measures. I explored the data, prepared numerical pixels, trained and validated the models, and evaluated the frozen choices on a separate test set.

**Simple background.** Data Science combines programming and statistical reasoning to answer a question using data. Machine Learning is a part of Artificial Intelligence that learns relationships from examples. A multilayer neural network is a Machine Learning model studied in Deep Learning. Here, visual summaries support analysis rather than a deployed business application.

**Classroom connection.** Sequential Jupyter cells use familiar aliases such as np, pd, plt and tf. NumPy handles arrays, Pandas supplies small tables, Matplotlib and Seaborn supply plots. The workflow follows the classroom methods inventory, with necessary adaptations explained explicitly.

**Sources.** Assignment pages 1–3. Technical report sections 1–5.2. docs/CLASSROOM_METHODS.md.

## Slide 3. MNIST and exploratory analysis

Suggested time: 45 seconds.

**What to say.** MNIST contains 70,000 labelled grayscale images, each 28 by 28 pixels. Its official partitions are 60,000 training images and 10,000 test images. I explored only the original training images. The fixed sample grid shows different writing styles. The distribution is approximately balanced, although the counts differ: digit one occurs 6,742 times and digit five occurs 5,421 times.

**How to read the figures.** The upper figure shows the first twenty training images in their original order, with their real labels. It is a fixed illustration, not a representative random sample. The lower bar chart counts the ten digit categories, starting the vertical axis at zero.

**Statistical explanation.** The global mean pixel intensity is 33.318421 and the population standard deviation is 78.567490 on the raw 0–255 scale. Dark backgrounds help explain the low overall mean. These summaries describe all 47,040,000 training pixels. Per-image brightness summaries in the report describe a different statistical population. Labels are categories, so their arithmetic average is not interpreted as an average digit.

**Classroom evidence.** linear regression.ipynb cell 4 demonstrates describe(), and cells 6 and 8 labelled plotting. The assignment page 2 demonstrates grayscale image display. Counting image classes and displaying grids are necessary extensions.

**Figures.** report/figures/training_digit_samples.png and training_class_distribution.png, unchanged saved outputs from main-notebook cells 16 and 30. Dataset definition: https://keras.io/api/datasets/mnist/.

## Slide 4. Data preparation and separation

Suggested time: 45 seconds.

**What to say.** I shuffled the indices of the original 60,000 training images once using NumPy seed 42. The first 54,000 formed the fitting partition and the remaining 6,000 formed validation. I verified zero overlap and complete coverage, then saved the exact indices. Both model families used those same image memberships and labels. The official 10,000 test images remained separate until final evaluation.

**Numerical representation.** Each image is a 28 by 28 matrix. A pixel is a number from zero for black to 255 for white. Converting to float32 and dividing by 255 maps values to zero through one. For example, 255/255 is one and 128/255 is approximately 0.502. This is fixed scaling, with no statistic estimated from test data. Labels stay as integers.

**Code and shapes.** Logistic Regression reshapes each image into one row of 784 features because 28 × 28 = 784. The network retains shape (28,28) until Flatten rearranges it internally. A single demonstration input has shapes (1,784) and (1,28,28). Reshape changes arrangement, not the underlying pixel meaning. Float32 and avoiding large feature DataFrames help conserve the computer's 8 GB RAM.

**Classroom evidence.** Untitled20.ipynb cells 6–7 demonstrate paired index shuffling and slicing. Untitled19.ipynb cell 3 demonstrates float32 and divide-by-255 scaling. Untitled13.ipynb demonstrates reshape. The 90/10 split, fixed seed and explicit overlap checks are project adaptations.

**Sources.** results/stage2/split_indices.npz and logistic_regression_run.json. Technical report sections 5.4–5.5. The split is random and unstratified.

## Slide 5. Logistic Regression

Suggested time: 40 seconds.

**What to say.** Logistic Regression is a classification model despite its name. My model takes 784 normalised pixel features. For each digit it calculates a weighted score plus an intercept. Multinomial probabilities identify the most likely digit. It has no hidden layers and provides a useful baseline.

**Mathematics.** For class k, z_k = sum_j(w_kj x_j) + b_k. Here x_j is a pixel feature, w_kj is its learned coefficient and b_k is an intercept. The probabilities are p_k = exp(z_k)/sum_l exp(z_l). The prediction is the digit with the largest probability. This ten-class model does not apply a binary 0.5 cutoff.

**Explain the actual code.** reg_log = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000), then reg_log.fit(x,y). C controls inverse regularisation strength. Smaller C means stronger regularisation. L2 regularisation and the fitted intercept follow the recorded defaults. The solver converged in 235 iterations with no convergence warnings. The 1,000 setting is a solver iteration ceiling, not an epoch count.

**Classroom evidence and adaptation.** Bank_Logistic_Regression.ipynb cells 5–6 demonstrate x/y, reg_log and fitting. The original uses binary Statsmodels Logit. The PDF requires Scikit-learn, so this project uses its simplest appropriate multiclass equivalent. Scikit-learn handles the intercept internally, so I do not add a constant column.

**Sources.** Main notebook cell 50 and results/stage2/logistic_regression_run.json. Technical report section 5.6. Official API: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html.

## Slide 6. Classroom-style dense neural networks

Suggested time: 45 seconds.

**What to say.** Both networks use Sequential, Flatten, two Dense hidden layers with ReLU, and exactly ten Softmax outputs for digits zero to nine. The first trial has fifty units in each hidden layer, while the second has one hundred in each. I kept the other settings the same: Adam, sparse categorical cross-entropy, five epochs, batch size one hundred and initialization seed 42.

**Weights and biases.** Each neuron combines inputs using learned weights, then adds a learned bias. Weights determine contributions and bias shifts the score. ReLU returns max(0,z), introducing a nonlinear transformation. Softmax converts the ten final scores to nonnegative values summing to one. Argmax selects the largest entry. These probabilities have not been calibrated as guarantees of correctness.

**Loss and optimisation.** Sparse cross-entropy for one example is -log(p_y), where p_y is the probability assigned to its correct integer label. A confident incorrect answer therefore receives a large loss. Keras uses backpropagation to calculate gradients and Adam to update weights. Adam's recorded learning rate is approximately 0.001. The project uses the standard build, compile, fit and evaluate sequence, without a custom training loop.

**Classroom evidence.** Untitled19.ipynb cells 7–9 demonstrate Flatten, two ReLU layers, Softmax, compile and five-epoch fit. Cells 10–11 demonstrate evaluation. Untitled18.ipynb cells 5–7 inspect weights and biases. Untitled20.ipynb cell 10 demonstrates explicit Input.

**Necessary corrections.** The classroom example accidentally used the hidden width at the output. I use exactly ten outputs. I also match the actual (28,28) NumPy image shape and pass the intended batch size consistently. No CNN or transfer learning was introduced.

**Sources.** Main notebook cells 69 and 73. results/stage3/selected_settings.json. Activation definitions: https://keras.io/api/layers/activations/. Loss: https://keras.io/api/losses/probabilistic_losses/#sparsecategoricalcrossentropy-class. Adam: https://keras.io/api/optimizers/adam/.

## Slide 7. Validation experiment and model selection

Suggested time: 40 seconds.

**What to say.** I trained the candidates sequentially to conserve memory. The fifty/fifty model achieved final validation accuracy of 96.4000% and validation loss of 0.118696. The hundred/hundred model achieved 97.2833% and loss of 0.093159. My rule selected the highest final-epoch validation accuracy, with the smaller model winning an exact tie. Therefore I selected one hundred units in each hidden layer. I did not consult the official test data for this choice.

**How to read the table.** Validation accuracy is a percentage. Validation loss is sparse categorical cross-entropy, where a smaller value indicates less loss on those examples. Times are measured seconds for each fit, including validation during epochs. Parameter counts include weights and biases.

**Further detail if asked.** The smaller network has 42,310 parameters and the larger 89,610. A dense layer with m inputs and h neurons has m×h+h parameters. The larger model has 784×100+100, then 100×100+100, then 100×10+10, totalling 89,610. Final training accuracy/loss were 97.3111%/0.091234 and 98.2148%/0.059637. Both candidate fits together took 47.077246 seconds.

**Classroom connection.** The build/compile/fit sequence follows Untitled19.ipynb cells 7–9. Recording a controlled comparison, parameter totals and the explicit selection rule are documented project additions.

**Sources.** results/stage3/neural_network_trials.csv and selected_settings.json. Table values are rounded, with unrounded values preserved in those files.

## Slide 8. Learning across five epochs

Suggested time: 40 seconds.

**What to say.** These are the actual saved learning curves for both widths. Each row belongs to one network. The left panels show accuracy and the right panels show loss. Training and validation accuracy increased across the five epochs, while both losses decreased. The final training accuracy is above validation accuracy.

**Interpretation.** An epoch is one full pass through the fitting examples. Batch size one hundred determines how many fitting images contribute to each parameter update. Validation checks progress after an epoch and does not update weights. A training–validation gap can indicate limited generalisation, but five epochs and one run cannot establish that overfitting is absent. Training metrics aggregate batches while parameters change, whereas validation uses the end-of-epoch weights.

**Avoid an unsupported claim.** I did not select an earlier checkpoint, use early stopping or experiment with additional epochs. I selected between the two final-epoch configurations using validation accuracy only.

**Figure and classroom evidence.** results/stage3/learning_curves.png, unchanged Stage 3 output. The histories are history_50.csv and history_100.csv. Labelled line plotting follows Untitled18.ipynb cell 10 and linear regression.ipynb cell 8. Plotting Keras History values is an explained project addition.

## Slide 9. Evaluation methods

Suggested time: 40 seconds.

**What to say.** Accuracy is the fraction of all predictions that are correct. For one digit treated as the positive class, precision asks how many predictions of that digit are correct. Recall asks how many actual examples of it were found. F1 combines precision and recall. I calculated each digit's measures, then used their unweighted average, called macro averaging, so each of the ten classes contributes equally.

**Explain the formulas.** TP is a true positive, FP a false positive and FN a false negative for the digit under discussion. Precision = TP/(TP+FP). Recall = TP/(TP+FN). F1 = 2PR/(P+R). Macro F1 is the mean of ten per-digit F1 scores, not generally the F1 computed from macro precision and macro recall.

**Confusion matrix.** Rows are actual labels and columns predicted labels. The diagonal counts correct answers and off-diagonal cells count particular mistakes. Each final matrix totals 10,000. The models see the same ordered test examples, with preprocessing unchanged and no additional fitting.

**Why separation matters.** Validation influenced network width, so it is not an independent final assessment. The untouched official test partition assessed the already frozen choices. After seeing test errors I did not alter settings.

**Classroom evidence.** Bank_Data_Logistic_Regression (1).ipynb cells 10–12 and 17–21 demonstrate confusion tables and held-out reasoning. Library multiclass macro metrics are necessary additions.

**Sources.** Technical report sections 5.8–5.9. https://scikit-learn.org/stable/modules/model_evaluation.html and https://scikit-learn.org/stable/modules/generated/sklearn.metrics/confusion_matrix.html.

## Slide 10. Final test comparison

Suggested time: 55 seconds.

**What to say.** On the same ten thousand official test images, Logistic Regression achieved 92.54% accuracy and macro F1 of 0.924303. The selected network achieved 97.53% and macro F1 of 0.975045. That means 9,254 correct answers versus 9,753. The network made 247 errors compared with 746, a reduction of 499 errors and an accuracy difference of 4.99 percentage points. These are the observed results of this experiment.

**How to read the table.** Accuracy uses percentages. Macro precision, recall and F1 use the zero-to-one scale. Correct and incorrect counts each sum to 10,000 per model. The table rounds proportion metrics to six decimal places. The network also has the higher recorded precision and recall.

**Timing qualification.** Logistic Regression's recorded fit was 52.845251 seconds and the selected network's 23.330180 seconds. Both neural-network candidates together took 47.077246 seconds. Neural-network fit included validation during each of five epochs. Logistic Regression's timer included fit only and excluded later validation prediction. Scheduling and different stopping rules also affect times. The smaller recorded network time does not establish a definitive training-speed advantage.

**Preliminary validation context.** On the shared 6,000 validation examples, Logistic Regression accuracy was 92.3667%, macro F1 0.922606. The selected network achieved 97.2833%, macro F1 0.972546. Validation helped select the network, whereas final test results assessed the frozen choice.

**Sources.** results/stage4/final_model_comparison.csv, test_metrics.json, Stage 2 run record, Stage 3 trial table and validation_model_comparison.csv.

## Slide 11. Logistic Regression confusion matrix

Suggested time: 40 seconds.

**What to say.** This is the actual Logistic Regression test confusion matrix. Rows show the real digit and columns its prediction. The diagonal totals 9,254 correct answers, and all cells together total ten thousand. The largest directional mistakes were five predicted as three, thirty-eight times, and two predicted as eight, thirty-six times. Seven predicted as nine occurred thirty-four times.

**Interpretation.** A count from five to three describes only that direction. It does not include three predicted as five. Digits with more examples have more opportunities for mistakes, so per-class precision and recall complement these counts.

**Worked mathematical example for questions.** Of 974 true eights, Logistic Regression recognised 859. It missed 115 eights and incorrectly called 119 other digits eight. Precision for eight is 859/(859+119) = 0.878323. Recall is 859/974 = 0.881930. F1 is 2×859/(2×859+119+115) = 0.880123.

**Possible explanation.** Some strokes and open loops may resemble other digits. Visual similarity is a plausible interpretation, not a proven explanation of a particular model decision.

**Figure.** results/stage4/logistic_test_confusion_matrix.png, unchanged original figure. Counts agree with its CSV and confusion_pairs.csv. For detailed cell-by-cell examination, open the source PNG at full size. The slide highlights selected counts in large text.

## Slide 12. Neural Network confusion matrix

Suggested time: 40 seconds.

**What to say.** The selected network's real matrix totals ten thousand examples with 9,753 on the diagonal. Its most frequent directional error was four predicted as nine, fifteen times. Five predicted as three and eight predicted as five each occurred nine times. The network made fewer errors overall, while still confusing some handwriting patterns.

**Comparison and caution.** Two nonlinear ReLU layers can represent more flexible combinations of pixels than linear class scores, which is a plausible reason for the better performance here. However, the model families also differ in optimisation, regularisation and parameter count. This experiment does not isolate one causal explanation or establish that neural networks always win.

**Further detail.** For digit eight, the network had 939 true positives, 30 false positives and 35 false negatives. Precision was 0.969040, recall 0.964066 and F1 0.966547. Both models' first-error grids remain available in the report, but those first errors need not be the globally most frequent confusion pairs. I did not tune models after seeing these errors.

**Figure.** results/stage4/neural_test_confusion_matrix.png, unchanged original figure. Counts agree with its CSV and confusion_pairs.csv. Open the source PNG at full size for detailed inspection.

## Slide 13. Handwritten digit demonstration

Suggested time: 45 seconds.

**What to say.** The demonstration uses MNIST_Demonstration.ipynb and the two frozen trained models. Changing sample_index selects another original training image by its index. The same image is normalised once and shaped appropriately for each model. For the saved example, index zero has actual label five. Logistic Regression predicts five and the network predicts five.

**Live sequence.** Before speaking, open the repository in Jupyter, select the documented Python 3.11 kernel and run the demonstration once with its existing local cache. During the demonstration, show sample_index = 0, the image and the two predictions. If time permits, change to predetermined index 1, rerun the selection cell and all following cells, and show actual zero with both predictions zero. Never run the training notebook for this demonstration.

**Interpret the example honestly.** The original training partition supplies these images. A displayed image may have helped fit the models. Therefore a correct displayed answer illustrates the pipeline and does not establish performance on an unseen image. The separate official test results supply that evidence. There is no user-drawn-digit interface or web application.

**Verified examples.** Stage 6A fixed indices 0, 1, 2, 10, 100, 1000, 10000 and 59999 before prediction. Actual digits were 5, 0, 4, 3, 5, 0, 3 and 8, respectively. Both models matched each supplied label. Do not call eight correct examples a new accuracy experiment.

**Classroom evidence.** Matplotlib display follows the assignment page 2 and classroom labelled plotting. Prediction follows Untitled18.ipynb cell 8 and the logistic lessons. Argmax, native artifact loading and hash checks are documented additions.

**Sources and image.** MNIST_Demonstration.ipynb cell 12 contains the displayed image, exported byte-for-byte for this presentation. The prediction evidence is docs/demonstration_verification.json and notebook cell 16. No new inference occurred during presentation preparation.

## Slide 14. Limitations, recommendations and conclusion

Suggested time: 40 seconds.

**What to say.** The selected network performed better on this official MNIST test set, achieving 97.53% against 92.54% for Logistic Regression. Logistic Regression remains a useful simple baseline. The project showed how to move from classroom array operations and sequential modelling to a reproducible comparison with validation-based selection.

**Limitations.** This is one split and one run per candidate on prepared MNIST images. It does not establish performance on photographs, arbitrary external handwriting or multi-digit strings. A common seed supports traceability but does not measure uncertainty. The different timing scopes prevent a definitive speed ranking. Softmax probabilities were not calibrated.

**Recommendations.** A future study could repeat runs across seeds, measure equivalent timing scopes and assess genuinely new handwriting with suitable preprocessing and a fresh independent evaluation set. Since these official test outcomes are now known, future tuning should not be presented as independent evaluation on that same untouched test set. These are recommendations, not experiments completed here.

**Closing.** My main conclusion is a conditional, measured comparison, supported by saved models, real figures, numerical records and a simple prediction demonstration. The technical report contains the full mathematical explanations, per-class results and references.

**Sources.** Technical report sections 7–9. Its verified research references include Keras MNIST, Scikit-learn LogisticRegression and metric documentation, Keras activations/loss/Adam, LeCun et al. (1998) and Kingma and Ba (2014, conference 2015). These explain background rather than supply this project's accuracy values.

## Figure provenance

1. Slide 3: [training samples](../report/figures/training_digit_samples.png), main-notebook cell 16.
2. Slide 3: [training class counts](../report/figures/training_class_distribution.png), main-notebook cell 30.
3. Slide 8: [both networks' learning curves](../results/stage3/learning_curves.png).
4. Slide 11: [Logistic Regression final confusion matrix](../results/stage4/logistic_test_confusion_matrix.png).
5. Slide 12: [selected network final confusion matrix](../results/stage4/neural_test_confusion_matrix.png).
6. Slide 13: the exact saved PNG output of [demonstration notebook](../MNIST_Demonstration.ipynb) cell 12, sample_index 0. Predictions agree with [Stage 6A verification](../docs/demonstration_verification.json).

All six embedded images are genuine existing project outputs, not regenerated experimental plots. The comparison and experiment tables are editable native PowerPoint tables populated from the authoritative CSV files.

## Supporting documents

- [Technical report and its organised research references](../report/TECHNICAL_REPORT.md)
- [Classroom-method evidence](../docs/CLASSROOM_METHODS.md)
- [Final measured comparison](../results/stage4/final_model_comparison.csv)
- [Neural-network trial table](../results/stage3/neural_network_trials.csv)
- [Project defense questions](PROJECT_DEFENSE_QA.md)

Background sources in the slide notes were previously verified for the technical report. This presentation does not substitute external benchmark findings for the project's actual results.
