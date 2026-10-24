# Comparative Analysis of Machine Learning and Deep Learning Techniques for Handwritten Digit Recognition

## 1. Project overview

This undergraduate Data Science project compares Scikit-learn Logistic Regression with a simple TensorFlow/Keras neural network on MNIST. The work covers exploration, introductory statistics, preparation, validation experiments and a final comparison using frozen models. It follows the lecturer's sequential Jupyter Notebook style.

The completed results are **92.54% final test accuracy for Logistic Regression** and **97.53% for the selected 100/100 neural network**. This repository is a local submission preparation. No application deployment or GitHub publication is claimed.

A working [handwritten digit demonstration](MNIST_Demonstration.ipynb) now displays an index-selected image and predictions from both saved models. It uses original training images for illustration and does not repeat final test evaluation.

## 2. Problem statement

Handwriting varies in shape, thickness and style. The task is to classify an image as a digit from 0 to 9 using numerical pixels and labelled examples, then compare the two methods on previously unused examples.

## 3. Objectives

- Explore image structure, class distribution and intensity statistics.
- Prepare numerical features and a reproducible training/validation partition.
- Fit the required Logistic Regression classifier and two small dense networks.
- Select network width using validation data only.
- Compare frozen models using genuine final test metrics, errors and recorded fit times.
- Explain the mathematics, classroom adaptations, limitations and reproducibility evidence.

## 4. Technologies and libraries

Python **3.11.17**, Jupyter notebooks, NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, TensorFlow/Keras and Joblib were used. Exact direct dependency versions are in [requirements.txt](requirements.txt). IPython and ipykernel pins were additionally verified in the existing environment for notebook use.

This is primarily a notebook project. It has no web application, CNN, transfer-learning model, custom training loop or application framework. [Classroom-method evidence](docs/CLASSROOM_METHODS.md) identifies the original notebooks and one-based cell positions; private lesson notebooks are not distributed.

## 5. MNIST dataset

The required loader is **tf.keras.datasets.mnist.load_data()**. MNIST supplies 60,000 original training images and 10,000 official test images. Images are 28 × 28 grayscale arrays, with uint8 pixels from 0 to 255 and integer labels from 0 to 9.

The original training data were used for exploration. The official test set was kept separate throughout development and used only after model selection. It has now been evaluated and must not become a tuning set.

The dataset itself is not committed. The two modelling/evaluation notebooks configure a repository-local **.keras/datasets/mnist.npz** cache before TensorFlow imports. See Section 14 for first-download and offline instructions. Dataset SHA-256:

731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1

The [official Keras MNIST documentation](https://keras.io/api/datasets/mnist/) provides dataset provenance and licensing information.

## 6. Methodology

1. Load MNIST and inspect shapes, dtypes, sample images, labels, class counts and intensity statistics.
2. Set NumPy seed **42**, shuffle indices once and slice the first **54,000** for fitting and the remaining **6,000** for validation.
3. Check disjointness and full coverage, saving the exact indices in results/stage2/split_indices.npz. The split is random and unstratified.
4. Convert pixels to float32 and divide by 255 exactly once; preserve integer targets.
5. Reshape to **784 features** for Logistic Regression and retain **28 × 28** images for the network's Flatten layer.
6. Train and compare candidates using validation data only.
7. Freeze selected artifacts, record hashes and evaluate the same **10,000** official test examples.

The 8 GB RAM constraint motivated small Pandas tables, float32 arrays, shared reshape views and sequential network fits. Peak memory was not benchmarked.

## 7. Logistic Regression implementation

The model is Scikit-learn LogisticRegression with **C=1.0**, **solver='lbfgs'**, **max_iter=1000**, default tolerance **0.0001**, an intercept and default L2 regularisation. It converged in **235 iterations** without convergence warnings.

The lecturer mainly demonstrated Statsmodels binary Logit. This project preserves feature/target separation, familiar names such as x, y and reg_log, fitting and prediction. The brief requires Scikit-learn, so its multinomial implementation and internally handled intercept replace the binary API. No constant feature column or binary 0.5 cutoff is added.

## 8. Neural Network implementation

Both networks use:

Input(28,28) → Flatten(784) → Dense(ReLU) → Dense(ReLU) → Dense(10, Softmax).

The two hidden-layer configurations are **50/50** and **100/100**. Both use Adam, learning rate approximately **0.001**, sparse categorical cross-entropy, integer labels, **five epochs**, **batch size 100**, and initialization seed **42** reset before each construction.

The ten-neuron output corrects the classroom example's output-width mistake. No early stopping, dropout or further search was added.

## 9. Validation experiment and selection

| Hidden units | Parameters | Training accuracy (%) | Final validation accuracy (%) | Training loss | Final validation loss | Fit time (s) |
|---|---:|---:|---:|---:|---:|---:|
| 50/50 | 42,310 | 97.3111 | 96.4000 | 0.091234 | 0.118696 | 23.747066 |
| 100/100 | 89,610 | 98.2148 | 97.2833 | 0.059637 | 0.093159 | 23.330180 |

The final fifth-epoch validation accuracy selected **100/100**; an exact tie would have favoured the smaller model. Final-epoch weights were retained. Both trials together took **47.077246 seconds** of fit time. The rejected 50/50 model was not evaluated on the official test set.

| Validation-only comparison | Accuracy (%) | Macro precision | Macro recall | Macro F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 92.3667 | 0.923005 | 0.922510 | 0.922606 |
| Selected Neural Network | 97.2833 | 0.972724 | 0.972400 | 0.972546 |

These are development results, not independent final scores. See [trial records](results/stage3/neural_network_trials.csv) and [selection settings](results/stage3/selected_settings.json).

## 10. Final evaluation results

| Frozen model | Test accuracy (%) | Macro precision | Macro recall | Macro F1 | Correct | Incorrect | Recorded fit time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | **92.54** | 0.924432 | 0.924348 | 0.924303 | 9,254 | 746 | 52.845251 |
| Selected Neural Network (100/100) | **97.53** | 0.975181 | 0.974954 | 0.975045 | 9,753 | 247 | 23.330180 |

Both confusion matrices total **10,000**, and both models used the same test-image order. Macro metrics average the ten per-digit scores equally. Full per-class precision, recall, F1 and support are saved alongside the headline comparison.

Authoritative records: [comparison CSV](results/stage4/final_model_comparison.csv), [metrics JSON](results/stage4/test_metrics.json) and [evaluation record](results/stage4/evaluation_record.json). Predictions, labels, original indices and network probabilities are retained for independent numerical checks.

## 11. Comparison and interpretation

The selected network achieved **4.99 percentage points** higher test accuracy and made **499 fewer mistakes**. Nonlinear hidden layers can represent more flexible pixel combinations, but this experiment does not isolate a causal explanation or prove that a network will always win.

Common directional confusions included **5→3 (38)** for Logistic Regression and **4→9 (15)** for the network. Stroke similarity is a possible interpretation, not proven model reasoning.

Timing scopes differ: network fitting included validation after every epoch; Logistic Regression's fit timer excluded subsequent validation predictions. The selected network's **23.330180 s** must also be distinguished from **47.077246 s** for both candidates. These single measurements do not establish a definitive training-speed advantage.

## 12. Important visualizations

![Final model comparison](results/stage4/final_model_comparison.png)

*Final test metrics and recorded fit times. Timing scopes differ.*

![Neural-network learning curves](results/stage3/learning_curves.png)

*Training and validation accuracy/loss for both candidates over five epochs.*

The [technical report](report/TECHNICAL_REPORT.md) includes eleven captioned figures: digit samples, class counts, intensity distribution, learning curves, two validation confusion matrices, two final confusion matrices, final comparison and two misclassification grids.

## 13. Folder structure

The complete file-by-file inventory is [SUBMISSION_FILES.txt](SUBMISSION_FILES.txt).

~~~text
github_submission/
├── README.md
├── requirements.txt
├── .gitignore
├── .gitattributes
├── MNIST_Final_Project.ipynb
├── MNIST_Final_Evaluation.ipynb
├── MNIST_Demonstration.ipynb
├── verify_submission.py
├── SUBMISSION_FILES.txt
├── FINAL_SUBMISSION_AUDIT.md
├── presentation/
│   ├── MNIST_Presentation_Compatible.pptx
│   ├── MNIST_Presentation_Compatible.pdf
│   ├── PRESENTATION_SPEAKER_NOTES.md
│   └── PROJECT_DEFENSE_QA.md
├── models/                  # classifier, both NN trials, selected NN, input contracts
├── results/
│   ├── stage2/              # split, Logistic Regression validation records
│   ├── stage3/              # NN trials, histories and validation comparison
│   └── stage4/              # frozen final results, reports, plots and predictions
├── report/
│   ├── TECHNICAL_REPORT.md
│   └── figures/             # unchanged exports of existing exploration figures
└── docs/
    ├── PROJECT_BRIEF.md
    ├── CLASSROOM_METHODS.md
    ├── PORTABILITY.md
    ├── REPRODUCIBILITY.md
    ├── source_artifacts.json
    ├── environment_versions.json
    ├── demonstration_verification.json
    ├── final_audit_verification.json
    ├── FILE_REVIEW.md
    └── verification.json
~~~

All four model artifacts are included for ordinary Git tracking:

| Artifact | Bytes | Purpose |
|---|---:|---|
| models/logistic_regression.joblib | 32,271 | Frozen Logistic Regression |
| models/neural_network.keras | 1,103,198 | Selected 100/100 network |
| models/neural_network_100.keras | 1,103,198 | Preserved 100/100 trial |
| models/neural_network_50.keras | 535,584 | Preserved 50/50 trial; not final-tested |

No LFS or separate download service is needed for these sizes. Only load trusted project model files; serialized model formats must not be treated as safe for arbitrary downloads.

## 14. Installation

These are instructions for a future fresh setup, not installations performed during repository preparation. Use a **new, unused** environment name. In an Anaconda Prompt, first change into the downloaded repository root, then:

~~~bat
conda create -n mnist_reproduce python=3.11.17 pip
conda activate mnist_reproduce
python -m pip install -r requirements.txt
python -m pip check
python -m ipykernel install --user --name mnist_reproduce --display-name "Python (MNIST reproduction)"
~~~

The exact Python patch is the recorded version; its availability in a fresh Conda solve has not been tested. Do not silently change versions and claim an identical environment. Transitive packages are resolved by pip; this direct-dependency list is not a complete platform lockfile.

Use your existing Anaconda Jupyter Notebook/JupyterLab frontend and choose **Python (MNIST reproduction)**. A frontend was not installed inside the verified dedicated environment; IPython and ipykernel were present. If no frontend is available, install one separately following [Jupyter's official instructions](https://jupyter.org/install). Its version was not part of the recorded experiment and is not invented here. Kernel registration follows the [official IPython guide](https://ipython.readthedocs.io/en/stable/install/kernel_install.html).

Before either modelling/evaluation notebook runs, prepare the ignored local cache from the repository root:

~~~bat
python -c "import os; from pathlib import Path; os.environ['KERAS_HOME']=str(Path('.keras').resolve()); import tensorflow as tf; tf.keras.datasets.mnist.load_data(); print('MNIST cache ready')"
~~~

This first cache setup needs internet access. For offline use, place an already obtained, hash-verified mnist.npz at **.keras/datasets/mnist.npz**. Do not commit the cache. The notebooks intentionally stop when it is missing; they do not silently switch to another user's cache. This preparation loads the dataset but does not train a model or make predictions.

## 15. Running the notebooks

Open the frontend from the repository root, select the documented kernel and restart it before running cells. All three notebooks derive paths from the working directory, not a fixed Windows username. Set up the cache first.

For a future training reproduction, set the recorded thread limits in the Anaconda Prompt that launches the frontend, before starting its kernel:

~~~bat
set OMP_NUM_THREADS=2
set OPENBLAS_NUM_THREADS=2
set MKL_NUM_THREADS=2
set TF_NUM_INTRAOP_THREADS=2
set TF_NUM_INTEROP_THREADS=2
~~~

These are Windows Anaconda Prompt commands. On another shell use its environment-variable syntax. An already running frontend/kernel must be restarted to inherit them.

- **Review completed work:** open either notebook and read its preserved outputs without executing it.
- **Review final results:** use the saved CSV/JSON files and verify_submission.py. No training or model inference is needed.
- **Run the demonstration:** open [MNIST_Demonstration.ipynb](MNIST_Demonstration.ipynb), select the Python 3.11 project kernel, restart it and run all cells. Change **sample_index = 0** in **Choose an image** to any integer from **0 to 59999**, then rerun that cell and all following cells. The notebook shows the image, actual label, both predictions and whether each matches. Models are loaded without fitting or saving.
- **Future training reproduction:** execute MNIST_Final_Project.ipynb sequentially only in a separate disposable working copy. Its fit/save cells overwrite that working copy's artifacts.
- **Frozen evaluation notebook:** it loads the included frozen models and normally reuses the shipped, hash-checked prediction records. Its saving cells can rewrite copied result records. Use a disposable working copy if executing it. Do not delete cached predictions merely to repeat final-test inference.

Neither notebook was rerun during repository preparation. The submission copies retain original recorded numerical outputs, with machine paths redacted. Earlier stage stopping notes describe historical checkpoints; they do not cancel the completed later stages.

The sentence above concerns the two original project/evaluation notebooks in Stage 5B. The new demonstration was executed in Stage 6A: all eleven code cells passed, followed by index-change tests at **0, 1, 2, 10, 100, 1000, 10000 and 59999**, chosen before inference. See [the demonstration record](docs/demonstration_verification.json). These original-training examples are not an unbiased accuracy sample, and the demonstration does not recalculate final metrics or alter models.

The demonstration defaults to the ignored repository cache and downloads MNIST on first use if absent. To reuse an existing local cache without editing the notebook, set **MNIST_KERAS_HOME** before launching its kernel to the Keras directory containing **datasets/mnist.npz**. The notebook then sets KERAS_HOME before TensorFlow imports. Testing used an existing cache through this override, without copying the dataset into this repository or accessing the network. Never commit the cache.

## 16. Reproducing and verifying results

For an audit of the shipped evidence, run:

~~~bat
python verify_submission.py
~~~

This checks notebook structure/syntax, copy integrity, models' hashes, split membership, metrics reconstructed from **saved predictions**, local figure links and the explicit file inventory. It performs no model fitting, model loading or new predictions.

Full reproduction is a separate question. Fresh environment installation, GUI notebook execution and training were **not** repeated in this preparation stage. A future training reproduction must use the same split, seed, preprocessing and settings in an isolated copy. Timings and floating-point results may vary. Native model archive hashes need not repeat across new training/saving runs.

The final notebook intentionally rejects newly trained artifacts whose hashes differ from the frozen records. Do not weaken that check or replace the historical records to force a match. Fresh training reproduction should stop after validation; any new final experiment needs a separate evaluation protocol. See [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the scope and remaining checks.

## 17. Limitations and possible improvements

This was one split and one run per configuration. No repeated-run uncertainty estimate, external handwriting evaluation, inference-speed benchmark or peak-memory profile was produced. MNIST performance does not establish readiness for arbitrary photographs or other handwriting sources.

Future work could repeat development experiments across seeds, test limited classroom-appropriate settings on validation data, evaluate independent handwriting and measure comparable resource scopes. These are recommendations, not changes made here. Do not tune against the already observed official test errors.

## 18. Technical report and references

The [complete technical report](report/TECHNICAL_REPORT.md) contains the abstract, nine required sections, mathematics, detailed findings, classroom-method evidence, limitations and verified references. Administrative information remains as six labelled placeholders.

The private assignment PDF and classroom sources are not redistributed. [PROJECT_BRIEF.md](docs/PROJECT_BRIEF.md) summarises the applicable requirements; the report's source link points there explicitly. Its classroom source link points to the [method inventory](docs/CLASSROOM_METHODS.md).

## 19. Submission status and privacy

The models, numerical results, predictions and figures were copied from verified records. Adaptations comprise notebook setup paths, associated environment explanation, kernel metadata, private paths, unavailable private-source links and one Stage 7 report-notation correction. [PORTABILITY.md](docs/PORTABILITY.md) lists these changes.

No classroom notebooks, original assignment PDF, agent configuration, memory files, credentials, Anaconda environments, backups or dataset caches are included. Privacy scanning and artifact hashes are recorded in the local verification documentation; automated scans are useful checks, not proof that no conceivable sensitive content exists.

This workspace has no commit and no remote. Publishing, account/repository selection, visibility and any licensing decision remain for a separately approved step. No GitHub URL has been invented.


## 20. Presentation and final audit

Use the repaired [PowerPoint](presentation/MNIST_Presentation_Compatible.pptx) and [PDF](presentation/MNIST_Presentation_Compatible.pdf), each containing 14 slides/pages. The student confirmed PowerPoint compatibility on Windows. All 14 PDF pages were rendered and visually inspected during Stage 7. The older presentation is backed up locally and excluded from the submission inventory.

[Speaker notes](presentation/PRESENTATION_SPEAKER_NOTES.md) match the 14 slides. [Project-defense questions](presentation/PROJECT_DEFENSE_QA.md) explain the methods, mathematics, real findings and demonstration limitations. The demo remains a Jupyter Notebook using index-selected MNIST training images, not a drawing interface or web application.

The [final submission audit](FINAL_SUBMISSION_AUDIT.md) maps each compulsory brief requirement to exact files and notebook cell positions. [Audit evidence](docs/final_audit_verification.json) records the current checks. Eight of the brief's nine final deliverable categories are complete as local artifacts. The hosted GitHub repository category remains pending separate approval, although its local source, README, dependencies and running instructions are prepared.

No new model training or inference occurred during this audit. Saved prediction records confirm the reported metrics. The normalization formula in the submission report received one notation-only correction, documented in the source catalog. Modelling notebooks, model weights, experimental records, original report and classroom sources remain unchanged.

Before academic submission, supply the six report administrative details and the three cover fields. Fresh-machine installation and complete training reproduction have not been tested. Before publication, approve the account, repository name, private visibility and exact staged file list. This local workspace still has no commit, remote or publication URL.
