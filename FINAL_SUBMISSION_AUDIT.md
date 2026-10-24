# Final Submission Audit

**Project:** Comparative Analysis of Machine Learning and Deep Learning Techniques for Handwritten Digit Recognition  
**Scope:** Stage 7, local submission preparation before any GitHub publication  
**Audit date:** 24 October 2026

## 1. Readiness and counting convention

The original four-page project PDF was read in full. Its page 4, section 10 lists **nine final deliverable categories**. Eight have complete local artifacts and supporting evidence. One remains incomplete in its literal hosted form: the GitHub repository has been prepared locally but has not been committed, assigned a remote or published, as instructed.

**Formal deliverables: 9 required, 8 complete as local artifacts, 1 pending publication.** The twenty detailed categories requested for this audit are separately mapped below: 19 complete and 1 pending hosted publication. These counts describe different levels of the same checklist and must not be added together. There is no missing modelling experiment, metric, saved model, figure, presentation file or demonstration notebook.

The package is technically prepared for an approved private GitHub repository. It is **not yet a fully personalised academic submission**: six report fields and three presentation-cover fields remain placeholders. Fresh-machine environment installation, interactive Jupyter rendering and a new full training reproduction have not been established. These limitations are documented rather than presented as completed tests.

The local filenames and earlier project stage numbers do not exactly match the brief's methodology stage numbering. Evidence below uses actual notebook section names and one-based cell positions, including Markdown and empty cells. Cell positions are not execution counters.

## 2. The nine compulsory final deliverables

| ID | PDF requirement, page 4 section 10 | Status | Supporting submission files and evidence |
|---|---|---|---|
| D1 | Technical report with all nine named sections | Complete content; administrative fields pending | [Technical report](report/TECHNICAL_REPORT.md), sections 1–9, abstract, references and appendices. All eleven figure links resolve. One normalization-expression notation error was corrected in the submission copy only. |
| D2 | Python scripts and Jupyter Notebooks used for the project | Complete | [Main notebook](MNIST_Final_Project.ipynb), [final evaluation](MNIST_Final_Evaluation.ipynb), [demonstration](MNIST_Demonstration.ipynb), [safe evidence verifier](verify_submission.py). The analytical implementation is notebook based. Private local authoring/execution helpers are excluded and are not additional model implementations. |
| D3 | Data exploration, statistical analysis and visualisations | Complete | Main notebook cells 8–34, report sections 5.3 and 6.1, report/figures and existing results figures. |
| D4 | Trained Logistic Regression model | Complete | [Frozen classifier](models/logistic_regression.joblib), [input contract](models/logistic_regression_preprocessing.md), Stage 2 reload verification and SHA-256 comparison. |
| D5 | Trained Neural Network model | Complete | [Selected model](models/neural_network.keras), both trial archives, [input contract](models/neural_network_preprocessing.md), [selected settings](results/stage3/selected_settings.json), Stage 3 reload verification and hash checks. |
| D6 | Evaluation metrics, confusion matrices and comparison | Complete | [Final comparison](results/stage4/final_model_comparison.csv), [metrics](results/stage4/test_metrics.json), classification reports, prediction records and confusion-matrix CSV/PNG files. Saved-array checks reproduce all reported scores. |
| D7 | GitHub repository with source, README, dependencies and instructions | **Pending hosted publication** | Local Git repository, [README](README.md), [requirements](requirements.txt), [.gitignore](.gitignore), [reproduction guidance](docs/REPRODUCIBILITY.md), [inventory](SUBMISSION_FILES.txt). No commits, staged files or remotes. Creation of a private hosted repository requires later approval. |
| D8 | Final presentation explaining problem, methods, findings and lessons | Complete local artifact | [Repaired PowerPoint](presentation/MNIST_Presentation_Compatible.pptx), [matching PDF](presentation/MNIST_Presentation_Compatible.pdf), [speaker notes](presentation/PRESENTATION_SPEAKER_NOTES.md), [defense questions](presentation/PROJECT_DEFENSE_QA.md). Fourteen slides and fourteen populated PDF pages. The student confirmed Windows opening. All PDF pages were rendered and inspected during this audit. |
| D9 | Demonstration predicting handwritten digits | Complete, with existing execution evidence | [Demonstration notebook](MNIST_Demonstration.ipynb), cells 11–18, and [Stage 6A verification](docs/demonstration_verification.json). Both models predict the same selected image. Live delivery of the academic defense remains a student activity. |

## 3. Detailed requirement-to-evidence checklist

The following twenty rows cover the requested audit categories and the brief's objectives and methodology on pages 1–3.

| ID | Requirement | Status | Exact notebook section/cells and supporting evidence |
|---|---|---|---|
| A1 | Introduction and problem statement, including basic concepts | Complete | Main notebook cell 1; report sections 1–2 and 4 explain Data Science, AI, Machine Learning, Deep Learning, BI context and handwriting variation. Presentation slides 1–2 introduce the comparison. |
| A2 | Project aim and objectives | Complete | Main notebook cell 1; report section 3; README sections 1–3. Aim matches PDF pages 1–2. |
| A3 | MNIST data collection through the required Keras loader | Complete | Main notebook section 2, cells 6–7: tf.keras.datasets.mnist.load_data(). The official 60,000/10,000 partition and 28 × 28 grayscale contract are recorded. Keras cache is configured before TensorFlow import in cell 4. |
| A4 | Dataset structure, image samples, labels and class distribution | Complete | Main notebook cells 8–17 inspect/display images, cells 25–30 count/plot classes. Report figures 1–2 and section 6.1 retain those real outputs. Exploration uses only original training images. |
| A5 | Probability, descriptive statistics and meaningful observations | Complete | Main notebook cells 18–24 and 25–33 use a small table, describe(), pixel mean/std, empirical class proportions and a brightness histogram. Report sections 5.3 and 6.1 distinguish pixel population statistics from sample statistics of per-image means. |
| A6 | Numerical preparation, labels, scaling and validation separation | Complete | Main notebook section 11, cells 39–40: one seed-42 index shuffle and 54,000/6,000 slicing. Cells 41–44: float32/divide by 255, integer targets, 784-feature views. Cells 45–46 save the split. Section 21, cell 67 verifies exact reuse for the networks. Indices are disjoint and cover all 60,000 training memberships. |
| A7 | Scikit-learn Logistic Regression and explanation of pixel prediction | Complete | Main notebook sections 15–20, cells 48–63. Cell 50 constructs and fits reg_log with C=1.0, lbfgs and max_iter=1000. Cell 53 predicts validation labels. Report section 5.6 explains scores, coefficients, intercept and multinomial probabilities. |
| A8 | Simple taught-framework Neural Network, weights and activations | Complete | Main notebook sections 22–24, cells 68–74. Cells 69 and 73 build Sequential/Input/Flatten/two Dense-ReLU layers/Dense(10,Softmax). Report section 5.7 explains weights, biases, ReLU, Softmax, cross-entropy, backpropagation and Adam. Saved configuration was inspected without model deserialization. |
| A9 | Test a selected neural setting and record the experiment | Complete | Main notebook cells 69–77 and 79; [trial table](results/stage3/neural_network_trials.csv), both histories and learning_curves.png. Hidden widths 50/50 and 100/100, five epochs, batch 100, same initialization seed 42 and shared split. No CNN or extra architecture search. |
| A10 | Validation-only model selection and independent final test | Complete | Main notebook section 27, cells 80–81 selects highest final validation accuracy, tie favouring smaller model. Final evaluation cells 6–16 load frozen artifacts, verify hashes, prepare official test images and use saved final predictions on repeat runs. Stage 4 records are consistent with one original prediction pass per selected model and no test-based tuning. |
| A11 | Accuracy, precision, recall and F1 with multiclass interpretation | Complete | Main notebook cells 52–54, 82–83 and 88–90 cover validation. Evaluation sections 7–8, cells 17–21 cover test metrics and per-class reports. Section 14, cells 36–37 provides a real digit-8 calculation. Macro means equal-weight averaging over ten digits. |
| A12 | Confusion matrices and error interpretation | Complete | Main notebook cells 55–58 and 84–87 cover validation. Evaluation section 9, cells 22–25 covers test matrices. Sections 12–14, cells 30–37 show real error images, confusion counts and the worked example. Matrix rows are actual labels and columns predictions. |
| A13 | Actual training times and fair interpretation | Complete | Main notebook cell 50 times LR fit, cells 71 and 74 time network fits. Evaluation section 10, cells 26–27 reuses recorded times. Report section 7.4 states the different timing scopes and combined NN candidate cost. |
| A14 | Comparison, strengths, limitations and possible explanations | Complete | Main notebook section 30, cells 88–90 is validation only. Evaluation cells 26–29 and 35–40 cover final comparison and findings. Report sections 6–9 distinguish observations, plausible explanations, limitations and future recommendations without universal superiority or causal claims. |
| A15 | Complete evidence-based technical report and research | Complete content | Report sections 1–9, abstract, references and appendices A–C. All nine required headings exist. Thirteen external references were rechecked using their primary publication/documentation pages. Administrative placeholders are listed below. |
| A16 | Saved trained models and documented preprocessing | Complete | models contains four archives: LR, selected NN and both NN trials. Main notebook cells 60–63 and 92–95 record saving/reloading. Hashes agree with the frozen Stage 4 and Stage 6A records. Keras ZIP configuration confirms Dense widths 100,100,10. |
| A17 | Hosted GitHub repository | **Pending approval/publication** | Only github_submission contains the local Git repository. No remote or commit exists. Local content and inventory are ready; a hosted private repository has not been created. |
| A18 | README, dependencies, running/reproduction instructions and portability | Complete locally | README sections 4–19 and 20, requirements.txt, .gitignore, .gitattributes, docs/PORTABILITY.md and docs/REPRODUCIBILITY.md. Paths derive from the repository root; dataset caches and private classroom materials are excluded. |
| A19 | Final presentation and defense preparation | Complete local artifact | presentation contains the repaired PPTX/PDF, corresponding notes and 58 defense answers. Slides 1–14 cover context, exploration, preparation, both model families, experimentation, metrics, results, confusion matrices, demo and limitations. |
| A20 | Actual digit-prediction demonstration | Complete with historical execution evidence | Demonstration sections 4–6, cells 10–18, and section 7, cells 19–20. Stage 6A verified eleven code cells and eight predetermined training indices. README section 15 explains running and changing sample_index. No fresh inference was needed for this audit. |

### Additional compulsory brief coverage

- **Programming and libraries:** the notebooks use Python/Jupyter, NumPy, small Pandas tables, Matplotlib, Seaborn, Scikit-learn and TensorFlow/Keras. They contain no custom functions or classes and retain sequential classroom-style cells.
- **Mathematical/statistical understanding:** report sections 5.3–5.8 and 6.5 explain means, standard deviation, matrices/vectors, weighted scores, learned parameters, activations, sparse loss, macro metrics and matrix counts. Parameter arithmetic and real per-class calculations agree with saved data.
- **Challenges and recommendations:** report sections 7.6 and 9, README sections 17–20, speaker notes and defense answers cover download/cache issues, classroom corrections, timing limits and future work.
- **Source and processing explanation:** dataset source and cache contract are explicit. PDF page 3 calls SQL, APIs and cloud integration optional. Their absence is not a missing compulsory requirement. No web application or drawing interface was required or claimed.
- **Timeline and assessment:** the four-week table is an organisational plan, not evidence that the work was completed on particular historical dates. All eight weighted assessment areas have supporting artifacts: problem/background, preparation/EDA, mathematics/statistics, ML implementation, NN implementation, evaluation/comparison, report/reproducibility, and presentation/demo. This audit does not predict grades or certify the student's oral understanding.

## 4. Notebook and classroom-method verification

| Submission notebook | Total cells | Code cells | Current language metadata | Current kernel metadata | Audit result |
|---|---:|---:|---|---|---|
| MNIST_Final_Project.ipynb | 96 | 40 | Python 3.11.17 | Python (MNIST reproduction), mnist_reproduce | Valid structure, syntax and saved outputs. No error output cells. |
| MNIST_Final_Evaluation.ipynb | 41 | 21 | Python 3.11.17 | Python (MNIST reproduction), mnist_reproduce | Valid structure, syntax and saved outputs. No error output cells. |
| MNIST_Demonstration.ipynb | 23 | 11 | Python 3.11.17 | Python (MNIST reproduction), mnist_reproduce | Valid structure, syntax and saved outputs. No error output cells. |

The actual dedicated interpreter identified itself as Python 3.11.17. All ten direct dependency pins matched installed versions and imported successfully. pip check reported no broken requirements. These checks did not load a trained model or the dataset.

The protected original project and evaluation notebooks outside this repository still contain Python 3.13.9 language metadata from previously documented editor saves. This is not the training runtime. Their recorded environment evidence and the submission snapshots identify Python 3.11.17. Stage 7 did not change original or submission notebook metadata, code, output representations or numerical results.

The existing safe verifier checked cell types, IDs, execution-count fields, output structure, code syntax, embedded PNG signatures and expected code-cell counts. Inspection confirmed the explanation/preparation/fit/validation/selection sequence. Code is statically reviewed here, not newly executed. Relative paths are appropriate when launching from the repository root as documented. The dataset cache is intentionally absent from the submission, so a new installation must prepare it separately.

Key classroom evidence, reinspected against the source inventory:

| Project technique | Lecturer source, one-based cells | Correct reuse/adaptation |
|---|---|---|
| Familiar imports and aliases | Bank_Logistic_Regression.ipynb 1; linear regression.ipynb 1 | np, pd, plt, seaborn and sequential plotting style. |
| Descriptive summaries and labelled plots | linear regression.ipynb 4, 6, 8 | Small describe() table and simple labelled plots, extended to image statistics. |
| Paired shuffle and slicing | Untitled20.ipynb 6–7 | Same indexed memberships for images/labels, adapted to fixed 90/10 training-only split. |
| Float32 and divide by 255 | Untitled19.ipynb 3 | Same scaling rule using NumPy arrays. |
| Logistic fit and prediction | Bank_Logistic_Regression.ipynb 5–6; Bank_Data_Logistic_Regression (1).ipynb 6–12 | Required Scikit-learn multinomial API replaces binary Statsmodels, retaining x/y/reg_log reasoning. |
| Keras architecture and fit/evaluate | Untitled19.ipynb 7–11; Untitled20.ipynb 10, 12 | Simple dense Sequential, ReLU, Softmax, Adam, sparse loss and five-epoch sequence. Output corrected to ten and batch setting applied consistently. |
| Weights, biases and prediction | Untitled18.ipynb 5–8 | Educational explanation and direct prediction calls. |

The [full inventory](docs/CLASSROOM_METHODS.md) distinguishes additions: Scikit-learn multiclass API, fixed-seed checks, macro metrics, timing, persistence, hashes, History plots and argmax were necessary documented adaptations. Completed AI-produced MNIST notebooks were not treated as lecturer examples.

## 5. Models, split and numerical findings

The safe verifier reconstructs metrics from existing stored labels/predictions. This is arithmetic verification of saved evidence, not a new model evaluation or prediction run.

| Frozen model | Test accuracy | Macro precision | Macro recall | Macro F1 | Correct | Incorrect | Recorded fit seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 92.54% | 0.924432 | 0.924348 | 0.924303 | 9,254 | 746 | 52.845251 |
| Selected NN, 100/100 | 97.53% | 0.975181 | 0.974954 | 0.975045 | 9,753 | 247 | 23.330180 |

All four required headline metrics, per-class reports and both final confusion matrices agree with the saved prediction arrays. Each test matrix is 10 × 10 and totals 10,000. Both prediction records contain the same ordered actual labels and original test indices. Predictions are valid digits 0–9.

The accuracy difference is 4.99 percentage points and the error difference 499. LR's largest directional confusion is 5 predicted as 3, 38 times. The selected network's largest is 4 predicted as 9, 15 times. These are observations rather than proven causal explanations.

| Network candidate | Parameters | Final validation accuracy | Final validation loss | Recorded fit seconds |
|---|---:|---:|---:|---:|
| 50/50 | 42,310 | 96.4000% | 0.118696 | 23.747066 |
| 100/100 | 89,610 | 97.2833% | 0.093159 | 23.330180 |

Both histories contain five epochs and agree with the trial summary. Selection uses final validation accuracy, not official test results. Both networks use initialization seed 42, Adam, sparse categorical cross-entropy and batch 100. Saved split hashes agree across Stage 2, selected settings and final records. The 54,000/6,000 indices have no overlap and cover all original training memberships. No rejected-network final test run is claimed.

Both neural fits together took 47.077246 seconds. Neural fit timers include epoch validation while LR fit timing excludes later validation predictions. These are not interchangeable speed benchmarks.

| Included model | Bytes | SHA-256 |
|---|---:|---|
| models/logistic_regression.joblib | 32,271 | d97566f2cbb699d3ec2f8f7ba2370314adab7b449788c2917863c65dbbac6379 |
| models/neural_network.keras | 1,103,198 | 04b9c0ac6dd5128127311aea2ddc400d601f282d250004914348e21b0a94ccde |
| models/neural_network_100.keras | 1,103,198 | dea2fbecb7798396799cdf3d88bf30e79df139ebd4c68fd413e6e81ff57c6b53 |
| models/neural_network_50.keras | 535,584 | 697dd36023e8a14a66a7ba08891e5ec70228fd38c3ace0bb09fea278cd443bd0 |

Hashes identify file bytes. They do not certify model correctness or make arbitrary serialized files trustworthy. Only trusted project artifacts were inspected, without executing the Joblib archive or loading Keras weights.

## 6. Report review and administrative fields

The nine required report sections exist: Introduction, Problem Statement, Project Objectives, Background Research, Methodology, Results, Discussion, Conclusion and Recommendations. The abstract, references, figure/table captions and classroom evidence are present. All eleven referenced images exist and local links resolve.

The report's numerical tables agree with saved records. Mathematical review covered the normalization rule, 784-feature representation, multinomial score/probability formulas, weights/biases, ReLU, Softmax, sparse cross-entropy, parameter counts, macro metrics and confusion-matrix calculations. The digit-8 example agrees with saved TP/FP/FN counts. Findings remain conditional on this experiment and distinguish possible reasons from causal evidence.

A single malformed normalization expression contained inline dollar delimiters inside display mathematics. The **submission copy only** now uses function parentheses around the pixel value. Its result still means float32(I)/255. The previous file was backed up, the source catalog records its previous and current hashes, and the protected original report is unchanged. No numerical finding, model description, caption or citation was rewritten.

The report was written during Stage 5A, so its closing statements that presentation/demo were future phases remain historical. The README and this final audit provide current completion status rather than rewriting earlier findings.

Outstanding report fields:

- [STUDENT NAME]
- [MATRICULATION NUMBER]
- [INSTITUTION]
- [DEPARTMENT]
- [SUPERVISOR]
- [SUBMISSION DATE]

The repaired presentation cover contains the first three placeholders. The same fields are mirrored in its PDF. No identity, institution, supervisor, date, declaration or academic-assistance disclosure was invented. Confirm institutional formatting, citation and disclosure rules before final academic submission.

## 7. Presentation integration and demonstration

Both explicitly named repaired files were found in Downloads, opened as readable files, and copied byte-for-byte into presentation. The superseded deck was backed up before integration and moved into ignored local audit storage. It is absent from the intended submission and no active document link points to it.

| Active file | Bytes | SHA-256 |
|---|---:|---|
| presentation/MNIST_Presentation_Compatible.pptx | 467,256 | afaf9a3ebf217be782cb62b4942f125c852da65ea8ce53fff41e8e22c610b643 |
| presentation/MNIST_Presentation_Compatible.pdf | 492,590 | 75b84d2e361dc816df39dce03cc0cfefb769f61fc1c051f0198d184bf75d2e73 |

The PPTX has fourteen slides, fourteen notes parts, native text/tables and embedded images. Structural package checks pass. Its slide content agrees with the verified experimental numbers and the speaker-note slide order. The PDF has fourteen pages. Windows' installed PDF API rendered every page; each was individually inspected for content, figures, tables, clipping and empty areas. No material visual issue was observed.

**Opening attribution:** the student confirmed the repaired PowerPoint opens correctly on Windows. This audit independently checked its package/content and rendered the matching PDF; it does not claim a new automated PowerPoint launch.

The notes and defense document were retained, with only active-file links/status updated. The saved demonstration example remains index 0, actual digit 5, with LR prediction 5 and NN prediction 5. Stage 6A's record confirms both model loads, correct shapes (1,784)/(1,28,28), eleven executed code cells and eight predetermined indices. No unnecessary inference was repeated.

These demonstration images come from the original training partition and may include fitting examples. Correct displayed predictions illustrate the pipeline rather than prove generalisation. The independent evidence remains the frozen official test comparison. README instructions explain the kernel, cache, sample_index and which cells to rerun.

## 8. Reproducibility, security and preservation

Checks performed with the documented Python 3.11.17 interpreter:

1. Existing verify_submission.py passed for notebook structure/syntax, expected saved outputs, copied-artifact hashes, split coverage/disjointness, selected settings, model hashes, metrics reconstructed from saved predictions, confusion matrices, figure links and exact file inventory.
2. Installed direct dependency versions matched requirements.txt. All imported in the existing environment, and pip check reported no broken requirements.
3. Additional saved-data checks compared per-class reports, trial summaries, histories and final prediction identifiers. No model fitting, loading for inference, dataset loading or fresh predictions occurred.
4. The repaired presentation copies match the Downloads source hashes, and the retired deck is excluded.
5. Public files, textual notebook content, archive metadata and readable binary strings were inspected for private absolute paths, credentials and unrelated material. No actionable secret or private-path finding remained. An automated scan is not proof against every possible disclosure.
6. Git ignore checks exclude classroom sources, dataset caches, environments, agent configuration, credentials, execution helpers and the retired presentation backup. Models and the small split/prediction NPZ records remain intentionally included.
7. Before/after hashes confirm original project files, original report, classroom sources, all modelling notebooks, models and experiment result files were preserved. Authorized changes concern presentation integration, documentation accuracy, provenance and this audit only.

The [machine-readable audit evidence](docs/final_audit_verification.json) contains measured check results and preservation details. The [file inventory](SUBMISSION_FILES.txt) is the complete intended version-control list. The individual purpose review is [FILE_REVIEW.md](docs/FILE_REVIEW.md).

Remaining reproducibility limits: no fresh Conda/PyPI solve, no installation on another machine, no new GUI notebook run and no training repetition during this audit. Direct pins are not a complete transitive lockfile. The dedicated environment lacks a notebook server and nbformat; an existing Anaconda frontend can host the documented kernel. No package was installed to perform this audit.

## 9. Corrections and issues discovered

| Issue | Resolution or remaining status |
|---|---|
| Older presentation had compatibility/empty-area issues | Student supplied repaired files. Backed up and excluded the superseded deck. Copied both repairs unchanged, verified slide/page counts and inspected all PDF pages. |
| README and requirement/reproduction summaries had obsolete phase status | Added repaired-file links and current status, preserving historical numerical records. |
| Submission file-purpose list omitted presentation deliverables | Regenerated the individual purpose list and inventory for the complete current package. |
| Normalization equation had malformed Markdown/LaTeX delimiters | Corrected notation only in the submission report. Preserved the original and previous hash. |
| Protected original notebooks identify Python 3.13.9 in editor metadata | Documented the discrepancy. The submission notebooks and runtime records correctly identify Python 3.11.17. No notebook edit was needed. |
| Initial Windows PDF rendering helper could not resolve an AsTask overload | Used the explicit existing Windows Runtime action overload. All fourteen pages then rendered successfully, without installation. |
| One primary reference fetch initially failed | Retried the official Scikit-learn linear-model guide successfully. No fabricated replacement citation. |
| TensorFlow import printed oneDNN informational messages | Imports completed successfully. No model settings or environment versions changed. |
| Administrative information is absent | Explicit placeholders retained; student must supply confirmed details. |
| Hosted repository does not yet exist | Pending separate approval. No staging, commit, remote or publication performed. |

## 10. Research-reference verification

All thirteen report reference identities/links were revisited using primary sources during Stage 7. These references support background definitions, not this project's measured scores.

| Reference | Verified source and scope |
|---|---|
| R1 | [Keras MNIST](https://keras.io/api/datasets/mnist/): official dimensions, partitions and cache contract. |
| R2 | [TensorFlow MNIST loader](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist/load_data): return arrays and labels. The retrieved page still labels itself v2.16.1; the experiment runtime is 2.21.0, as the report explicitly distinguishes. |
| R3 | [LeCun, Bottou, Bengio and Haffner publication record](https://bottou.org/papers/lecun-98h): 1998 title, authors, Proceedings of IEEE 86(11), 2278–2324. |
| R4 | [Scikit-learn LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html): API, solver and regularisation settings. |
| R5 | [Official linear-model guide](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression): multinomial formulation. |
| R6 | [Keras activations](https://keras.io/api/layers/activations/): ReLU and Softmax. |
| R7 | [Keras sparse categorical cross-entropy](https://keras.io/api/losses/probabilistic_losses/#sparsecategoricalcrossentropy-class): integer-target loss. |
| R8 | [Keras Adam](https://keras.io/api/optimizers/adam/): optimizer and default learning rate 0.001. |
| R9 | [Kingma and Ba, arXiv 1412.6980](https://arxiv.org/abs/1412.6980): initial submission in 2014 and conference publication in 2015. |
| R10 | [Scikit-learn data leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage): test independence and consistent preparation. |
| R11 | [Scikit-learn evaluation guide](https://scikit-learn.org/stable/modules/model_evaluation.html): classification metrics. |
| R12 | [Scikit-learn F1](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html): F1 and macro averaging. |
| R13 | [Scikit-learn confusion matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html): actual-row/predicted-column convention. |

No unverified DOI, invented publication or external benchmark accuracy was introduced.

## 11. Final folder structure and remaining publication steps

~~~text
github_submission/
├── FINAL_SUBMISSION_AUDIT.md
├── README.md
├── SUBMISSION_FILES.txt
├── requirements.txt
├── .gitignore
├── .gitattributes
├── verify_submission.py
├── MNIST_Final_Project.ipynb
├── MNIST_Final_Evaluation.ipynb
├── MNIST_Demonstration.ipynb
├── models/                 # four model archives and two input contracts
├── results/
│   ├── stage2/             # 7 saved evidence files
│   ├── stage3/             # 11 saved evidence files
│   └── stage4/             # 20 saved evidence files
├── report/
│   ├── TECHNICAL_REPORT.md
│   └── figures/            # three saved exploration figures
├── presentation/
│   ├── MNIST_Presentation_Compatible.pptx
│   ├── MNIST_Presentation_Compatible.pdf
│   ├── PRESENTATION_SPEAKER_NOTES.md
│   └── PROJECT_DEFENSE_QA.md
└── docs/                   # methods, brief, provenance and verification records
~~~

Ignored local Git internals, caches and backups are not intended submission artifacts. Use SUBMISSION_FILES.txt for every individual filename.

Before publication:

1. Supply confirmed administrative details and decide whether they should appear in a private repository as well as the academic hand-in.
2. Confirm any institution-specific submission/disclosure rules and rehearse the presentation/demo.
3. Approve the GitHub account, repository name and **private** visibility.
4. Review the exact intended files and eventual staged diff before authorising a commit.
5. Only after separate approval, create the private remote and publish. Then verify remote privacy and uploaded artifact integrity.

**Final assessment:** local technical artifacts and documented evidence are ready for the separately approved private-repository step. Personalisation and hosted publication remain outstanding. No missing model experiment or fabricated result was identified. Stage 7 stops here without staging, committing, pushing or publishing.
