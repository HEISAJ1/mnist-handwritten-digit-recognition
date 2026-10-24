# Reproducibility scope

## What the evidence establishes

The supplied artifacts come from completed local experiments: seed-42 fitting/validation memberships, float32/divide-by-255 preprocessing, the fixed Logistic Regression settings, both five-epoch networks and validation selection. The final models were frozen before official test evaluation. Historical result records contain actual versions, fit scopes, metrics and integrity identifiers.

The submission preparation checks import compatibility in the existing dedicated environment, notebook JSON structure and Python syntax, all referenced figures, copied binary hashes, result arithmetic from saved arrays, explicit file inclusion and private-path removal. It does not call fit, predict, evaluate, model-loading APIs or the dataset loader.

Run **python verify_submission.py** from the repository root for the saved-artifact checks. The verifier uses NumPy and Python's standard library. It does not need the dataset cache or a loaded model. The JSON review record describes preparation-time checks.

## Installation and notebook frontend

The exact versions of the modelling libraries and Joblib/IPython/ipykernel are specified in requirements.txt. Python 3.11.17 was recorded. The existing dedicated environment has no JupyterLab/Notebook server or nbformat package. Use an Anaconda frontend with the registered kernel, or install a frontend separately following its official instructions.

No installation was attempted during preparation. A fresh Conda/PyPI solve, package availability for another operating system and a GUI launch have not been verified. Direct pins are not a full transitive lockfile. Do not present the static notebook structural checks as a new execution or GUI test.

## Two different meanings of reproduction

**Audit the historical results:** review notebook outputs and reconstruct all metrics from the supplied saved labels/predictions. This avoids extra final inference. The final evaluation notebook also reuses its supplied prediction records after checking dataset/model identifiers when executed in a disposable copy.

**Repeat model training:** use a separate, disposable copy of the repository and a fresh kernel with the same recorded settings. Main-notebook fitting/saving cells replace artifacts in that copy. Stop after validation. Seed control does not guarantee identical floating-point values or timings on different platforms, and saved native archives can have different hashes even when numerical behaviour agrees.

The frozen final-evaluation guards are tied to the shipped artifacts. Freshly trained models must not be mixed with those historical prediction records. Do not change the guards to force a mismatch through. A future new final experiment requires a separately defined protocol, with test independence considered because the original test outcomes are now known.

## Remaining verification

- Fresh environment creation and dependency installation on a clean machine.
- Interactive Jupyter opening/rendering with the documented kernel.
- Full training execution in an isolated reproduction copy, only when authorised.
- Confirm any institution-specific formatting, citation or disclosure rules. Confirmed academic details are complete in Stage 9.
- The project is already hosted in the private GitHub repository linked from README. Stage 9 separately authorises a final cleanup commit and push. Confirm lecturer access while retaining private visibility.

No full fresh-install or training reproducibility claim is made at this stage. The supplied working results and import/static checks are identified separately.

## Stage 6A demonstration

The no-inference preparation statements above describe Stage 5B and the saved-artifact verifier. Stage 6A added MNIST_Demonstration.ipynb and executed all eleven new code cells using the verified Python 3.11.17 environment. The notebook loads the two frozen models and predicts only selected original-training images. Eight indices were fixed before inference and retained regardless of outcome; changing the selection variable and rerunning the following cells was verified.

Both representations use the same float32 pixels divided by 255, shaped as (1,784) and (1,28,28). The official test arrays returned by the required loader are discarded, without new final test predictions or metrics. Existing model hashes and weights were verified unchanged. The existing local cache was used offline through the optional MNIST_KERAS_HOME override; the dataset is not distributed.

The new notebook's saved outputs and docs/demonstration_verification.json provide the executed evidence. This verifies the demonstration cells in fresh IPython, not a new GUI frontend installation or a new full training experiment.


## Stage 7 final audit

The repaired presentation and PDF are linked in the README. The student confirmed that the PowerPoint opens on Windows. Stage 7 rendered and visually inspected all 14 PDF pages using the existing Windows PDF API. Opening the PowerPoint was student-verified rather than repeated through automation.

The three submission notebooks retain Python 3.11.17 metadata and their existing code and outputs. Python 3.13.9 metadata remains only in the protected original project notebooks, a previously documented editor-save discrepancy. No notebook was rewritten to address it.

The saved-artifact verifier and current-environment import checks were run without loading models, training, predicting or loading MNIST. Direct dependency pins matched the dedicated Python 3.11.17 environment, and pip check reported no broken requirements. This is not a fresh-machine installation or a new full notebook execution. See [the final audit](../FINAL_SUBMISSION_AUDIT.md) and [machine-readable evidence](final_audit_verification.json).

## Stage 9 academic cleanup

The personalised presentation was checked against the approved deck: only the cover and its notes changed; slides 2–14 and the embedded project images remain unchanged. The report body from Abstract through the coverage table is unchanged; its administrative header and closing checklist were updated. A complete PDF was typeset from the Markdown using installed Pandoc and Matplotlib, without installing packages. The PDF was rendered with the Windows PDF API for visual review. This document conversion does not rerun notebooks or models. Safe saved-array verification is distinct from fresh-machine training reproduction, which remains untested.
