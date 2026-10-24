# Submission-copy portability changes

Preparation writes target this submission workspace and local temporary helpers, not the original notebooks, models, results, report, classroom sources, assignment PDF or AGENTS.md. Copy hashes and individual changed-file flags are in [source_artifacts.json](source_artifacts.json).

The following seven copied files differ from their sources. Stage 7 also records the notation-only report correction below:

1. **MNIST_Final_Project.ipynb:** cell 4 now obtains the project root from the working directory and builds the local Keras path with os.path.join. The exact-machine interpreter assertion becomes a Python 3.11 check, with a repository-root check. Cell 3 explains the portable check. The kernel name matches the documented registration. Absolute machine paths in historical text outputs, installed-library citations and metadata are redacted.
2. **MNIST_Final_Evaluation.ipynb:** cell 4 uses the same portable root and Python check. Kernel metadata and historical absolute paths are sanitised. No inference, metrics or cache-branch code is changed.
3. **report/TECHNICAL_REPORT.md:** the interpreter path is redacted; links to the unavailable private PDF/classroom folder point to explicit repository summaries. Findings, numerical tables, mathematical explanations and figure links are unchanged.
4. **results/stage2/logistic_regression_run.json:** the private interpreter path is redacted.
5. **results/stage3/neural_network_run.json:** the private interpreter path is redacted.
6. **results/stage4/evaluation_record.json:** private interpreter/cache paths are replaced by placeholders or repository-relative paths, including the nested historical verification metadata.
7. **results/stage4/frozen_models_before_evaluation.json:** model paths become repository-relative. Frozen model, settings and dataset SHA-256 identifiers remain unchanged.

Path redactions affect documentary strings, not numbers. Source and submission hashes therefore differ for these text files. Models, CSV metrics, NPZ prediction/split arrays, settings and figures retain identical bytes.

The repository's .gitattributes disables automatic newline conversion so that checkout on another platform does not invalidate artifact checksums. Binary models, arrays and images are also explicitly marked binary. No existing artifact was normalised or rewritten for line endings.

Every code cell other than the two setup cells is unchanged. Both copies carry a metadata note explicitly stating that the portable copies were not executed again. Their numerical outputs come from the original verified experiments.

The final audit detected concurrent saves of both original notebooks after the submission copies were made. Differences were limited to text/plain display representation and language_info.version changing from 3.11.17 to 3.13.9. Code, execution counts, numerical results and PNG bytes were unchanged. The preparation did not write or restore those original files; the cause of the concurrent saves is unconfirmed. The submission retains the approved starting snapshots and their correct recorded training environment. All other protected originals remain byte-identical. Details are in verification.json.

Launch from the repository root and select the registered kernel. The dataset cache is deliberately absent; obtain it separately using the README instructions before either notebook executes. No training, inference, library installation or model modification was needed for these changes.


## Stage 7 documentation correction

The submission report's display-math normalization formula contained inline dollar delimiters inside its float32 numerator. Stage 7 replaced them with parentheses, expressing float32(I) divided by 255 correctly. This is a notation-only correction. Model code, numerical values, figures and original report bytes remain unchanged. The source catalog retains the previous submission hash and the corrected copy's hash, together with the reason. No notebook code or metadata was edited in Stage 7.

## Stage 9 administrative changes

Only the submission report header and closing administrative checklist were personalised using confirmed student details. The entire analytical body, equations, results, references and figure links were preserved. Optional unconfirmed supervisor and official submission-date fields were omitted. The source catalog retains the previous report-copy hash and its original source hash, and records the new submission hash with this reason. The complete PDF is a typeset derivative of this Markdown, not a new experiment. Models, all three notebooks and saved results remain byte-identical.
