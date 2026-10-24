"""Verify shipped evidence using saved arrays; no models or dataset are loaded."""
from pathlib import Path
from urllib.parse import unquote
import ast
import base64
import csv
import hashlib
import json
import re
import struct
import sys
import numpy as np

root = Path(__file__).resolve().parent
local_only = {'.git', '.keras', '.cache', '.ipynb_checkpoints', '__pycache__'}
assert sys.version_info[:2] == (3, 11), 'Use the documented Python 3.11 kernel.'
catalog = json.loads((root / 'docs/source_artifacts.json').read_text(encoding='utf-8'))
for item in catalog['copied_files']:
    path = root / item['file']
    assert path.is_file()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item['submission_sha256'], item['file']

notebooks = {}
for name, expected_count in [('MNIST_Final_Project.ipynb', 40), ('MNIST_Final_Evaluation.ipynb', 21),
                             ('MNIST_Demonstration.ipynb', 11)]:
    nb = json.loads((root / name).read_text(encoding='utf-8'))
    assert nb['nbformat'] == 4 and nb['nbformat_minor'] == 5
    assert isinstance(nb['metadata'], dict) and isinstance(nb['cells'], list)
    assert len({c['id'] for c in nb['cells']}) == len(nb['cells'])
    count = 0
    for cell in nb['cells']:
        assert cell['cell_type'] in ['code', 'markdown', 'raw']
        assert isinstance(cell['metadata'], dict)
        assert isinstance(cell['source'], (str, list))
        if cell['cell_type'] != 'code':
            continue
        count += 1
        ast.parse(''.join(cell['source']))
        assert isinstance(cell['execution_count'], int)
        assert isinstance(cell['outputs'], list)
        for output in cell['outputs']:
            assert output['output_type'] in ['stream', 'display_data', 'execute_result']
            if output['output_type'] == 'stream':
                assert output['name'] in ['stdout', 'stderr']
                assert isinstance(output['text'], (str, list))
            else:
                assert isinstance(output['data'], dict) and isinstance(output['metadata'], dict)
                if output['output_type'] == 'execute_result':
                    assert isinstance(output['execution_count'], int)
            if 'image/png' in output.get('data', {}):
                encoded = output['data']['image/png']
                assert base64.b64decode(''.join(encoded) if isinstance(encoded, list) else encoded).startswith(b'\x89PNG')
    assert count == expected_count
    notebooks[name] = {'code_cells': count, 'check': 'JSON structure and syntax; no execution'}

lr = json.loads((root / 'results/stage2/logistic_regression_run.json').read_text())
nn = json.loads((root / 'results/stage3/neural_network_run.json').read_text())
final = json.loads((root / 'results/stage4/evaluation_record.json').read_text())
with np.load(root / 'results/stage2/split_indices.npz', allow_pickle=False) as data:
    train, validation = data['train'], data['validation']
assert train.shape == (54000,) and validation.shape == (6000,)
assert np.intersect1d(train, validation).size == 0
assert np.array_equal(np.sort(np.concatenate((train, validation))), np.arange(60000))
assert hashlib.sha256(train.tobytes()).hexdigest() == lr['train_indices_sha256']
assert hashlib.sha256(validation.tobytes()).hexdigest() == lr['validation_indices_sha256']
settings = json.loads((root / 'results/stage3/selected_settings.json').read_text())
assert settings == nn['selected_settings'] == final['selected_nn_settings']
assert settings['hidden_units_per_layer'] == 100 and settings['output_units'] == 10
assert settings['epochs'] == 5 and settings['batch_size'] == 100 and settings['initialization_seed'] == 42
for filename, digest in final['model_hashes_before'].items():
    assert hashlib.sha256((root / 'models' / filename).read_bytes()).hexdigest() == digest

checked = {}
for stage, prefix, expected in [
    ('stage2', 'logistic_validation', lr['validation_metrics']),
    ('stage3', 'neural_validation', nn['selected_validation_metrics']),
    ('stage4', 'logistic_test', final['test_metrics']['Logistic Regression']),
    ('stage4', 'neural_test', final['test_metrics']['Selected Neural Network'])]:
    with np.load(root / 'results' / stage / (prefix + '_predictions.npz'), allow_pickle=False) as data:
        actual, predicted = data['actual'], data['predicted']
        count = 10000 if stage == 'stage4' else 6000
        assert actual.shape == predicted.shape == (count,)
        assert np.isin(actual, np.arange(10)).all() and np.isin(predicted, np.arange(10)).all()
        if stage != 'stage4':
            assert np.array_equal(data['validation_indices'], validation)
        else:
            assert np.array_equal(data['test_indices'], np.arange(10000))
        if 'probabilities' in data.files:
            assert np.array_equal(data['probabilities'].argmax(axis=1), predicted)
        if prefix == 'logistic_test':
            test_labels = actual.copy()
        if prefix == 'neural_test':
            assert np.array_equal(test_labels, actual)
    matrix = np.zeros((10, 10), dtype=np.int64)
    np.add.at(matrix, (actual, predicted), 1)
    assert matrix.sum() == count
    tp = np.diag(matrix).astype(float)
    support, totals = matrix.sum(axis=1), matrix.sum(axis=0)
    precision = np.divide(tp, totals, out=np.zeros(10), where=totals > 0)
    recall = np.divide(tp, support, out=np.zeros(10), where=support > 0)
    f1 = np.divide(2 * tp, support + totals, out=np.zeros(10), where=(support + totals) > 0)
    metrics = {'accuracy': float(tp.sum() / count), 'macro_precision': float(precision.mean()),
               'macro_recall': float(recall.mean()), 'macro_f1': float(f1.mean())}
    for key, value in metrics.items():
        assert np.isclose(value, expected[key], rtol=0, atol=1e-12), (prefix, key)
    saved = np.loadtxt(root / 'results' / stage / (prefix + '_confusion_matrix.csv'),
                      delimiter=',', skiprows=1, dtype=int)[:, 1:]
    assert np.array_equal(matrix, saved)
    checked[prefix] = dict(metrics, correct=int(tp.sum()), incorrect=int(count - tp.sum()))

with (root / 'results/stage4/final_model_comparison.csv').open(newline='') as file:
    rows = list(csv.DictReader(file))
for row, key in zip(rows, ['logistic_test', 'neural_test']):
    for label, metric in [('Test accuracy', 'accuracy'), ('Macro precision', 'macro_precision'),
                          ('Macro recall', 'macro_recall'), ('Macro F1', 'macro_f1')]:
        assert np.isclose(float(row[label]), checked[key][metric], rtol=0, atol=1e-12)
    assert int(row['Correct predictions']) == checked[key]['correct']
    assert int(row['Incorrect predictions']) == checked[key]['incorrect']

links_checked = 0
for path in root.rglob('*.md'):
    if local_only.intersection(path.relative_to(root).parts):
        continue
    contents = path.read_text(encoding='utf-8')
    for link in re.findall(r'!?\[[^\]]*\]\(([^)\s]+)\)', contents):
        if link.startswith(('https://', 'http://', '#')):
            continue
        target = (path.parent / unquote(link.split('#')[0])).resolve()
        assert target.is_relative_to(root) and target.exists(), (path.name, link)
        links_checked += 1
    if path.name == 'TECHNICAL_REPORT.md':
        assert len(re.findall(r'!\[[^\]]+\]\(([^)]+)\)', contents)) == 11
for path in root.rglob('*.png'):
    if local_only.intersection(path.relative_to(root).parts):
        continue
    raw = path.read_bytes()
    assert raw.startswith(b'\x89PNG\r\n\x1a\n')
    assert all(n > 0 for n in struct.unpack('>II', raw[16:24]))

files = sorted(path.relative_to(root).as_posix() for path in root.rglob('*')
               if path.is_file() and not local_only.intersection(path.relative_to(root).parts))
assert files == (root / 'SUBMISSION_FILES.txt').read_text().splitlines()
forbidden = {'.keras', '.cache', '.ipynb_checkpoints', '__pycache__', 'classroom_reference',
             '.codex', '.agents', '.aws', '.venv', 'venv', 'envs', 'memories'}
for name in files:
    assert not forbidden.intersection(Path(name).parts), name
    assert Path(name).name not in ['AGENTS.md', 'Data Science Project.pdf']
    if Path(name).suffix in ['.md', '.json', '.txt', '.py', '.ipynb']:
        contents = (root / name).read_text(encoding='utf-8')
        assert not re.search(r'(?<![A-Za-z0-9])[A-Za-z]:[\\/]|/(?:Users|home)/', contents), name

print(json.dumps({'status': 'passed', 'notebooks': notebooks,
                  'copied_artifacts': len(catalog['copied_files']), 'submission_files': len(files),
                  'valid_local_links': links_checked, 'report_figures': 11,
                  'metrics_from_saved_predictions': checked,
                  'models_loaded': False, 'training_calls': 0, 'prediction_calls': 0,
                  'fresh_install_and_training_reproduction': 'Not performed'}, indent=2))
