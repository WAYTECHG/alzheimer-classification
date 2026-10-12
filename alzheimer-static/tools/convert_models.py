"""Development-only conversion. Hosting the static folder requires no Python."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
import json
import argparse
from pathlib import Path
import joblib
import numpy as np
import tensorflow as tf
import tf2onnx
import onnxruntime as ort
from PIL import Image
import zipfile

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model-directory', type=Path, default=ROOT.parent / 'alzheimer-deploy/models')
parser.add_argument('--test-archive', type=Path, default=ROOT.parent / 'upload/test.zip')
args = parser.parse_args()
SOURCE = args.model_directory
model = tf.keras.models.load_model(SOURCE / 'alzheimer_mri.keras', compile=False)
signature = [tf.TensorSpec([1, 176, 208, 3], tf.float32, name='images')]

@tf.function(input_signature=signature)
def inference(images):
    return model(images, training=False)

tf2onnx.convert.from_function(inference, input_signature=signature, opset=17,
                             output_path=str(ROOT / 'models/mri.onnx'))
session = ort.InferenceSession(str(ROOT / 'models/mri.onnx'), providers=['CPUExecutionProvider'])
validation = {'architecture': 'Custom Sequential CNN', 'parameters': model.count_params(),
              'input_shape': [1, 176, 208, 3], 'input_scale': 'raw RGB 0..255',
              'checks': [], 'source': 'Public repository checkpoint; no retraining performed'}
with zipfile.ZipFile(args.test_archive) as archive:
    names = [n for n in archive.namelist() if n.lower().endswith('.jpg')][:3]
    for i, name in enumerate(names):
        with archive.open(name) as file:
            pixels = np.asarray(Image.open(file).convert('RGB').resize((208, 176), Image.Resampling.LANCZOS), dtype=np.float32)[None]
        expected = model(pixels, training=False).numpy()
        actual = session.run(None, {session.get_inputs()[0].name: pixels})[0]
        difference = float(np.max(np.abs(expected - actual)))
        assert np.allclose(expected, actual, atol=1e-4, rtol=1e-4), difference
        validation['checks'].append({'image': name, 'maximum_absolute_difference': difference,
                                     'tensorflow_scores': expected.tolist(), 'onnx_scores': actual.tolist()})
        if i == 0:
            pixels.tofile(ROOT / 'validation/mri-input.f32')
            (ROOT / 'validation/mri-expected.json').write_text(json.dumps(expected.tolist()))
            Image.fromarray(pixels[0].astype('uint8')).save(ROOT / 'validation/sample-mri.png')

clinical = joblib.load(SOURCE / 'alzheimer_info_model.pkl')
export = {'kind': 'LogisticRegressionCV', 'classes': clinical.classes_.tolist(),
          'features': clinical.feature_names_in_.tolist(), 'coefficients': clinical.coef_.tolist(),
          'intercept': clinical.intercept_.tolist(),
          'notes': 'Inference uses saved named feature order. Original preprocessing pipeline is unavailable.'}
(ROOT / 'models/clinical.json').write_text(json.dumps(export, indent=2))
validation['clinical'] = {'classes': export['classes'], 'features': len(export['features'])}
rng = np.random.default_rng(2024)
x = rng.normal(size=(10, len(export['features'])))
scores = x @ clinical.coef_.T + clinical.intercept_
scores -= scores.max(axis=1, keepdims=True)
prob = np.exp(scores); prob /= prob.sum(axis=1, keepdims=True)
expected = clinical.predict_proba(x)
assert np.allclose(prob, expected, atol=1e-12)
validation['clinical']['maximum_absolute_difference'] = float(np.max(np.abs(prob - expected)))
(ROOT / 'validation/clinical-fixtures.json').write_text(json.dumps({'features': x.tolist(), 'expected': expected.tolist()}))
(ROOT / 'validation/conversion.json').write_text(json.dumps(validation, indent=2))
print(json.dumps(validation, indent=2))
