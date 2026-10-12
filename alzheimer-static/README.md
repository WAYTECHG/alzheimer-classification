---
title: Alzheimer Classification
emoji: 🧠
colorFrom: blue
colorTo: yellow
sdk: static
app_file: index.html
pinned: false
license: mit
---

# Alzheimer Classification — Static Edition

A browser-only version of the existing Streamlit project, preserving its dark navy theme, blue headings, yellow accents, sidebar page selector, forms, brain animation, and team page. It runs the supplied MRI and clinical classifiers in the browser. **No Docker, Python web server, API key, or paid inference endpoint is needed for deployment.**

Project: [WAYTECHG/alzheimer-classification](https://github.com/WAYTECHG/alzheimer-classification)

## Features

| Page | Static implementation |
| --- | --- |
| Home | Definition, Reason and Aim tabs, original local brain animation |
| Chatbot | Clearly labelled offline FAQ chat with suggestion buttons; no Gemini requests |
| Alzheimer Prediction Data Info | Saved LogisticRegressionCV inference in JavaScript; optional local CSV charts |
| Alzheimer Prediction MRI | Saved custom CNN converted to ONNX; inference with ONNX Runtime Web / WebAssembly |
| About us | Original group members and roles |

The sidebar becomes a menu button on smaller screens. Layouts adapt to phones without fixed-height cropping.

## Hugging Face — Static Space

1. Create a new Space and select **Static** as its SDK.
2. Upload the **contents** of this folder to the Space repository root. `README.md` and `index.html` must be at the root, not inside an extra folder.
3. Include `models/`, `vendor/`, `assets/`, `app.js`, `models.js`, and `styles.css`. Wait for the Space to publish.

This README already contains `sdk: static` and `app_file: index.html`. No build command is needed. Do not select Docker or Streamlit.

[Official Static Spaces documentation](https://huggingface.co/docs/hub/spaces-sdks-static) states that Static Spaces are free and served without compute.

For command-line Git uploads, model and WASM files may need the Hub's large-file tooling. The web upload interface is the simplest option. Do not upload just a Git LFS pointer instead of the actual binary content.

## Render — Static Site

1. Add this folder as `alzheimer-static/` in your GitHub repository and commit its contents, including the actual model and WASM files.
2. In Render, choose **New → Static Site** and connect the repository.
3. Use these settings:

| Setting | Value |
| --- | --- |
| Root Directory | `alzheimer-static` |
| Build Command | `true` |
| Publish Directory | `.` |

If you create a separate repository containing this folder's contents at its root, leave **Root Directory** blank instead.

An optional `render.yaml` blueprint is included for the `alzheimer-static/` subfolder layout. For Blueprint deployment, copy it to the GitHub repository root. For a separate repository, remove its `rootDir` setting.

Render Static Sites are free to deploy, subject to the workspace's included bandwidth and pipeline limits. See [official documentation](https://render.com/docs/static-sites). The approximately 16 MB model and 11 MB WASM runtime are downloaded to each visitor's browser on first use; normal browser HTTP caching may reduce repeat downloads.

## Preview on your laptop

From the extracted `alzheimer-static` folder:

```powershell
python -m http.server 8000
```

Open `http://localhost:8000`. Python is only a convenient local file server here. The deployed app does not run Python. Opening `index.html` directly with `file://` will not reliably load the modules and model files.

Use a modern desktop browser with WebAssembly support. Older browsers and devices with limited memory may fail to load the MRI model. Browser speed and available memory determine inference performance. No shared compute server is required.

## Models and preprocessing

### MRI classifier

- The **actual supplied checkpoint is a custom Sequential CNN**, not a VGG16 checkpoint, although the presentation discusses VGG16.
- Parameter count: **4,152,356**.
- Architecture: initial convolution layers, four separable-convolution blocks with batch normalization and pooling, followed by dense layers and a four-class softmax output.
- Export: `models/mri.onnx`, converted from the supplied Keras checkpoint with inference mode enabled. No retraining was performed.
- Input tensor: `[1, 176, 208, 3]` (NHWC: height 176, width 208, RGB).
- Input values: raw RGB **0–255**, not divided by 255.
- Class order follows the supplied app: Mild Dementia, Moderate Dementia, Non Dementia, Very Mild Dementia. The original training encoder was not supplied to independently verify this order.

The browser resizes uploaded images using Canvas high-quality smoothing. This differs from Pillow Lanczos used in the Python evaluation, so scores may differ for images requiring resizing. Conversion checks compare identical input tensors separately from resizing behavior. Browser inference is not a clinically validated assessment.

### Clinical classifier

- The supplied model is **LogisticRegressionCV** with 22 named features and classes `AD`, `CN`, `LMCI`.
- `models/clinical.json` contains the saved coefficients, intercepts, classes, and original feature order. JavaScript computes the same multinomial softmax inference as the checkpoint.
- Inputs are placed in the model's **named feature order**, correcting the original app's inconsistent positional one-hot order.
- The original fitted preprocessing pipeline was not supplied. The form uses raw numeric values and named one-hot indicators, so matching the saved estimator's math does not establish that the entire original training pipeline is reproduced.
- Displayed percentages are model output scores, not calibrated diagnostic confidence.

## What results can be reported?

Keep presentation results separate from the included checkpoint's measured behavior:

| Evidence | Result | Limitation |
| --- | --- | --- |
| Presentation MRI baseline | Approximately 87% evaluation accuracy | Slides describe a VGG16 workflow |
| Presentation best tuning trial | 91.54% test accuracy; 91.523% F1 | Not reproduced with the supplied checkpoint; evaluation splits vary |
| Presentation manual MRI rerun | 76.25% accuracy | Different run/settings |
| Clinical notebook | 71.58% test accuracy; 72.38% weighted F1 | SMOTE before splitting creates a data-leakage risk |
| Supplied custom CNN on 1,279 extracted test images | 52.78% accuracy with raw RGB and Pillow Lanczos resizing | Exploratory checkpoint evaluation; original training preprocessing and encoder unavailable |
| Training accuracy | Not documented in the supplied materials | Do not invent a training score |

Converting to ONNX preserves the checkpoint's inference; it does **not** improve accuracy or validate the presentation scores.

## Dataset sources

MRI dataset confirmed by the project owner:

[Alzheimer's Dataset — 4 Class of Images (Kaggle)](https://www.kaggle.com/datasets/psraju2024/alzheimers-dataset-4-class-of-images)

The supplied archives contain 5,121 training and 1,279 test images across MildDemented, ModerateDemented, NonDemented, and VeryMildDemented. These counts describe the supplied archives; they need not match every version of the Kaggle listing. Full train/test folders are not required to deploy this inference app.

The clinical presentation cites [ADNI](https://adni.loni.usc.edu/). This is a separate dataset from the Kaggle MRI images. The original clinical CSV was not supplied; obtain it through its authorized access process. Clinical charts accept a locally selected CSV with these exact fields:

```text
Age,Education_Year,MMSE,Gender,Ethnicity,APOE4_Status,Imputed_Genotype,APOE_Genotype,DX.bl
```

`DX.bl` uses `AD`, `CN`, or `LMCI`. Numeric values belong in `Age`, `Education_Year`, and `MMSE`. All counts and correlations use loaded records; scatter/strip plots sample at most 10,000 rows for browser performance. Clinical model prediction works without a CSV.

## Privacy and FAQ chat

MRI images, clinical form values, selected CSVs, and chat messages are processed locally in the browser. The app has no upload endpoint, analytics, or third-party CDN scripts. The static host still receives normal requests for public app files. Messages reset when the tab is reloaded; New Chat clears them immediately.

The static FAQ is a collection of fixed educational responses. It does not generate answers, browse current research, or contact Gemini. A live Gemini integration requires a backend or other secure proxy; do not put a private Gemini API key into public JavaScript or static Space variables.

Background educational reading: [Mayo Clinic — Alzheimer's disease](https://www.mayoclinic.org/diseases-conditions/alzheimers-disease/symptoms-causes/syc-20350447). The app is an educational demonstration and cannot diagnose, prescribe treatment, or estimate validated personal risk.

## Files and dependencies

There is no runtime `requirements.txt`. All browser libraries are included locally:

- ONNX Runtime Web 1.20.1
- Lottie Web 5.12.2
- Plotly.js 2.35.2
- PapaParse 5.4.1

Their license files are in `vendor/`. Project code follows the original repository's MIT license; dataset access and reuse are subject to the dataset providers' terms.

`tools/` and `validation/` contain development conversion/check scripts and evidence. They are not needed to host the app and can be omitted when uploading. Do not use the Python conversion tools as a deployment build step.

Optional developer reconversion: use Python 3.11 or 3.12, install `tools/requirements-conversion.txt` in a separate environment, then run `python tools/convert_models.py --model-directory PATH_TO_MODELS --test-archive PATH_TO_TEST_ZIP`. The model directory must contain `alzheimer_mri.keras` and `alzheimer_info_model.pkl`. This regenerates the local image/tensor fixtures used by `node tools/browser-check.cjs` (requires Playwright and its Chromium browser). Dataset images, tensor fixtures, and internal browser screenshots are omitted from the downloadable package; numerical validation reports are included.

## Validation

`validation/conversion.json` records TensorFlow versus ONNX checks and the clinical export check. `validation/browser.json` records local Chromium testing, including real browser MRI inference, clinical inference, CSV charts, chat, navigation, and desktop/mobile layout checks. The direct browser model check uses the same input tensor as TensorFlow.

Hosting configuration is prepared for both platforms. Public Render and Hugging Face deployments have not been created or verified from this package.
