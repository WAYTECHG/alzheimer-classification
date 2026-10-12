# NeuroInsight: Alzheimer’s MRI and Clinical Classification

An educational team project combining VGG16-based brain MRI classification and clinical-data machine learning in a multi-page Streamlit application. The project explores model selection, preprocessing, hyperparameter tuning and interactive presentation of predictions.

## Results

The original coursework presentation reports the following results. These scores have not been independently reproduced from the current repository.

| Workflow | Model | Accuracy | Precision | Recall | F1 score |
|---|---|---:|---:|---:|---:|
| Brain MRI | VGG16 CNN | 91% | 92% | 92% | 92% |
| Clinical data | Logistic Regression | 72% | 72% | 71% | 71% |

GridSearchCV tuning reduced clinical Logistic Regression accuracy from 72% to 69%; it did not improve this baseline. The project therefore demonstrates comparing experimental outcomes rather than assuming tuning always improves performance.

## My contribution

The presentation credits **Wilbert Andrew Yonathan** with MRI prediction and interface development. My work focused on the MRI classification workflow and the Streamlit application interface. Clinical model comparisons and the complete application were collaborative team work.

## Methods

### MRI classification

- Kaggle MRI dataset with more than 6,400 images, as described in the presentation.
- Four application labels: Mild Dementia, Moderate Dementia, Non Dementia and Very Mild Dementia.
- VGG16 CNN trained with Adam, with resizing, normalization and augmentation explored during preprocessing.
- Bayesian hyperparameter optimization using GPyOpt to explore learning rate, trainable layers, epochs and split settings.
- Evaluation using accuracy, precision, recall, F1 score and confusion matrices.

### Clinical-data classification

- ADNI structured features including demographic, cognitive and genetic information.
- Three labels: Alzheimer’s disease, late mild cognitive impairment and cognitively normal.
- Missing-value handling, one-hot encoding, exploratory analysis and SMOTE experiments.
- Nine classifier families: LDA, Logistic Regression, Random Forest, Ridge, Gradient Boosting, MLP, XGBoost, SVM and SGD. Linear and polynomial SVM variants were compared.
- Baseline and GridSearchCV-tuned model comparisons.

## Application

The Streamlit interface includes a home page, chatbot, clinical-data prediction, MRI upload and prediction, and team information. It brings both research workflows into one navigable prototype.

## Datasets

### Brain MRI images

The uploaded `train.zip` and `test.zip` contain **6,400 JPEG images**, all with native dimensions of **176 × 208 pixels**. The following counts were measured directly from those archives.

| Folder label | Training | Test | Total |
|---|---:|---:|---:|
| MildDemented | 717 | 179 | 896 |
| ModerateDemented | 52 | 12 | 64 |
| NonDemented | 2,560 | 640 | 3,200 |
| VeryMildDemented | 1,792 | 448 | 2,240 |
| **Total** | **5,121** | **1,279** | **6,400** |

Dataset download: [Alzheimer’s Dataset (4 Class of Images) — Kaggle, psraju2024](https://www.kaggle.com/datasets/psraju2024/alzheimers-dataset-4-class-of-images).

The counts above were verified from the project’s supplied archives. They also match the historical dataset description in [A deep learning framework for early diagnosis of Alzheimer’s disease on MRI images](https://doi.org/10.1007/s11042-023-15738-7), Section 3.1. The linked Kaggle copy was identified by the project owner; a byte-for-byte comparison against its download has not been performed.

Keep the supplied training and test split for comparable experiments and create a validation split within training data. The archives alone do not establish a patient-level split.

### Structured clinical data

The coursework presentation identifies the clinical-data source as the **Alzheimer’s Disease Neuroimaging Initiative (ADNI)**. The notebook reads a project-specific file named `alzheimer_dataset.csv`, using demographic, cognitive and genetic features with the target `DX.bl`.

- [ADNI official website](https://adni.loni.usc.edu/).
- [ADNI data access information](https://adni.loni.usc.edu/data-samples/adni-data/).

The exact ADNI export, version and processing steps that produced the local CSV have not been confirmed. The MRI ZIP files do not contain that CSV. Do not describe the CSV as an official ADNI release filename without its original export record.

### Local file locations

| Local path | Purpose |
|---|---|
| `main.py` | Streamlit entry point |
| `data/alzheimer_dataset.csv` | CSV expected by the clinical prediction page |
| `models/alzheimer_info_model.pkl` | Saved clinical classifier |
| `models/alzheimer_mri.pkl` | Saved MRI classifier |
| `train/<class>/*.jpg` | MRI training images |
| `test/<class>/*.jpg` | MRI test images |
| `.env` | Local chatbot configuration |

Place the CSV inside `data/`, not beside `main.py`. Extract the image archives so `train/` and `test/` each contain the four class folders directly. The Streamlit MRI page loads a saved model and an uploaded image; it does not retrain from these folders when launched.

## Run locally

Clone the renamed repository:

```bash
git clone https://github.com/WAYTECHG/alzheimer-classification.git
cd alzheimer-classification
```

Use Python 3.11 in a dedicated Conda environment. If the `alzheimer` environment already exists, skip its creation.

```powershell
conda create -n alzheimer python=3.11 -y
conda activate alzheimer
$env:PYTHONNOUSERSITE = "1"
python -m pip install --no-user -r requirements.txt
python -m pip check
```

The environment variable prevents older packages in the Windows user-site directory from overriding the isolated environment. Set it again when opening a new terminal.

Configure the existing chatbot locally in `.env`:

```dotenv
GOOGLE_API_KEY=your_api_key_here
```

Use a currently available model supporting `generateContent`. The original chatbot hard-codes `gemini-pro`, which failed with 404 during local troubleshooting. Listing models or migrating the chatbot code is required; dependency installation alone does not replace that model name. The requirements preserve the existing `google.generativeai` imports for compatibility; this SDK is deprecated.

Start from the directory containing `main.py`:

```powershell
python -m streamlit run main.py
```

The accompanying requirements target the existing app and clinical notebook on Python 3.11/Windows. The repository MRI artifact’s embedded metadata records Keras 3.3.3 and a save date of 22 June 2024. The requirements match that Keras version and select TensorFlow 2.16.1 as a compatible backend; the original TensorFlow training version remains unknown. Native Windows uses CPU inference with this TensorFlow version. GPyOpt, mentioned in the MRI presentation, is not needed to launch the app and is not included; reproducing those optimization experiments requires the original MRI training source and its separate environment.

The requirements passed a pip dependency-resolution dry run targeting CPython 3.11 on Windows x86-64, with Windows environment markers, on 12 October 2026. No packages were installed for that check. A clean installation and successful inference with the actual saved artifacts must still be verified locally. The clinical scikit-learn artifact’s original training version still needs verification. If upgrading an existing environment from Keras 2.15, stop Streamlit and uninstall `tensorflow`, `tensorflow-intel` and `keras` before installing this requirements file, to prevent stale shared TensorFlow files.

## Evaluation and implementation limitations

The clinical notebook applies SMOTE before the train/test split. Re-evaluation should split data first and restrict resampling to training folds to avoid contamination of the held-out evaluation.

The current MRI preprocessing function returns an uninitialized array rather than the converted uploaded image. Its dimensions also differ from the presentation’s training resize. The inference path needs correction against the saved model’s input specification and validation before uploaded-image predictions can support the reported training results.

The presentation’s scores describe the original coursework experiments; they do not establish clinical validity. This project is for education and research demonstration, not medical diagnosis.

## License

The repository includes an [MIT License](LICENSE). Dataset and third-party model terms remain applicable to their respective assets. The code’s MIT License does not grant rights to redistribute the MRI images or ADNI clinical records. Verify each source’s terms before adding datasets to the public repository, and keep `.env` out of Git.
