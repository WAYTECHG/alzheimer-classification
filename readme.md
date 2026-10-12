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

## Run locally

Clone the renamed repository and enter its directory:

```bash
git clone https://github.com/WAYTECHG/alzheimer-classification.git
cd alzheimer-classification
```

Use an environment containing the dependencies imported by the application and the required model artifacts. Start the repository’s actual entry point:

```bash
streamlit run main.py
```

The application expects saved clinical and MRI models under `models/`. Check `config.py` and the prediction pages for their configured paths. A complete, pinned dependency file and reproducible MRI training pipeline are not currently documented, so this is not a verified clean-install procedure.

## Evaluation and implementation limitations

The clinical notebook applies SMOTE before the train/test split. Re-evaluation should split data first and restrict resampling to training folds to avoid contamination of the held-out evaluation.

The current MRI preprocessing function returns an uninitialized array rather than the converted uploaded image. Its dimensions also differ from the presentation’s training resize. The inference path needs correction against the saved model’s input specification and validation before uploaded-image predictions can support the reported training results.

The presentation’s scores describe the original coursework experiments; they do not establish clinical validity. This project is for education and research demonstration, not medical diagnosis.

## License

The repository includes an [MIT License](LICENSE). Dataset and third-party model terms remain applicable to their respective assets.
