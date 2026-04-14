# 🫀 Heart Disease Predictor

A clinical decision-support web app that predicts heart disease risk using **Logistic Regression with Bootstrap Uncertainty Quantification** — trained on the Cleveland Heart Disease dataset.

> Built as a portfolio project from a Jupyter Notebook ML study exploring Naive Bayes, Logistic Regression, Random Forest, XGBoost, and Bayesian Neural Networks.

---

## Live Demo
Deploy instantly on [Streamlit Community Cloud](https://streamlit.io/cloud) (free).

---

## Features

- **13-feature patient input form** (age, cholesterol, chest pain type, ECG results, etc.)
- **Bootstrap UQ (n=200)** — 200 resampled models produce a mean probability + 90% confidence interval
- **Confidence tiers** — High / Medium / Low based on CI width
- **Threshold = 0.30** — tuned to maximize recall (minimize missed diagnoses)
- Clean medical UI built with Streamlit

---

## Model Performance

| Model | Accuracy | Recall (Disease) |
|---|---|---|
| Naive Bayes | 86.7% | 0.71 |
| Logistic Regression (default) | ~88% | 0.82 |
| **LR + Bootstrap UQ (threshold=0.30)** | **~88–89%** | **0.89** |
| Random Forest | ~88% | 0.79 |
| XGBoost | ~85% | 0.75 |

**Key insight:** Simpler linear models generalized better than ensemble methods on this dataset. Adding bootstrap uncertainty quantification improved recall while providing prediction reliability scores.

---

## Run Locally

```bash
git clone https://github.com/apoorva-rapolu/heart-disease-predictor
cd heart-disease-predictor
pip install -r requirements.txt
streamlit run app.py
```

The app auto-downloads the Cleveland dataset from UCI on first run. No manual data setup needed.

---

## Project Structure

```
heart-disease-predictor/
├── app.py                  # Streamlit app
├── requirements.txt
├── notebooks/
│   └── Final.ipynb         # Full ML analysis notebook
└── README.md
```

---

## Dataset

[Cleveland Heart Disease — UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/heart+disease)  
303 samples · 13 features · Binary classification (disease present/absent)

---

## Tech Stack

`Python` `Streamlit` `Scikit-Learn` `Pandas` `NumPy`

---

## Disclaimer

This tool is for **educational purposes only** and is not a substitute for professional medical advice.
