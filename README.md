# Grammar Scoring Engine for Spoken Data Samples

## Project Overview

This Grammar Scoring Engine is a machine learning-driven system developed to assess spoken audio clips by predicting their grammar proficiency scores. It processes audio samples ranging from 45 to 60 seconds and outputs a continuous grammar score between 0 and 5, aligned with the MOS Likert Grammar Scale. The solution utilizes Mel-Frequency Cepstral Coefficients (MFCCs) for extracting speech-relevant features and employs a Random Forest Regressor for prediction.

---

## Key Features

- Handles `.wav` format audio files using `librosa` for preprocessing and feature extraction.
- Computes 13 MFCCs per audio file, then summarizes them using **mean** and **standard deviation** to form feature vectors.
- Leverages a **Random Forest Regressor** model to estimate grammar scores.
- Evaluates model performance using **Pearson correlation**, **Mean Squared Error (MSE)**, **Mean Absolute Error (MAE)**, and **R² score**.
- Outputs predicted scores for test samples and saves them in a `submission.csv` file.

---

## Dataset

- **Training Set:** Contains 444 audio files labeled with grammar scores (`train.csv`).
- **Test Set:** Consists of 195 audio files (`test.csv`) with placeholder labels for prediction.
- **Audio Format:** All files are stored in `.wav` format.

---

## Requirements

To run this project, make sure the following tools are installed:

- **Python**: Version 3.8 or later
- **Jupyter Notebook** or **JupyterLab**
- All packages listed in the `requirements.txt` file

---

## Dependencies

This project relies on the following Python libraries:

- `pandas`
- `numpy`
- `librosa`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `scipy`
