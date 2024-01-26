# Rock vs Mine Prediction Project

## Overview

This project focuses on predicting whether an object is a rock or a mine based on sonar data. It utilizes a logistic regression model to analyze patterns in the data and make predictions.

## Code

### Dependencies

Make sure you have the following dependencies installed:

```bash
pip install numpy pandas scikit-learn
```

## Data Collection and Processing
The dataset (sonar.csv) is loaded into a Pandas DataFrame. The dataset contains sonar readings, and the labels denote whether the object is a rock ('R') or a mine ('M').

## Model Training
A logistic regression model is used for training. The dataset is split into training and test sets. The model is then trained on the training data.

## Model Evaluation
The accuracy of the model is evaluated on both the training and test data.

## Predictive System
A predictive system is implemented to make predictions on new data. You can use the provided example input data:

## Flowchart
![alt text](https://github.com/atharv-451/Rock-vs-Mine-Prediction/blob/main/screenshots/Flowchart.jpg)

## Usage
- Clone the repository: 'git clone https://github.com/atharv-451/Rock-vs-Mine-Prediction.git'
- Navigate to the project directory: 'cd rock-vs-mine-prediction'
- Run the main script: 'python main.py'

## Contributing
Feel free to contribute to the project by opening issues or submitting pull requests.
