## Table of Contents
1 [Overview](#2-overview)  
2. [Repository Contents](#3-repository-contents)  
3. [How to Run the Code](#4-how-to-run-the-code)  
4. [Usage](#5-usage)  

---



## 1. Overview
This repository contains the code for the final project of the Intro to AI course. The project aims to predict stock prices using machine learning models, specifically Feedforward Neural Networks (FFNN) and Long Short-Term Memory (LSTM) networks.

## 2. Repository Contents

| File/Folder            | Description                                                                                            |
|------------------------|--------------------------------------------------------------------------------------------------------|
| `stock_prediction.py`  | The `stock_prediction.py` loads stock data, removes the DATE column, splits the data into 80% training and 20% testing, and normalizes it to the range [-1, 1] using MinMaxScaler. It  compiled with the Adam optimizer and MSE loss. The FFNN and LSTM train iteratively until the test MSE falls below 0.003 or reaches 100 epochs, outputting MSE per epoch and plotting predictions against actual values. |
| `test_tensorflow.py`   | File to verify TensorFlow version and GPU availability.                                             |
| `yFinance_API_test.py` | File to download stock data from Yahoo Finance API and save it as a CSV file.                      |
| `stock_data.csv`       | CSV file containing stock data downloaded from Yahoo Finance API.                                     |

---

## 3. How to Run the Code
This project may require specific environment that you may not have. If you encounter any issues, please try to follow the steps below to ensure a running successfully.

1. **Install Dependencies**:
   - Ensure you have Python installed.
   - Install the required libraries using pip via the command line:
     ```
     pip install pandas numpy tensorflow scikit-learn matplotlib yfinance
     ```

2. **Download Stock Data**:
   - Run `yFinance_API_test.py` manually to download the latest stock data from Yahoo Finance API and save it as `stock_data.csv`. Or you can run it automatically by adding it to your IDE's run configuration.
     ```
     python yFinance_API_test.py
     ```

3. **Check TensorFlow Version and GPU Availability**:
   - Run `test_tensorflow.py` manually to verify the TensorFlow version and check if a GPU is available.
   Or you can run it automatically by adding it to your IDE's run configuration.
     ```
     python test_tensorflow.py
     ```

4. **Train Models**:
   - Run `stock_prediction.py` manually to train the FFNN and LSTM models using the downloaded `stock_data.csv` file. Or you can run it automatically by adding it to your IDE's run configuration.
     ```
     python stock_prediction.py
     ```


## 4. Usage
- **Data Preprocessing**:
  - The `stock_prediction.py` loads the `stock_data.csv` file, preprocesses the data by normalizing it, and splits it into training sets and testing sets.

- **Model Training**:
  - The script defines and trains an FFNN model with four hidden layers.
  - It also defines and trains an LSTM model with a sequence length of 5 and a single hidden layer.
  - Both models are trained until the test Mean Squared Error (MSE) falls below a specified threshold or a maximum number of epochs is reached.

- **Model Evaluation**:
  - The plots the actual and predicted values for both the FFNN and LSTM models.
