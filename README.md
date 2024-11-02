# Bitcoin Price Forecasting

<img src="https://contenthub-static.crypto.com/wp_media/2023/05/image-34.png" width=80%>

### Project Overview

This project focuses on **predicting Bitcoin prices** using a variety of **time series forecasting techniques**, including replicating the **N-BEATS architecture** for time series forecasting. The implementation is based on TensorFlow and includes models like Dense Neural Networks, Recurrent Neural Networks (RNN), Long Short-Term Memory (LSTM), Convolutional Neural Networks (CNN), and **N-BEATS**.

The goal is to forecast future Bitcoin prices based on historical data, providing insights into model performance and challenges associated with time series forecasting in volatile financial markets. **Note**: This is for educational purposes and should not be used for financial advice.

- **Model Development**:
    - Baseline models
    - Dense Neural Networks (DNN)
    - Recurrent Neural Networks (RNN)
    - Long Short-Term Memory (LSTM) models
    - Convolutional Neural Networks (CNN) for time series data
    - **N-BEATS (Neural Basis Expansion Analysis of Time Series)**: A state-of-the-art model specifically designed for time series forecasting, known for its accuracy and scalability.

- **Evaluation and Comparison**:
  - Each model is evaluated using metrics such as Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE). The performance of **N-BEATS** is compared with other models to demonstrate its effectiveness.


#### N-BEATS Replication
This project replicates the **N-BEATS (Neural Basis Expansion Analysis of Time Series)** model, which is a powerful deep learning architecture designed for time series forecasting. N-BEATS is a general-purpose model that does not require feature engineering or domain knowledge, making it flexible and scalable.

- **Paper Title**: [N-BEATS: Neural Basis Expansion Analysis for Time Series Forecasting](https://arxiv.org/abs/1905.10437)
- **Authors**: Boris Oreshkin, Dmitri Carpov, Nicolas Chapados, Yoshua Bengio
- **Published in**: ICLR 2020

#### Dataset
The dataset used for this project is publicly available and includes historical Bitcoin prices, along with daily high, low, and opening prices. The dataset spans from 2013 to 2021, making it suitable for time series forecasting tasks.

- **Dataset Link**: [Bitcoin Historical Price Data](https://www.kaggle.com/mczielinski/bitcoin-historical-data)

#### Libraries and Dependencies
- **TensorFlow**: For building and training deep learning models.
- **Pandas**: For handling and preprocessing time series data.
- **Matplotlib/Seaborn**: For visualizing data and results.
- **NumPy**: For numerical computations.
- **Sklearn**: For preprocessing and evaluation metrics.

#### Project Structure
1. **Data Loading**: The dataset is loaded from a CSV file containing historical Bitcoin prices.
2. **Exploratory Data Analysis (EDA)**: Basic statistical analysis and data visualization.
3. **Model Building**:
   - **Baseline Model**: Simple forecasting based on previous values.
   - **DNN, RNN, LSTM, CNN Models**: Deep learning models built and trained on the time series data.
   - **N-BEATS**: Replication of the N-BEATS model, designed for highly accurate time series forecasting.
4. **Model Evaluation**: Each model is evaluated based on its prediction accuracy using metrics like MAE and RMSE.
5. **Visualization**: Plotting actual vs. predicted values to visually compare model performance.

---
Feel free to customize and extend it according to your specific needs!
