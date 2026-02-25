📊 Sales Forecasting and Interactive Dashboard in Python
📌 Overview

This project is an end-to-end sales forecasting application built using Python and Streamlit. It analyzes historical sales data and predicts future sales using multiple time series models such as ARIMA, Prophet, and LSTM. The project also provides an interactive dashboard to explore sales trends, seasonality, and forecasts.

🎯 Objective

Forecast future sales using time series analysis

Compare statistical and deep learning models

Visualize actual vs predicted sales interactively

Support data-driven business decision making

🚀 Features

Upload CSV sales data or use synthetic sample data

Interactive date range selection

Exploratory sales trend visualization

Time series decomposition (trend, seasonality, residual)

Forecasting using:

ARIMA

Prophet

LSTM (Deep Learning)

Model evaluation using MAE and RMSE

Combined forecast comparison chart

Future sales forecast table

Interactive dashboard built with Streamlit & Plotly

🛠️ Tech Stack

Programming Language: Python

Framework: Streamlit

Libraries:

Pandas, NumPy

Plotly

Statsmodels

Prophet

TensorFlow / Keras

Scikit-learn

📂 Dataset Format

The CSV file should contain the following columns:

date,sales
2023-01-01,250
2023-01-02,270

date → Date column

sales → Sales amount

⚙️ Installation

Clone the repository:

git clone https://github.com/harshsaini11/sales-forecasting-dashboard-python.git
cd sales-forecasting-dashboard-python

Install required packages:

pip install -r requirements.txt
▶️ How to Run the App
streamlit run app.py

The dashboard will open in your browser.

📈 Model Workflow

Data loading & preprocessing

Exploratory data analysis

Time series decomposition

Model training (ARIMA, Prophet, LSTM)

Model evaluation (MAE, RMSE)

Future sales forecasting

Interactive visualization

📊 Output

Sales trend plot

Seasonal decomposition charts

Actual vs forecast comparison

Model performance table

Future forecast table

🎓 Use Cases

Retail sales forecasting

Inventory & demand planning

Time series learning project

Data science portfolio project

🔮 Future Improvements

Add holiday & promotional effects

Support multiple products/stores

Add MAPE metric

Deploy dashboard on Streamlit Cloud

🤝 Contributing

Contributions are welcome!
Feel free to fork the repository and submit a pull request.

📄 License

This project is licensed under the MIT License.
