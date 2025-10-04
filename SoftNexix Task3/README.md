#  Time Series Analysis and Forecasting

This project demonstrates **Time Series Analysis** using Python to explore **trend**, **seasonality**, and **forecasting** through the **ARIMA model**. The analysis focuses on datasets such as **Apple Stock Prices**, **Energy Consumption**, and **Supermarket Sales**.

---

##  Objective

To analyze time-dependent data, identify key temporal patterns (trend and seasonality), and forecast future values using statistical time series modeling (ARIMA/SARIMA).

---

##  Datasets Used

1. **Apple Stock Prices (1980–2023)**  
   - Historical daily stock data of Apple Inc. including `Open`, `High`, `Low`, `Close`, and `Volume`.  
   

2. **Energy Consumption (2004–2018)**  
   - Hourly electricity demand data (`AEP_MW`) for regional power usage analysis.  
  

3. **Supermarket Sales (Optional)**  
   - Point-of-sale transactional data for retail sales analysis.  
   - **Note:** This dataset was unavailable during Trying to get the dataset for the link provided in the Pdf Task Given To us  
   -

---

##  Operations Performed

###  **Data Inspection and Cleaning**
- Verified dataset shape, data types, and checked for missing values.  
- Converted appropriate columns to `datetime` format and set as index.  
- Handled outliers or missing timestamps where applicable.

---

###  **Time Series Resampling**
- **Monthly resampling** for smoother visualization and trend detection.  
- **Quarterly resampling** to observe seasonal variations.  



---

###  **Visualization**
- Line plots of actual values.  
- Moving averages (6-month and 12-month) to visualize trends.  
- Seasonal decomposition to observe trend, seasonality, and residuals.

 **Insights:**  
- 12-month moving average highlights long-term growth trends.  
- Seasonal decomposition exposes cyclical variations in stock/energy usage patterns.

---

###  **ARIMA Forecasting**
- Split data into training and testing sets (e.g., last 12 months as test).  
- Fitted ARIMA/SARIMA model for forecasting future values.  
- Evaluated forecast accuracy using **Root Mean Squared Error (RMSE)**.  



##  Key Insights

| Dataset | Observation | Forecasting Result |
|----------|--------------|--------------------|
| **Apple Stock Prices** | Long-term bullish trend with minor seasonal fluctuations | ARIMA captured growth trend; ~5% RMSE |
| **Energy Consumption** | Daily cycles with strong weekly seasonality | SARIMA handled periodic demand patterns effectively |
| **Supermarket Sales** | (Skipped due to missing Datsets Due to Not Being Avaliable in the Link Provided!) | — |

---

## Skills Demonstrated

| Category | Techniques Used |
|-----------|------------------|
| **Time Series Manipulation** | Resampling (`M`, `Q`), rolling averages, datetime indexing |
| **Visualization** | Trend plots, moving averages, seasonal decomposition |
| **Forecasting** | ARIMA/SARIMA model fitting, residual analysis, auto_arima |
| **Evaluation** | RMSE accuracy metric, forecast vs actual comparison |



 **Note:** The analyses were performed using monthly resampled data for smoother trends. Forecast accuracy and seasonality detection depend on data granularity and preprocessing quality.
