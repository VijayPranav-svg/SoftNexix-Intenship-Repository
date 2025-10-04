import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error


AppleStrFactors = pd.read_csv('AAPL.csv')
print(AppleStrFactors.head().to_string())
print("\n")
print(AppleStrFactors.info())
print("\n")
print(AppleStrFactors.shape)

print("Data Shape:", AppleStrFactors.shape)
print("\nData Types:\n", AppleStrFactors.dtypes)
print("\nMissing Values:\n", AppleStrFactors.isnull().sum())
print("\nSummary Statistics Of Data:\n", AppleStrFactors.describe(include='all'))

AppleStrFactors['Date'] = pd.to_datetime(AppleStrFactors['Date'])
AppleStrFactors.set_index('Date', inplace=True)


AppleModstock = AppleStrFactors['Close']
print("Data Head:\n", AppleModstock.head().to_string())


def OrgainisingDatasets(AppleStrFactors):
    if not isinstance(AppleStrFactors.index, pd.DatetimeIndex):
        AppleStrFactors.index = pd.to_datetime(AppleStrFactors.index)
    ts = AppleStrFactors['Close']
    applemontly = ts.resample('M').mean()
    return applemontly




def StocksDartingData(series, period=12, model_type='multiplicative'):
    DecaptedSet = seasonal_decompose(series, model=model_type, period=period)
    Vtsets = DecaptedSet.plot()
    Vtsets.set_size_inches(12, 8)
    plt.show()


    
def PltsShifitingAvgs(series):
    AppleStrFactors = series.to_frame(name='Close')
    AppleStrFactors['MA_6'] = AppleStrFactors['Close'].rolling(window=6).mean()
    AppleStrFactors['MA_12'] = AppleStrFactors['Close'].rolling(window=12).mean()

    plt.figure(figsize=(12, 6))
    plt.plot(AppleStrFactors['Close'], label='Actual')
    plt.plot(AppleStrFactors['MA_6'], label='6-Month MA', linestyle='--')
    plt.plot(AppleStrFactors['MA_12'], label='12-Month MA', color='red')
    plt.legend()
    plt.title('Moving Averages Smoothing')
    plt.show()
    return AppleStrFactors


def QuarterlyResampling(df):
    quarterly = df['Close'].resample('Q').mean()
    plt.figure(figsize=(12, 5))
    plt.plot(quarterly, label="Quarterly Average Close")
    plt.legend()
    plt.title("Quarterly Resampled Close Price")
    plt.show()
    return quarterly


def ForecastingFuncts(series, order=(2,1,1), test_size=12):
    TrainDatasets = series.iloc[:-test_size]
    TestDatasets = series.iloc[-test_size:]

    model = ARIMA(TrainDatasets, order=order)
    result = model.fit()

    forecast = result.forecast(steps=test_size)
    rmse = np.sqrt(mean_squared_error(TestDatasets, forecast))
    print(f"RMSE: {rmse:.4f}")

    plt.figure(figsize=(12,6))
    plt.plot(TrainDatasets.index, TrainDatasets, label='Training Data')
    plt.plot(TestDatasets.index, TestDatasets, label='Actual', color='blue')
    plt.plot(TestDatasets.index, forecast, label='Forecast', color='red', linestyle='--')
    plt.fill_between(TestDatasets.index, forecast*0.9, forecast*1.1, alpha=0.2) # ±10% range
    plt.title(f"ARIMA Forecast (RMSE={rmse:.2f})")
    plt.legend()
    plt.show()


DistuAppleSets = OrgainisingDatasets(AppleStrFactors)
QuarterlyResampling(AppleStrFactors)  # new step
StocksDartingData(DistuAppleSets, period=12)
DistuAppleSets_df = PltsShifitingAvgs(DistuAppleSets)
ForecastingFuncts(DistuAppleSets_df['Close'], order=(2,1,1), test_size=12)

print("Completed Successfully")
print("The First Few Rows of the Final DataFrame:")
print(DistuAppleSets_df.head().to_string())
print(AppleStrFactors.columns)
