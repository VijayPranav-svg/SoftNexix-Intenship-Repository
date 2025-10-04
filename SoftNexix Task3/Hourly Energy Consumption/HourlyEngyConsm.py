import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error


HourlyEngryComps = pd.read_csv("AEP_hourly.csv", parse_dates=['Datetime'], index_col='Datetime')

print(HourlyEngryComps.head().to_string())
print("\n")
print(HourlyEngryComps.info())
print("\n")
print(HourlyEngryComps.shape)

print("Data Shape:", HourlyEngryComps.shape)
print("\nData Types:\n", HourlyEngryComps.dtypes)
print("\nMissing Values:\n", HourlyEngryComps.isnull().sum())
print("\nSummary Statistics Of Data:\n", HourlyEngryComps.describe(include='all'))

# Extract the main series
EngrSers = HourlyEngryComps['AEP_MW']
print("Data Head:\n", EngrSers.head().to_string())



def OrganisingDatasets(Horengys):
    if not isinstance(Horengys.index, pd.DatetimeIndex):
        Horengys.index = pd.to_datetime(Horengys.index)
    ts = Horengys['AEP_MW']
    EnergyRatemonts = ts.resample('ME').mean()  # 'ME' = month end
    return EnergyRatemonts

def DecmpSeasnlFunc(series, period=12, model_type='multiplicative'):
    DecaptedSet = seasonal_decompose(series, model=model_type, period=period)
    Vtsets = DecaptedSet.plot()
    Vtsets.set_size_inches(12, 8)
    plt.show()




def PltsShifitingAvgs(series):
    MontlyEngrs = series.to_frame(name='AEP_MW')
    MontlyEngrs['MA_6'] = MontlyEngrs['AEP_MW'].rolling(window=6, min_periods=1).mean()
    MontlyEngrs['MA_12'] = MontlyEngrs['AEP_MW'].rolling(window=12, min_periods=1).mean()

    plt.figure(figsize=(12, 6))
    plt.plot(MontlyEngrs['AEP_MW'], label='Actual')
    plt.plot(MontlyEngrs['MA_6'], label='6-Month MA', linestyle='--')
    plt.plot(MontlyEngrs['MA_12'], label='12-Month MA', color='red')
    plt.legend()
    plt.title('Moving Averages Smoothing')
    plt.show()
    return MontlyEngrs

def HalyEngrsDataResampling(MontlyEngrs):
    HalyEngrsData = MontlyEngrs['AEP_MW'].resample('QE').mean()  
    plt.figure(figsize=(12, 5))
    plt.plot(HalyEngrsData, label="HalyEngrsData Average AEP_MW")
    plt.legend()
    plt.title("HalyEngrsData Resampled AEP_MW")
    plt.show()
    return HalyEngrsData


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

DiminishedEneragdarat = OrganisingDatasets(HourlyEngryComps)       
HalyEngrsDataResampling(HourlyEngryComps)                 
DecmpSeasnlFunc(DiminishedEneragdarat, period=12)       
DiminishedEneragdarat_df = PltsShifitingAvgs(DiminishedEneragdarat)    
ForecastingFuncts(DiminishedEneragdarat_df['AEP_MW'], order=(2,1,1), test_size=12)  

print("Completed Successfully")
print("The First Few Rows of the Final DataFrame:")
print(DiminishedEneragdarat_df.head().to_string())
print(HourlyEngryComps.columns)
