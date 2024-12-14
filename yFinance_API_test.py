import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Test stocks
tickers = [
    'AAPL', 'MSFT', 'GOOG', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'ASML', 'AVGO',
     'COST', #'PEP', 'CSCO', 'CMCSA', 'VZ', 'AZN', 'INTC', 'ADBE', 'QCOM', 'TXN',
    # 'TMUS', 'NFLX', 'AMGN', 'PYPL', 'SNY', 'AMD', 'INTU', 'AMAT', 'SBUX', 'ABNB',
    # 'CHTR', 'MRNA', 'MDLZ', 'BKNG', 'ADI', 'ISRG', 'LRCX', 'REGN', 'MU', 'VRTX',
    # 'PANW', 'GILD', 'ADP', 'SNPS', 'KLAC', 'MELI', 'ATVI', 'CSX', 'IDXX', 'MNST',
    # 'ORLY', 'KDP', 'CTAS', 'MAR', 'FTNT', 'NXPI', 'EXC', 'CDNS', 'MCHP', 'AEP',
    # 'ODFL', 'BIIB', 'PCAR', 'WBD', 'CTSH', 'PAYX', 'DLTR', 'XEL', 'FAST', 'ROST',
    # 'EBAY', 'ANSS', 'SWKS', 'VRSK', 'SIRI', 'LULU', 'ALGN', 'JD', 'NTES', 'DOCU',
    # 'MTCH', 'SGEN', 'LCID', 'ZM', 'OKTA', 'BIDU', 'VRSN', 'ENPH', 'PDD', 'CPRT',
    # 'CEG', 'RIVN', 'DDOG', 'ON', 'GEHC', 'TTD', 'ARM', 'SMCI', 'APP'
]


# Test range
end_date = datetime.now()
start_date = end_date - timedelta(days=7)

# Pull data
data = yf.download(tickers, start=start_date, end=end_date, interval='1m')

data = data['Close']

# Format data
data.reset_index(inplace=True)
data['DATE'] = data['Datetime'].apply(lambda x: int(x.timestamp()) if isinstance(x, pd.Timestamp) else None)
data.drop('Datetime', axis=1, inplace=True)
columns_order = ['DATE'] + [col for col in data.columns if col != 'DATE']
data = data[columns_order]
# To CSV file
csv_file_path = 'stock_data.csv'
# Rename columns to add "NASDAQ." prefix
data.rename(columns={col: f"NASDAQ.{col}" for col in data.columns if col != 'DATE' }, inplace=True)
# Identify missing data and fill forward
data = data.sort_values('DATE') 
data.fillna(method='ffill', inplace=True)
data.fillna(method='bfill', inplace=True)  # Backward fill if there are more than one missing data
data.to_csv(csv_file_path, index=False)


print(f'Data has been saved to {csv_file_path}')

# Test case 2
'''
start_date = '2017-08-31'
end_date = '2017-09-01'

data = yf.download(tickers, start=start_date, end=end_date, interval='1d')
print(data)

# Unix time
timestamp1 = [1491226200, 1491226260, 1491226320, 1504209600]

# Convert Unix time to datetime object
for i in timestamp1:
    date = datetime.fromtimestamp(i)
    print(date)'''