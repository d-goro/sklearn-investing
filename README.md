# sklearn-investing
identify stocks that should outperform S&amp;P 500

used python3, sklearn, pandas, quandl

Before running project extract intraQuarter.zip to the same folder. It contains key statistics of stocks from 2000 till 2013


GSPC.csv - S&P 500 prices from 2000 up to 30/03/2018 downloaded from https://finance.yahoo.com/quote/^GSPC/history?period1=946677600&period2=1522357200&interval=1d&filter=history&frequency=1d

for most recent results one should open yahoo finance, GBTC, define interval and download data as csv.

api_key.txt - API key for QUANDL


My code parses fueatures from key statistics and decides if specific stock ouperforms S&P500 if it's price change after specified period (quarter, half year, year) more than price chnage of S&P 500. So, outperforming Yes/No gives us label: 1/0;
X - it's following features:
'Total Debt/Equity',
             'Trailing P/E',
             'Price/Sales',
             'Price/Book',
             'Profit Margin',
             'Operating Margin',
             'Return on Assets',
             'Return on Equity',
             'Revenue Per Share',
             'Market Cap',
             'Enterprise Value',
             'Forward P/E',
             'PEG Ratio',
             'Enterprise Value/Revenue',
             'Enterprise Value/EBITDA',
             'Revenue',
             'Gross Profit',
             'EBITDA',
             'Net Income Avl to Common ',
             'Diluted EPS',
             'Earnings Growth',
             'Revenue Growth',
             'Total Cash',
             'Total Cash Per Share',
             'Total Debt',
             'Current Ratio',
             'Book Value Per Share',
             'Cash Flow',
             'Beta',
             'Held by Insiders',
             'Held by Institutions',
             'Shares Short (as of',
             'Short Ratio',
             'Short % of Float',
             'Shares Short (prior '
y - Status (1/0)

Stock prices (Adj. Close) pulled from Quandl for each ticker.
Next, it pulls from yahoo key statics for each ticker.

KNeighborsClassifier gives the best results (0.7-0.8 accuracy).
X and y divided to train and test sets.
KNeighborsClassifier trained with X_train (historical key stats) and y_train (labels - 1/0 regarding outperforming S&P 500). 
And tested with X_test and y_test.

Next, for each ticker fresh key statistics is taken and parsed and with this X trained model is used. If specific ticker is predicted to outperform - it's printed
