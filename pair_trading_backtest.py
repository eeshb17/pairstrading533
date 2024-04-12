import numpy as np
import refinitiv.dataplatform.eikon as ek
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

# Set the Eikon App Key
ek.set_app_key('9b8ddd220e9145aeb7d0c34ec1aee644c4e6431a')

def fetch_eikon_data(instrument, start_date, end_date):
    df = ek.get_timeseries(instrument,
                           start_date=start_date,
                           end_date=end_date,
                           fields=['Open'],
                           interval='daily')
    df.rename(columns={'OPEN': 'Open Price'}, inplace=True)
    return df

def calculate_z_score(spread_values):
    z_scores = []

    for i in range(len(spread_values)):
        spread_avg = np.mean(spread_values[max(0, i - 21):i + 1])
        spread_std = np.std(spread_values[max(0, i - 21):i + 1])

        if spread_std != 0:  # Check for division by zero
            z_score = (spread_values[i] - spread_avg) / spread_std
        else:
            z_score = np.nan  # Set to NaN if spread_std is zero

        z_scores.append(z_score)

    return z_scores

def main():
    stock_a = 'DFS'
    stock_b = 'COF'
    start_date = '2023-01-21'
    end_date = '2023-12-31'

    # Get dataframes for both stocks
    df_dis = fetch_eikon_data(stock_a, start_date, end_date)
    df_cof = fetch_eikon_data(stock_b, start_date, end_date)

    # Calculate spread values
    spread_values_dis = df_dis['Open Price'].values - df_cof['Open Price'].values
    spread_values_cof = df_cof['Open Price'].values - df_dis['Open Price'].values

    # Calculate z-scores
    z_scores_dis = calculate_z_score(spread_values_dis)
    z_scores_cof = calculate_z_score(spread_values_cof)

    # Strategy parameters
    signal_entry = 1.2
    signal_exit = 0.5

    # Initialize position variables
    position_dis = None
    position_cof = None

    # Initialize an empty list to store trade dictionaries
    trade_data = []

    # Apply strategy for Discover
    for i in range(len(z_scores_dis)):
        if z_scores_dis[i] > signal_entry and position_dis != 'long':
            # Process buy signals
            if position_dis == 'short':
                # Update trades for closing short position
                trade_data.append({
                    'Identifier': 'DIS',
                    'Entry Date': entry_date_dis,
                    'Entry Price': entry_price_dis,
                    'Exit Date': df_dis.index[i],
                    'Exit Price': df_dis.iloc[i]['Open Price'],
                    'Profit/Loss': (entry_price_dis - df_dis.iloc[i]['Open Price']) if position_dis == 'short' else (
                                df_dis.iloc[i]['Open Price'] - entry_price_dis)
                })
                position_dis = None

            # Update entry price and position for opening long position
            entry_date_dis = df_dis.index[i]
            entry_price_dis = df_dis.iloc[i]['Open Price']
            position_dis = 'long'

        elif z_scores_dis[i] < -signal_entry and position_dis != 'short':
            # Process sell signals
            if position_dis == 'long':
                # Update trades for closing long position
                trade_data.append({
                    'Identifier': 'DIS',
                    'Entry Date': entry_date_dis,
                    'Entry Price': entry_price_dis,
                    'Exit Date': df_dis.index[i],
                    'Exit Price': df_dis.iloc[i]['Open Price'],
                    'Profit/Loss': (df_dis.iloc[i]['Open Price'] - entry_price_dis) if position_dis == 'long' else (
                                entry_price_dis - df_dis.iloc[i]['Open Price'])
                })
                position_dis = None

            # Update entry price and position for opening short position
            entry_date_dis = df_dis.index[i]
            entry_price_dis = df_dis.iloc[i]['Open Price']
            position_dis = 'short'

    # Apply strategy for Capital One Financial
    for i in range(len(z_scores_cof)):
        if z_scores_cof[i] > signal_entry and position_cof != 'long':
            # Process buy signals
            if position_cof == 'short':
                # Update trades for closing short position
                trade_data.append({
                    'Identifier': 'COF',
                    'Entry Date': entry_date_cof,
                    'Entry Price': entry_price_cof,
                    'Exit Date': df_cof.index[i],
                    'Exit Price': df_cof.iloc[i]['Open Price'],
                    'Profit/Loss': (entry_price_cof - df_cof.iloc[i]['Open Price']) if position_cof == 'short' else (
                                df_cof.iloc[i]['Open Price'] - entry_price_cof)
                })
                position_cof = None

            # Update entry price and position for opening long position
            entry_date_cof = df_cof.index[i]
            entry_price_cof = df_cof.iloc[i]['Open Price']
            position_cof = 'long'

        elif z_scores_cof[i] < -signal_entry and position_cof != 'short':
            # Process sell signals
            if position_cof == 'long':
                # Update trades for closing long position
                trade_data.append({
                    'Identifier': 'COF',
                    'Entry Date': entry_date_cof,
                    'Entry Price': entry_price_cof,
                    'Exit Date': df_cof.index[i],
                    'Exit Price': df_cof.iloc[i]['Open Price'],
                    'Profit/Loss': (df_cof.iloc[i]['Open Price'] - entry_price_cof) if position_cof == 'long' else (
                                entry_price_cof - df_cof.iloc[i]['Open Price'])
                })
                position_cof = None

            # Update entry price and position for opening short position
            entry_date_cof = df_cof.index[i]
            entry_price_cof = df_cof.iloc[i]['Open Price']
            position_cof = 'short'

    # Create DataFrame from trade_data list
    trades = pd.DataFrame(trade_data)

    # Save trades DataFrame to Excel
    trades.to_excel('trades.xlsx', index=False)

    # Plotting
    plt.figure(figsize=(12, 6))

    # Plot Discover prices
    plt.plot(df_dis.index, df_dis['Open Price'], label='Discover', color='blue')

    # Plot buy and sell signals for Discover
    for index, row in trades.loc[trades['Identifier'] == 'DIS'].iterrows():
        if not pd.isnull(row['Entry Price']):  # Plot buy signal
            plt.scatter(pd.to_datetime(row['Entry Date']), row['Entry Price'], color='green', marker='^',
                        label='Buy Signal (Discover)')
        if not pd.isnull(row['Exit Price']):  # Plot sell signal
            plt.scatter(pd.to_datetime(row['Exit Date']), row['Exit Price'], color='red', marker='v',
                        label='Sell Signal (Discover)')

    # Plot Capital One Financial prices
    plt.plot(df_cof.index, df_cof['Open Price'], label='Capital One Financial', color='orange')

    # Plot buy and sell signals for Capital One Financial
    for index, row in trades.loc[trades['Identifier'] == 'COF'].iterrows():
        if not pd.isnull(row['Entry Price']):  # Plot buy signal
            plt.scatter(pd.to_datetime(row['Entry Date']), row['Entry Price'], color='green', marker='^',
                        label='_nolegend_')
        if not pd.isnull(row['Exit Price']):  # Plot sell signal
            plt.scatter(pd.to_datetime(row['Exit Date']), row['Exit Price'], color='red', marker='v',
                        label='_nolegend_')

    # Set legend with only one signal for both stocks
    plt.legend(['Discover', 'Capital One Financial', 'Buy Signal', 'Sell Signal'])

    plt.title('Stock Prices and Trading Signals')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


main()
