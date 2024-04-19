import pandas as pd
import numpy as np
import statsmodels.api as sm

# Define constants
SIGNAL_ENTRY = 0.7
SIGNAL_EXIT = 0.2


def load_data_from_excel(filename):
    """
    Load data from Excel file into a DataFrame.
    """
    df = pd.read_excel(filename)
    df.set_index('Date', inplace=True)  # Set index to date column
    return df

def calculate_z_score(spread_values):
    """
    Calculate z-scores based on spread values.
    """
    z_scores = []
    for i in range(len(spread_values)):
        spread_avg = np.mean(spread_values[max(0, i - 21):i + 1])
        spread_std = np.std(spread_values[max(0, i - 21):i + 1])

        if spread_std != 0:
            z_score = (spread_values[i] - spread_avg) / spread_std
        else:
            z_score = np.nan

        z_scores.append(z_score)

    return z_scores


def enter_long_position(z_score, position, entry_date, entry_price, identifier, df, trades, i):
    """
    Enter a long position if the z-score crosses 1.2 from below and there is no existing position.
    """
    if z_score > SIGNAL_ENTRY and position != 'long':
        # Process buy signals
        if position == 'short':
            # Update trades for closing short position
            trades.append({
                'Identifier': identifier,
                'Entry Date': entry_date,
                'Entry Price': entry_price,
                'Exit Date': df.index[i],
                'Exit Price': df.loc[df.index[i], 'OPEN'],
                'Profit/Loss': (entry_price - df.loc[df.index[i], 'OPEN']) if position == 'short' else (
                        df.loc[df.index[i], 'OPEN'] - entry_price)
            })
            position = None

        # Update entry price and position for opening long position
        entry_date = df.index[i]
        entry_price = df.loc[df.index[i], 'OPEN']
        position = 'long'

    return position, entry_date, entry_price


def enter_short_position(z_score, position, entry_date, entry_price, identifier, df, trades, i):
    """
    Enter a short position if the z-score crosses 1.2 from above and there is no existing position.
    """
    if z_score < -SIGNAL_ENTRY and position != 'short':
        # Process sell signals
        if position == 'long':
            # Update trades for closing long position
            trades.append({
                'Identifier': identifier,
                'Entry Date': entry_date,
                'Entry Price': entry_price,
                'Exit Date': df.index[i],
                'Exit Price': df.loc[df.index[i], 'OPEN'],
                'Profit/Loss': (df.loc[df.index[i], 'OPEN'] - entry_price) if position == 'long' else (
                        entry_price - df.loc[df.index[i], 'OPEN'])
            })
            position = None

        # Update entry price and position for opening short position
        entry_date = df.index[i]
        entry_price = df.loc[df.index[i], 'OPEN']
        position = 'short'

    return position, entry_date, entry_price


def exit_position(z_score, position, entry_date, entry_price, identifier, df, trades, i):
    """
    Exit the position if the z-score crosses 0 and there is an existing position.
    """
    if abs(z_score) < SIGNAL_EXIT and position is not None:
        # Process exit signals
        trades.append({
            'Identifier': identifier,
            'Entry Date': entry_date,
            'Entry Price': entry_price,
            'Exit Date': df.index[i],
            'Exit Price': df.loc[df.index[i], 'OPEN'],
            'Profit/Loss': (df.loc[df.index[i], 'OPEN'] - entry_price) if position == 'long' else (
                    entry_price - df.loc[df.index[i], 'OPEN'])
        })
        position = None
        entry_date = None
        entry_price = None

    return position, entry_date, entry_price


def run_regression(daily_return, risk_free_rate, market_return):
    # Calculate excess returns for the strategy
    excess_returns = daily_return - risk_free_rate

    # Add a constant term for the intercept
    X = sm.add_constant(excess_returns)

    # Fit the linear regression model
    model = sm.OLS(excess_returns, X)
    results = model.fit()

    # Extract alpha and beta from the results
    alpha = results.params[0]
    beta = results.params[1]
    return alpha, beta


def calculate_distance(expected_strategy_return, risk_free_rate, market_return, beta):
    # Calculate the expected return from Security Market Line
    expected_sml_return = risk_free_rate + beta * (market_return - risk_free_rate)

    # Calculate the distance
    distance = abs(expected_strategy_return - expected_sml_return)

    return distance


def main():
    stock_a_file = 'DFS_data.xlsx'
    stock_b_file = 'COF_data.xlsx'
    sp500_file = 'sp500.xlsx'

    # Load dataframes for both stocks from Excel files
    df_dis = load_data_from_excel(stock_a_file)
    df_cof = load_data_from_excel(stock_b_file)
    # Load historical S&P 500 data
    sp500_data = load_data_from_excel(sp500_file)

    # Check if the DataFrame columns contain 'OPEN'
    if 'OPEN' not in df_dis.columns or 'OPEN' not in df_cof.columns:
        print("Error: DataFrame columns do not contain 'OPEN'")
        return

    # Calculate spread values
    spread_values_dis = df_dis['OPEN'].values - df_cof['OPEN'].values
    spread_values_cof = df_cof['OPEN'].values - df_dis['OPEN'].values

    # Calculate z-scores
    z_scores_dis = calculate_z_score(spread_values_dis)
    z_scores_cof = calculate_z_score(spread_values_cof)

    # Initialize entry variables for Discover
    position_dis = None
    entry_date_dis = None
    entry_price_dis = None

    # Initialize entry variables for Capital One Financial
    position_cof = None
    entry_date_cof = None
    entry_price_cof = None

    # Initialize empty list to store trade dictionaries
    all_trades = []

    # Apply strategy for Discover and Capital One Financial
    for i in range(len(z_scores_dis)):
        # Process trades for Discover
        position_dis, entry_date_dis, entry_price_dis = enter_long_position(z_scores_dis[i], position_dis,
                                                                            entry_date_dis, entry_price_dis, 'DIS',
                                                                            df_dis, all_trades, i)
        position_dis, entry_date_dis, entry_price_dis = enter_short_position(z_scores_dis[i], position_dis,
                                                                             entry_date_dis, entry_price_dis, 'DIS',
                                                                             df_dis, all_trades, i)
        position_dis, entry_date_dis, entry_price_dis = exit_position(z_scores_dis[i], position_dis, entry_date_dis,
                                                                      entry_price_dis, 'DIS', df_dis, all_trades, i)

        # Process trades for Capital One Financial
        position_cof, entry_date_cof, entry_price_cof = enter_long_position(z_scores_cof[i], position_cof,
                                                                            entry_date_cof, entry_price_cof, 'COF',
                                                                            df_cof, all_trades, i)
        position_cof, entry_date_cof, entry_price_cof = enter_short_position(z_scores_cof[i], position_cof,
                                                                             entry_date_cof, entry_price_cof, 'COF',
                                                                             df_cof, all_trades, i)
        position_cof, entry_date_cof, entry_price_cof = exit_position(z_scores_cof[i], position_cof, entry_date_cof,
                                                                      entry_price_cof, 'COF', df_cof, all_trades, i)

    # Create DataFrame from trade dictionaries
    all_trades_df = pd.DataFrame(all_trades)

    # Calculate profit/loss for all trades
    all_trades_df['Profit/Loss'] = all_trades_df.apply(
        lambda row: row['Exit Price'] - row['Entry Price'] if row['Entry Price'] and row['Exit Price'] else 0, axis=1)

    # Save all trades DataFrame to Excel
    all_trades_df.to_excel('all_trades.xlsx', index=False)

    # Calculate additional variables
    # Sort trades by date
    all_trades_df['Entry Date'] = pd.to_datetime(all_trades_df['Entry Date'])
    all_trades_df['Exit Date'] = pd.to_datetime(all_trades_df['Exit Date'])
    all_trades_df = all_trades_df.sort_values(by=['Entry Date'])

    # Calculate time held
    all_trades_df['Time Held'] = (all_trades_df['Exit Date'] - all_trades_df['Entry Date']).dt.days.clip(lower=1)

    # Calculate daily return per trade
    all_trades_df['Rtn Per Trade'] = np.log(all_trades_df['Exit Price'] / all_trades_df['Entry Price'])
    all_trades_df['Daily Return'] = all_trades_df['Rtn Per Trade'] / all_trades_df['Time Held']
    all_trades_df.loc[all_trades_df['Time Held'] == 0, 'Daily Return'] = 0  # Handling division by zero
    all_trades_df = all_trades_df.sort_values(by='Exit Date')
    average_daily_return = all_trades_df.groupby('Exit Date')['Daily Return'].mean()
    # Create a new DataFrame with date and average daily return for that date
    average_daily_return_df = pd.DataFrame({'Date': average_daily_return.index,
                                            'Average Daily Return': average_daily_return})
    print(average_daily_return_df)
    average_daily_return_df.to_excel('avg_returns.xlsx', index=False)
    # Calculate and print the mean of the average daily return
    mean_average_daily_return = average_daily_return_df['Average Daily Return'].mean()
    print("Mean of the Average Daily Return:", mean_average_daily_return)

    # Convert 'Close' column to numeric format, forcing errors to coerce non-numeric values to NaN
    sp500_data['Open'] = pd.to_numeric(sp500_data['Open'], errors='coerce')
    sp500_data = sp500_data.dropna(subset=['Open'])
    sp500_data['Log Return'] = sp500_data['Open'].pct_change().apply(lambda x: np.log(1 + x))
    # Select logarithmic returns for the available dates
    available_dates = all_trades_df['Exit Date'].unique()
    log_returns = sp500_data.loc[available_dates, 'Log Return']
    # Calculate the average logarithmic return
    average_log_return = log_returns.mean()
    # Annualize the average logarithmic return
    market_return = average_log_return * 252  # Assuming 252 trading days in a year

    risk_free_rate = 5.44

    # Run regression to calculate alpha and beta
    beta, alpha = run_regression(log_returns, risk_free_rate, average_daily_return)

    print("Alpha:", alpha)
    print("Beta:", beta)

    # Assess strategy performance
    expected_strategy_return = risk_free_rate + beta * (market_return - risk_free_rate)
    distance_to_sml = calculate_distance(expected_strategy_return, risk_free_rate, market_return, beta)

    print("Expected Strategy Return:", expected_strategy_return)
    print("Distance to Security Market Line:", distance_to_sml)


main()
