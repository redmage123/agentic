#!C:/Users/bbrel/agentic/.venv/Scripts/python
import yfinance as yf


def fetch_portfolio_data(tickers, start_date, end_date):
    portfolio_data = {}
    for ticker in tickers:
        # Download historical data, including adjusted prices
        data = yf.download(ticker, start=start_date, end=end_date)
        # Select adjusted open and close prices along with other key columns
        data = data[["Open", "Adj Close", "High", "Low", "Volume"]]
        # Renaming columns for clarity
        data.rename(
            columns={"Adj Open": "Adjusted Open", "Adj Close": "Adjusted Close"},
            inplace=True,
        )
        portfolio_data[ticker] = data
    return portfolio_data


def save_portfolio_data(portfolio_data, folder_path="./"):
    for ticker, data in portfolio_data.items():
        data.to_csv(f"{folder_path}{ticker}_data.csv")


if __name__ == "__main__":
    # Example portfolio of tickers
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    start_date = "2020-01-01"
    end_date = "2023-01-01"

    # Fetch and save data for each ticker in the portfolio
    portfolio_data = fetch_portfolio_data(tickers, start_date, end_date)
    save_portfolio_data(portfolio_data)
