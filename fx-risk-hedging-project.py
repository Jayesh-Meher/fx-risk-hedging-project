import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# 1. Download FX data
# -----------------------------
tickers = ["EURUSD=X", "GBPUSD=X", "USDINR=X"]

data = yf.download(tickers, start="2023-01-01", auto_adjust=False)["Close"]
data.columns = ["EURUSD", "GBPUSD", "USDINR"]
data = data.dropna()

print("\n=== FX Prices (Last 5 Rows) ===")
print(data.tail())


# -----------------------------
# 2. Calculate daily returns
# -----------------------------
returns = data.pct_change().dropna()

print("\n=== Daily Returns (Last 5 Rows) ===")
print(returns.tail())


# -----------------------------
# 3. Risk metrics
# -----------------------------
daily_volatility = returns.std()
annualised_volatility = returns.std() * np.sqrt(252)
correlation_matrix = returns.corr()

print("\n=== Daily Volatility ===")
print(daily_volatility.round(6))

print("\n=== Annualised Volatility ===")
print(annualised_volatility.round(6))

print("\n=== Correlation Matrix ===")
print(correlation_matrix.round(4))


# -----------------------------
# 4. Normalized price chart
# -----------------------------
normalized_prices = data / data.iloc[0]

plt.figure(figsize=(10, 5))
for column in normalized_prices.columns:
    plt.plot(normalized_prices.index, normalized_prices[column], label=column)

plt.title("Normalized FX Prices (Base = 1)")
plt.xlabel("Date")
plt.ylabel("Relative Change")
plt.legend()
plt.grid(True)
plt.show()


# -----------------------------
# 5. Returns chart
# -----------------------------
plt.figure(figsize=(10, 5))
for column in returns.columns:
    plt.plot(returns.index, returns[column], label=column)

plt.title("Daily FX Returns")
plt.xlabel("Date")
plt.ylabel("Return")
plt.legend()
plt.grid(True)
plt.show()


# -----------------------------
# 6. Equal-weight portfolio analysis
# -----------------------------
weights = np.array([1/3, 1/3, 1/3])
portfolio_returns = returns.dot(weights)

mean_daily_return = portfolio_returns.mean()
portfolio_daily_volatility = portfolio_returns.std()
portfolio_annual_return = mean_daily_return * 252
portfolio_annual_volatility = portfolio_daily_volatility * np.sqrt(252)

print("\n=== Portfolio Metrics ===")
print("Mean Daily Return:", round(mean_daily_return, 6))
print("Daily Volatility:", round(portfolio_daily_volatility, 6))
print("Annual Return:", round(portfolio_annual_return, 6))
print("Annual Volatility:", round(portfolio_annual_volatility, 6))


# -----------------------------
# 7. Sharpe ratio
# -----------------------------
risk_free_rate = 0.02
sharpe_ratio = (portfolio_annual_return - risk_free_rate) / portfolio_annual_volatility

print("\n=== Sharpe Ratio ===")
print("Sharpe Ratio:", round(sharpe_ratio, 4))


# -----------------------------
# 8. Historical Value at Risk (VaR)
# -----------------------------
portfolio_value = 1_000_000
var_95_percent = np.percentile(portfolio_returns, 5)
var_95_amount = abs(var_95_percent * portfolio_value)

print("\n=== 95% Historical Daily VaR ===")
print("VaR (%):", round(var_95_percent, 6))
print("VaR (Amount):", round(var_95_amount, 2))


# -----------------------------
# 9. FX shock scenario and hedging
# Example: company must pay USD 100,000
# -----------------------------
usd_payment = 100000
current_usdinr = data["USDINR"].iloc[-1]
shock_usdinr = current_usdinr * 1.05

# Unhedged cost after 5% USD appreciation
unhedged_cost = usd_payment * shock_usdinr

# Futures hedge: lock current rate today
futures_cost = usd_payment * current_usdinr

# Option hedge:
# Strike = current rate
# Premium = 1.5% of notional hedged amount
strike_rate = current_usdinr
premium_rate = 0.015
option_premium = usd_payment * strike_rate * premium_rate

# If future spot is above strike, exercise the option
if shock_usdinr > strike_rate:
    option_cost = usd_payment * strike_rate + option_premium
else:
    option_cost = usd_payment * shock_usdinr + option_premium

print("\n=== FX Shock Scenario ===")
print("Current USD/INR:", round(current_usdinr, 4))
print("Shocked USD/INR (5% increase):", round(shock_usdinr, 4))

print("\n=== Hedging Cost Comparison ===")
print("Unhedged Cost:", round(unhedged_cost, 2))
print("Futures Hedged Cost:", round(futures_cost, 2))
print("Option Hedged Cost:", round(option_cost, 2))


# -----------------------------
# 10. Hedging comparison table
# -----------------------------
comparison_df = pd.DataFrame({
    "Strategy": ["Unhedged", "Futures Hedge", "Option Hedge"],
    "Cost_INR": [unhedged_cost, futures_cost, option_cost]
})

comparison_df["Cost_INR"] = comparison_df["Cost_INR"].round(2)

print("\n=== Hedging Strategy Comparison Table ===")
print(comparison_df)

best_strategy = comparison_df.loc[comparison_df["Cost_INR"].idxmin(), "Strategy"]
print("\nBest Strategy Based on Lowest Cost:", best_strategy)


# -----------------------------
# 11. Bar chart for hedging strategies
# -----------------------------
plt.figure(figsize=(8, 5))
plt.bar(comparison_df["Strategy"], comparison_df["Cost_INR"])
plt.title("Comparison of FX Hedging Strategies")
plt.xlabel("Strategy")
plt.ylabel("Cost in INR")
plt.grid(axis="y")
plt.show()


# -----------------------------
# 12. Monte Carlo simulation for USD/INR
# -----------------------------
usd_inr_returns = returns["USDINR"]
starting_rate = data["USDINR"].iloc[-1]
mu = usd_inr_returns.mean()
sigma = usd_inr_returns.std()

time_horizon = 30
num_simulations = 1000

simulated_prices = np.zeros((time_horizon, num_simulations))

for i in range(num_simulations):
    price_path = [starting_rate]

    for t in range(1, time_horizon):
        random_shock = np.random.normal(mu, sigma)
        next_price = price_path[-1] * (1 + random_shock)
        price_path.append(next_price)

    simulated_prices[:, i] = price_path

# Plot only first 100 paths for cleaner chart
plt.figure(figsize=(10, 5))
plt.plot(simulated_prices[:, :100], alpha=0.15)
plt.title("Monte Carlo Simulation of USD/INR (30 Days)")
plt.xlabel("Days")
plt.ylabel("Simulated USD/INR Rate")
plt.grid(True)
plt.show()

final_simulated_prices = simulated_prices[-1, :]
expected_price_30d = final_simulated_prices.mean()
worst_case_95 = np.percentile(final_simulated_prices, 95)

print("\n=== Monte Carlo Simulation Results ===")
print("Expected USD/INR in 30 days:", round(expected_price_30d, 4))
print("95th Percentile USD/INR in 30 days:", round(worst_case_95, 4))


# -----------------------------
# 13. Monte Carlo distribution histogram
# -----------------------------
plt.figure(figsize=(10, 5))
plt.hist(final_simulated_prices, bins=30, edgecolor="black")
plt.title("Distribution of Simulated USD/INR After 30 Days")
plt.xlabel("USD/INR Rate")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()


# -----------------------------
# 14. Final summary
# -----------------------------
summary_df = pd.DataFrame({
    "Metric": [
        "Portfolio Annual Return",
        "Portfolio Annual Volatility",
        "Sharpe Ratio",
        "95% Historical Daily VaR",
        "Expected USD/INR in 30 days",
        "95th Percentile USD/INR in 30 days",
        "Best Hedging Strategy"
    ],
    "Value": [
        round(portfolio_annual_return, 4),
        round(portfolio_annual_volatility, 4),
        round(sharpe_ratio, 4),
        round(var_95_amount, 2),
        round(expected_price_30d, 4),
        round(worst_case_95, 4),
        best_strategy
    ]
})

print("\n=== Final Project Summary ===")
print(summary_df)