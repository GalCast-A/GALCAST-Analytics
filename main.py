from flask import Flask, request
import json
import pandas as pd
import numpy as np

app = Flask(__name__)

class PortfolioAnalyzer:
    def __init__(self):
        self.today_date = "2025-04-17"
        self.default_start_date = "2015-04-17"
        self.data_cache = {}

    def fetch_treasury_yield(self):
        return 0.04

    def fetch_stock_data(self, stocks, start=None, end=None):
        if start is None:
            start = self.default_start_date
        if end is None:
            end = self.today_date
        cache_key = (tuple(sorted(stocks)), start, end)
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        error_tickers = {}
        earliest_dates = {}
        return None, error_tickers, earliest_dates

    def compute_returns(self, prices):
        if prices is None:
            return pd.DataFrame()
        returns = pd.DataFrame(prices).pct_change()
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna(how='any')
        return returns

    def compute_max_drawdown(self, returns):
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min() if not drawdown.empty else 0

    def compute_sortino_ratio(self, returns, risk_free_rate):
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        annualized_return = returns.mean() * 252
        return (annualized_return - risk_free_rate) / downside_std if downside_std != 0 else 0

    def compute_beta(self, portfolio_returns, benchmark_returns):
        covariance = portfolio_returns.cov(benchmark_returns)
        benchmark_variance = benchmark_returns.var()
        return covariance / benchmark_variance if benchmark_variance != 0 else 0

    def portfolio_performance(self, weights, returns, risk_free_rate):
        portfolio_returns = returns.dot(weights)
        portfolio_return = portfolio_returns.mean() * 252
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility != 0 else 0
        return portfolio_return, portfolio_volatility, sharpe_ratio

    def compute_var(self, returns, confidence_level=0.90):
        sorted_returns = np.sort(returns)
        index = int((1 - confidence_level) * len(sorted_returns))
        return sorted_returns[index] if len(sorted_returns) > 0 else 0

    def compute_avg_correlation(self, returns_df, weights):
        weighted_corr_sum = 0
        num_assets = returns_df.shape[1]
        corr_matrix = returns_df.corr()
        for i in range(num_assets):
            for j in range(i + 1, num_assets):
                weighted_corr_sum += weights[i] * weights[j] * corr_matrix.iloc[i, j]
        avg_corr = 2 * weighted_corr_sum
        return avg_corr

    def optimize_portfolio(self, returns, risk_free_rate, objective='sharpe', min_allocation=0.0, max_allocation=1.0):
        num_assets = returns.shape[1]
        initial_weights = np.ones(num_assets) / num_assets
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((min_allocation, max_allocation) for _ in range(num_assets))

        def negative_sharpe(weights):
            r, v, s = self.portfolio_performance(weights, returns, risk_free_rate)
            return -s

        def negative_sortino(weights):
            portfolio_returns = returns.dot(weights)
            return -self.compute_sortino_ratio(portfolio_returns, risk_free_rate)

        def max_drawdown(weights):
            portfolio_returns = returns.dot(weights)
            drawdown = -self.compute_max_drawdown(portfolio_returns)
            return drawdown

        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))

        def negative_var(weights):
            portfolio_returns = returns.dot(weights)
            return -self.compute_var(portfolio_returns)

        objective_functions = {
            'sharpe': negative_sharpe,
            'sortino': negative_sortino,
            'max_drawdown': max_drawdown,
            'volatility': portfolio_volatility,
            'value_at_risk': negative_var
        }
        obj_fun = objective_functions.get(objective, negative_sharpe)

        try:
            from scipy.optimize import minimize
            result = minimize(obj_fun, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
            if not result.success:
                return initial_weights
            weights = result.x
            weights[weights < 0.001] = 0
            weights /= weights.sum() if weights.sum() != 0 else 1
            return weights
        except Exception:
            return initial_weights

analyzer = PortfolioAnalyzer()

@app.route('/analyze_portfolio', methods=['POST'])
def analyze_portfolio():
    request_json = request.get_json(silent=True)
    if not request_json:
        return json.dumps({"error": "No data provided"}), 400

    tickers = request_json.get("tickers", [])
    weights = request_json.get("weights", [])
    start_date = request_json.get("start_date", analyzer.default_start_date)
    end_date = request_json.get("end_date", analyzer.today_date)
    risk_tolerance = request_json.get("risk_tolerance", "medium")
    benchmarks = request_json.get("benchmarks", ["^GSPC"])
    risk_free_rate = request_json.get("risk_free_rate", analyzer.fetch_treasury_yield())

    if not tickers or not weights:
        return json.dumps({"error": "Tickers and weights required"}), 400

    weights_dict = dict(zip(tickers, weights))
    stock_prices = request_json.get("stock_prices", None)
    benchmark_prices = request_json.get("benchmark_prices", {})

    if stock_prices is None:
        return json.dumps({"error": "Stock prices required"}), 400

    stock_prices_df = pd.DataFrame(stock_prices, index=[pd.to_datetime(date) for date in request_json.get("dates", [])])
    returns = analyzer.compute_returns(stock_prices_df)
    if returns.empty:
        return json.dumps({"error": "Failed to compute returns"}), 400

    benchmark_returns = {}
    benchmark_metrics = {}
    for bench in benchmarks:
        if bench in benchmark_prices:
            bench_data = pd.Series(benchmark_prices[bench], index=[pd.to_datetime(date) for date in request_json.get("dates", [])])
            bench_ret = analyzer.compute_returns(bench_data)
            if not bench_ret.empty:
                benchmark_returns[bench] = bench_ret.tolist()
                benchmark_metrics[bench] = {
                    "annual_return": float(bench_ret.mean() * 252),
                    "annual_volatility": float(bench_ret.std() * np.sqrt(252)),
                    "sharpe_ratio": float(analyzer.portfolio_performance(np.array([1.0]), pd.DataFrame(bench_ret), risk_free_rate)[2]),
                    "maximum_drawdown": float(analyzer.compute_max_drawdown(bench_ret)),
                    "value_at_risk": float(analyzer.compute_var(bench_ret, 0.90))
                }

    original_weights = np.array(list(weights_dict.values()))
    portfolio_returns = returns.dot(original_weights)
    original_metrics = {
        "annual_return": float(portfolio_returns.mean() * 252),
        "annual_volatility": float(portfolio_returns.std() * np.sqrt(252)),
        "sharpe_ratio": float(analyzer.portfolio_performance(original_weights, returns, risk_free_rate)[2]),
        "maximum_drawdown": float(analyzer.compute_max_drawdown(portfolio_returns)),
        "value_at_risk": float(analyzer.compute_var(portfolio_returns, 0.90))
    }

    opt_weights = analyzer.optimize_portfolio(returns, risk_free_rate, "sharpe")
    optimized_metrics = {
        "annual_return": float(analyzer.portfolio_performance(opt_weights, returns, risk_free_rate)[0]),
        "annual_volatility": float(analyzer.portfolio_performance(opt_weights, returns, risk_free_rate)[1]),
        "sharpe_ratio": float(analyzer.portfolio_performance(opt_weights, returns, risk_free_rate)[2]),
        "maximum_drawdown": float(analyzer.compute_max_drawdown(returns.dot(opt_weights))),
        "value_at_risk": float(analyzer.compute_var(returns.dot(opt_weights), 0.90))
    }

    strategies = {
        "Original Portfolio": original_weights.tolist(),
        "Max Sharpe": opt_weights.tolist()
    }
    cumulative_returns = {
        label: (1 + returns.dot(weights)).cumprod().tolist()
        for label, weights in strategies.items()
    }
    for bench, bench_ret in benchmark_returns.items():
        cumulative_returns[bench] = (1 + pd.Series(bench_ret)).cumprod().tolist()

    corr_matrix = returns.corr().values.tolist()
    rolling_volatility = {
        label: (pd.Series(returns.dot(weights)).rolling(window=252).std() * np.sqrt(252)).tolist()
        for label, weights in strategies.items()
    }

    return json.dumps({
        "original_metrics": original_metrics,
        "optimized_metrics": optimized_metrics,
        "benchmark_metrics": benchmark_metrics,
        "cumulative_returns": cumulative_returns,
        "correlation_matrix": corr_matrix,
        "rolling_volatility": rolling_volatility
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
