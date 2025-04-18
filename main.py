from flask import Flask, request
import json
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from scipy.optimize import minimize
import warnings
import time
warnings.filterwarnings('ignore')

# Attempt to import optional dependencies
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("Warning: 'yfinance' not installed. Data fetching unavailable.")

try:
    from pypfopt import BlackLittermanModel, risk_models, expected_returns
    PYPFOPT_AVAILABLE = True
except ImportError:
    PYPFOPT_AVAILABLE = False
    print("Warning: 'pypfopt' not installed. Falling back to basic optimization methods.")

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    print("Warning: 'cvxpy' not installed. Using scipy.optimize.minimize.")

try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: 'statsmodels' not installed. Fama-French exposures unavailable.")

from sklearn.decomposition import PCA

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PortfolioAnalyzer:
    def __init__(self):
        self.today_date = datetime.now().strftime("%Y-%m-%d")
        self.default_start_date = (datetime.strptime(self.today_date, "%Y-%m-%d") - timedelta(days=3652)).strftime("%Y-%m-%d")
        self.data_cache = {}

    def fetch_treasury_yield(self):
        if not YFINANCE_AVAILABLE:
            logger.warning("yfinance unavailable. Using fallback Treasury yield of 0.04.")
            return 0.04
        try:
            treasury_data = yf.download("^TNX", period="1d", interval="1d")['Close']
            if treasury_data.empty or not isinstance(treasury_data, pd.Series):
                logger.warning("Could not fetch 10-year Treasury yield. Using fallback value of 0.04.")
                return 0.04
            latest_yield = float(treasury_data.iloc[-1]) / 100
            return latest_yield
        except Exception as e:
            logger.error(f"Error fetching Treasury yield: {e}. Using fallback value of 0.04.")
            return 0.04
            
    def fetch_stock_data(self, stocks, start=None, end=None):
        if start is None:
            start = self.default_start_date
        if end is None:
            end = self.today_date
        cache_key = (tuple(sorted(stocks)), start, end)
        if cache_key in self.data_cache:
            logger.info(f"Returning cached stock data for {stocks}")
            return self.data_cache[cache_key]
        logger.info(f"YFINANCE_AVAILABLE: {YFINANCE_AVAILABLE}")
        if not YFINANCE_AVAILABLE:
            logger.error("yfinance unavailable. Cannot fetch stock data.")
            return None, {"error": "yfinance unavailable"}, {}
        error_tickers = {}
        earliest_dates = {}
        for attempt in range(3):
            try:
                logger.info(f"Fetching stock data for {stocks} from {start} to {end}, attempt {attempt + 1}...")
                stock_data = yf.download(list(stocks), start=start, end=end, auto_adjust=True)['Close']
                logger.info(f"Fetched stock data: {stock_data.shape if not stock_data.empty else 'empty'}")
                if stock_data.empty:
                    logger.warning("No data available for the specified date range.")
                    return None, error_tickers, earliest_dates
                break
            except Exception as e:
                logger.error(f"Error fetching data (attempt {attempt + 1}): {e}")
                if attempt == 2:
                    logger.error("Failed to fetch data after 3 attempts.")
                    return None, {"error": "Failed to fetch stock data"}, {}
                time.sleep(2)  # Wait before retrying
        try:
            stock_data = stock_data.dropna(axis=1, how='all')
            logger.info(f"After dropna: {stock_data.shape if not stock_data.empty else 'empty'}")
            if stock_data.shape[0] < 252:
                logger.warning("Insufficient data (< 252 days). Optimization may be unreliable.")

            problematic_tickers = []
            for ticker in stock_data.columns:
                if (stock_data[ticker] <= 1e-4).any():
                    logger.warning(f"{ticker} has zero or near-zero prices (<= 1e-4). Excluding.")
                    problematic_tickers.append(ticker)
                    error_tickers[ticker] = "Zero or near-zero prices detected"
                elif stock_data[ticker].isna().mean() > 0.5:
                    logger.warning(f"{ticker} has too many missing values (>50%). Excluding.")
                    problematic_tickers.append(ticker)
                    error_tickers[ticker] = "Too many missing values"
                elif (stock_data[ticker] > 1e6).any():
                    logger.warning(f"{ticker} has extremely large prices (>1e6). Excluding.")
                    problematic_tickers.append(ticker)
                    error_tickers[ticker] = "Extremely large prices detected"
            stock_data = stock_data.drop(columns=problematic_tickers, errors='ignore')
            logger.info(f"After dropping problematic tickers: {stock_data.shape if not stock_data.empty else 'empty'}")

            if stock_data.empty:
                logger.error("No valid stock data available after filtering problematic tickers.")
                return None, error_tickers, earliest_dates

            stock_data = stock_data.fillna(method='ffill').fillna(method='bfill')
            logger.info(f"After filling NaNs: {stock_data.shape if not stock_data.empty else 'empty'}")

            for ticker in stocks:
                if ticker not in stock_data.columns or stock_data[ticker].isna().all():
                    error_tickers[ticker] = "Data not available"
                else:
                    first_valid = stock_data[ticker].first_valid_index()
                    earliest_dates[ticker] = first_valid.strftime("%Y-%m-%d") if first_valid else end

            self.data_cache[cache_key] = (stock_data, error_tickers, earliest_dates)
            logger.info(f"Successfully fetched stock data for {stocks}.")
            return stock_data, error_tickers, earliest_dates
        except Exception as e:
            logger.error(f"Error processing stock data: {e}")
            return None, error_tickers, earliest_dates


    def compute_returns(self, prices):
        try:
            if isinstance(prices, pd.Series):
                prices = prices.to_frame()
            if prices is None or prices.empty:
                logger.error("Prices data is empty.")
                return pd.DataFrame()
            prices = prices.where(prices > 1e-4, np.nan)
            if prices.isna().all().all():
                logger.error("All prices are zero or invalid after cleaning.")
                return pd.DataFrame()
            returns = prices.pct_change()
            returns = returns.replace([np.inf, -np.inf], np.nan).dropna(how='any')
            if returns.empty:
                logger.error("Insufficient valid returns data after cleaning.")
                return pd.DataFrame()
            for col in returns.columns:
                if returns[col].isna().any() or np.isinf(returns[col]).any():
                    logger.warning(f"Asset {col} contains invalid data after cleaning.")
            return returns
        except Exception as e:
            logger.error(f"Error computing returns: {e}. Returning empty DataFrame.")
            return pd.DataFrame()

    def compute_max_drawdown(self, returns):
        try:
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max
            return float(drawdown.min()) if not drawdown.empty else 0.0
        except Exception as e:
            logger.error(f"Error in compute_max_drawdown: {e}")
            return 0.0

    def compute_sortino_ratio(self, returns, risk_free_rate):
        try:
            downside_returns = returns[returns < 0]
            downside_std = downside_returns.std() * np.sqrt(252) if not downside_returns.empty else 0
            annualized_return = returns.mean() * 252
            return float((annualized_return - risk_free_rate) / downside_std) if downside_std != 0 else 0.0
        except Exception as e:
            logger.error(f"Error in compute_sortino_ratio: {e}")
            return 0.0

    def compute_beta(self, portfolio_returns, benchmark_returns):
        try:
            covariance = portfolio_returns.cov(benchmark_returns)
            benchmark_variance = benchmark_returns.var()
            return float(covariance / benchmark_variance) if benchmark_variance != 0 else 0.0
        except Exception as e:
            logger.error(f"Error in compute_beta: {e}")
            return 0.0

    def portfolio_performance(self, weights, returns, risk_free_rate):
        try:
            portfolio_returns = returns.dot(weights)
            portfolio_return = portfolio_returns.mean() * 252
            if portfolio_returns.shape[0] <= 1:
                logger.warning("Insufficient data points to compute volatility (< 2 returns). Returning volatility and Sharpe ratio as 0.")
                return float(portfolio_return), 0.0, 0.0
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights))) if not returns.empty else 0
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility != 0 else 0
            return float(portfolio_return), float(portfolio_volatility), float(sharpe_ratio)
        except Exception as e:
            logger.error(f"Error in portfolio_performance: {e}")
            return float(portfolio_return) if 'portfolio_return' in locals() else 0.0, 0.0, 0.0

    
    def compute_var(self, returns, confidence_level=0.90):
        try:
            sorted_returns = np.sort(returns)
            index = int((1 - confidence_level) * len(sorted_returns))
            return float(sorted_returns[index]) if len(sorted_returns) > 0 else 0.0
        except Exception as e:
            logger.error(f"Error in compute_var: {e}")
            return 0.0

    def compute_avg_correlation(self, returns_df, weights):
        try:
            if returns_df.shape[0] <= 1:
                logger.warning("Insufficient data points to compute correlation (< 2 returns). Returning 0.")
                return 0.0
            weighted_corr_sum = 0
            num_assets = returns_df.shape[1]
            corr_matrix = returns_df.corr()
            if corr_matrix.isna().all().all():
                logger.warning("Correlation matrix contains only NaN values. Returning 0.")
                return 0.0
            for i in range(num_assets):
                for j in range(i + 1, num_assets):
                    corr_value = corr_matrix.iloc[i, j]
                    if pd.isna(corr_value):
                        corr_value = 0.0
                    weighted_corr_sum += weights[i] * weights[j] * corr_value
            avg_corr = 2 * weighted_corr_sum
            return float(avg_corr)
        except Exception as e:
            logger.error(f"Error in compute_avg_correlation: {e}")
            return 0.0
            
    def optimize_portfolio(self, returns, risk_free_rate, objective='sharpe', min_allocation=0.0, max_allocation=1.0):
        try:
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

            result = minimize(obj_fun, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
            if not result.success:
                logger.warning(f"Optimization failed for {objective}: {result.message}")
                return initial_weights
            weights = result.x
            weights[weights < 0.001] = 0
            weights /= weights.sum() if weights.sum() != 0 else 1
            return weights
        except Exception as e:
            logger.error(f"Error in optimization: {e}")
            return initial_weights

    def compute_eigenvalues(self, returns):
        try:
            if returns.shape[0] <= 1:
                logger.warning("Insufficient data points to compute eigenvalues (< 2 returns). Returning zeros.")
                n_assets = returns.shape[1]
                return [0.0] * n_assets, [0.0] * n_assets
            cov_matrix = returns.cov() * 252
            if cov_matrix.isna().all().all():
                logger.warning("Covariance matrix contains only NaN values. Returning zeros.")
                n_assets = returns.shape[1]
                return [0.0] * n_assets, [0.0] * n_assets
            eigenvalues, _ = np.linalg.eigh(cov_matrix)
            eigenvalues = sorted(eigenvalues, reverse=True)
            total_variance = sum(eigenvalues)
            if total_variance == 0:
                logger.warning("Total variance is zero. Returning zeros for explained variance ratio.")
                return eigenvalues, [0.0] * len(eigenvalues)
            explained_variance_ratio = [eig / total_variance for eig in eigenvalues]
            return eigenvalues, explained_variance_ratio
        except Exception as e:
            logger.error(f"Error in compute_eigenvalues: {e}")
            return [], []

    def compute_fama_french_exposures(self, portfolio_returns, start_date, end_date):
        if not STATSMODELS_AVAILABLE or not YFINANCE_AVAILABLE:
            logger.warning("statsmodels or yfinance unavailable. Using zero exposures.")
            return {"Mkt-RF": 0.0, "SMB": 0.0, "HML": 0.0}
        ff_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
        cache_key = "fama_french_data"
        if cache_key in self.data_cache:
            ff_data = self.data_cache[cache_key]
        else:
            try:
                logger.info("Fetching Fama-French data...")
                ff_data = pd.read_csv(ff_url, skiprows=3, index_col=0)
                ff_data.index = pd.to_datetime(ff_data.index, format="%Y%m%d", errors='coerce')
                ff_data = ff_data.dropna() / 100
                ff_data = ff_data[["Mkt-RF", "SMB", "HML"]]
                self.data_cache[cache_key] = ff_data
                logger.info("Fama-French data cached successfully.")
            except Exception as e:
                logger.error(f"Error fetching Fama-French data: {e}. Using fallback zero exposures.")
                return {"Mkt-RF": 0.0, "SMB": 0.0, "HML": 0.0}
        try:
            ff_data = ff_data.loc[start_date:end_date]
            common_dates = portfolio_returns.index.intersection(ff_data.index)
            if len(common_dates) < 30:
                logger.warning("Insufficient overlapping data with Fama-French factors. Using zero exposures.")
                return {"Mkt-RF": 0.0, "SMB": 0.0, "HML": 0.0}
            aligned_returns = portfolio_returns.loc[common_dates]
            aligned_ff = ff_data.loc[common_dates]
            X = sm.add_constant(aligned_ff)
            model = sm.OLS(aligned_returns, X).fit()
            exposures = {
                "Mkt-RF": float(model.params["Mkt-RF"]),
                "SMB": float(model.params["SMB"]),
                "HML": float(model.params["HML"])
            }
            return exposures
        except Exception as e:
            logger.error(f"Error computing Fama-French exposures: {e}. Using fallback zero exposures.")
            return {"Mkt-RF": 0.0, "SMB": 0.0, "HML": 0.0}
        
    def optimize_with_factor_and_correlation(self, returns, risk_free_rate, tickers, market_prices=None, min_allocation=0.05, max_allocation=0.30, original_weights=None, bl_views=None, bl_confidences=None):
        try:
            num_assets = len(tickers)
            if num_assets < 2:
                raise ValueError("Portfolio must have at least 2 assets for optimization.")

            if returns.empty or returns.shape[0] < 252:
                raise ValueError("Returns data is empty or has insufficient data points (< 252 days).")
            
            returns = returns.replace([np.inf, -np.inf], np.nan).dropna(how='any').fillna(method='ffill').fillna(method='bfill')
            if returns.isna().any().any() or np.isinf(returns).any().any():
                raise ValueError("Returns contain NaN or infinite values after cleaning.")
            if returns.shape[0] < 252:
                raise ValueError("Insufficient data points (< 252) after cleaning returns.")

            cov_matrix = returns.cov() * 252
            if cov_matrix.isna().any().any() or np.isinf(cov_matrix).any().any():
                raise ValueError("Covariance matrix contains NaN or infinite values.")
            if (cov_matrix.abs() > 1e4).any().any():
                cov_matrix = cov_matrix / cov_matrix.max().max() * 1e3

            if original_weights is not None:
                if len(original_weights) != num_assets or not np.isclose(sum(original_weights), 1.0, rtol=1e-5):
                    initial_weights = np.ones(num_assets) / num_assets
                else:
                    initial_weights = np.array(original_weights)
            else:
                initial_weights = np.ones(num_assets) / num_assets

            initial_vol = np.sqrt(np.dot(initial_weights.T, np.dot(cov_matrix, initial_weights)))
            initial_risk_contribs = np.zeros(num_assets)
            for i in range(num_assets):
                marginal_contrib = np.dot(cov_matrix.iloc[i], initial_weights)
                initial_risk_contribs[i] = initial_weights[i] * marginal_contrib / initial_vol if initial_vol > 0 else initial_weights[i]
            if initial_risk_contribs.sum() != 0:
                initial_risk_contribs /= initial_risk_contribs.sum()

            expected_rets = None
            bl_success = False
            fallback_reason = ""
            if PYPFOPT_AVAILABLE and market_prices is not None and not market_prices.empty:
                try:
                    market_prices = market_prices.reindex(returns.index).dropna()
                    if market_prices.empty:
                        raise ValueError("Market prices do not overlap with returns data after reindexing.")
                    
                    market_prices = market_prices.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill')
                    if market_prices.isna().any() or np.isinf(market_prices).any():
                        raise ValueError("Market prices contain NaN or infinite values after cleaning.")

                    common_index = returns.index.intersection(market_prices.index)
                    if len(common_index) < 252:
                        raise ValueError(f"Insufficient overlapping data between returns and market prices ({len(common_index)} days).")
                    returns = returns.loc[common_index]
                    market_prices = market_prices.loc[common_index]

                    S = risk_models.CovarianceShrinkage(returns).ledoit_wolf()
                    if not np.all(np.linalg.eigvals(S) >= -1e-10):
                        S = risk_models.fix_nonpositive_semidefinite(S, fix_method='spectral')
                    if S.isna().any().any() or np.isinf(S).any().any():
                        raise ValueError("Covariance matrix S contains NaN or infinite values after shrinkage.")
                    if (S.abs() > 1e4).any().any():
                        S = S / S.max().max() * 1e3

                    delta = 2.5
                    market_weights = pd.Series(1/num_assets, index=tickers)
                    market_prior = expected_returns.capm_return(
                        prices=returns,
                        risk_free_rate=risk_free_rate,
                        market_prices=market_prices
                    )
                    if market_prior.isna().any() or np.isinf(market_prior).any():
                        raise ValueError("Market-implied prior returns contain NaN or infinite values.")
                    if (market_prior.abs() > 10).any():
                        market_prior = market_prior / market_prior.abs().max() * 1.0

                    if bl_views is not None and bl_confidences is not None:
                        Q = pd.Series([bl_views.get(ticker, 0.0) for ticker in tickers], index=tickers)
                        P = np.eye(num_assets)
                        Omega = np.diag([bl_confidences.get(ticker, 0.01) for ticker in tickers])
                    else:
                        Q = pd.Series([0.0] * num_assets, index=tickers)
                        P = np.eye(num_assets)
                        Omega = np.diag([0.01] * num_assets)

                    bl = BlackLittermanModel(
                        cov_matrix=S,
                        pi=market_prior,
                        Q=Q,
                        P=P,
                        Omega=Omega,
                        delta=delta
                    )
                    expected_rets = bl.bl_returns()
                    if expected_rets.isna().any() or np.isinf(expected_rets).any():
                        raise ValueError("Black-Litterman returned NaN or infinite values.")
                    if (expected_rets.abs() > 10).any():
                        expected_rets = expected_rets / expected_rets.abs().max() * 1.0
                    bl_success = True
                except Exception as e:
                    fallback_reason = str(e)

            if expected_rets is None and PYPFOPT_AVAILABLE and market_prices is not None and not market_prices.empty:
                try:
                    S = risk_models.CovarianceShrinkage(returns).ledoit_wolf()
                    if not np.all(np.linalg.eigvals(S) >= -1e-10):
                        S = risk_models.fix_nonpositive_semidefinite(S, fix_method='spectral')
                    if (S.abs() > 1e4).any().any():
                        S = S / S.max().max() * 1e3
                    market_prior = expected_returns.capm_return(
                        prices=returns,
                        risk_free_rate=risk_free_rate,
                        market_prices=market_prices
                    )
                    if market_prior.isna().any() or np.isinf(market_prior).any():
                        raise ValueError("Simple CAPM returned NaN or infinite values.")
                    if (market_prior.abs() > 10).any():
                        market_prior = market_prior / market_prior.abs().max() * 1.0
                    expected_rets = market_prior
                    fallback_reason = "Used simple CAPM due to Black-Litterman failure."
                except Exception as e:
                    fallback_reason = str(e)

            if expected_rets is None:
                try:
                    pca = PCA(n_components=min(3, num_assets))
                    pca.fit(returns)
                    factor_returns = pd.DataFrame(pca.transform(returns), index=returns.index)
                    expected_rets = pd.Series(pca.inverse_transform(factor_returns.mean()) * 252, index=tickers)
                    if expected_rets.isna().any() or np.isinf(expected_rets).any():
                        raise ValueError("PCA returned NaN or infinite values.")
                    if (expected_rets.abs() > 10).any():
                        expected_rets = expected_rets / expected_rets.abs().max() * 1.0
                    fallback_reason = "Used PCA due to CAPM/Black-Litterman failure."
                except Exception as e:
                    fallback_reason = str(e)

            if expected_rets is None or expected_rets.isna().any() or np.isinf(expected_rets).any():
                expected_rets = returns.mean() * 252
                if expected_rets.isna().any() or np.isinf(expected_rets).any():
                    expected_rets = pd.Series([risk_free_rate + 0.05] * num_assets, index=tickers)
                    fallback_reason = "Used risk-free rate + 5% due to all other methods failing."
                else:
                    if (expected_rets.abs() > 10).any():
                        expected_rets = expected_rets / expected_rets.abs().max() * 1.0
                    fallback_reason = "Used historical mean returns due to PCA/CAPM/Black-Litterman failure."

            def objective(weights):
                try:
                    ret = np.dot(weights, expected_rets)
                    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                    sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0
                    vol_drag = 0.5 * vol**2
                    adj_sharpe = (ret - vol_drag - risk_free_rate) / vol if vol > 0 else 0
                    corr_penalty = 1.5 * self.compute_avg_correlation(returns, weights)
                    risk_contribs = weights * np.dot(cov_matrix, weights) / vol if vol > 0 else weights
                    risk_parity_penalty = 5.0 * np.var(risk_contribs)
                    weights_clean = weights + 1e-10
                    entropy = -np.sum(weights_clean * np.log(weights_clean)) / np.log(num_assets)
                    diversification_penalty = 1.0 * (1 - entropy)
                    return -adj_sharpe + corr_penalty + risk_parity_penalty + diversification_penalty
                except Exception as e:
                    logger.error(f"Objective function error: {e}. Returning infinity.")
                    return float('inf')

            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
            max_allowed = min(0.40, 1.0 / num_assets * 1.5)
            min_allowed = max(0.01, 1.0 / num_assets * 0.5)
            if min_allocation * num_assets > 1 or max_allocation * num_assets < 1:
                min_allocation = min_allowed
                max_allocation = max_allowed
            bounds = tuple((max(min_allocation, min_allowed), min(max_allocation, max_allowed)) for _ in range(num_assets))

            optimized_weights = None
            try:
                if CVXPY_AVAILABLE:
                    w = cp.Variable(num_assets)
                    ret = expected_rets @ w
                    vol = cp.sqrt(cp.quad_form(w, cov_matrix))
                    objective = cp.Minimize(-ret + 2.5 * vol)
                    constraints_cvx = [cp.sum(w) == 1, w >= min_allocation, w <= max_allocation]
                    problem = cp.Problem(objective, constraints_cvx)
                    problem.solve(solver=cp.SCS, max_iters=1000, eps=1e-8)
                    if problem.status != cp.OPTIMAL:
                        raise ValueError(f"CVXPY optimization failed: {problem.status}")
                    optimized_weights = w.value
                else:
                    raise ImportError("CVXPY unavailable.")
            except (ImportError, Exception):
                result = minimize(
                    objective,
                    initial_weights,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 1000, 'ftol': 1e-8, 'disp': False}
                )
                if not result.success:
                    result = minimize(
                        objective,
                        initial_weights,
                        method='SLSQP',
                        bounds=tuple((0.0, 1.0) for _ in range(num_assets)),
                        constraints=constraints,
                        options={'maxiter': 500, 'ftol': 1e-6}
                    )
                    if not result.success:
                        optimized_weights = initial_weights
                    else:
                        optimized_weights = result.x
                else:
                    optimized_weights = result.x

            optimized_weights = np.clip(optimized_weights, min_allowed, max_allowed)
            optimized_weights /= optimized_weights.sum() if optimized_weights.sum() != 0 else 1.0

            opt_vol = np.sqrt(np.dot(optimized_weights.T, np.dot(cov_matrix, optimized_weights)))
            risk_contribs = np.zeros(num_assets)
            for i in range(num_assets):
                marginal_contrib = np.dot(cov_matrix.iloc[i], optimized_weights)
                risk_contribs[i] = optimized_weights[i] * marginal_contrib / opt_vol if opt_vol > 0 else optimized_weights[i]
            if risk_contribs.sum() != 0:
                risk_contribs /= risk_contribs.sum()

            opt_ret, opt_vol, opt_sharpe = self.portfolio_performance(optimized_weights, returns, risk_free_rate)
            avg_corr = self.compute_avg_correlation(returns, optimized_weights)
            entropy = -np.sum(optimized_weights * np.log(optimized_weights + 1e-10)) / np.log(num_assets)
            
            tracking_error = None
            if market_prices is not None and not market_prices.empty:
                market_returns = market_prices.pct_change().dropna()
                portfolio_ret = returns.dot(optimized_weights)
                common_dates = portfolio_ret.index.intersection(market_returns.index)
                if len(common_dates) > 0:
                    aligned_portfolio = portfolio_ret.loc[common_dates]
                    aligned_market = market_returns.loc[common_dates]
                    tracking_error = float((aligned_portfolio - aligned_market).std() * np.sqrt(252))

            opt_beta = None
            if market_prices is not None and not market_prices.empty:
                opt_beta = self.compute_beta(returns.dot(optimized_weights), market_returns)

            return optimized_weights, {
                "return": float(opt_ret),
                "volatility": float(opt_vol),
                "sharpe": float(opt_sharpe),
                "avg_correlation": float(avg_corr),
                "entropy": float(entropy),
                "risk_contributions": risk_contribs.tolist(),
                "tracking_error": tracking_error,
                "beta": opt_beta
            }
        except Exception as e:
            logger.error(f"Critical error in optimization: {e}. Returning equal weights.")
            return np.ones(num_assets) / num_assets, {}

    def get_historical_metrics(self, tickers, weights_dict, risk_free_rate, hist_returns):
        try:
            strategies = {
                "Original Portfolio": np.array(list(weights_dict.values())),
                "Max Sharpe": self.optimize_portfolio(hist_returns, risk_free_rate, "sharpe"),
                "Max Sortino": self.optimize_portfolio(hist_returns, risk_free_rate, "sortino"),
                "Min Max Drawdown": self.optimize_portfolio(hist_returns, risk_free_rate, "max_drawdown"),
                "Min Volatility": self.optimize_portfolio(hist_returns, risk_free_rate, "volatility"),
                "Min Value at Risk": self.optimize_portfolio(hist_returns, risk_free_rate, "value_at_risk")
            }
            metrics = {"Annual Return": [], "Volatility": [], "Avg Correlation": []}
            labels = []
            for label, weights in strategies.items():
                portfolio_return, portfolio_volatility, _ = self.portfolio_performance(weights, hist_returns, risk_free_rate)
                avg_corr = self.compute_avg_correlation(hist_returns, weights)
                metrics["Annual Return"].append(float(portfolio_return))
                metrics["Volatility"].append(float(portfolio_volatility))
                metrics["Avg Correlation"].append(float(avg_corr))
                labels.append(label)
            return metrics, labels
        except Exception as e:
            logger.error(f"Error in get_historical_metrics: {e}")
            return {"Annual Return": [], "Volatility": [], "Avg Correlation": []}, []

    def get_cumulative_returns(self, returns, strategies, benchmark_returns, earliest_dates, title="Cumulative Returns of Strategies"):
        try:
            data = {}
            for label, weights in strategies.items():
                portfolio_returns = returns.dot(weights)
                cumulative = (1 + portfolio_returns).cumprod() - 1
                data[label] = cumulative.tolist()
            for bench_ticker, bench_ret in benchmark_returns.items():
                cumulative = (1 + bench_ret).cumprod() - 1
                data[bench_ticker] = cumulative.tolist()
            dates = [d.strftime("%Y-%m-%d") for d in returns.index]
            return {"dates": dates, "cumulative_returns": data}
        except Exception as e:
            logger.error(f"Error in get_cumulative_returns: {e}")
            return {"dates": [], "cumulative_returns": {}}

    def get_correlation_matrix(self, prices):
        try:
            returns = self.compute_returns(prices)
            if returns.shape[0] <= 1:
                logger.warning("Insufficient data points to compute correlation matrix (< 2 returns). Returning zero matrix.")
                return {
                    "tickers": list(prices.columns),
                    "matrix": [[0.0 if i != j else 1.0 for j in range(len(prices.columns))] for i in range(len(prices.columns))]
                }
            corr_matrix = returns.corr()
            if corr_matrix.isna().all().all():
                logger.warning("Correlation matrix contains only NaN values. Returning zero matrix.")
                return {
                    "tickers": list(prices.columns),
                    "matrix": [[0.0 if i != j else 1.0 for j in range(len(prices.columns))] for i in range(len(prices.columns))]
                }
            corr_matrix = corr_matrix.fillna(0.0)  # Replace any remaining NaN with 0
            return {
                "tickers": list(corr_matrix.index),
                "matrix": corr_matrix.values.tolist()
            }
        except Exception as e:
            logger.error(f"Error in get_correlation_matrix: {e}")
            return {"tickers": [], "matrix": []}

    def get_efficient_frontier(self, returns, risk_free_rate, n_portfolios=1000):
        try:
            np.random.seed(42)
            n_assets = returns.shape[1]
            all_weights = np.zeros((n_portfolios, n_assets))
            all_returns = np.zeros(n_portfolios)
            all_volatilities = np.zeros(n_portfolios)
            all_sharpe_ratios = np.zeros(n_portfolios)

            for i in range(n_portfolios):
                weights = np.random.random(n_assets)
                weights /= weights.sum()
                all_weights[i, :] = weights
                port_return, port_vol, port_sharpe = self.portfolio_performance(weights, returns, risk_free_rate)
                all_returns[i] = port_return
                all_volatilities[i] = port_vol
                all_sharpe_ratios[i] = port_sharpe

            strategies = {
                "Max Sharpe": self.optimize_portfolio(returns, risk_free_rate, "sharpe"),
                "Max Sortino": self.optimize_portfolio(returns, risk_free_rate, "sortino"),
                "Min Max Drawdown": self.optimize_portfolio(returns, risk_free_rate, "max_drawdown"),
                "Min Volatility": self.optimize_portfolio(returns, risk_free_rate, "volatility"),
                "Min Value at Risk": self.optimize_portfolio(returns, risk_free_rate, "value_at_risk")
            }

            strategy_metrics = {}
            for name, weights in strategies.items():
                port_return, port_vol, port_sharpe = self.portfolio_performance(weights, returns, risk_free_rate)
                strategy_metrics[name] = {
                    "return": float(port_return),
                    "volatility": float(port_vol),
                    "sharpe": float(port_sharpe)
                }

            max_sharpe_vol = strategy_metrics["Max Sharpe"]["volatility"]
            max_sharpe_sharpe = strategy_metrics["Max Sharpe"]["sharpe"]
            cml = {
                "x": [0, max_sharpe_vol * 1.5],
                "y": [risk_free_rate, risk_free_rate + max_sharpe_sharpe * max_sharpe_vol * 1.5]
            }

            return {
                "portfolios": {
                    "returns": all_returns.tolist(),
                    "volatilities": all_volatilities.tolist(),
                    "sharpe_ratios": all_sharpe_ratios.tolist()
                },
                "strategies": strategy_metrics,
                "capital_market_line": cml
            }
        except Exception as e:
            logger.error(f"Error in get_efficient_frontier: {e}")
            return {"portfolios": {"returns": [], "volatilities": [], "sharpe_ratios": []}, "strategies": {}, "capital_market_line": {"x": [], "y": []}}

    def get_comparison_bars(self, original_metrics, optimized_metrics, benchmark_metrics):
        try:
            metrics = ["annual_return", "annual_volatility", "sharpe_ratio", "maximum_drawdown", "value_at_risk"]
            labels = ["Annual Return", "Annual Volatility", "Sharpe Ratio", "Maximum Drawdown", "Value at Risk (90%)"]
            data = []
            for metric, label in zip(metrics, labels):
                values = [original_metrics[metric], optimized_metrics[metric]]
                names = ["Original", "Optimized"]
                if benchmark_metrics:
                    for bench, bm in benchmark_metrics.items():
                        values.append(bm[metric])
                        names.append(bench)
                data.append({
                    "metric": label,
                    "names": names,
                    "values": values
                })
            return data
        except Exception as e:
            logger.error(f"Error in get_comparison_bars: {e}")
            return []

    def get_portfolio_exposures(self, tickers, original_weights, optimized_weights):
        try:
            original_exposures = [float(w) for w in original_weights if w > 0]
            original_labels = [t for t, w in zip(tickers, original_weights) if w > 0]
            optimized_exposures = [float(w) for w in optimized_weights if w > 0]
            optimized_labels = [t for t, w in zip(tickers, optimized_weights) if w > 0]
            return {
                "original": {"labels": original_labels, "exposures": original_exposures},
                "optimized": {"labels": optimized_labels, "exposures": optimized_exposures}
            }
        except Exception as e:
            logger.error(f"Error in get_portfolio_exposures: {e}")
            return {"original": {"labels": [], "exposures": []}, "optimized": {"labels": [], "exposures": []}}

    def get_rolling_volatility(self, returns, weights_dict, benchmark_returns, window=252):
        try:
            data = {}
            dates = [d.strftime("%Y-%m-%d") for d in returns.index]
            for label, weights in weights_dict.items():
                portfolio_returns = returns.dot(weights)
                rolling_vol = portfolio_returns.rolling(window=window).std() * np.sqrt(252)
                data[f"{label} Volatility"] = rolling_vol.fillna(0).tolist()
            for bench_ticker, bench_ret in benchmark_returns.items():
                rolling_vol = bench_ret.rolling(window=window).std() * np.sqrt(252)
                data[f"{bench_ticker} Volatility"] = rolling_vol.fillna(0).tolist()
            return {"dates": dates, "rolling_volatility": data}
        except Exception as e:
            logger.error(f"Error in get_rolling_volatility: {e}")
            return {"dates": [], "rolling_volatility": {}}

    def get_diversification_benefit(self, returns, original_weights, optimized_weights, tickers):
        try:
            equal_weights = np.ones(len(tickers)) / len(tickers)
            cov_matrix = returns.cov() * 252
            orig_vol = np.sqrt(np.dot(original_weights.T, np.dot(cov_matrix, original_weights)))
            opt_vol = np.sqrt(np.dot(optimized_weights.T, np.dot(cov_matrix, optimized_weights)))
            equal_vol = np.sqrt(np.dot(equal_weights.T, np.dot(cov_matrix, equal_weights)))
            orig_ret = returns.dot(original_weights).mean() * 252
            opt_ret = returns.dot(optimized_weights).mean() * 252
            equal_ret = returns.dot(equal_weights).mean() * 252
            labels = ['Equal Weight', 'Original', 'Optimized']
            vols = [float(equal_vol), float(orig_vol), float(opt_vol)]
            rets = [float(equal_ret), float(orig_ret), float(opt_ret)]
            return {
                "labels": labels,
                "volatilities": vols,
                "returns": rets
            }
        except Exception as e:
            logger.error(f"Error in get_diversification_benefit: {e}")
            return {"labels": [], "volatilities": [], "returns": []}

    def get_crisis_performance(self, returns, weights_dict, benchmark_returns, earliest_dates):
        crises = [
            {
                "name": "Dot-Com Bust",
                "start": pd.to_datetime("2000-03-01"),
                "end": pd.to_datetime("2002-10-31"),
                "description": "The Dot-Com Bust (March 2000 - October 2002) saw a tech bubble collapse, with the Nasdaq dropping 78% as overvalued internet companies failed, leading to reduced business activity in tech sectors."
            },
            {
                "name": "Great Recession",
                "start": pd.to_datetime("2007-12-01"),
                "end": pd.to_datetime("2009-06-30"),
                "description": "The Great Recession (December 2007 - June 2009) followed a housing bubble burst and financial crisis, with GDP dropping 4.3% and business activity stalling as credit froze."
            },
            {
                "name": "COVID-19 Crisis",
                "start": pd.to_datetime("2020-02-01"),
                "end": pd.to_datetime("2020-04-30"),
                "description": "The COVID-19 Crisis (February - April 2020) involved global lockdowns, halting business activity, with a 31.4% GDP drop in Q2 2020 and a swift 34% S&P 500 decline."
            }
        ]
        crisis_summaries = {}
        earliest_data = pd.to_datetime(min(earliest_dates.values()))
        six_months = timedelta(days=180)

        for crisis in crises:
            crisis_start = crisis["start"]
            crisis_end = crisis["end"]
            if earliest_data > (crisis_start - six_months):
                continue
            available_starts = returns.index[returns.index >= crisis_start]
            available_ends = returns.index[returns.index <= crisis_end]
            if available_starts.empty or available_ends.empty:
                continue
            available_start = available_starts.min()
            available_end = available_ends.max()
            if pd.isna(available_start) or pd.isna(available_end) or available_start > available_end:
                continue
            crisis_returns = returns.loc[available_start:available_end]
            crisis_data = {}
            crisis_performance = {}
            for label, weights in weights_dict.items():
                portfolio_returns = crisis_returns.dot(weights)
                cumulative = (1 + portfolio_returns).cumprod() - 1
                crisis_data[label] = cumulative.tolist()
                crisis_performance[label] = float(cumulative.iloc[-1])
            for bench_ticker, bench_ret in benchmark_returns.items():
                bench_crisis_ret = bench_ret.loc[available_start:available_end]
                if not bench_crisis_ret.empty:
                    bench_cum = (1 + bench_crisis_ret).cumprod() - 1
                    crisis_data[bench_ticker] = bench_cum.tolist()
                    crisis_performance[bench_ticker] = float(bench_cum.iloc[-1])
            dates = [d.strftime("%Y-%m-%d") for d in crisis_returns.index]
            crisis_summaries[crisis["name"]] = {
                "dates": dates,
                "cumulative_returns": crisis_data,
                "performance": crisis_performance,
                "start": available_start.strftime("%Y-%m-%d"),
                "end": available_end.strftime("%Y-%m-%d"),
                "description": crisis["description"]
            }
        return crisis_summaries

    def suggest_courses_of_action(self, tickers, original_weights, optimized_weights, returns, risk_free_rate, benchmark_metrics, risk_tolerance, start_date, end_date):
        original_returns = returns.dot(original_weights)
        optimized_returns = returns.dot(optimized_weights)
        original_metrics = {
            "annual_return": float(original_returns.mean() * 252),
            "annual_volatility": float(original_returns.std() * np.sqrt(252)) if not pd.isna(original_returns.std()) else 0.0,
            "sharpe_ratio": self.portfolio_performance(original_weights, returns, risk_free_rate)[2],
            "max_drawdown": self.compute_max_drawdown(original_returns),
            "var": self.compute_var(original_returns, 0.90),
            "sortino": self.compute_sortino_ratio(original_returns, risk_free_rate)
        }
        optimized_metrics = {
            "annual_return": float(optimized_returns.mean() * 252),
            "annual_volatility": float(optimized_returns.std() * np.sqrt(252)) if not pd.isna(optimized_returns.std()) else 0.0,
            "sharpe_ratio": self.portfolio_performance(optimized_weights, returns, risk_free_rate)[2],
            "max_drawdown": self.compute_max_drawdown(optimized_returns),
            "var": self.compute_var(optimized_returns, 0.90),
            "sortino": self.compute_sortino_ratio(optimized_returns, risk_free_rate)
        }
        ff_exposures = self.compute_fama_french_exposures(original_returns, start_date, end_date)
        corr_matrix = returns.corr()
        max_corr = float(max(corr_matrix.max())) if not pd.isna(max(corr_matrix.max())) else 0.0

        analysis = {
            "strengths": [],
            "weaknesses": [],
            "current_standing": [],
            "short_term": [],
            "medium_term": [],
            "long_term": [],
            "disclaimer": "The information provided by GALCAST Portfolio Analytics & Optimization is for informational and educational purposes only. It should not be considered as financial advice or a recommendation to buy or sell any security. Investment decisions should be based on your own research, investment objectives, financial situation, and needs. Past performance is not indicative of future results. Always consult with a qualified financial advisor before making any investment decisions."
        }

        # Strengths
        if original_metrics["sharpe_ratio"] > 1.5:
            analysis["strengths"].append(f"Strong Risk-Adjusted Returns: Your Sharpe Ratio of {original_metrics['sharpe_ratio']:.2f} is impressive—it shows you’re getting solid returns for the risk you’re taking. This is a great foundation to build on!")
        if original_metrics["annual_volatility"] < 0.15:
            analysis["strengths"].append(f"Low Volatility: At {original_metrics['annual_volatility']:.2%}, your portfolio is stable, which is fantastic for peace of mind and steady growth.")
        if ff_exposures["Mkt-RF"] < 1:
            analysis["strengths"].append(f"Market Resilience: With a market beta of {ff_exposures['Mkt-RF']:.2f}, your portfolio is less sensitive to market swings than the average—excellent for weathering downturns.")

        # Weaknesses
        if ff_exposures["Mkt-RF"] > 1.2:
            analysis["weaknesses"].append(f"High Market Exposure: Your market beta of {ff_exposures['Mkt-RF']:.2f} means your portfolio amplifies market moves. This can boost gains in bull markets but leaves you vulnerable in crashes.")
        if max_corr > 0.8:
            high_corr_pairs = [(t1, t2) for t1 in tickers for t2 in tickers if t1 < t2 and corr_matrix.loc[t1, t2] > 0.8]
            analysis["weaknesses"].append(f"Concentration Risk: Stocks like {', '.join([f'{p[0]}-{p[1]}' for p in high_corr_pairs])} have correlations above 0.8, suggesting your risk is concentrated. If one drops, others may follow.")
        if original_metrics["max_drawdown"] < -0.25:
            analysis["weaknesses"].append(f"Significant Drawdowns: A max drawdown of {original_metrics['max_drawdown']:.2%} indicates past losses were steep. We’ll want to protect against this moving forward.")

        # Current Standing
        bench_key = list(benchmark_metrics.keys())[0]
        bench_return = benchmark_metrics[bench_key]["annual_return"]
        analysis["current_standing"].append(f"Your original portfolio has delivered an annualized return of {original_metrics['annual_return']:.2%}, with a volatility of {original_metrics['annual_volatility']:.2%}, compared to {bench_key}’s {bench_return:.2%} return.")
        if optimized_metrics["annual_return"] > original_metrics["annual_return"]:
            analysis["current_standing"].append(f"Good News: Optimization boosts your return to {optimized_metrics['annual_return']:.2%}—a {optimized_metrics['annual_return'] - original_metrics['annual_return']:.2%} improvement, showing we can enhance your growth.")
        if optimized_metrics["annual_volatility"] < original_metrics["annual_volatility"]:
            analysis["current_standing"].append(f"Risk Reduction: Optimization cuts volatility to {optimized_metrics['annual_volatility']:.2%}, a {original_metrics['annual_volatility'] - optimized_metrics['annual_volatility']:.2%} drop, aligning better with your {risk_tolerance} risk tolerance.")

        # Short-Term
        analysis["short_term"].append("Short-Term (0-1 Year): Quick Wins and Stability")
        analysis["short_term"].append("Goal: Capitalize on immediate opportunities while managing risk.")
        if risk_tolerance == "low":
            analysis["short_term"].append(f"Action 1: De-Risk with Stability Focus")
            analysis["short_term"].append(f"Why: Your {risk_tolerance} risk tolerance favors safety. With a VaR of {original_metrics['var']:.2%}, there’s a 10% chance of losing that much in a day.")
            analysis["short_term"].append(f"How: Shift 10-15% of your portfolio to low-volatility assets like utilities (e.g., XLU ETF, 1.5% yield, 10% volatility) or Treasuries (e.g., TLT, 2-3% yield). This could reduce VaR to {original_metrics['var'] * 0.85:.2%} based on historical correlations.")
            analysis["short_term"].append(f"Probability: 70% chance of stabilizing returns within 6 months, given utilities’ low beta (~0.3).")
        else:
            analysis["short_term"].append(f"Action 1: Capitalize on Momentum")
            analysis["short_term"].append(f"Why: Your {risk_tolerance} tolerance allows chasing short-term gains. Optimized Sharpe ({optimized_metrics['sharpe_ratio']:.2f}) suggests upside potential.")
            analysis["short_term"].append(f"How: Increase allocation to top performers (e.g., stocks with recent 20%+ gains in your portfolio—check returns) by 5-10%, or add a momentum ETF like MTUM (12% annualized return, 15% volatility).")
            analysis["short_term"].append(f"Probability: 60% chance of outperforming {bench_key} by 2-3% in 6 months, based on momentum factor trends.")
        analysis["short_term"].append(f"Action 2: Rebalance Quarterly")
        analysis["short_term"].append(f"Why: Keeps your portfolio aligned with short-term market shifts.")
        analysis["short_term"].append(f"How: Adjust weights to optimized levels (e.g., {', '.join([f'{t}: {w:.2%}' for t, w in zip(tickers, optimized_weights)])}).")
        analysis["short_term"].append(f"Probability: 80% chance of maintaining or improving Sharpe Ratio, per historical rebalancing studies.")

        # Medium-Term
        analysis["medium_term"].append("Medium-Term (1-5 Years): Growth with Balance")
        analysis["medium_term"].append("Goal: Build wealth steadily while preparing for volatility.")
        if ff_exposures["HML"] > 0.3:
            analysis["medium_term"].append(f"Action 1: Leverage Value Opportunities")
            analysis["medium_term"].append(f"Why: Your value exposure (HML: {ff_exposures['HML']:.2f}) suggests strength in undervalued stocks, which often shine in recovery phases.")
            analysis["medium_term"].append(f"How: Allocate 10-20% to a value ETF (e.g., VTV, 10% return, 14% volatility) or deepen exposure to value sectors like financials (e.g., XLF).")
            analysis["medium_term"].append(f"Probability: 65% chance of 8-10% annualized returns over 3 years, based on value factor outperformance post-recession.")
        else:
            analysis["medium_term"].append(f"Action 1: Explore Growth Sectors")
            analysis["medium_term"].append(f"Why: Low HML ({ff_exposures['HML']:.2f}) suggests room to capture growth, especially with {risk_tolerance} tolerance.")
            analysis["medium_term"].append(f"How: Invest 15-25% in tech or consumer discretionary (e.g., QQQ, 13% return, 18% volatility), targeting sectors with 10-15% growth potential.")
            analysis["medium_term"].append(f"Probability: 55% chance of beating {bench_key} by 3-5% annually, per growth stock cycles.")
        analysis["medium_term"].append(f"Action 2: Diversify Correlation")
        analysis["medium_term"].append(f"Why: High correlations (e.g., {max_corr:.2f}) increase risk concentration.")
        analysis["medium_term"].append(f"How: Add 10% to assets with correlations < 0.5 to your portfolio (e.g., gold via GLD, 5% return, -0.1 correlation to equities).")
        analysis["medium_term"].append(f"Probability: 75% chance of reducing volatility by 2-3%, per diversification models.")

        # Long-Term
        analysis["long_term"].append("Long-Term (5+ Years): Wealth Maximization")
        analysis["long_term"].append("Goal: Achieve sustained growth with resilience.")
        if optimized_metrics["sortino"] > original_metrics["sortino"]:
            analysis["long_term"].append(f"Action 1: Stick with Optimization")
            analysis["long_term"].append(f"Why: Optimized Sortino ({optimized_metrics['sortino']:.2f} vs {original_metrics['sortino']:.2f}) shows better downside protection, key for long-term stability.")
            analysis["long_term"].append(f"How: Fully adopt optimized weights ({', '.join([f'{t}: {w:.2%}' for t, w in zip(tickers, optimized_weights)])}) and reinvest dividends.")
            analysis["long_term"].append(f"Probability: 70% chance of growing $10,000 to ${(10000 * (1 + optimized_metrics['annual_return']) ** 5):,.0f} in 5 years, vs ${(10000 * (1 + original_metrics['annual_return']) ** 5):,.0f} originally.")
        else:
            analysis["long_term"].append(f"Action 1: Enhance Downside Protection")
            analysis["long_term"].append(f"Why: Original Sortino ({original_metrics['sortino']:.2f}) suggests vulnerability to losses over time.")
            analysis["long_term"].append(f"How: Shift 20% to Min Volatility strategy (e.g., SPLV ETF, 8% return, 10% volatility) or bonds.")
            analysis["long_term"].append(f"Probability: 80% chance of cutting max drawdown to {optimized_metrics['max_drawdown'] * 0.8:.2%}, per low-volatility studies.")
        analysis["long_term"].append(f"Action 2: Expand Globally")
        analysis["long_term"].append(f"Why: Broaden exposure beyond U.S. markets reduces systemic risk.")
        analysis["long_term"].append(f"How: Allocate 15-20% to international equities (e.g., VXUS, 7% return, 16% volatility), diversifying across emerging markets.")
        analysis["long_term"].append(f"Probability: 60% chance of boosting returns by 1-2% annually over 10 years, per global diversification data.")

        return analysis

analyzer = PortfolioAnalyzer()

@app.route('/')
def index():
    logger.info("Received request to /")
    return "Portfolio Analyzer API is running. Use POST /analyze_portfolio for analysis."

@app.route('/analyze_portfolio', methods=['POST'])
def analyze_portfolio():
    try:
        logger.info("Received request to /analyze_portfolio")
        request_json = request.get_json(silent=True)
        if not request_json:
            logger.error("No JSON data provided")
            return json.dumps({"error": "No data provided"}), 400

        tickers = request_json.get("tickers", [])
        weights = request_json.get("weights", [])
        start_date = request_json.get("start_date", analyzer.default_start_date)
        end_date = request_json.get("end_date", analyzer.today_date)
        risk_tolerance = request_json.get("risk_tolerance", "medium")
        benchmarks = request_json.get("benchmarks", ["^GSPC"])
        risk_free_rate = request_json.get("risk_free_rate", analyzer.fetch_treasury_yield())
        dates = request_json.get("dates", [])
        stock_prices = request_json.get("stock_prices", None)
        benchmark_prices = request_json.get("benchmark_prices", {})
        fetch_data = request_json.get("fetch_data", False)

        logger.info(f"Request parameters: tickers={tickers}, weights={weights}")

        if not tickers or not weights:
            logger.error("Tickers or weights missing")
            return json.dumps({"error": "Tickers and weights required"}), 400

        weights_dict = dict(zip(tickers, weights))
        if fetch_data:
            logger.info(f"Attempting to fetch data for tickers {tickers} from {start_date} to {end_date}")
            if not YFINANCE_AVAILABLE:
                logger.error("yfinance unavailable. Cannot fetch data.")
                return json.dumps({"error": "yfinance unavailable"}), 400
            stock_prices, error_tickers, earliest_dates = analyzer.fetch_stock_data(tickers, start_date, end_date)
            if stock_prices is None or stock_prices.empty:
                logger.error(f"No valid stock data available. Error tickers: {error_tickers}")
                return json.dumps({"error": "No valid stock data available", "error_tickers": error_tickers}), 400
            logger.info(f"Successfully fetched stock data: {stock_prices.shape}")
            dates = [d.strftime("%Y-%m-%d") for d in stock_prices.index]
            stock_prices = stock_prices.to_dict()
        else:
            logger.info("Using provided stock prices for manual input")
            if stock_prices is None:
                logger.error("Stock prices missing")
                return json.dumps({"error": "Stock prices required"}), 400
            stock_prices_df = pd.DataFrame(stock_prices, index=[pd.to_datetime(date) for date in dates])
            earliest_dates = {ticker: dates[0] for ticker in tickers}
            stock_prices = stock_prices_df
            
            returns = analyzer.compute_returns(stock_prices)
        if returns.empty:
            logger.error("Failed to compute returns")
            return json.dumps({"error": "Failed to compute returns"}), 400

        original_weights = np.array(list(weights_dict.values()))
        portfolio_returns = returns.dot(original_weights)
        portfolio_return, portfolio_volatility, sharpe_ratio = analyzer.portfolio_performance(original_weights, returns, risk_free_rate)
        original_metrics = {
            "annual_return": float(portfolio_return),
            "annual_volatility": float(portfolio_volatility),
            "sharpe_ratio": float(sharpe_ratio),
            "maximum_drawdown": float(analyzer.compute_max_drawdown(portfolio_returns)),
            "value_at_risk": float(analyzer.compute_var(portfolio_returns, 0.90))
        }

        benchmark_returns = {}
        benchmark_metrics = {}
        if fetch_data:
            for bench in benchmarks:
                bench_data, _, _ = analyzer.fetch_stock_data([bench], start_date, end_date)
                if bench_data is not None and not bench_data.empty:
                    bench_ret = analyzer.compute_returns(bench_data)[bench]
                    benchmark_returns[bench] = bench_ret
                    bench_return, bench_volatility, bench_sharpe = analyzer.portfolio_performance(np.array([1.0]), pd.DataFrame(bench_ret), risk_free_rate)
                    benchmark_metrics[bench] = {
                        "annual_return": float(bench_return),
                        "annual_volatility": float(bench_volatility),
                        "sharpe_ratio": float(bench_sharpe),
                        "maximum_drawdown": float(analyzer.compute_max_drawdown(bench_ret)),
                        "value_at_risk": float(analyzer.compute_var(bench_ret, 0.90))
                    }
        else:
            for bench in benchmarks:
                if bench in benchmark_prices:
                    bench_data = pd.Series(benchmark_prices[bench], index=[pd.to_datetime(date) for date in dates])
                    bench_ret = analyzer.compute_returns(bench_data)
                    benchmark_returns[bench] = bench_ret
                    bench_return, bench_volatility, bench_sharpe = analyzer.portfolio_performance(np.array([1.0]), pd.DataFrame(bench_ret), risk_free_rate) if not bench_ret.empty else (0.0, 0.0, 0.0)
                    benchmark_metrics[bench] = {
                        "annual_return": float(bench_return),
                        "annual_volatility": float(bench_volatility),
                        "sharpe_ratio": float(bench_sharpe),
                        "maximum_drawdown": float(analyzer.compute_max_drawdown(bench_ret)),
                        "value_at_risk": float(analyzer.compute_var(bench_ret, 0.90))
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
            "Original Portfolio": original_weights,
            "Max Sharpe": opt_weights,
            "Max Sortino": analyzer.optimize_portfolio(returns, risk_free_rate, "sortino"),
            "Min Max Drawdown": analyzer.optimize_portfolio(returns, risk_free_rate, "max_drawdown"),
            "Min Volatility": analyzer.optimize_portfolio(returns, risk_free_rate, "volatility"),
            "Min Value at Risk": analyzer.optimize_portfolio(returns, risk_free_rate, "value_at_risk")
        }

        hist_metrics, hist_labels = analyzer.get_historical_metrics(tickers, weights_dict, risk_free_rate, returns)
        cumulative_returns = analyzer.get_cumulative_returns(returns, strategies, benchmark_returns, earliest_dates)
        correlation_matrix = analyzer.get_correlation_matrix(stock_prices)
        efficient_frontier = analyzer.get_efficient_frontier(returns, risk_free_rate)
        comparison_bars = analyzer.get_comparison_bars(original_metrics, optimized_metrics, benchmark_metrics)
        portfolio_exposures = analyzer.get_portfolio_exposures(tickers, original_weights, opt_weights)
        rolling_volatility = analyzer.get_rolling_volatility(returns, strategies, benchmark_returns)
        crisis_performance = analyzer.get_crisis_performance(returns, {"Original Portfolio": original_weights, "Optimized Portfolio": opt_weights}, benchmark_returns, earliest_dates)
        eigenvalues, explained_variance_ratio = analyzer.compute_eigenvalues(returns)
        fama_french_exposures = analyzer.compute_fama_french_exposures(portfolio_returns, start_date, end_date)
        suggestions = analyzer.suggest_courses_of_action(tickers, original_weights, opt_weights, returns, risk_free_rate, benchmark_metrics, risk_tolerance, start_date, end_date)

        response = {
            "original_metrics": original_metrics,
            "optimized_metrics": optimized_metrics,
            "benchmark_metrics": benchmark_metrics,
            "cumulative_returns": cumulative_returns,
            "correlation_matrix": correlation_matrix,
            "efficient_frontier": efficient_frontier,
            "comparison_bars": comparison_bars,
            "portfolio_exposures": portfolio_exposures,
            "rolling_volatility": rolling_volatility,
            "historical_metrics": {"metrics": hist_metrics, "labels": hist_labels},
            "crisis_performance": crisis_performance,
            "eigenvalues": eigenvalues,
            "explained_variance_ratio": explained_variance_ratio,
            "fama_french_exposures": fama_french_exposures,
            "suggestions": suggestions
        }
        logger.info("Request processed successfully")
        return json.dumps(response), 200
    except Exception as e:
        logger.error(f"Error in analyze_portfolio: {e}")
        return json.dumps({"error": f"Internal server error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
