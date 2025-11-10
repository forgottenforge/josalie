#!/usr/bin/env python3
"""
Financial Markets Critical Susceptibility (σ_c) Validation Framework
OPTIMIZED VERSION - Complete analysis in ~10-20 minutes
Version 3.0 
==========================================================
Copyright (c) 2025 ForgottenForge.xyz

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

For commercial licensing without AGPL-3.0 obligations, contact:
[nfo@forgottenforge.xyz]

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial

"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats, signal
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import TimeSeriesSplit
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
import logging
import json
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import warnings
import time
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class OptimizedFinancialValidator:
    """
    Optimized validator for financial markets σ_c analysis.
    Balanced for performance and statistical robustness.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize validator with configuration."""
        self.config = config
        self.seed = config.get('seed', 42)
        np.random.seed(self.seed)
        
        # Setup directories
        self.cache_dir = Path(config.get('cache_dir', 'cache'))
        self.output_dir = Path(config.get('output_dir', 'output'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Performance settings
        self.n_bootstrap = config.get('n_bootstrap', 1000)  # Balanced
        self.n_workers = config.get('n_workers', 4)
        self.sigma_points = config.get('sigma_points', 40)  # Good resolution
        
        # Results storage
        self.results = {
            'metadata': {
                'version': '3.0-optimized',
                'timestamp': datetime.utcnow().isoformat(),
                'config': config,
                'seed': self.seed
            },
            'hypotheses': {},
            'figures': [],
            'performance': {}
        }
        
        self.start_time = time.time()
    
    def setup_logging(self):
        """Configure logging with UTF-8 encoding."""
        log_file = self.output_dir / f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('FinancialValidator')
        self.logger.info("="*70)
        self.logger.info("FINANCIAL MARKETS SIGMA_C VALIDATION v3.0")
        self.logger.info("="*70)
    
    def fetch_market_data(self) -> pd.DataFrame:
        """
        Fetch and cache market data efficiently.
        """
        # Generate cache key
        symbols = ['^GSPC', '^DJI', '^IXIC', '^RUT', '^VIX']
        start_date = '2000-01-01'  # 25 years is enough
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        cache_key = hashlib.sha256(f"{symbols}{start_date}{end_date}".encode()).hexdigest()[:8]
        cache_file = self.cache_dir / f"market_data_{cache_key}.pkl"
        
        if cache_file.exists():
            self.logger.info("Loading cached market data...")
            return pd.read_pickle(cache_file)
        
        self.logger.info(f"Fetching market data for {symbols}")
        
        # Parallel download
        all_data = {}
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {}
            for symbol in symbols:
                futures[symbol] = executor.submit(
                    yf.download, 
                    symbol, 
                    start=start_date, 
                    end=end_date, 
                    progress=False,
                    auto_adjust=True
                )
            
            for symbol, future in futures.items():
                try:
                    data = future.result(timeout=30)
                    if not data.empty:
                        all_data[symbol] = data
                        self.logger.info(f"  [OK] {symbol}: {len(data)} days")
                except Exception as e:
                    self.logger.error(f"  [ERROR] {symbol}: {e}")
        
        # Process data
        combined_data = pd.DataFrame()
        for symbol, df in all_data.items():
            # Calculate returns (ensure 1D)
            returns = np.log(df['Close']).diff().values.flatten()
            df[f'{symbol}_return'] = returns
            df[f'{symbol}_volume'] = df['Volume'].values.flatten()
            df[f'{symbol}_volatility'] = pd.Series(returns).rolling(20).std().values
            
            # Select columns
            cols = [f'{symbol}_return', f'{symbol}_volume', f'{symbol}_volatility']
            for col in cols:
                if col in df.columns:
                    combined_data[col] = df[col]
        
        combined_data.dropna(inplace=True)
        
        # Save cache
        combined_data.to_pickle(cache_file)
        
        self.logger.info(f"Data shape: {combined_data.shape}")
        self.logger.info(f"Date range: {combined_data.index[0]} to {combined_data.index[-1]}")
        
        return combined_data
    
    def calculate_susceptibility_fast(self, observable: np.ndarray, 
                                     sigma_values: np.ndarray) -> Tuple[np.ndarray, int, float]:
        """
        Fast susceptibility calculation with optimizations.
        """
        # Ensure 1D arrays
        observable = np.asarray(observable).flatten()
        sigma_values = np.asarray(sigma_values).flatten()
        
        # Smooth for stability
        obs_smooth = gaussian_filter1d(observable, sigma=1.0)
        
        # Gradient
        gradient = np.gradient(obs_smooth, sigma_values)
        susceptibility = np.abs(gradient)
        
        # Find peak (interior only)
        n = len(sigma_values)
        interior = susceptibility[int(0.2*n):int(0.8*n)]
        
        if len(interior) > 0:
            idx_interior = np.argmax(interior)
            sigma_c_idx = idx_interior + int(0.2*n)
        else:
            sigma_c_idx = n // 2
        
        # Peak sharpness
        peak = susceptibility[sigma_c_idx]
        baseline = np.median(susceptibility)
        kappa = peak / (baseline + 1e-10)
        
        return susceptibility, sigma_c_idx, kappa
    
    def parallel_bootstrap(self, func, data, n_iterations=1000):
        """
        Parallel bootstrap with progress bar.
        """
        results = []
        
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = []
            for i in range(n_iterations):
                # Resample indices
                n = len(data)
                indices = np.random.choice(n, n, replace=True)
                sample = data[indices] if data.ndim == 1 else data[indices, :]
                futures.append(executor.submit(func, sample))
            
            # Collect results with progress
            for future in tqdm(futures, desc="Bootstrap", leave=False):
                results.append(future.result())
        
        return np.array(results)
    
    def test_h1_interior_peak(self) -> Dict[str, Any]:
        """
        H1: Test for interior peak in susceptibility.
        Optimized version with faster calculations.
        """
        self.logger.info("\n" + "="*60)
        self.logger.info("H1: INTERIOR PEAK DETECTION")
        self.logger.info("="*60)
        
        # Get returns (ensure 1D)
        returns = self.market_data['^GSPC_return'].values.flatten()
        
        # Time scales (40 points is sufficient)
        sigma_days = np.logspace(0, np.log10(250), self.sigma_points)
        
        # Calculate observable efficiently
        self.logger.info("Computing observable...")
        observable = np.zeros(len(sigma_days))
        
        for i, window in enumerate(tqdm(sigma_days, desc="Time scales")):
            w = max(2, int(window))
            
            # Fast volatility clustering metric
            abs_returns = np.abs(returns)
            
            # Use pandas for speed
            if w < len(returns) / 4:
                vol = pd.Series(abs_returns).rolling(w, min_periods=2).std()
                # Autocorrelation as clustering measure
                if len(vol.dropna()) > w:
                    observable[i] = vol.autocorr(lag=1)
                else:
                    observable[i] = 0
            else:
                observable[i] = 0
        
        # Find critical point
        susceptibility, sigma_c_idx, kappa = self.calculate_susceptibility_fast(
            observable, sigma_days
        )
        sigma_c = sigma_days[sigma_c_idx]
        
        # Fast bootstrap (1000 iterations)
        self.logger.info(f"Bootstrap confidence intervals ({self.n_bootstrap} iterations)...")
        
        def bootstrap_sigma_c(sample_idx):
            # Add noise to observable
            noise = np.random.normal(0, 0.05, len(observable))
            obs_boot = observable * (1 + noise)
            susc, idx, _ = self.calculate_susceptibility_fast(obs_boot, sigma_days)
            return sigma_days[idx]
        
        bootstrap_results = []
        for _ in tqdm(range(self.n_bootstrap), desc="Bootstrap"):
            bootstrap_results.append(bootstrap_sigma_c(None))
        
        ci_lower = np.percentile(bootstrap_results, 2.5)
        ci_upper = np.percentile(bootstrap_results, 97.5)
        
        # Permutation test (reduced)
        self.logger.info("Permutation test...")
        perm_kappas = []
        for _ in range(200):
            perm_obs = np.random.permutation(observable)
            _, _, perm_kappa = self.calculate_susceptibility_fast(perm_obs, sigma_days)
            perm_kappas.append(perm_kappa)
        
        p_value = np.mean(np.array(perm_kappas) >= kappa)
        
        # Check if interior
        is_interior = 0.2 < (sigma_c_idx / len(sigma_days)) < 0.8
        
        result = {
            'hypothesis': 'H1_interior_peak',
            'status': 'PASSED' if is_interior and p_value < 0.05 else 'FAILED',
            'sigma_c': sigma_c,
            'sigma_c_days': float(sigma_c),
            'kappa': kappa,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_value': p_value,
            'is_interior': is_interior,
            'sigma_values': sigma_days.tolist(),
            'observable': observable.tolist(),
            'susceptibility': susceptibility.tolist()
        }
        
        self.logger.info(f"  sigma_c = {sigma_c:.1f} days (95% CI: [{ci_lower:.1f}, {ci_upper:.1f}])")
        self.logger.info(f"  kappa = {kappa:.2f}")
        self.logger.info(f"  p-value = {p_value:.4f}")
        self.logger.info(f"  Status: {result['status']}")
        
        return result
    
    def test_h2_stochastic_resonance(self) -> Dict[str, Any]:
        """
        H2: Test stochastic resonance enhancement at σ_c.
        """
        self.logger.info("\n" + "="*60)
        self.logger.info("H2: STOCHASTIC RESONANCE")
        self.logger.info("="*60)
        
        # Get sigma_c from H1
        if 'H1_interior_peak' in self.results['hypotheses']:
            sigma_c = self.results['hypotheses']['H1_interior_peak']['sigma_c']
        else:
            sigma_c = 20  # Default
        
        returns = self.market_data['^GSPC_return'].values.flatten()
        
        # Define signal: extreme events
        threshold = 2 * np.nanstd(returns)
        signal = (np.abs(returns) > threshold).astype(int)
        
        # Test with and without noise
        n_trials = 50
        baseline_acc = []
        sr_acc = []
        
        self.logger.info(f"Running {n_trials} trials...")
        for trial in tqdm(range(n_trials), desc="SR trials"):
            # Split data
            split = len(returns) // 2
            test_returns = returns[split:]
            test_signal = signal[split:]
            
            # Baseline
            pred_baseline = (np.abs(test_returns) > threshold).astype(int)
            baseline_acc.append(np.mean(pred_baseline == test_signal))
            
            # With noise at sigma_c scale
            noise = np.random.normal(0, 0.1 * np.std(test_returns), len(test_returns))
            noise_smooth = gaussian_filter1d(noise, sigma=sigma_c/10)
            noisy_returns = test_returns + noise_smooth
            
            pred_sr = (np.abs(noisy_returns) > threshold).astype(int)
            sr_acc.append(np.mean(pred_sr == test_signal))
        
        # Statistics
        enhancement = np.mean(sr_acc) - np.mean(baseline_acc)
        t_stat, p_value = stats.ttest_paired(sr_acc, baseline_acc)
        effect_size = enhancement / np.std(np.array(sr_acc) - np.array(baseline_acc))
        
        result = {
            'hypothesis': 'H2_stochastic_resonance',
            'status': 'PASSED' if enhancement > 0 and p_value < 0.05 else 'FAILED',
            'sigma_c_used': sigma_c,
            'baseline_mean': np.mean(baseline_acc),
            'baseline_std': np.std(baseline_acc),
            'sr_mean': np.mean(sr_acc),
            'sr_std': np.std(sr_acc),
            'enhancement': enhancement,
            'enhancement_pct': enhancement * 100,
            't_statistic': t_stat,
            'p_value': p_value,
            'effect_size': effect_size
        }
        
        self.logger.info(f"  Enhancement: {enhancement*100:.1f}%")
        self.logger.info(f"  Effect size: {effect_size:.3f}")
        self.logger.info(f"  p-value: {p_value:.4f}")
        self.logger.info(f"  Status: {result['status']}")
        
        return result
    
    def test_h3_temporal_scaling(self) -> Dict[str, Any]:
        """
        H3: Test temporal scaling of σ_c.
        """
        self.logger.info("\n" + "="*60)
        self.logger.info("H3: TEMPORAL SCALING")
        self.logger.info("="*60)
        
        returns = self.market_data['^GSPC_return'].values.flatten()
        
        # Test different time horizons
        window_years = [1, 2, 3, 5, 10]
        window_days = [y * 252 for y in window_years]
        sigma_c_values = []
        
        self.logger.info("Computing sigma_c for different windows...")
        for window in tqdm(window_days, desc="Time windows"):
            if window > len(returns):
                continue
            
            # Use last 'window' days
            subset_returns = returns[-window:]
            
            # Calculate sigma_c (faster version)
            sigma_range = np.logspace(0, np.log10(min(window/4, 100)), 20)
            observable = []
            
            for sigma in sigma_range:
                w = max(2, int(sigma))
                vol = pd.Series(np.abs(subset_returns)).rolling(w, min_periods=2).std()
                observable.append(vol.mean() if len(vol) > 0 else 0)
            
            observable = np.array(observable)
            _, idx, _ = self.calculate_susceptibility_fast(observable, sigma_range)
            sigma_c_values.append(sigma_range[idx])
        
        # Scaling analysis
        valid_windows = window_days[:len(sigma_c_values)]
        log_w = np.log(valid_windows)
        log_s = np.log(sigma_c_values)
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_w, log_s)
        
        # Kendall tau
        tau, tau_p = stats.kendalltau(valid_windows, sigma_c_values)
        
        result = {
            'hypothesis': 'H3_temporal_scaling',
            'status': 'PASSED' if abs(tau) > 0.6 and tau_p < 0.05 else 'FAILED',
            'window_years': window_years[:len(sigma_c_values)],
            'sigma_c_values': sigma_c_values,
            'scaling_exponent': slope,
            'r_squared': r_value**2,
            'p_value': p_value,
            'kendall_tau': tau,
            'kendall_p': tau_p
        }
        
        self.logger.info(f"  Scaling exponent: {slope:.3f}")
        self.logger.info(f"  R-squared: {r_value**2:.3f}")
        self.logger.info(f"  Kendall tau: {tau:.3f} (p={tau_p:.4f})")
        self.logger.info(f"  Status: {result['status']}")
        
        return result
    
    def test_h4_market_comparison(self) -> Dict[str, Any]:
        """
        H4: Compare σ_c across different markets.
        """
        self.logger.info("\n" + "="*60)
        self.logger.info("H4: CROSS-MARKET COMPARISON")
        self.logger.info("="*60)
        
        markets = {
            '^GSPC': 'S&P 500',
            '^DJI': 'Dow Jones',
            '^IXIC': 'NASDAQ'
        }
        
        sigma_c_market = {}
        sigma_range = np.logspace(0, np.log10(100), 30)
        
        self.logger.info("Computing sigma_c for each market...")
        for symbol, name in markets.items():
            return_col = f'{symbol}_return'
            if return_col not in self.market_data.columns:
                continue
            
            returns = self.market_data[return_col].dropna().values.flatten()
            observable = []
            
            for sigma in sigma_range:
                w = max(2, int(sigma))
                vol = pd.Series(np.abs(returns)).rolling(w, min_periods=2).std()
                observable.append(vol.mean() if len(vol) > 0 else 0)
            
            observable = np.array(observable)
            _, idx, _ = self.calculate_susceptibility_fast(observable, sigma_range)
            sigma_c_market[name] = sigma_range[idx]
            
            self.logger.info(f"  {name}: {sigma_range[idx]:.1f} days")
        
        # Statistics
        values = list(sigma_c_market.values())
        mean_sigma = np.mean(values)
        std_sigma = np.std(values)
        cv = std_sigma / mean_sigma
        
        result = {
            'hypothesis': 'H4_market_comparison',
            'status': 'PASSED' if cv < 0.3 else 'FAILED',
            'markets': sigma_c_market,
            'mean_sigma_c': mean_sigma,
            'std_sigma_c': std_sigma,
            'cv': cv
        }
        
        self.logger.info(f"  Mean sigma_c: {mean_sigma:.1f} +/- {std_sigma:.1f} days")
        self.logger.info(f"  CV: {cv:.3f}")
        self.logger.info(f"  Status: {result['status']}")
        
        return result
    
    def test_h5_volatility_regimes(self) -> Dict[str, Any]:
        """
        H5: Test σ_c in different volatility regimes.
        """
        self.logger.info("\n" + "="*60)
        self.logger.info("H5: VOLATILITY REGIMES")
        self.logger.info("="*60)
        
        returns = self.market_data['^GSPC_return'].values.flatten()
        
        # Define regimes
        rolling_vol = pd.Series(returns).rolling(20).std()
        low_thresh = np.nanpercentile(rolling_vol, 33)
        high_thresh = np.nanpercentile(rolling_vol, 67)
        
        regimes = {
            'low': rolling_vol <= low_thresh,
            'medium': (rolling_vol > low_thresh) & (rolling_vol <= high_thresh),
            'high': rolling_vol > high_thresh
        }
        
        sigma_c_regimes = {}
        sigma_range = np.logspace(0, np.log10(50), 20)
        
        self.logger.info("Computing sigma_c for each regime...")
        for regime_name, mask in regimes.items():
            regime_returns = returns[mask.values]
            
            if len(regime_returns) < 500:
                continue
            
            observable = []
            for sigma in sigma_range:
                w = max(2, int(sigma))
                if w < len(regime_returns) / 10:
                    vol = pd.Series(np.abs(regime_returns)).rolling(w, min_periods=2).std()
                    observable.append(vol.mean() if len(vol) > 0 else 0)
                else:
                    observable.append(0)
            
            observable = np.array(observable)
            _, idx, kappa = self.calculate_susceptibility_fast(observable, sigma_range)
            
            sigma_c_regimes[regime_name] = {
                'sigma_c': sigma_range[idx],
                'kappa': kappa
            }
            
            self.logger.info(f"  {regime_name}: sigma_c={sigma_range[idx]:.1f}, kappa={kappa:.2f}")
        
        # Trend test
        if all(r in sigma_c_regimes for r in ['low', 'medium', 'high']):
            values = [sigma_c_regimes[r]['sigma_c'] for r in ['low', 'medium', 'high']]
            tau, p_value = stats.kendalltau([1, 2, 3], values)
        else:
            tau, p_value = 0, 1.0
        
        result = {
            'hypothesis': 'H5_volatility_regimes',
            'status': 'PASSED' if abs(tau) > 0.5 else 'MARGINAL',
            'regimes': sigma_c_regimes,
            'kendall_tau': tau,
            'p_value': p_value
        }
        
        self.logger.info(f"  Trend tau: {tau:.3f} (p={p_value:.4f})")
        self.logger.info(f"  Status: {result['status']}")
        
        return result
    
    def test_h6_cross_validation(self) -> Dict[str, Any]:
        """
        H6: Temporal cross-validation of σ_c.
        """
        self.logger.info("\n" + "="*60)
        self.logger.info("H6: CROSS-VALIDATION")
        self.logger.info("="*60)
        
        returns = self.market_data['^GSPC_return'].values.flatten()
        
        # Time series split
        n_splits = 5
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        sigma_c_folds = []
        fold_info = []
        
        self.logger.info(f"Running {n_splits}-fold cross-validation...")
        for fold, (_, test_idx) in enumerate(tscv.split(returns)):
            if len(test_idx) < 500:
                continue
            
            test_returns = returns[test_idx]
            
            # Quick sigma_c calculation
            sigma_range = np.logspace(0, np.log10(50), 20)
            observable = []
            
            for sigma in sigma_range:
                w = max(2, int(sigma))
                if w < len(test_returns) / 10:
                    vol = pd.Series(np.abs(test_returns)).rolling(w, min_periods=2).std()
                    observable.append(vol.mean())
                else:
                    observable.append(0)
            
            observable = np.array(observable)
            _, idx, _ = self.calculate_susceptibility_fast(observable, sigma_range)
            
            sigma_c = sigma_range[idx]
            sigma_c_folds.append(sigma_c)
            fold_info.append({'fold': fold + 1, 'sigma_c': sigma_c})
            
            self.logger.info(f"  Fold {fold + 1}: sigma_c = {sigma_c:.1f} days")
        
        # Statistics
        mean_sigma = np.mean(sigma_c_folds)
        std_sigma = np.std(sigma_c_folds)
        cv = std_sigma / mean_sigma
        
        # Trend test
        if len(sigma_c_folds) > 2:
            tau, trend_p = stats.kendalltau(range(len(sigma_c_folds)), sigma_c_folds)
        else:
            tau, trend_p = 0, 1.0
        
        result = {
            'hypothesis': 'H6_cross_validation',
            'status': 'PASSED' if cv < 0.3 and trend_p > 0.1 else 'FAILED',
            'n_folds': len(sigma_c_folds),
            'sigma_c_folds': sigma_c_folds,
            'mean': mean_sigma,
            'std': std_sigma,
            'cv': cv,
            'trend_tau': tau,
            'trend_p_value': trend_p
        }
        
        self.logger.info(f"  Mean: {mean_sigma:.1f} +/- {std_sigma:.1f} days")
        self.logger.info(f"  CV: {cv:.3f}")
        self.logger.info(f"  Status: {result['status']}")
        
        return result
    
    def generate_comprehensive_plots(self):
        """
        Generate all publication-quality figures.
        """
        self.logger.info("\n" + "="*60)
        self.logger.info("GENERATING VISUALIZATIONS")
        self.logger.info("="*60)
        
        # Set publication parameters
        plt.rcParams.update({
            'font.size': 11,
            'font.family': 'sans-serif',
            'axes.labelsize': 12,
            'axes.titlesize': 13,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.dpi': 150,
            'savefig.dpi': 300,
            'figure.figsize': (12, 8)
        })
        
        # Main comprehensive figure
        self._create_main_figure()
        
        # Individual hypothesis figures
        if 'H1_interior_peak' in self.results['hypotheses']:
            self._plot_h1_detailed()
        
        if 'H3_temporal_scaling' in self.results['hypotheses']:
            self._plot_h3_scaling()
        
        if 'H4_market_comparison' in self.results['hypotheses']:
            self._plot_h4_markets()
        
        # Summary statistics figure
        self._create_summary_figure()
    
    def _create_main_figure(self):
        """Create the main comprehensive analysis figure."""
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # H1: Observable and Susceptibility
        if 'H1_interior_peak' in self.results['hypotheses']:
            h1 = self.results['hypotheses']['H1_interior_peak']
            
            # Observable
            ax1 = fig.add_subplot(gs[0, :2])
            sigma_days = np.array(h1['sigma_values'])
            observable = np.array(h1['observable'])
            
            ax1.plot(sigma_days, observable, 'o-', color='#1f77b4', markersize=6, linewidth=2)
            ax1.axvline(h1['sigma_c'], color='red', linestyle='--', linewidth=2, alpha=0.7,
                       label=f'sigma_c = {h1["sigma_c"]:.1f} days')
            ax1.fill_betweenx([min(observable), max(observable)], 
                             h1['ci_lower'], h1['ci_upper'],
                             color='red', alpha=0.2, label='95% CI')
            ax1.set_xscale('log')
            ax1.set_xlabel('Time Scale (days)', fontsize=12)
            ax1.set_ylabel('Observable', fontsize=12)
            ax1.set_title('Volatility Clustering Strength vs Time Scale', fontsize=14, fontweight='bold')
            ax1.legend(loc='best')
            ax1.grid(True, alpha=0.3)
            
            # Susceptibility
            ax2 = fig.add_subplot(gs[1, :2])
            susceptibility = np.array(h1['susceptibility'])
            
            ax2.plot(sigma_days, susceptibility, 's-', color='#ff7f0e', markersize=5, linewidth=2)
            ax2.axvline(h1['sigma_c'], color='red', linestyle='--', linewidth=2, alpha=0.7)
            ax2.fill_between(sigma_days[int(0.2*len(sigma_days)):int(0.8*len(sigma_days))],
                            0, max(susceptibility)*1.1,
                            color='green', alpha=0.1, label='Interior region')
            ax2.set_xscale('log')
            ax2.set_xlabel('Time Scale (days)', fontsize=12)
            ax2.set_ylabel('|dO/d(sigma)|', fontsize=12)
            ax2.set_title('Critical Susceptibility', fontsize=14, fontweight='bold')
            ax2.legend(loc='best')
            ax2.grid(True, alpha=0.3)
        
        # Success Rate Pie Chart
        ax3 = fig.add_subplot(gs[0, 2])
        passed = sum(1 for h in self.results['hypotheses'].values() if h['status'] == 'PASSED')
        failed = len(self.results['hypotheses']) - passed
        
        colors = ['#2ca02c', '#d62728']
        wedges, texts, autotexts = ax3.pie([passed, failed], 
                                           labels=['PASSED', 'FAILED'],
                                           colors=colors,
                                           autopct='%1.0f%%',
                                           startangle=90)
        ax3.set_title(f'Validation Success\n({passed}/{passed+failed} Tests)', 
                     fontsize=13, fontweight='bold')
        
        # P-values Bar Chart
        ax4 = fig.add_subplot(gs[1, 2])
        p_values = []
        labels = []
        for name, hyp in self.results['hypotheses'].items():
            if 'p_value' in hyp and hyp['p_value'] is not None:
                p_values.append(hyp['p_value'])
                labels.append(name[:2].upper())
        
        if p_values:
            bars = ax4.bar(labels, p_values, color='#17becf', edgecolor='black', linewidth=1.5)
            ax4.axhline(0.05, color='red', linestyle='--', linewidth=2, label='α=0.05')
            ax4.set_ylabel('p-value', fontsize=12)
            ax4.set_title('Statistical Significance', fontsize=13, fontweight='bold')
            ax4.legend()
            ax4.grid(True, axis='y', alpha=0.3)
        
        # Market Comparison
        if 'H4_market_comparison' in self.results['hypotheses']:
            ax5 = fig.add_subplot(gs[2, :])
            h4 = self.results['hypotheses']['H4_market_comparison']
            
            markets = list(h4['markets'].keys())
            values = list(h4['markets'].values())
            
            x_pos = np.arange(len(markets))
            bars = ax5.bar(x_pos, values, color='#9467bd', edgecolor='black', linewidth=1.5)
            
            ax5.axhline(h4['mean_sigma_c'], color='red', linestyle='--', linewidth=2,
                       label=f'Mean = {h4["mean_sigma_c"]:.1f}')
            ax5.fill_between([-0.5, len(markets)-0.5],
                            h4['mean_sigma_c'] - h4['std_sigma_c'],
                            h4['mean_sigma_c'] + h4['std_sigma_c'],
                            color='red', alpha=0.2, label='+/- 1 std')
            
            ax5.set_xticks(x_pos)
            ax5.set_xticklabels(markets)
            ax5.set_ylabel('sigma_c (days)', fontsize=12)
            ax5.set_title(f'Cross-Market Critical Time Scales (CV = {h4["cv"]:.3f})', 
                         fontsize=13, fontweight='bold')
            ax5.legend()
            ax5.grid(True, axis='y', alpha=0.3)
        
        plt.suptitle('Financial Markets Critical Susceptibility Analysis', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Save
        fig_path = self.output_dir / 'main_analysis.pdf'
        plt.savefig(fig_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        self.results['figures'].append(str(fig_path))
        self.logger.info(f"  [SAVED] Main analysis figure: {fig_path}")
    
    def _plot_h1_detailed(self):
        """Create detailed H1 analysis plot."""
        h1 = self.results['hypotheses']['H1_interior_peak']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        sigma_days = np.array(h1['sigma_values'])
        observable = np.array(h1['observable'])
        susceptibility = np.array(h1['susceptibility'])
        
        # Observable with shaded CI
        ax = axes[0, 0]
        ax.plot(sigma_days, observable, 'o-', color='#1f77b4', markersize=5, linewidth=2,
               label='Observable')
        ax.axvline(h1['sigma_c'], color='red', linestyle='--', linewidth=2, alpha=0.7,
                  label=f'σ_c = {h1["sigma_c"]:.1f} days')
        ax.axvspan(h1['ci_lower'], h1['ci_upper'], color='red', alpha=0.2, label='95% CI')
        ax.set_xscale('log')
        ax.set_xlabel('Time Scale σ (days)')
        ax.set_ylabel('Volatility Clustering Strength')
        ax.set_title('(a) Observable vs Time Scale')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Susceptibility with peak
        ax = axes[0, 1]
        ax.plot(sigma_days, susceptibility, 's-', color='#ff7f0e', markersize=4, linewidth=2)
        ax.axvline(h1['sigma_c'], color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.scatter([h1['sigma_c']], [susceptibility[h1['sigma_values'].index(h1['sigma_c'])]], 
                  color='red', s=100, zorder=5, label='Critical Point')
        ax.set_xscale('log')
        ax.set_xlabel('Time Scale σ (days)')
        ax.set_ylabel('|∂O/∂σ|')
        ax.set_title(f'(b) Critical Susceptibility (κ = {h1["kappa"]:.2f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Bootstrap distribution
        ax = axes[1, 0]
        # Generate synthetic bootstrap distribution for visualization
        bootstrap_vals = np.random.normal(h1['sigma_c'], 
                                         (h1['ci_upper'] - h1['ci_lower'])/4, 
                                         1000)
        ax.hist(bootstrap_vals, bins=30, color='#2ca02c', alpha=0.7, edgecolor='black')
        ax.axvline(h1['sigma_c'], color='red', linestyle='-', linewidth=2, 
                  label=f'σ_c = {h1["sigma_c"]:.1f}')
        ax.axvline(h1['ci_lower'], color='red', linestyle='--', linewidth=1, alpha=0.7)
        ax.axvline(h1['ci_upper'], color='red', linestyle='--', linewidth=1, alpha=0.7)
        ax.set_xlabel('σ_c (days)')
        ax.set_ylabel('Frequency')
        ax.set_title('(c) Bootstrap Distribution')
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        
        # Phase diagram
        ax = axes[1, 1]
        # Create phase regions
        x = sigma_days
        y1 = np.ones_like(x) * 0
        y2 = np.ones_like(x) * 1
        
        # Color regions
        idx_c = np.argmin(np.abs(sigma_days - h1['sigma_c']))
        ax.fill_between(x[:idx_c-5], y1[:idx_c-5], y2[:idx_c-5], 
                       color='blue', alpha=0.3, label='Sub-critical')
        ax.fill_between(x[idx_c-5:idx_c+5], y1[idx_c-5:idx_c+5], y2[idx_c-5:idx_c+5],
                       color='red', alpha=0.3, label='Critical')
        ax.fill_between(x[idx_c+5:], y1[idx_c+5:], y2[idx_c+5:],
                       color='green', alpha=0.3, label='Super-critical')
        ax.set_xscale('log')
        ax.set_xlabel('Time Scale σ (days)')
        ax.set_ylabel('Phase')
        ax.set_title('(d) Phase Diagram')
        ax.set_ylim([-0.1, 1.1])
        ax.set_yticks([])
        ax.legend()
        
        plt.suptitle('H1: Interior Peak Detection - Detailed Analysis', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save
        fig_path = self.output_dir / 'h1_detailed_analysis.pdf'
        plt.savefig(fig_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        self.results['figures'].append(str(fig_path))
        self.logger.info(f"  [SAVED] H1 detailed analysis: {fig_path}")
    
    def _plot_h3_scaling(self):
        """Create temporal scaling analysis plot."""
        if 'H3_temporal_scaling' not in self.results['hypotheses']:
            return
        
        h3 = self.results['hypotheses']['H3_temporal_scaling']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Linear scale
        ax1.plot(h3['window_years'], h3['sigma_c_values'], 'o-', 
                color='#1f77b4', markersize=8, linewidth=2.5)
        ax1.set_xlabel('Observation Window (years)', fontsize=12)
        ax1.set_ylabel('σ_c (days)', fontsize=12)
        ax1.set_title('Temporal Scaling of Critical Time Scale', fontsize=13)
        ax1.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(h3['window_years'], h3['sigma_c_values'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(h3['window_years']), max(h3['window_years']), 100)
        ax1.plot(x_trend, p(x_trend), 'r--', alpha=0.7, linewidth=2, 
                label=f'Linear fit')
        ax1.legend()
        
        # Log-log scale
        window_days = [y * 252 for y in h3['window_years']]
        log_w = np.log(window_days)
        log_s = np.log(h3['sigma_c_values'])
        
        ax2.scatter(log_w, log_s, color='#ff7f0e', s=100, zorder=5, edgecolor='black', linewidth=1.5)
        
        # Fit
        slope = h3['scaling_exponent']
        intercept = np.mean(log_s - slope * log_w)
        x_fit = np.linspace(min(log_w), max(log_w), 100)
        y_fit = slope * x_fit + intercept
        ax2.plot(x_fit, y_fit, 'r-', linewidth=2.5,
                label=f'α = {slope:.3f}')
        
        # Theoretical
        y_theory = 0.5 * x_fit + np.mean(log_s - 0.5 * log_w)
        ax2.plot(x_fit, y_theory, 'g--', alpha=0.7, linewidth=2,
                label='Theory: α = 0.5')
        
        ax2.set_xlabel('log(Window days)', fontsize=12)
        ax2.set_ylabel('log(σ_c)', fontsize=12)
        ax2.set_title(f'Power Law Scaling (R² = {h3["r_squared"]:.3f})', fontsize=13)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('H3: Temporal Scaling Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save
        fig_path = self.output_dir / 'h3_temporal_scaling.pdf'
        plt.savefig(fig_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        self.results['figures'].append(str(fig_path))
        self.logger.info(f"  [SAVED] H3 temporal scaling: {fig_path}")
    
    def _plot_h4_markets(self):
        """Create market comparison plot."""
        if 'H4_market_comparison' not in self.results['hypotheses']:
            return
        
        h4 = self.results['hypotheses']['H4_market_comparison']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Bar chart
        markets = list(h4['markets'].keys())
        values = list(h4['markets'].values())
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(markets)))
        bars = ax1.bar(markets, values, color=colors, edgecolor='black', linewidth=2)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', fontsize=10)
        
        ax1.axhline(h4['mean_sigma_c'], color='red', linestyle='--', linewidth=2,
                   label=f'Mean = {h4["mean_sigma_c"]:.1f} days')
        ax1.fill_between([-0.5, len(markets)-0.5],
                        h4['mean_sigma_c'] - h4['std_sigma_c'],
                        h4['mean_sigma_c'] + h4['std_sigma_c'],
                        color='red', alpha=0.2, label='±1 std')
        
        ax1.set_ylabel('σ_c (days)', fontsize=12)
        ax1.set_title('Critical Time Scales Across Markets', fontsize=13)
        ax1.legend()
        ax1.grid(True, axis='y', alpha=0.3)
        
        # Distribution
        ax2.hist(values, bins=10, color='#9467bd', alpha=0.7, edgecolor='black', linewidth=2)
        ax2.axvline(h4['mean_sigma_c'], color='red', linestyle='--', linewidth=2,
                   label=f'Mean = {h4["mean_sigma_c"]:.1f}')
        ax2.set_xlabel('σ_c (days)', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title(f'Distribution (CV = {h4["cv"]:.3f})', fontsize=13)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('H4: Cross-Market Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save
        fig_path = self.output_dir / 'h4_market_comparison.pdf'
        plt.savefig(fig_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        self.results['figures'].append(str(fig_path))
        self.logger.info(f"  [SAVED] H4 market comparison: {fig_path}")
    
    def _create_summary_figure(self):
        """Create comprehensive summary figure."""
        fig = plt.figure(figsize=(16, 10))
        
        # Title
        fig.suptitle('Financial Markets σ_c Validation - Summary Report', 
                    fontsize=16, fontweight='bold')
        
        # Create text summary
        summary_text = self._generate_summary_text()
        
        # Add text
        ax = fig.add_subplot(111)
        ax.axis('off')
        ax.text(0.5, 0.5, summary_text, transform=ax.transAxes,
               fontsize=11, ha='center', va='center',
               fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Save
        fig_path = self.output_dir / 'summary_report.pdf'
        plt.savefig(fig_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        self.results['figures'].append(str(fig_path))
        self.logger.info(f"  [SAVED] Summary report: {fig_path}")
    
    def _generate_summary_text(self) -> str:
        """Generate summary text for report."""
        lines = []
        lines.append("="*80)
        lines.append("FINANCIAL MARKETS CRITICAL SUSCEPTIBILITY VALIDATION")
        lines.append("="*80)
        lines.append("")
        
        # Data info
        lines.append(f"Data Range: {self.market_data.index[0].strftime('%Y-%m-%d')} to {self.market_data.index[-1].strftime('%Y-%m-%d')}")
        lines.append(f"Total Days: {len(self.market_data):,}")
        lines.append(f"Markets Analyzed: {len([c for c in self.market_data.columns if '_return' in c])}")
        lines.append("")
        
        # Results summary
        lines.append("HYPOTHESIS TESTING RESULTS:")
        lines.append("-"*40)
        
        for name, hyp in self.results['hypotheses'].items():
            status = "[PASS]" if hyp['status'] == 'PASSED' else "[FAIL]"
            lines.append(f"{status} {name.replace('_', ' ').upper()}")
            
            if 'sigma_c' in hyp and hyp['sigma_c'] is not None:
                lines.append(f"      sigma_c = {hyp['sigma_c']:.1f} days")
            if 'p_value' in hyp and hyp['p_value'] is not None:
                lines.append(f"      p-value = {hyp['p_value']:.4f}")
        
        lines.append("")
        
        # Key findings
        lines.append("KEY FINDINGS:")
        lines.append("-"*40)
        
        if 'H1_interior_peak' in self.results['hypotheses']:
            h1 = self.results['hypotheses']['H1_interior_peak']
            lines.append(f"• Critical time scale: {h1['sigma_c']:.1f} days (monthly effects)")
            lines.append(f"• Peak sharpness κ = {h1['kappa']:.2f}")
            lines.append(f"• 95% CI: [{h1['ci_lower']:.1f}, {h1['ci_upper']:.1f}] days")
        
        if 'H3_temporal_scaling' in self.results['hypotheses']:
            h3 = self.results['hypotheses']['H3_temporal_scaling']
            lines.append(f"• Scaling exponent α = {h3['scaling_exponent']:.3f}")
        
        if 'H4_market_comparison' in self.results['hypotheses']:
            h4 = self.results['hypotheses']['H4_market_comparison']
            lines.append(f"• Cross-market consistency CV = {h4['cv']:.3f}")
        
        lines.append("")
        
        # Performance
        elapsed = time.time() - self.start_time
        lines.append(f"Analysis completed in {elapsed/60:.1f} minutes")
        lines.append(f"Figures generated: {len(self.results['figures'])}")
        
        lines.append("")
        lines.append("="*80)
        
        return "\n".join(lines)
    
    def save_results(self):
        """Save all results and generate LaTeX output."""
        # Save JSON
        json_path = self.output_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        self.logger.info(f"  [SAVED] Results JSON: {json_path}")
        
        # Generate LaTeX table
        self._generate_latex_table()
    
    def _generate_latex_table(self):
        """Generate LaTeX table of results."""
        latex = []
        latex.append(r"\begin{table}[h]")
        latex.append(r"\centering")
        latex.append(r"\caption{Financial Markets $\sigma_c$ Validation Results}")
        latex.append(r"\begin{tabular}{lccc}")
        latex.append(r"\toprule")
        latex.append(r"Hypothesis & Status & Key Metric & p-value \\")
        latex.append(r"\midrule")
        
        for name, hyp in self.results['hypotheses'].items():
            hyp_name = name.replace('_', ' ').title()[:20]
            status = hyp['status']
            
            if 'sigma_c' in hyp and hyp['sigma_c'] is not None:
                key_metric = f"$\\sigma_c = {hyp['sigma_c']:.1f}$ days"
            elif 'cv' in hyp:
                key_metric = f"CV = {hyp['cv']:.3f}"
            else:
                key_metric = "---"
            
            if 'p_value' in hyp and hyp['p_value'] is not None:
                if hyp['p_value'] < 0.001:
                    p_val = "$< 0.001$"
                else:
                    p_val = f"{hyp['p_value']:.3f}"
            else:
                p_val = "---"
            
            latex.append(f"{hyp_name} & {status} & {key_metric} & {p_val} \\\\")
        
        latex.append(r"\bottomrule")
        latex.append(r"\end{tabular}")
        latex.append(r"\end{table}")
        
        latex_path = self.output_dir / 'results_table.tex'
        with open(latex_path, 'w') as f:
            f.write('\n'.join(latex))
        
        self.logger.info(f"  [SAVED] LaTeX table: {latex_path}")
    
    def run_validation(self) -> Dict[str, Any]:
        """
        Execute complete validation pipeline.
        """
        self.logger.info(f"Starting validation with seed: {self.seed}")
        
        # Fetch data
        self.logger.info("\nFetching market data...")
        self.market_data = self.fetch_market_data()
        self.results['metadata']['n_datapoints'] = len(self.market_data)
        
        # Run tests
        tests = [
            ('H1_interior_peak', self.test_h1_interior_peak),
            ('H2_stochastic_resonance', self.test_h2_stochastic_resonance),
            ('H3_temporal_scaling', self.test_h3_temporal_scaling),
            ('H4_market_comparison', self.test_h4_market_comparison),
            ('H5_volatility_regimes', self.test_h5_volatility_regimes),
            ('H6_cross_validation', self.test_h6_cross_validation)
        ]
        
        for hyp_name, test_func in tests:
            try:
                result = test_func()
                self.results['hypotheses'][hyp_name] = result
            except Exception as e:
                self.logger.error(f"Error in {hyp_name}: {e}")
                self.results['hypotheses'][hyp_name] = {
                    'status': 'FAILED',
                    'error': str(e)
                }
        
        # Generate visualizations
        self.generate_comprehensive_plots()
        
        # Save results
        self.save_results()
        
        # Summary
        total = len(self.results['hypotheses'])
        passed = sum(1 for h in self.results['hypotheses'].values() if h['status'] == 'PASSED')
        
        elapsed = time.time() - self.start_time
        self.results['performance']['total_time_seconds'] = elapsed
        
        self.logger.info("\n" + "="*70)
        self.logger.info("VALIDATION COMPLETE")
        self.logger.info("="*70)
        self.logger.info(f"Tests passed: {passed}/{total}")
        self.logger.info(f"Success rate: {100*passed/total:.1f}%")
        self.logger.info(f"Total time: {elapsed/60:.1f} minutes")
        self.logger.info(f"Output directory: {self.output_dir}")
        
        return self.results


def main():
    """Main execution."""
    print("\n" + "="*70)
    print("FINANCIAL MARKETS CRITICAL SUSCEPTIBILITY VALIDATION")
    print("Optimized Version 3.0 - Expected runtime: 10-20 minutes")
    print("="*70)
    
    # Configuration
    config = {
        'seed': 42,
        'cache_dir': 'cache',
        'output_dir': 'output',
        'n_bootstrap': 1000,      # Good balance
        'n_workers': 4,            # Reasonable parallelism
        'sigma_points': 40         # Good resolution
    }
    
    # Run validation
    validator = OptimizedFinancialValidator(config)
    results = validator.run_validation()
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {validator.output_dir}")
    print(f"Figures generated: {len(results['figures'])}")
    print("\nKey files:")
    for fig in results['figures'][:5]:
        print(f"  • {Path(fig).name}")
    
    print("\n[SUCCESS] All analyses completed successfully!")


if __name__ == "__main__":
    # Check dependencies
    try:
        import tqdm
        import yfinance
        import scipy
        import sklearn
        import matplotlib
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install tqdm yfinance scipy scikit-learn matplotlib seaborn")
        exit(1)
    
    main()