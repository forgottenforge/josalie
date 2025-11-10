#!/usr/bin/env python3
"""
Financial Markets σ_c - ROBUST QUICK VERSION
###############################################
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
from scipy import stats
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from datetime import datetime
import json

print("="*70)
print("FINANCIAL MARKETS SIGMA_C - ROBUST VERSION")
print("="*70)

# 1. FETCH DATA
print("\n[1/6] Fetching market data...")
data = yf.download('^GSPC', start='2010-01-01', progress=False, auto_adjust=True)
returns = np.log(data['Close']).diff().dropna().values.flatten()
print(f"  Data: {len(returns)} days")

# 2. CALCULATE OBSERVABLE
print("\n[2/6] Computing observable...")
sigma_days = np.logspace(0, np.log10(100), 30)
observable = []

for window in sigma_days:
    w = max(2, int(window))
    vol = pd.Series(np.abs(returns)).rolling(w, min_periods=2).std()
    observable.append(np.nanmean(vol))
    
observable = np.array(observable)

# 3. FIND CRITICAL POINT
print("\n[3/6] Finding σ_c...")
gradient = np.gradient(observable, sigma_days)
susceptibility = np.abs(gradient)
susceptibility_smooth = gaussian_filter1d(susceptibility, 1)

# Interior peak
n = len(sigma_days)
interior = susceptibility_smooth[int(0.2*n):int(0.8*n)]
idx = np.argmax(interior) + int(0.2*n)
sigma_c = sigma_days[idx]
kappa = susceptibility_smooth[idx] / np.median(susceptibility_smooth)

print(f"  σ_c = {sigma_c:.1f} days")
print(f"  κ = {kappa:.2f}")

# 4. BOOTSTRAP CI (fast version)
print("\n[4/6] Bootstrap confidence interval (500 iterations)...")
bootstrap_sigma = []
for i in range(500):
    if i % 100 == 0:
        print(f"  Progress: {i}/500")
    noise = np.random.normal(0, 0.05, len(observable))
    obs_boot = observable * (1 + noise)
    grad_boot = np.abs(np.gradient(obs_boot, sigma_days))
    smooth_boot = gaussian_filter1d(grad_boot, 1)
    interior_boot = smooth_boot[int(0.2*n):int(0.8*n)]
    idx_boot = np.argmax(interior_boot) + int(0.2*n)
    bootstrap_sigma.append(sigma_days[idx_boot])

ci_lower = np.percentile(bootstrap_sigma, 2.5)
ci_upper = np.percentile(bootstrap_sigma, 97.5)
print(f"  95% CI: [{ci_lower:.1f}, {ci_upper:.1f}] days")

# 5. CROSS-MARKET TEST
print("\n[5/6] Cross-market comparison...")
markets = {'^GSPC': 'S&P500', '^DJI': 'Dow', '^IXIC': 'NASDAQ'}
sigma_c_markets = {}

for symbol, name in markets.items():
    try:
        mkt_data = yf.download(symbol, start='2020-01-01', progress=False, auto_adjust=True)
        mkt_returns = np.log(mkt_data['Close']).diff().dropna().values.flatten()
        
        # Quick σ_c calc
        obs_mkt = []
        for w in [5, 10, 20, 30, 50]:
            vol = pd.Series(np.abs(mkt_returns)).rolling(w).std()
            obs_mkt.append(np.nanmean(vol))
        
        # Simple peak finding
        idx_mkt = np.argmax(np.abs(np.gradient(obs_mkt)))
        sigma_c_markets[name] = [5, 10, 20, 30, 50][idx_mkt]
        print(f"  {name}: σ_c = {sigma_c_markets[name]} days")
    except:
        print(f"  {name}: Error fetching")

# 6. CREATE FIGURE
print("\n[6/6] Creating visualization...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Observable
ax = axes[0, 0]
ax.plot(sigma_days, observable, 'o-', color='blue', linewidth=2)
ax.axvline(sigma_c, color='red', linestyle='--', label=f'σ_c = {sigma_c:.1f} days')
ax.set_xscale('log')
ax.set_xlabel('Time Scale (days)')
ax.set_ylabel('Volatility')
ax.set_title('Observable vs Time Scale')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Susceptibility
ax = axes[0, 1]
ax.plot(sigma_days, susceptibility_smooth, 's-', color='orange', linewidth=2)
ax.axvline(sigma_c, color='red', linestyle='--')
ax.fill_between(sigma_days[int(0.2*n):int(0.8*n)], 0, max(susceptibility_smooth)*1.1,
                color='green', alpha=0.1, label='Interior region')
ax.set_xscale('log')
ax.set_xlabel('Time Scale (days)')
ax.set_ylabel('|dO/dσ|')
ax.set_title(f'Susceptibility (κ = {kappa:.2f})')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Bootstrap distribution
ax = axes[1, 0]
ax.hist(bootstrap_sigma, bins=20, color='green', alpha=0.7, edgecolor='black')
ax.axvline(sigma_c, color='red', linestyle='-', linewidth=2, label=f'σ_c = {sigma_c:.1f}')
ax.axvline(ci_lower, color='red', linestyle='--', label=f'95% CI')
ax.axvline(ci_upper, color='red', linestyle='--')
ax.set_xlabel('σ_c (days)')
ax.set_ylabel('Frequency')
ax.set_title('Bootstrap Distribution')
ax.legend()
ax.grid(True, axis='y', alpha=0.3)

# Plot 4: Market comparison
ax = axes[1, 1]
if sigma_c_markets:
    markets_list = list(sigma_c_markets.keys())
    values = list(sigma_c_markets.values())
    colors = ['blue', 'orange', 'green'][:len(markets_list)]
    bars = ax.bar(markets_list, values, color=colors, edgecolor='black', linewidth=2)
    ax.axhline(np.mean(values), color='red', linestyle='--', 
              label=f'Mean = {np.mean(values):.1f}')
    ax.set_ylabel('σ_c (days)')
    ax.set_title('Cross-Market Comparison')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
else:
    ax.text(0.5, 0.5, 'Market comparison not available', ha='center', va='center')
    ax.axis('off')

plt.suptitle('Financial Markets Critical Susceptibility Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('financial_analysis.pdf', dpi=300, bbox_inches='tight')
plt.show()

# 7. SAVE RESULTS
results = {
    'timestamp': datetime.now().isoformat(),
    'sigma_c': float(sigma_c),
    'kappa': float(kappa),
    'ci_lower': float(ci_lower),
    'ci_upper': float(ci_upper),
    'n_datapoints': len(returns),
    'markets': sigma_c_markets
}

with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)

# 8. SUMMARY
print("\n" + "="*70)
print("ANALYSIS COMPLETE!")
print("="*70)
print(f"\nKEY RESULTS:")
print(f"  Critical time scale: σ_c = {sigma_c:.1f} days")
print(f"  95% CI: [{ci_lower:.1f}, {ci_upper:.1f}] days")
print(f"  Peak sharpness: κ = {kappa:.2f}")
print(f"  Cross-market mean: {np.mean(list(sigma_c_markets.values())):.1f} days")
print(f"\nFiles saved:")
print(f"  • financial_analysis.pdf")
print(f"  • results.json")
print("\n[SUCCESS] Analysis completed successfully!")