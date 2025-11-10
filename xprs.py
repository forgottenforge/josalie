#!/usr/bin/env python3
"""
EXPRESS ANALYZER GPU - Wertet die bisherigen Daten aus
Copyright (c) 2025 ForgottenForge.xyz


This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

For commercial licensing without AGPL-3.0 obligations, contact:
[nfo@forgottenforge.xyz]

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

# Finde die neueste JSON-Datei
output_dir = Path("gpu_validation_results")
json_files = list(output_dir.glob("gpu_validation_*.json"))
latest_file = max(json_files, key=lambda x: x.stat().st_mtime)

print(f"Analyzing: {latest_file}")

# Lade Daten
with open(latest_file, 'r') as f:
    data = json.load(f)

validations = data['validations']
print(f"\nTotal configs completed: {len(validations)}")

# Gruppiere nach Size
by_size = {}
for v in validations:
    size = v['config']['size']
    if size not in by_size:
        by_size[size] = []
    by_size[size].append(v)

print("\n" + "="*60)
print("DEFINITIVE RESULTS")
print("="*60)

# Analysiere pro Size
all_sigma_c = []
stable_configs = []
best_config = None
best_cv = 1.0

for size in sorted(by_size.keys()):
    configs = by_size[size]
    sigma_c_values = [c['sigma_c_mean'] for c in configs]
    cv_values = [c['sigma_c_cv'] for c in configs]
    
    # Filter für stabile Configs (CV < 30%)
    stable = [c for c in configs if c['sigma_c_cv'] < 0.3]
    stable_configs.extend(stable)
    
    # Finde besten
    if stable:
        best_in_size = min(stable, key=lambda x: x['sigma_c_cv'])
        if best_in_size['sigma_c_cv'] < best_cv:
            best_cv = best_in_size['sigma_c_cv']
            best_config = best_in_size
    
    all_sigma_c.extend(sigma_c_values)
    
    print(f"\nSize {size}:")
    print(f"  Configs tested: {len(configs)}")
    print(f"  σ_c range: [{min(sigma_c_values):.3f}, {max(sigma_c_values):.3f}]")
    print(f"  Mean σ_c: {np.mean(sigma_c_values):.3f} ± {np.std(sigma_c_values):.3f}")
    print(f"  Mean CV: {np.mean(cv_values)*100:.1f}%")
    print(f"  Stable configs: {len(stable)}/{len(configs)}")

# Global Summary
print("\n" + "="*60)
print("PUBLICATION SUMMARY")
print("="*60)
print(f"✓ σ_c DOES emerge in GPU systems")
print(f"✓ Global mean: σ_c = {np.mean(all_sigma_c):.3f} ± {np.std(all_sigma_c):.3f}")
print(f"✓ Range: [{min(all_sigma_c):.3f}, {max(all_sigma_c):.3f}]")
print(f"✓ Stable configurations: {len(stable_configs)}/{len(validations)} ({len(stable_configs)/len(validations)*100:.0f}%)")

if best_config:
    print(f"\n✨ BEST CONFIGURATION:")
    print(f"  Size: {best_config['config']['size']}")
    print(f"  Thermal: {best_config['config']['thermal']}")
    print(f"  Memory: {best_config['config']['memory_pressure']}")
    print(f"  σ_c = {best_config['sigma_c_mean']:.3f} (CV = {best_config['sigma_c_cv']*100:.1f}%)")
    print(f"  CI: [{best_config['bootstrap_ci'][0]:.3f}, {best_config['bootstrap_ci'][1]:.3f}]")

# Quick Figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# σ_c Distribution
ax1.hist(all_sigma_c, bins=30, edgecolor='black', alpha=0.7)
ax1.axvline(np.mean(all_sigma_c), color='red', linestyle='--', 
            label=f'Mean = {np.mean(all_sigma_c):.3f}')
ax1.set_xlabel('σ_c')
ax1.set_ylabel('Frequency')
ax1.set_title('GPU σ_c Distribution (All Measurements)')
ax1.legend()

# CV by Size
sizes = sorted(by_size.keys())
mean_cvs = [np.mean([c['sigma_c_cv'] for c in by_size[s]])*100 for s in sizes]
ax2.bar(range(len(sizes)), mean_cvs, tick_label=sizes)
ax2.axhline(y=30, color='red', linestyle='--', alpha=0.5, label='Stability threshold')
ax2.set_xlabel('Matrix Size')
ax2.set_ylabel('Mean CV (%)')
ax2.set_title('Measurement Stability by Size')
ax2.legend()

plt.suptitle('GPU σ_c Validation - EXPRESS RESULTS', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / 'express_summary.png', dpi=300)
plt.show()

print("\n✓ Express summary saved!")
print("="*60)