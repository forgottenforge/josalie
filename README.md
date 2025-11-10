# ForgottenForge - SigmaSuite
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![AWS](https://img.shields.io/badge/AWS-Cloud-orange?logo=amazonaws)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0.en.html)
[![License: Commercial](https://img.shields.io/badge/License-Commercial-orange)](LICENSE-COMMERCIAL.txt)
![Status: Early Alpha](https://img.shields.io/badge/status-early--alpha-red)

This repository contains all code, data, documentation, and resources associated with the paper "A Susceptibility-Based Methodology for Characteristic Scale Identification: Preliminary Validation Across Five Complex Systems" by ForgottenForge.

## 🎓 Theoretical Background

The susceptibility χ(σ) quantifies system response to scale changes:

```
χ(σ) = |∂O/∂σ|
```

where:
- O: Observable quantity (energy, performance, variance, etc.)
- σ: Scale parameter (distance, time, noise level, etc.)
- σ_c: Critical scale where χ peaks

Key insight: **Gradient-based observables outperform absolute measures by up to 28×**

## 💻 Example Usage

```python
import numpy as np
from analysis.valivali import find_critical_scale

# Your data
scales = np.logspace(0, 2, 50)  # 1 to 100 km
measurements = your_observable_function(scales)

# Find critical scale
sigma_c, confidence_interval, stats = find_critical_scale(
    scales, 
    measurements,
    observable_type='gradient',  # Recommended!
    bootstrap_n=10000
)

print(f"Critical scale: {sigma_c:.2f} [{confidence_interval[0]:.2f}, {confidence_interval[1]:.2f}]")
print(f"Peak clarity κ = {stats['kappa']:.2f}")
print(f"Significance p = {stats['p_value']:.4f}")
```

## License
Copyright (c) 2025 ForgottenForge.xyz

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

For commercial licensing without AGPL-3.0 obligations, contact:
[nfo@forgottenforge.xyz]

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial


## Contact

- Email: nfo@forgottenforge.xyz
- Web: https://www.forgottenforge.xyz
