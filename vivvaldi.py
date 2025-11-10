#!/usr/bin/env python3
"""
Possible REVIEWER-REQUESTED QPU EXPERIMENTS (Simulator Only)
==============================================
Copyright (c) 2025 ForgottenForge.xyz

Addresses experimental gaps - solvable with QPU-access:
1. Smoothing bandwidth sensitivity analysis
2. ε-grid robustness analysis  
3. Ablation studies (idle-only, noise-only)
4. Readout confusion dose-response curve
5. Nonlinear observable (Purity) for κ > 1
6. 2D multi-parameter sweep demonstration

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

For commercial licensing without AGPL-3.0 obligations, contact:
[nfo@forgottenforge.xyz]

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial

"""

import numpy as np
from braket.devices import LocalSimulator
from braket.circuits import Circuit
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import json
from datetime import datetime
from typing import Dict, List, Tuple
from tqdm import tqdm

plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10


class ReviewerExperiments:
    def __init__(self):
        self.device = LocalSimulator("braket_dm")
        self.results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'device': 'simulator',
                'purpose': 'reviewer_requests'
            },
            'experiments': {}
        }
    
    # ========== CORE METHODS ==========
    
    def add_depolarizing_noise(self, circuit: Circuit, epsilon: float, layer: int = 0) -> Circuit:
        """Add depolarizing noise after all gates."""
        if epsilon <= 0:
            return circuit
        
        seed = (layer * 6364136223846793005 + 1442695040888963407) % (2**63 - 1)
        rng = np.random.default_rng(seed)
        
        qubits = set()
        for instr in circuit.instructions:
            tgt = instr.target
            if isinstance(tgt, (list, tuple)):
                qubits.update(tgt)
            elif tgt is not None:
                qubits.add(tgt)
        
        if not qubits:
            qubits = {0}
        
        for q in qubits:
            if rng.random() < epsilon:
                angle = rng.uniform(0.03, 0.05) * np.pi * (1 if rng.random() < 0.5 else -1)
                if rng.random() < 0.5:
                    circuit.rx(q, angle)
                else:
                    circuit.ry(q, angle)
        
        return circuit
    
    def add_idle_only(self, circuit: Circuit, n_qubits: int, idle_level: float, seed: int = 0) -> Circuit:
        """Pure idle dephasing without other noise."""
        if idle_level <= 0:
            return circuit
        
        rng = np.random.default_rng(seed + int(1e6 * idle_level))
        p_dephase = min(0.95, idle_level * 0.6)
        
        for q in range(n_qubits):
            if rng.random() < p_dephase:
                angle = rng.uniform(-0.06, 0.06) * np.pi
                circuit.ry(q, angle)
        
        return circuit
    
    def create_grover_2q(self, epsilon: float = 0.0, idle_frac: float = 0.0, 
                         batch_seed: int = 0, noise_type: str = 'both') -> Circuit:
        """Grover with flexible noise injection."""
        circuit = Circuit()
        
        # Initialization
        for i in range(2):
            circuit.h(i)
        
        if noise_type in ['noise', 'both']:
            circuit = self.add_depolarizing_noise(circuit, epsilon, 0 + batch_seed)
        
        # Idle layers
        if idle_frac > 0 and noise_type in ['idle', 'both']:
            n_idle = max(1, int(12 * idle_frac))
            for k in range(n_idle):
                circuit = self.add_idle_only(circuit, 2, idle_frac, batch_seed + k)
                if noise_type == 'both':
                    circuit = self.add_depolarizing_noise(circuit, epsilon, k + batch_seed)
        
        # Oracle
        circuit.cnot(0, 1)
        circuit.rz(1, np.pi)
        circuit.cnot(0, 1)
        
        if noise_type in ['noise', 'both']:
            circuit = self.add_depolarizing_noise(circuit, epsilon, 10 + batch_seed)
        
        # Diffusion
        for i in range(2):
            circuit.h(i)
        for i in range(2):
            circuit.x(i)
        circuit.cnot(0, 1)
        circuit.rz(1, np.pi)
        circuit.cnot(0, 1)
        for i in range(2):
            circuit.x(i)
        for i in range(2):
            circuit.h(i)
        
        if noise_type in ['noise', 'both']:
            circuit = self.add_depolarizing_noise(circuit, epsilon, 20 + batch_seed)
        
        return circuit
    
    def run_circuit_batch(self, circuit_func, epsilon: float, shots: int = 500, 
                          reps: int = 3, **kwargs) -> Dict[str, int]:
        """Run circuit with multiple repetitions."""
        merged_counts = {}
        
        for rep in range(reps):
            circuit = circuit_func(epsilon=epsilon, batch_seed=rep * 1000, **kwargs)
            result = self.device.run(circuit, shots=shots).result()
            
            for bitstring, count in result.measurement_counts.items():
                merged_counts[bitstring] = merged_counts.get(bitstring, 0) + count
        
        return merged_counts
    
    def compute_observable_prob(self, counts: Dict[str, int]) -> float:
        """P(|11⟩) for Grover."""
        total = sum(counts.values())
        if total == 0:
            return 0.0
        return counts.get('11', 0) / total
    
    def compute_observable_purity(self, counts: Dict[str, int]) -> float:
        """Purity = Tr[ρ²]."""
        total = sum(counts.values())
        if total == 0:
            return 0.0
        
        # Approximate purity from measurement statistics
        # Purity = Σ_x p_x²
        purity = sum((count / total)**2 for count in counts.values())
        return purity
    
    def compute_susceptibility(self, eps: np.ndarray, obs: np.ndarray, 
                               kernel_sigma: float = 0.5) -> Tuple[np.ndarray, float, float]:
        """Compute χ, σc, κ with specified smoothing."""
        obs_smooth = gaussian_filter1d(obs, sigma=kernel_sigma)
        chi = np.gradient(obs_smooth, eps)
        abs_chi = np.abs(chi)
        
        # Edge damping
        if len(eps) >= 2:
            abs_chi[0] *= 0.5
            abs_chi[-1] *= 0.5
        
        # Robust baseline (10th percentile)
        interior = abs_chi[1:-1] if len(abs_chi) > 2 else abs_chi
        interior_pos = interior[interior > 1e-9]
        
        if interior_pos.size > 0:
            baseline = float(np.percentile(interior_pos, 10))
            baseline = max(baseline, 1e-5)
        else:
            baseline = 1e-5
        
        idx_max = int(np.argmax(abs_chi))
        sigma_c = float(eps[idx_max])
        kappa = float(abs_chi[idx_max] / baseline)
        
        return chi, sigma_c, min(kappa, 200.0)
    
    def apply_readout_confusion(self, counts: Dict[str, int], p0to1: float, p1to0: float) -> Dict[str, int]:
        """Apply readout error."""
        rng = np.random.default_rng(20251024)
        noisy = {}
        
        for bitstring, count in counts.items():
            for _ in range(count):
                out = []
                for b in bitstring:
                    if b == '0':
                        flip = rng.random() < p0to1
                        out.append('1' if flip else '0')
                    else:
                        flip = rng.random() < p1to0
                        out.append('0' if flip else '1')
                bs = ''.join(out)
                noisy[bs] = noisy.get(bs, 0) + 1
        
        return noisy
    
    # ========== EXPERIMENT R1: SMOOTHING SENSITIVITY ==========
    
    def experiment_r1_smoothing_sensitivity(self) -> Dict:
        """Test σc vs. smoothing bandwidth."""
        print("\n" + "="*70)
        print("R1: SMOOTHING BANDWIDTH SENSITIVITY")
        print("="*70)
        
        eps_test = np.linspace(0.04, 0.20, 10)
        
        # Collect data once
        print("Collecting data...")
        obs_data = []
        for eps in tqdm(eps_test, desc="Data collection"):
            counts = self.run_circuit_batch(
                self.create_grover_2q, eps, shots=800, reps=3
            )
            obs_data.append(self.compute_observable_prob(counts))
        
        obs_data = np.array(obs_data)
        
        # Test multiple bandwidths
        kernel_sigmas = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5]
        sigma_c_values = []
        kappa_values = []
        
        print("Testing bandwidths...")
        for ks in tqdm(kernel_sigmas, desc="Smoothing"):
            _, sc, kap = self.compute_susceptibility(eps_test, obs_data, kernel_sigma=ks)
            sigma_c_values.append(sc)
            kappa_values.append(kap)
        
        # Relative shift
        sigma_c_arr = np.array(sigma_c_values)
        rel_shift = np.std(sigma_c_arr) / np.mean(sigma_c_arr)
        
        res = {
            'kernel_sigmas': kernel_sigmas,
            'sigma_c_values': sigma_c_values,
            'kappa_values': kappa_values,
            'relative_shift': float(rel_shift),
            'max_shift': float(np.max(sigma_c_arr) - np.min(sigma_c_arr))
        }
        
        print(f"✓ Relative σc shift: {rel_shift:.1%}")
        print(f"✓ Max absolute shift: {res['max_shift']:.4f}")
        
        self.results['experiments']['r1'] = res
        return res
    
    # ========== EXPERIMENT R2: GRID ROBUSTNESS ==========
    
    def experiment_r2_grid_robustness(self) -> Dict:
        """Test σc with different ε-grids."""
        print("\n" + "="*70)
        print("R2: ε-GRID ROBUSTNESS")
        print("="*70)
        
        # Define different grids
        grids = {
            'original': np.linspace(0.04, 0.20, 10),
            'shifted_low': np.linspace(0.02, 0.18, 10),
            'shifted_high': np.linspace(0.06, 0.22, 10),
            'dense': np.linspace(0.04, 0.20, 15),
            'sparse': np.linspace(0.04, 0.20, 6)
        }
        
        sigma_c_grid = {}
        
        for grid_name, eps_grid in grids.items():
            print(f"Testing grid: {grid_name}")
            obs_data = []
            
            for eps in tqdm(eps_grid, desc=f"  {grid_name}", leave=False):
                counts = self.run_circuit_batch(
                    self.create_grover_2q, eps, shots=500, reps=2
                )
                obs_data.append(self.compute_observable_prob(counts))
            
            _, sc, _ = self.compute_susceptibility(eps_grid, np.array(obs_data))
            sigma_c_grid[grid_name] = float(sc)
            print(f"  σc = {sc:.4f}")
        
        # Compute relative deviations
        sc_mean = np.mean(list(sigma_c_grid.values()))
        rel_devs = {k: abs(v - sc_mean) / sc_mean for k, v in sigma_c_grid.items()}
        
        res = {
            'grids': {k: v.tolist() for k, v in grids.items()},
            'sigma_c_values': sigma_c_grid,
            'relative_deviations': rel_devs,
            'max_deviation': float(max(rel_devs.values()))
        }
        
        print(f"✓ Max relative deviation: {res['max_deviation']:.1%}")
        
        self.results['experiments']['r2'] = res
        return res
    
    # ========== EXPERIMENT R3: ABLATION STUDIES ==========
    
    def experiment_r3_ablation(self) -> Dict:
        """Ablation: idle-only, noise-only, both."""
        print("\n" + "="*70)
        print("R3: ABLATION STUDIES")
        print("="*70)
        
        eps_test = np.linspace(0.04, 0.20, 8)
        idle_frac = 0.20
        
        conditions = {
            'both': {'noise_type': 'both', 'idle_frac': idle_frac},
            'noise_only': {'noise_type': 'noise', 'idle_frac': 0.0},
            'idle_only': {'noise_type': 'idle', 'idle_frac': idle_frac}
        }
        
        results_cond = {}
        
        for cond_name, kwargs in conditions.items():
            print(f"\nCondition: {cond_name}")
            obs_data = []
            
            for eps in tqdm(eps_test, desc=f"  {cond_name}", leave=False):
                counts = self.run_circuit_batch(
                    self.create_grover_2q, eps, shots=600, reps=3, **kwargs
                )
                obs_data.append(self.compute_observable_prob(counts))
            
            _, sc, kap = self.compute_susceptibility(eps_test, np.array(obs_data))
            
            results_cond[cond_name] = {
                'sigma_c': float(sc),
                'kappa': float(kap),
                'observable': [float(x) for x in obs_data]
            }
            
            print(f"  σc = {sc:.4f}, κ = {kap:.2f}")
        
        res = {
            'conditions': results_cond,
            'epsilon': eps_test.tolist(),
            'idle_frac_tested': idle_frac
        }
        
        self.results['experiments']['r3'] = res
        return res
    
    # ========== EXPERIMENT R4: READOUT DOSE-RESPONSE ==========
    
    def experiment_r4_readout_dose_response(self) -> Dict:
        """κ reduction vs. readout confusion levels."""
        print("\n" + "="*70)
        print("R4: READOUT CONFUSION DOSE-RESPONSE")
        print("="*70)
        
        eps_test = np.array([0.04, 0.08, 0.12, 0.16, 0.20])
        
        # Collect clean data once
        print("Collecting clean data...")
        obs_clean = []
        counts_clean = []
        
        for eps in tqdm(eps_test, desc="Clean"):
            counts = self.run_circuit_batch(
                self.create_grover_2q, eps, shots=800, reps=3
            )
            counts_clean.append(counts)
            obs_clean.append(self.compute_observable_prob(counts))
        
        _, _, kappa_clean = self.compute_susceptibility(eps_test, np.array(obs_clean))
        
        # Test different confusion levels
        confusion_levels = [
            (0.0, 0.0),   # Clean
            (0.05, 0.02),
            (0.10, 0.05),
            (0.15, 0.08),
            (0.22, 0.12),
            (0.30, 0.15)
        ]
        
        kappa_confused = []
        reductions = []
        
        print("Testing confusion levels...")
        for p0, p1 in tqdm(confusion_levels, desc="Confusion"):
            obs_noisy = []
            for counts in counts_clean:
                noisy_counts = self.apply_readout_confusion(counts, p0, p1)
                obs_noisy.append(self.compute_observable_prob(noisy_counts))
            
            _, _, kap = self.compute_susceptibility(eps_test, np.array(obs_noisy))
            kappa_confused.append(float(kap))
            
            reduction = (kappa_clean - kap) / kappa_clean if kappa_clean > 0 else 0.0
            reductions.append(float(reduction))
        
        res = {
            'confusion_levels': [{'p0to1': p0, 'p1to0': p1} for p0, p1 in confusion_levels],
            'kappa_values': kappa_confused,
            'reductions': reductions,
            'kappa_clean': float(kappa_clean)
        }
        
        print(f"✓ κ_clean = {kappa_clean:.2f}")
        print(f"✓ Maximum reduction: {max(reductions):.1%} at confusion {confusion_levels[-1]}")
        
        self.results['experiments']['r4'] = res
        return res
    
    # ========== EXPERIMENT R5: NONLINEAR OBSERVABLE ==========
    
    def experiment_r5_nonlinear_observable(self) -> Dict:
        """Compare P(|11⟩) vs. Purity for κ."""
        print("\n" + "="*70)
        print("R5: NONLINEAR OBSERVABLE (PURITY)")
        print("="*70)
        
        eps_test = np.linspace(0.04, 0.20, 10)
        
        print("Collecting data for both observables...")
        obs_prob = []
        obs_purity = []
        
        for eps in tqdm(eps_test, desc="Data"):
            counts = self.run_circuit_batch(
                self.create_grover_2q, eps, shots=1000, reps=4
            )
            obs_prob.append(self.compute_observable_prob(counts))
            obs_purity.append(self.compute_observable_purity(counts))
        
        obs_prob = np.array(obs_prob)
        obs_purity = np.array(obs_purity)
        
        # Compute susceptibility for both
        chi_p, sc_p, kappa_p = self.compute_susceptibility(eps_test, obs_prob)
        chi_pur, sc_pur, kappa_pur = self.compute_susceptibility(eps_test, obs_purity)
        
        res = {
            'epsilon': eps_test.tolist(),
            'probability': {
                'observable': obs_prob.tolist(),
                'sigma_c': float(sc_p),
                'kappa': float(kappa_p)
            },
            'purity': {
                'observable': obs_purity.tolist(),
                'sigma_c': float(sc_pur),
                'kappa': float(kappa_pur)
            },
            'kappa_ratio': float(kappa_pur / kappa_p) if kappa_p > 0 else 0.0
        }
        
        print(f"✓ P(|11⟩): σc = {sc_p:.4f}, κ = {kappa_p:.2f}")
        print(f"✓ Purity:  σc = {sc_pur:.4f}, κ = {kappa_pur:.2f}")
        print(f"✓ κ enhancement: {res['kappa_ratio']:.2f}×")
        
        self.results['experiments']['r5'] = res
        return res
    
    # ========== EXPERIMENT R6: 2D MULTI-PARAMETER ==========
    
    def experiment_r6_multiparameter_2d(self) -> Dict:
        """2D sweep: ε1 (depolarizing) vs ε2 (idle)."""
        print("\n" + "="*70)
        print("R6: 2D MULTI-PARAMETER SWEEP")
        print("="*70)
        
        eps1_grid = np.linspace(0.02, 0.16, 5)  # Depolarizing
        eps2_grid = np.linspace(0.0, 0.30, 5)   # Idle fraction
        
        obs_2d = np.zeros((len(eps1_grid), len(eps2_grid)))
        
        print("Running 2D sweep (25 points)...")
        for i, eps1 in enumerate(tqdm(eps1_grid, desc="ε1 (noise)")):
            for j, eps2 in enumerate(eps2_grid):
                counts = self.run_circuit_batch(
                    self.create_grover_2q, eps1, shots=400, reps=2,
                    idle_frac=eps2
                )
                obs_2d[i, j] = self.compute_observable_prob(counts)
        
        # Find maximum gradient direction
        grad_eps1 = np.gradient(obs_2d, axis=0)
        grad_eps2 = np.gradient(obs_2d, axis=1)
        grad_mag = np.sqrt(grad_eps1**2 + grad_eps2**2)
        
        idx_max = np.unravel_index(np.argmax(grad_mag), grad_mag.shape)
        sigma_c_2d = (float(eps1_grid[idx_max[0]]), float(eps2_grid[idx_max[1]]))
        
        res = {
            'eps1_grid': eps1_grid.tolist(),
            'eps2_grid': eps2_grid.tolist(),
            'observable_2d': obs_2d.tolist(),
            'sigma_c_2d': sigma_c_2d,
            'max_gradient': float(grad_mag[idx_max])
        }
        
        print(f"✓ 2D threshold: σc = ({sigma_c_2d[0]:.3f}, {sigma_c_2d[1]:.3f})")
        print(f"✓ Max gradient magnitude: {res['max_gradient']:.4f}")
        
        self.results['experiments']['r6'] = res
        return res
    
    # ========== PLOTTING ==========
    
    def plot_all_results(self, output_dir: str = "."):
        """Generate all reviewer-requested plots."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # R1: Smoothing sensitivity
        if 'r1' in self.results['experiments']:
            r1 = self.results['experiments']['r1']
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            ax1.plot(r1['kernel_sigmas'], r1['sigma_c_values'], 'o-', color='#2E86AB')
            ax1.axhline(np.mean(r1['sigma_c_values']), color='red', linestyle='--', 
                       label=f'Mean = {np.mean(r1["sigma_c_values"]):.4f}')
            ax1.set_xlabel('Smoothing Bandwidth σ')
            ax1.set_ylabel('Estimated σc')
            ax1.set_title('R1: σc vs. Smoothing Bandwidth')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(r1['kernel_sigmas'], r1['kappa_values'], 's-', color='#F18F01')
            ax2.set_xlabel('Smoothing Bandwidth σ')
            ax2.set_ylabel('Peak Clarity κ')
            ax2.set_title('R1: κ vs. Smoothing Bandwidth')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/r1_smoothing_sensitivity.pdf', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved: {output_dir}/r1_smoothing_sensitivity.pdf")
        
        # R2: Grid robustness (bar chart)
        if 'r2' in self.results['experiments']:
            r2 = self.results['experiments']['r2']
            
            fig, ax = plt.subplots(figsize=(8, 5))
            
            names = list(r2['sigma_c_values'].keys())
            values = list(r2['sigma_c_values'].values())
            colors = ['#2E86AB' if n == 'original' else '#A6CEE3' for n in names]
            
            ax.bar(names, values, color=colors, edgecolor='black', linewidth=1.5)
            ax.axhline(np.mean(values), color='red', linestyle='--', 
                      label=f'Mean = {np.mean(values):.4f}')
            ax.set_ylabel('σc')
            ax.set_title('R2: σc Across Different ε-Grids')
            ax.legend()
            ax.grid(True, axis='y', alpha=0.3)
            plt.xticks(rotation=45, ha='right')
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/r2_grid_robustness.pdf', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved: {output_dir}/r2_grid_robustness.pdf")
        
        # R3: Ablation (observable curves)
        if 'r3' in self.results['experiments']:
            r3 = self.results['experiments']['r3']
            
            fig, ax = plt.subplots(figsize=(8, 5))
            
            eps = np.array(r3['epsilon'])
            colors = {'both': '#2E86AB', 'noise_only': '#F18F01', 'idle_only': '#A23B72'}
            
            for cond_name, data in r3['conditions'].items():
                obs = np.array(data['observable'])
                ax.plot(eps, obs, 'o-', color=colors[cond_name], 
                       label=f"{cond_name}: σc={data['sigma_c']:.3f}, κ={data['kappa']:.2f}")
            
            ax.set_xlabel('Noise Parameter ε')
            ax.set_ylabel('Observable O(ε)')
            ax.set_title('R3: Ablation Study (Idle vs. Noise)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/r3_ablation.pdf', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved: {output_dir}/r3_ablation.pdf")
        
        # R4: Readout dose-response
        if 'r4' in self.results['experiments']:
            r4 = self.results['experiments']['r4']
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Extract confusion means
            conf_means = [np.mean([c['p0to1'], c['p1to0']]) for c in r4['confusion_levels']]
            
            ax1.plot(conf_means, r4['kappa_values'], 'D-', color='#2E86AB', markersize=8)
            ax1.set_xlabel('Mean Readout Error Rate')
            ax1.set_ylabel('Peak Clarity κ')
            ax1.set_title('R4: κ vs. Readout Confusion')
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(conf_means[1:], [r*100 for r in r4['reductions'][1:]], 's-', color='#A23B72', markersize=8)
            ax2.set_xlabel('Mean Readout Error Rate')
            ax2.set_ylabel('κ Reduction (%)')
            ax2.set_title('R4: Peak Clarity Degradation')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/r4_readout_dose_response.pdf', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved: {output_dir}/r4_readout_dose_response.pdf")
        
        # R5: Nonlinear observable comparison
        if 'r5' in self.results['experiments']:
            r5 = self.results['experiments']['r5']
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            eps = np.array(r5['epsilon'])
            
            # Probability
            obs_p = np.array(r5['probability']['observable'])
            ax1.plot(eps, obs_p, 'o-', color='#2E86AB', label='P(|11⟩)')
            ax1.axvline(r5['probability']['sigma_c'], color='#2E86AB', linestyle='--', 
                       label=f"σc = {r5['probability']['sigma_c']:.3f}")
            ax1.set_xlabel('Noise Parameter ε')
            ax1.set_ylabel('Observable')
            ax1.set_title(f"R5: Probability (κ = {r5['probability']['kappa']:.2f})")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Purity
            obs_pur = np.array(r5['purity']['observable'])
            ax2.plot(eps, obs_pur, 's-', color='#F18F01', label='Purity')
            ax2.axvline(r5['purity']['sigma_c'], color='#F18F01', linestyle='--', 
                       label=f"σc = {r5['purity']['sigma_c']:.3f}")
            ax2.set_xlabel('Noise Parameter ε')
            ax2.set_ylabel('Observable')
            ax2.set_title(f"R5: Purity (κ = {r5['purity']['kappa']:.2f})")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/r5_nonlinear_observable.pdf', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved: {output_dir}/r5_nonlinear_observable.pdf")
        
        # R6: 2D heatmap
        if 'r6' in self.results['experiments']:
            r6 = self.results['experiments']['r6']
            
            fig, ax = plt.subplots(figsize=(8, 6))
            
            obs_2d = np.array(r6['observable_2d'])
            eps1 = np.array(r6['eps1_grid'])
            eps2 = np.array(r6['eps2_grid'])
            
            im = ax.contourf(eps2, eps1, obs_2d, levels=15, cmap='viridis')
            ax.plot(r6['sigma_c_2d'][1], r6['sigma_c_2d'][0], 'r*', markersize=20, 
                   label=f"σc = ({r6['sigma_c_2d'][0]:.3f}, {r6['sigma_c_2d'][1]:.3f})")
            
            ax.set_xlabel('ε2 (Idle Fraction)')
            ax.set_ylabel('ε1 (Depolarizing Noise)')
            ax.set_title('R6: 2D Multi-Parameter Threshold')
            ax.legend()
            plt.colorbar(im, ax=ax, label='Observable O(ε1, ε2)')
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/r6_multiparameter_2d.pdf', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved: {output_dir}/r6_multiparameter_2d.pdf")
    
    def save_results(self, filename: str = "reviewer_experiments.json"):
        """Save all results to JSON."""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✓ Results saved: {filename}")
    
    def run_all_experiments(self):
        """Run all reviewer-requested experiments."""
        print("="*70)
        print("RUNNING ALL REVIEWER-REQUESTED EXPERIMENTS")
        print("="*70)
        print("Device: Local Simulator (braket_dm)")
        print("Cost: €0.00")
        print("="*70)
        
        self.experiment_r1_smoothing_sensitivity()
        self.experiment_r2_grid_robustness()
        self.experiment_r3_ablation()
        self.experiment_r4_readout_dose_response()
        self.experiment_r5_nonlinear_observable()
        self.experiment_r6_multiparameter_2d()
        
        self.save_results()
        self.plot_all_results(output_dir="reviewer_results")
        
        print("\n" + "="*70)
        print("✅ ALL EXPERIMENTS COMPLETE")
        print("="*70)
        print(f"Total experiments: 6")
        print(f"Total cost: €0.00 (simulator only)")
        print(f"Output directory: reviewer_results/")


if __name__ == "__main__":
    experiments = ReviewerExperiments()
    experiments.run_all_experiments()