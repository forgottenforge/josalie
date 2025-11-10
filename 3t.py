#!/usr/bin/env python3
"""
reviewer_analysis.py 
==========================================================
Copyright (c) 2025 ForgottenForge.xyz

- Umfassende Analyse der drei potentieller Reviewer-Fragen
1. Renormalization Group Theory Connection
2. Machine Learning Observable Selection
3. Harmonic Structure Analysis

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
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
from scipy.stats import kstest, pearsonr, linregress
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Matplotlib settings
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 11
plt.rcParams['lines.linewidth'] = 2

# ============================================================================
# PART 1: RENORMALIZATION GROUP THEORY CONNECTION
# ============================================================================

@dataclass
class SystemData:
    """Datencontainer für System-Parameter"""
    name: str
    sigma_c: float
    system_size: float  # Charakteristische Längenskala
    dimension: int      # Räumliche Dimension d
    dynamic_exp: float  # Dynamischer Exponent z
    uncertainty: float  # Fehlerbalken für sigma_c
    
class RGAnalysis:
    """Renormalization Group Analysis für σ_c Framework"""
    
    def __init__(self):
        """Initialisiere mit allen experimentellen Daten"""
        self.systems = self._load_experimental_data()
        self.results = {}
        
    def _load_experimental_data(self) -> List[SystemData]:
        """Lade alle σ_c Messungen aus unseren Experimenten"""
        return [
            # Quantum System
            SystemData('Quantum', 0.080, 20e-6, 0, 2, 0.005),  # T2* = 20μs
            
            # GPU Cache Hierarchie
            SystemData('GPU-L1', 0.030, 32e3, 1, 1, 0.008),    # 32KB L1
            SystemData('GPU-L2', 0.080, 256e3, 1, 1, 0.012),   # 256KB L2
            SystemData('GPU-L3', 0.167, 12e6, 1, 1, 0.025),    # 12MB L3
            
            # Seismological
            SystemData('Seismic', 2.85, 316, 2, 2, 0.56),      # 316km characteristic
            
            # Financial 
            SystemData('Financial-3d', 3.0, 252, 0, 1, 0.8),   # 252 trading days/year
            SystemData('Financial-10d', 10.0, 252, 0, 1, 2.0),
            
            # Climate
            SystemData('Climate-meso', 54, 1000, 2, 2, 7),     # 1000km domain
            SystemData('Climate-synop', 750, 5000, 2, 2, 150), # 5000km continental
        ]
    
    def test_scaling_hypothesis(self):
        """Test: σ_c ~ L^(d/z) Skalierungshypothese"""
        print("\n" + "="*60)
        print("RG SCALING ANALYSIS: σ_c ~ L^(d/z)")
        print("="*60)
        
        x_data = []
        y_data = []
        labels = []
        
        for sys in self.systems:
            if sys.dynamic_exp > 0:
                exponent = sys.dimension / sys.dynamic_exp
                x_val = sys.system_size ** exponent
                y_val = sys.sigma_c
                
                x_data.append(x_val)
                y_data.append(y_val)
                labels.append(sys.name)
                
                print(f"{sys.name:15s}: L={sys.system_size:.2e}, "
                      f"d={sys.dimension}, z={sys.dynamic_exp:.1f}, "
                      f"σ_c={sys.sigma_c:.3f}")
        
        # Log-log Regression
        log_x = np.log10(x_data)
        log_y = np.log10(y_data)
        
        slope, intercept, r_value, p_value, std_err = linregress(log_x, log_y)
        
        print(f"\nLog-log Regression:")
        print(f"  σ_c ~ L^({slope:.3f})")
        print(f"  R² = {r_value**2:.4f}")
        print(f"  p-value = {p_value:.4e}")
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.scatter(x_data, y_data, s=100, alpha=0.7)
        for i, label in enumerate(labels):
            ax1.annotate(label, (x_data[i], y_data[i]), fontsize=8)
        ax1.set_xlabel('L^(d/z)')
        ax1.set_ylabel('σ_c')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_title('Universal Scaling Test')
        ax1.grid(True, alpha=0.3)
        
        # Fit line
        x_fit = np.logspace(np.log10(min(x_data)), np.log10(max(x_data)), 100)
        y_fit = 10**intercept * x_fit**slope
        ax1.plot(x_fit, y_fit, 'r--', label=f'σ_c ~ L^{slope:.2f}')
        ax1.legend()
        
        # Data collapse attempt
        ax2.scatter(log_x, log_y, s=100, alpha=0.7)
        ax2.plot(log_x, slope*log_x + intercept, 'r--')
        ax2.set_xlabel('log₁₀(L^(d/z))')
        ax2.set_ylabel('log₁₀(σ_c)')
        ax2.set_title(f'Data Collapse (R²={r_value**2:.3f})')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('rg_scaling_analysis.pdf', dpi=150, bbox_inches='tight')
        plt.show()
        
        self.results['scaling'] = {
            'exponent': slope,
            'r_squared': r_value**2,
            'p_value': p_value
        }
        
    def compute_beta_function(self):
        """Berechne β-Funktion: dσ_c/d(ln L) = β(σ_c)"""
        print("\n" + "="*60)
        print("BETA FUNCTION ANALYSIS")
        print("="*60)
        
        # Nutze GPU Daten mit verschiedenen Cache-Größen
        gpu_data = [(0.030, 32e3), (0.080, 256e3), (0.167, 12e6)]
        
        sigma_vals = [d[0] for d in gpu_data]
        L_vals = [d[1] for d in gpu_data]
        
        # Berechne β = dσ_c/d(ln L)
        log_L = np.log(L_vals)
        beta_empirical = np.gradient(sigma_vals, log_L)
        
        # Fitte β(σ) = -α σ^ν (RG Fluss)
        def beta_form(sigma, alpha, nu):
            return -alpha * sigma**nu
        
        try:
            popt, pcov = curve_fit(beta_form, sigma_vals[:-1], 
                                  beta_empirical[:-1], p0=[1.0, 1.0])
            alpha, nu = popt
            
            print(f"β(σ) = -{alpha:.3f} * σ^{nu:.3f}")
            print(f"Critical exponent ν = {nu:.3f}")
            
            # Fixpunkte
            if nu > 1:
                print("UV-stable fixed point at σ=0 (Gaussian)")
                print(f"IR-unstable fixed point at σ_* = (α/ν)^(1/(ν-1))")
            
            # Plot
            fig, ax = plt.subplots(figsize=(8, 6))
            sigma_range = np.linspace(0.01, 0.2, 100)
            beta_fit = beta_form(sigma_range, alpha, nu)
            
            ax.scatter(sigma_vals, beta_empirical, s=100, label='Empirical', zorder=5)
            ax.plot(sigma_range, beta_fit, 'r-', label=f'β = -{alpha:.2f}σ^{nu:.2f}')
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax.set_xlabel('σ_c')
            ax.set_ylabel('β(σ_c) = dσ_c/d(ln L)')
            ax.set_title('RG Beta Function')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.savefig('beta_function.pdf', dpi=150, bbox_inches='tight')
            plt.show()
            
            self.results['beta_function'] = {
                'alpha': alpha,
                'nu': nu,
                'critical_exponent': nu
            }
            
        except Exception as e:
            print(f"Beta function fitting failed: {e}")
    
    def test_universality_class(self):
        """Teste ob alle Systeme zur selben Universalitätsklasse gehören"""
        print("\n" + "="*60)
        print("UNIVERSALITY CLASS TEST")
        print("="*60)
        
        # Normiere σ_c mit system-spezifischen Skalen
        normalized = []
        
        for sys in self.systems:
            if sys.dynamic_exp > 0:
                # Verschiedene Normierungsansätze
                norm1 = sys.sigma_c / sys.system_size**(sys.dimension/sys.dynamic_exp)
                normalized.append(norm1)
                print(f"{sys.name:15s}: σ_c/L^(d/z) = {norm1:.3e}")
        
        # Kolmogorov-Smirnov Test auf Normalverteilung
        normalized_log = np.log10(normalized)
        ks_stat, p_value = kstest(normalized_log, 'norm')
        
        print(f"\nKS-Test für Universalität:")
        print(f"  KS-Statistik: {ks_stat:.4f}")
        print(f"  p-value: {p_value:.4f}")
        
        if p_value > 0.05:
            print("  → Daten konsistent mit universeller Verteilung!")
        else:
            print("  → Hinweise auf multiple Universalitätsklassen")
        
        self.results['universality'] = {
            'ks_statistic': ks_stat,
            'p_value': p_value,
            'is_universal': p_value > 0.05
        }

# ============================================================================
# PART 2: MACHINE LEARNING OBSERVABLE SELECTION
# ============================================================================

class MLObservableOptimizer:
    """Machine Learning basierte Observable-Optimierung"""
    
    def __init__(self):
        self.results = {}
        
    def generate_synthetic_data(self, n_samples=100):
        """Erzeuge synthetische Trainingsdaten"""
        np.random.seed(42)
        
        # Features: System-Eigenschaften
        features = {
            'noise_level': np.random.uniform(0.01, 0.5, n_samples),
            'system_size': np.random.uniform(10, 1000, n_samples),
            'dimension': np.random.randint(0, 3, n_samples),
            'nonlinearity': np.random.uniform(0, 1, n_samples),
            'sparsity': np.random.uniform(0, 1, n_samples)
        }
        
        # Target: Optimale Observable-Parameter (aus unseren Experimenten)
        # Gradient-basierte Observablen geben κ ~ 15-85
        # Absolute Werte geben κ ~ 2-5
        
        targets = {
            'use_gradient': (features['noise_level'] < 0.2).astype(float),
            'power_transform': 1.0 + 0.3 * features['nonlinearity'],
            'smoothing_scale': 0.5 + features['system_size'] / 1000,
            'expected_kappa': 10 * (1 + features['dimension']) * \
                            (1 - features['noise_level'])
        }
        
        return pd.DataFrame(features), pd.DataFrame(targets)
    
    def bayesian_optimization(self, test_function, n_trials=50):
        """Bayesian Optimization für Observable-Parameter"""
        from scipy.stats import norm
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern
        
        print("\n" + "="*60)
        print("BAYESIAN OPTIMIZATION FOR OBSERVABLE SELECTION")
        print("="*60)
        
        # Suchraum
        bounds = {
            'gradient_weight': (0.0, 1.0),
            'power_exp': (0.5, 2.0),
            'smooth_scale': (0.1, 2.0),
            'kernel_width': (0.5, 3.0)
        }
        
        # Initialisierung
        X_sample = []
        y_sample = []
        
        # Latin Hypercube Sampling für initiale Punkte
        n_init = 10
        for i in range(n_init):
            x = [np.random.uniform(b[0], b[1]) for b in bounds.values()]
            y = test_function(x)
            X_sample.append(x)
            y_sample.append(y)
            print(f"Initial {i+1}/{n_init}: κ = {y:.2f}")
        
        # Gaussian Process
        kernel = Matern(nu=2.5)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
        
        # Optimization loop
        for i in range(n_trials - n_init):
            # Fit GP
            gp.fit(X_sample, y_sample)
            
            # Acquisition function (Expected Improvement)
            def acquisition(x):
                mu, sigma = gp.predict([x], return_std=True)
                mu, sigma = mu[0], sigma[0]
                
                y_max = max(y_sample)
                Z = (mu - y_max) / (sigma + 1e-9)
                ei = sigma * (Z * norm.cdf(Z) + norm.pdf(Z))
                return -ei  # Minimize negative EI
            
            # Find next point
            x0 = [np.random.uniform(b[0], b[1]) for b in bounds.values()]
            res = minimize(acquisition, x0, 
                          bounds=list(bounds.values()),
                          method='L-BFGS-B')
            
            x_next = res.x
            y_next = test_function(x_next)
            
            X_sample.append(x_next)
            y_sample.append(y_next)
            
            if (i+1) % 10 == 0:
                print(f"Iteration {i+1+n_init}/{n_trials}: "
                      f"Best κ = {max(y_sample):.2f}")
        
        # Beste Parameter
        best_idx = np.argmax(y_sample)
        best_params = dict(zip(bounds.keys(), X_sample[best_idx]))
        best_kappa = y_sample[best_idx]
        
        print(f"\nOptimale Observable-Parameter:")
        for key, val in best_params.items():
            print(f"  {key}: {val:.3f}")
        print(f"  → κ = {best_kappa:.2f}")
        
        # Konvergenz-Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.plot(range(1, len(y_sample)+1), y_sample, 'b.-', alpha=0.5)
        ax1.plot(range(1, len(y_sample)+1), 
                np.maximum.accumulate(y_sample), 'r-', linewidth=2,
                label='Best so far')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('κ (Peak Clarity)')
        ax1.set_title('Bayesian Optimization Convergence')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Parameter Evolution
        params_array = np.array(X_sample)
        for i, key in enumerate(bounds.keys()):
            ax2.plot(params_array[:, i], label=key, alpha=0.7)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Parameter Value (normalized)')
        ax2.set_title('Parameter Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('ml_optimization.pdf', dpi=150, bbox_inches='tight')
        plt.show()
        
        self.results = {
            'best_params': best_params,
            'best_kappa': best_kappa,
            'convergence': y_sample,
            'all_params': X_sample
        }
        
        return best_params, best_kappa
    
    def neural_predictor(self):
        """Neural Network zur Vorhersage optimaler Observable"""
        try:
            import torch
            import torch.nn as nn
            from torch.optim import Adam
            
            print("\n" + "="*60)
            print("NEURAL NETWORK OBSERVABLE PREDICTOR")
            print("="*60)
            
            # Generiere Trainingsdaten
            X_df, y_df = self.generate_synthetic_data(1000)
            
            # Convert to tensors
            X = torch.FloatTensor(X_df.values)
            y = torch.FloatTensor(y_df[['use_gradient', 'power_transform', 
                                       'smoothing_scale']].values)
            
            # Split train/test
            n_train = 800
            X_train, X_test = X[:n_train], X[n_train:]
            y_train, y_test = y[:n_train], y[n_train:]
            
            # Define network
            class ObservableNet(nn.Module):
                def __init__(self, input_dim=5, hidden_dim=64):
                    super().__init__()
                    self.fc1 = nn.Linear(input_dim, hidden_dim)
                    self.fc2 = nn.Linear(hidden_dim, hidden_dim)
                    self.fc3 = nn.Linear(hidden_dim, 3)
                    self.relu = nn.ReLU()
                    self.sigmoid = nn.Sigmoid()
                    
                def forward(self, x):
                    x = self.relu(self.fc1(x))
                    x = self.relu(self.fc2(x))
                    x = self.fc3(x)
                    # Aktivierungen für verschiedene Outputs
                    x[:, 0] = self.sigmoid(x[:, 0])  # use_gradient ∈ [0,1]
                    x[:, 1] = 1.0 + 0.5 * torch.tanh(x[:, 1])  # power ∈ [0.5,1.5]
                    x[:, 2] = 0.1 + 1.9 * self.sigmoid(x[:, 2])  # scale ∈ [0.1,2.0]
                    return x
            
            model = ObservableNet()
            optimizer = Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            # Training
            print("Training Neural Network...")
            losses = []
            for epoch in range(200):
                optimizer.zero_grad()
                y_pred = model(X_train)
                loss = criterion(y_pred, y_train)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                
                if (epoch+1) % 50 == 0:
                    print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")
            
            # Evaluation
            model.eval()
            with torch.no_grad():
                y_pred_test = model(X_test)
                test_loss = criterion(y_pred_test, y_test)
                print(f"\nTest Loss: {test_loss.item():.4f}")
            
            # Beispiel-Vorhersagen für unsere Systeme
            print("\nVorhersagen für reale Systeme:")
            test_systems = {
                'Quantum': [0.05, 50, 0, 0.3, 0.1],
                'GPU': [0.1, 1000, 1, 0.5, 0.3],
                'Seismic': [0.3, 316, 2, 0.7, 0.6],
                'Climate': [0.2, 1000, 2, 0.4, 0.4]
            }
            
            for name, features in test_systems.items():
                x = torch.FloatTensor([features])
                pred = model(x)[0].detach().numpy()
                print(f"{name:10s}: gradient={pred[0]:.2f}, "
                      f"power={pred[1]:.2f}, scale={pred[2]:.2f}")
            
            # Plot training
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(losses, alpha=0.7)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('MSE Loss')
            ax.set_title('Neural Network Training')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            
            plt.savefig('nn_training.pdf', dpi=150, bbox_inches='tight')
            plt.show()
            
        except ImportError:
            print("PyTorch nicht installiert. Überspringe Neural Network Analyse.")

# ============================================================================
# PART 3: HARMONIC STRUCTURE ANALYSIS
# ============================================================================

class HarmonicAnalysis:
    """Analyse der hierarchischen/harmonischen Struktur von σ_c"""
    
    def __init__(self):
        """Lade alle Peak-Daten aus Experimenten"""
        self.peak_data = self._load_peak_data()
        self.results = {}
        
    def _load_peak_data(self) -> Dict:
        """Sammle alle primären und sekundären Peaks"""
        return {
            'quantum': {
                'peaks': [0.080],
                'labels': ['primary']
            },
            'gpu': {
                'peaks': [0.030, 0.080, 0.167],
                'labels': ['L1 cache', 'L2 cache', 'L3/Memory']
            },
            'seismic': {
                'peaks': [2.85, 4.2],  # Zweiter Peak aus Susceptibility
                'labels': ['primary', 'secondary']
            },
            'financial': {
                'peaks': [3.0, 10.0, 21.0],  # 3-day, 10-day, monthly
                'labels': ['settlement', 'options', 'monthly']
            },
            'climate': {
                'peaks': [54, 750],  # Mesoscale, Synoptic
                'labels': ['mesoscale', 'synoptic']
            }
        }
    
    def analyze_ratios(self):
        """Analysiere Verhältnisse zwischen Peaks"""
        print("\n" + "="*60)
        print("HARMONIC STRUCTURE ANALYSIS")
        print("="*60)
        
        all_ratios = []
        
        for system, data in self.peak_data.items():
            peaks = data['peaks']
            if len(peaks) > 1:
                print(f"\n{system.upper()}:")
                for i in range(len(peaks)-1):
                    ratio = peaks[i+1] / peaks[i]
                    all_ratios.append(ratio)
                    print(f"  {data['labels'][i+1]}/{data['labels'][i]} = "
                          f"{peaks[i+1]:.3f}/{peaks[i]:.3f} = {ratio:.3f}")
        
        # Teste gegen mathematische Konstanten
        print("\n" + "-"*40)
        print("Vergleich mit mathematischen Konstanten:")
        
        constants = {
            'φ (golden ratio)': (1 + np.sqrt(5))/2,
            '√2': np.sqrt(2),
            '√3': np.sqrt(3),
            '2': 2.0,
            'e': np.e,
            'π': np.pi,
            '3': 3.0,
            '4': 4.0,
            '2φ': 2*(1 + np.sqrt(5))/2,
            'φ²': ((1 + np.sqrt(5))/2)**2
        }
        
        ratio_matches = []
        for ratio in all_ratios:
            best_match = min(constants.items(), 
                           key=lambda x: abs(x[1] - ratio))
            error = abs(best_match[1] - ratio) / ratio * 100
            ratio_matches.append((ratio, best_match[0], best_match[1], error))
            print(f"  {ratio:.3f} ≈ {best_match[0]} = {best_match[1]:.3f} "
                  f"(error: {error:.1f}%)")
        
        self.results['ratios'] = all_ratios
        self.results['matches'] = ratio_matches
        
        # Histogram
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.hist(all_ratios, bins=20, alpha=0.7, edgecolor='black')
        for name, value in constants.items():
            if 1 < value < 15:
                ax1.axvline(value, color='r', alpha=0.3, linestyle='--',
                          label=name if len(name) < 4 else None)
        ax1.set_xlabel('Ratio σ_(n+1)/σ_n')
        ax1.set_ylabel('Count')
        ax1.set_title('Distribution of Peak Ratios')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Scatter plot
        systems_list = []
        ratios_list = []
        for sys, data in self.peak_data.items():
            if len(data['peaks']) > 1:
                for i in range(len(data['peaks'])-1):
                    systems_list.append(sys)
                    ratios_list.append(data['peaks'][i+1]/data['peaks'][i])
        
        y_pos = np.arange(len(systems_list))
        ax2.barh(y_pos, ratios_list, alpha=0.7)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([f"{s}" for s in systems_list])
        ax2.set_xlabel('Ratio')
        ax2.set_title('Peak Ratios by System')
        ax2.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig('harmonic_structure.pdf', dpi=150, bbox_inches='tight')
        plt.show()
    
    def test_multiplicative_cascade(self):
        """Teste ob Hierarchie einer multiplikativen Kaskade folgt"""
        print("\n" + "-"*40)
        print("TEST: Multiplikative Kaskade")
        print("-"*40)
        
        # Hypothese: σ_n = σ_0 * r^n
        # log(σ_n) = log(σ_0) + n*log(r)
        
        for system, data in self.peak_data.items():
            peaks = data['peaks']
            if len(peaks) > 2:
                log_peaks = np.log(peaks)
                indices = np.arange(len(peaks))
                
                slope, intercept, r_value, p_value, std_err = \
                    linregress(indices, log_peaks)
                
                r_cascade = np.exp(slope)
                
                print(f"{system:10s}: r = {r_cascade:.3f}, "
                      f"R² = {r_value**2:.3f}, p = {p_value:.3f}")
                
                if r_value**2 > 0.9:
                    print(f"  → Konsistent mit multiplikativer Kaskade!")
    
    def find_universal_generator(self):
        """Suche nach universellem Generator der Hierarchie"""
        print("\n" + "-"*40)
        print("UNIVERSAL GENERATOR SEARCH")
        print("-"*40)
        
        # Sammle alle normalisierten Verhältnisse
        normalized_ratios = []
        
        for system, data in self.peak_data.items():
            peaks = np.array(data['peaks'])
            if len(peaks) > 1:
                # Normalisiere auf ersten Peak
                norm_peaks = peaks / peaks[0]
                normalized_ratios.extend(norm_peaks[1:])
        
        # Suche nach gemeinsamem Faktor (GCD-like für reelle Zahlen)
        def find_generator(values, tolerance=0.1):
            """Finde kleinsten gemeinsamen Generator"""
            # Teste verschiedene Kandidaten
            candidates = [np.sqrt(2), (1+np.sqrt(5))/2, np.e, 2.0, 3.0]
            
            best_gen = None
            best_error = float('inf')
            
            for gen in candidates:
                # Teste ob alle Werte ≈ gen^k für ganzzahlige k
                errors = []
                for val in values:
                    k_float = np.log(val) / np.log(gen)
                    k_int = round(k_float)
                    error = abs(val - gen**k_int) / val
                    errors.append(error)
                
                mean_error = np.mean(errors)
                if mean_error < best_error:
                    best_error = mean_error
                    best_gen = gen
            
            return best_gen, best_error
        
        generator, error = find_generator(normalized_ratios)
        
        print(f"Bester universeller Generator: {generator:.4f}")
        print(f"Mittlerer relativer Fehler: {error*100:.1f}%")
        
        if generator is not None:
            # Zeige wie gut der Generator funktioniert
            print("\nRekonstruktion mit Generator:")
            for ratio in sorted(set(normalized_ratios)):
                k = round(np.log(ratio) / np.log(generator))
                reconstructed = generator**k
                err = abs(ratio - reconstructed) / ratio * 100
                print(f"  {ratio:.3f} ≈ {generator:.3f}^{k} = "
                      f"{reconstructed:.3f} (error: {err:.1f}%)")
        
        self.results['generator'] = generator
        self.results['generator_error'] = error
    
    def test_golden_ratio_hypothesis(self):
        """Spezialtest: Goldener Schnitt in der Hierarchie"""
        print("\n" + "-"*40)
        print("GOLDEN RATIO HYPOTHESIS TEST")
        print("-"*40)
        
        phi = (1 + np.sqrt(5)) / 2
        
        # Teste Fibonacci-Sequenz in Peaks
        fib_ratios = [1, 1, 2, 3, 5, 8, 13, 21, 34]
        fib_normalized = [f/fib_ratios[0] for f in fib_ratios]
        
        print(f"φ = {phi:.6f}")
        print("\nFibonacci-Verhältnisse:")
        for i in range(1, len(fib_ratios)-1):
            ratio = fib_ratios[i+1] / fib_ratios[i]
            print(f"  F_{i+1}/F_{i} = {fib_ratios[i+1]}/{fib_ratios[i]} "
                  f"= {ratio:.3f} → φ")
        
        # Teste ob unsere Peaks Fibonacci-ähnlich sind
        print("\nTest auf Fibonacci-Struktur in Peaks:")
        
        for system, data in self.peak_data.items():
            peaks = data['peaks']
            if len(peaks) >= 2:
                # Teste ob Verhältnis ≈ φ^n
                for i, peak in enumerate(peaks):
                    for n in range(-2, 5):
                        if abs(peak - peaks[0] * phi**n) / peak < 0.15:
                            print(f"  {system}: σ_{i} ≈ σ_0 * φ^{n}")
                            break

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def test_observable_function(params):
    """Testfunktion für Bayesian Optimization"""
    # Simuliere Observable-Performance basierend auf Parametern
    gradient_weight = params[0]
    power_exp = params[1]
    smooth_scale = params[2]
    kernel_width = params[3]
    
    # Empirische Formel basierend auf unseren Ergebnissen
    kappa = 10 * gradient_weight**2 + \
            5 * np.exp(-(power_exp - 1.0)**2) + \
            3 * np.exp(-(smooth_scale - 0.8)**2) + \
            2 * np.exp(-(kernel_width - 1.5)**2) + \
            np.random.normal(0, 0.5)
    
    return max(0, kappa)

def main():
    """Hauptanalyse aller drei Reviewer-Punkte"""
    
    print("="*60)
    print("COMPLETE REVIEWER ANALYSIS")
    print("σ_c Framework: RG Theory, ML Selection, Harmonic Structure")
    print("="*60)
    
    # Create output directory
    import os
    os.makedirs('reviewer_analysis', exist_ok=True)
    os.chdir('reviewer_analysis')
    
    # ========================================
    # PART 1: Renormalization Group Analysis
    # ========================================
    print("\n" + "="*60)
    print("PART 1: RENORMALIZATION GROUP THEORY")
    print("="*60)
    
    rg = RGAnalysis()
    rg.test_scaling_hypothesis()
    rg.compute_beta_function()
    rg.test_universality_class()
    
    # ========================================
    # PART 2: Machine Learning Observable Selection  
    # ========================================
    print("\n" + "="*60)
    print("PART 2: MACHINE LEARNING OBSERVABLE SELECTION")
    print("="*60)
    
    ml = MLObservableOptimizer()
    best_params, best_kappa = ml.bayesian_optimization(
        test_observable_function, n_trials=30)
    ml.neural_predictor()
    
    # ========================================
    # PART 3: Harmonic Structure Analysis
    # ========================================
    print("\n" + "="*60)
    print("PART 3: HARMONIC STRUCTURE ANALYSIS")
    print("="*60)
    
    harmonic = HarmonicAnalysis()
    harmonic.analyze_ratios()
    harmonic.test_multiplicative_cascade()
    harmonic.find_universal_generator()
    harmonic.test_golden_ratio_hypothesis()
    
    # ========================================
    # FINAL SUMMARY
    # ========================================
    print("\n" + "="*60)
    print("SUMMARY OF FINDINGS")
    print("="*60)
    
    print("\n1. RENORMALIZATION GROUP:")
    print(f"   - Scaling exponent: {rg.results.get('scaling', {}).get('exponent', 'N/A'):.3f}")
    print(f"   - R² for scaling: {rg.results.get('scaling', {}).get('r_squared', 'N/A'):.3f}")
    print(f"   - Universality test p-value: {rg.results.get('universality', {}).get('p_value', 'N/A'):.4f}")
    if rg.results.get('beta_function'):
        print(f"   - Critical exponent ν: {rg.results['beta_function']['nu']:.3f}")
    
    print("\n2. MACHINE LEARNING:")
    print(f"   - Best κ achieved: {ml.results.get('best_kappa', 'N/A'):.2f}")
    if ml.results.get('best_params'):
        print("   - Optimal parameters:")
        for key, val in ml.results['best_params'].items():
            print(f"     * {key}: {val:.3f}")
    
    print("\n3. HARMONIC STRUCTURE:")
    if harmonic.results.get('ratios'):
        print(f"   - Number of ratios found: {len(harmonic.results['ratios'])}")
        print(f"   - Universal generator: {harmonic.results.get('generator', 'N/A'):.3f}")
        print(f"   - Generator error: {harmonic.results.get('generator_error', 'N/A')*100:.1f}%")
    
    # Save all results
    all_results = {
        'rg_analysis': rg.results,
        'ml_optimization': ml.results,
        'harmonic_analysis': harmonic.results
    }
    
    with open('reviewer_analysis_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print("\n" + "="*60)
    print("Analysis complete! Results saved to reviewer_analysis/")
    print("="*60)

if __name__ == "__main__":
    main()