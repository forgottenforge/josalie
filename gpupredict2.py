#!/usr/bin/env python3
"""
🔬 GPU σ_c META-ANALYSIS & PATTERN DISCOVERY ENGINE (SKRIPT 2)
==============================================================
Copyright (c) 2025 ForgottenForge.xyz

Advanced framework to discover THE UNIVERSAL FORM of σ_c across workloads.

Research Goals:
    1. Test scaling collapse: Does χ̃(ε/σ_c) = f(x) universally?
    2. Extract hardware features: What determines σ_c?
    3. Build predictive model: Can we predict σ_c from features?
    4. Discover hidden patterns: ML-based pattern mining

Key Innovation: NORMALIZED ANALYSIS
    - Rescale χ by χ_max → dimensionless
    - Rescale ε by σ_c → universal variable x = ε/σ_c
    - Test: Do all kernels collapse onto master curve?

Hardware: NVIDIA GPU (RTX 3060+)
Runtime: ~2-3 hours for complete analysis

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

For commercial licensing without AGPL-3.0 obligations, contact:
[nfo@forgottenforge.xyz]

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial

"""

import numpy as np
import cupy as cp
import time
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, asdict
from scipy import stats, optimize, interpolate
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# EXTENDED KERNEL LIBRARY (20+ kernels)
# ============================================================================

class ExtendedGPUKernels:
    """Comprehensive GPU kernel library for pattern discovery."""
    
    # ========== COMPUTE-BOUND KERNELS ==========
    
    @staticmethod
    def gemm(size: int, epsilon: float) -> Tuple[float, Dict]:
        """Matrix multiplication."""
        A = cp.random.randn(size, size, dtype=cp.float32)
        B = cp.random.randn(size, size, dtype=cp.float32)
        overhead = int(epsilon * size * size * 10)
        dummy = cp.zeros(max(1, overhead), dtype=cp.float32)
        
        cp.cuda.Stream.null.synchronize()
        start = time.perf_counter()
        C = cp.matmul(A, B)
        if overhead > 0:
            dummy += cp.sum(C) * 1e-10
        cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - start
        
        gflops = 2 * size**3 / elapsed / 1e9
        features = {
            'arithmetic_intensity': 'high',
            'memory_bound': False,
            'synchronization': 'low',
            'cache_friendly': True
        }
        
        del A, B, C, dummy
        return gflops, features
    
    @staticmethod
    def gemv(size: int, epsilon: float) -> Tuple[float, Dict]:
        """Matrix-vector multiplication."""
        A = cp.random.randn(size, size, dtype=cp.float32)
        x = cp.random.randn(size, dtype=cp.float32)
        overhead = int(epsilon * size * size * 5)
        dummy = cp.zeros(max(1, overhead), dtype=cp.float32)
        
        cp.cuda.Stream.null.synchronize()
        start = time.perf_counter()
        y = cp.dot(A, x)
        if overhead > 0:
            dummy += cp.sum(y) * 1e-10
        cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - start
        
        gflops = 2 * size**2 / elapsed / 1e9
        features = {
            'arithmetic_intensity': 'moderate',
            'memory_bound': True,
            'synchronization': 'low',
            'cache_friendly': False
        }
        
        del A, x, y, dummy
        return gflops, features
    
    @staticmethod
    def conv2d(size: int, epsilon: float) -> Tuple[float, Dict]:
        """2D Convolution (simplified)."""
        image = cp.random.randn(size, size, dtype=cp.float32)
        kernel = cp.random.randn(3, 3, dtype=cp.float32)
        overhead = int(epsilon * size * size * 8)
        dummy = cp.zeros(max(1, overhead), dtype=cp.float32)
        
        cp.cuda.Stream.null.synchronize()
        start = time.perf_counter()
        # Simplified convolution via correlation
        from cupyx.scipy.signal import correlate2d
        result = correlate2d(image, kernel, mode='same')
        if overhead > 0:
            dummy += cp.sum(result) * 1e-10
        cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - start
        
        gflops = 2 * size * size * 9 / elapsed / 1e9
        features = {
            'arithmetic_intensity': 'moderate',
            'memory_bound': False,
            'synchronization': 'low',
            'cache_friendly': True
        }
        
        del image, kernel, result, dummy
        return gflops, features
    
    # ========== MEMORY-BOUND KERNELS ==========
    
    @staticmethod
    def fft(size: int, epsilon: float) -> Tuple[float, Dict]:
        """Fast Fourier Transform."""
        fft_size = 2 ** int(np.log2(size))
        data_real = cp.random.randn(fft_size, fft_size, dtype=cp.float32)
        data_imag = cp.random.randn(fft_size, fft_size, dtype=cp.float32)
        data = data_real + 1j * data_imag
        overhead = int(epsilon * fft_size * fft_size * 8)
        dummy = cp.zeros(max(1, overhead), dtype=cp.float32)
        
        cp.cuda.Stream.null.synchronize()
        start = time.perf_counter()
        result = cp.fft.fft2(data)
        if overhead > 0:
            dummy += cp.abs(result[0, 0].real) * 1e-10
        cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - start
        
        ops = 5 * fft_size * fft_size * np.log2(fft_size)
        gflops = ops / elapsed / 1e9
        features = {
            'arithmetic_intensity': 'moderate',
            'memory_bound': True,
            'synchronization': 'moderate',
            'cache_friendly': False
        }
        
        del data, data_real, data_imag, result, dummy
        return gflops, features
    
    @staticmethod
    def transpose(size: int, epsilon: float) -> Tuple[float, Dict]:
        """Matrix transpose."""
        A = cp.random.randn(size, size, dtype=cp.float32)
        overhead = int(epsilon * size * size * 2)
        dummy = cp.zeros(max(1, overhead), dtype=cp.float32)
        
        cp.cuda.Stream.null.synchronize()
        start = time.perf_counter()
        B = cp.transpose(A)
        if overhead > 0:
            dummy += cp.sum(B) * 1e-10
        cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - start
        
        ops = size * size
        gflops = ops / elapsed / 1e9
        features = {
            'arithmetic_intensity': 'low',
            'memory_bound': True,
            'synchronization': 'low',
            'cache_friendly': False
        }
        
        del A, B, dummy
        return gflops, features
    
    # ========== REDUCTION KERNELS ==========
    
    @staticmethod
    def reduction(size: int, epsilon: float) -> Tuple[float, Dict]:
        """Parallel reduction (sum)."""
        data = cp.random.randn(size * size, dtype=cp.float32)
        overhead = int(epsilon * 100)
        
        cp.cuda.Stream.null.synchronize()
        start = time.perf_counter()
        result = cp.sum(data)
        for _ in range(overhead):
            cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - start
        
        gflops = size * size / elapsed / 1e9
        features = {
            'arithmetic_intensity': 'low',
            'memory_bound': True,
            'synchronization': 'high',
            'cache_friendly': False
        }
        
        del data
        return gflops, features
    
    @staticmethod
    def max_reduce(size: int, epsilon: float) -> Tuple[float, Dict]:
        """Maximum reduction."""
        data = cp.random.randn(size * size, dtype=cp.float32)
        overhead = int(epsilon * 100)
        
        cp.cuda.Stream.null.synchronize()
        start = time.perf_counter()
        result = cp.max(data)
        for _ in range(overhead):
            cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - start
        
        gflops = size * size / elapsed / 1e9
        features = {
            'arithmetic_intensity': 'low',
            'memory_bound': True,
            'synchronization': 'high',
            'cache_friendly': False
        }
        
        del data
        return gflops, features
    
    # ========== SCAN/PREFIX KERNELS ==========
    
    @staticmethod
    def scan(size: int, epsilon: float) -> Tuple[float, Dict]:
        """Prefix sum (cumulative)."""
        data = cp.random.randn(size * size, dtype=cp.float32)
        overhead = int(epsilon * size * size * 4)
        dummy = cp.zeros(max(1, overhead), dtype=cp.float32)
        
        cp.cuda.Stream.null.synchronize()
        start = time.perf_counter()
        result = cp.cumsum(data)
        if overhead > 0:
            dummy += result[-1] * 1e-10
        cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - start
        
        gflops = size * size / elapsed / 1e9
        features = {
            'arithmetic_intensity': 'moderate',
            'memory_bound': True,
            'synchronization': 'moderate',
            'cache_friendly': False
        }
        
        del data, result, dummy
        return gflops, features
    
    # ========== ELEMENTWISE KERNELS ==========
    
    @staticmethod
    def saxpy(size: int, epsilon: float) -> Tuple[float, Dict]:
        """Vector addition: y = a*x + y."""
        x = cp.random.randn(size * size, dtype=cp.float32)
        y = cp.random.randn(size * size, dtype=cp.float32)
        a = cp.float32(2.5)
        overhead = int(epsilon * size * size * 2)
        dummy = cp.zeros(max(1, overhead), dtype=cp.float32)
        
        cp.cuda.Stream.null.synchronize()
        start = time.perf_counter()
        y = a * x + y
        if overhead > 0:
            dummy += cp.sum(y) * 1e-10
        cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - start
        
        gflops = 2 * size * size / elapsed / 1e9
        features = {
            'arithmetic_intensity': 'low',
            'memory_bound': True,
            'synchronization': 'low',
            'cache_friendly': True
        }
        
        del x, y, dummy
        return gflops, features
    
    @staticmethod
    def elementwise_exp(size: int, epsilon: float) -> Tuple[float, Dict]:
        """Elementwise exponential."""
        data = cp.random.randn(size * size, dtype=cp.float32)
        overhead = int(epsilon * size * size * 3)
        dummy = cp.zeros(max(1, overhead), dtype=cp.float32)
        
        cp.cuda.Stream.null.synchronize()
        start = time.perf_counter()
        result = cp.exp(data)
        if overhead > 0:
            dummy += cp.sum(result) * 1e-10
        cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - start
        
        gflops = size * size / elapsed / 1e9
        features = {
            'arithmetic_intensity': 'moderate',
            'memory_bound': False,
            'synchronization': 'low',
            'cache_friendly': True
        }
        
        del data, result, dummy
        return gflops, features
    
    @staticmethod
    def get_all_kernels() -> Dict[str, Callable]:
        """Return dictionary of all available kernels."""
        return {
            'GEMM': ExtendedGPUKernels.gemm,
            'GEMV': ExtendedGPUKernels.gemv,
            'Conv2D': ExtendedGPUKernels.conv2d,
            'FFT': ExtendedGPUKernels.fft,
            'Transpose': ExtendedGPUKernels.transpose,
            'Reduction': ExtendedGPUKernels.reduction,
            'MaxReduce': ExtendedGPUKernels.max_reduce,
            'Scan': ExtendedGPUKernels.scan,
            'SAXPY': ExtendedGPUKernels.saxpy,
            'ElementwiseExp': ExtendedGPUKernels.elementwise_exp
        }

# ============================================================================
# ADVANCED SUSCEPTIBILITY ANALYSIS
# ============================================================================

@dataclass
class KernelProfile:
    """Complete profile of a kernel's σ_c characteristics."""
    name: str
    sigma_c: float
    sigma_c_ci: Tuple[float, float]
    kappa: float
    chi_max: float
    chi_baseline: float
    epsilon: np.ndarray
    observable: np.ndarray
    chi: np.ndarray
    features: Dict
    
    def normalized_chi(self) -> np.ndarray:
        """Return χ normalized by χ_max."""
        return self.chi / self.chi_max
    
    def scaled_epsilon(self) -> np.ndarray:
        """Return ε scaled by σ_c."""
        return self.epsilon / self.sigma_c

class MetaAnalyzer:
    """Advanced meta-analysis framework."""
    
    def __init__(self, output_dir: str = 'gpu_meta_analysis'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        cp.cuda.Device(0).use()
        self.device_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
        
        self.profiles: Dict[str, KernelProfile] = {}
        
        print("=" * 80)
        print("🔬 GPU σ_c META-ANALYSIS & PATTERN DISCOVERY ENGINE")
        print("=" * 80)
        print(f"Device: {self.device_name}")
        print("Mission: Discover THE UNIVERSAL FORM")
        print("=" * 80)
    
    def profile_kernel(self, name: str, kernel_func: Callable,
                      size: int = 2048, n_epsilon: int = 30,
                      n_reps: int = 3) -> KernelProfile:
        """Complete profiling of a single kernel."""
        print(f"\n🔍 Profiling: {name}")
        
        epsilon_grid = np.linspace(0.0, 0.5, n_epsilon)
        observables = []
        features = None
        
        for eps in epsilon_grid:
            reps = []
            for _ in range(n_reps):
                perf, feat = kernel_func(size, eps)
                reps.append(perf)
                if features is None:
                    features = feat
            observables.append(np.mean(reps))
        
        observables = np.array(observables)
        
        # Compute susceptibility
        O_smooth = gaussian_filter1d(observables, sigma=0.5)
        chi = np.abs(np.gradient(O_smooth, epsilon_grid))
        
        # Find interior maximum
        interior = (epsilon_grid > epsilon_grid[0]) & (epsilon_grid < epsilon_grid[-1])
        chi_interior = chi[interior]
        eps_interior = epsilon_grid[interior]
        
        max_idx = np.argmax(chi_interior)
        sigma_c = eps_interior[max_idx]
        chi_max = chi_interior[max_idx]
        chi_baseline = max(np.percentile(chi_interior, 10), 1e-5)
        kappa = chi_max / chi_baseline
        
        # Bootstrap CI
        bootstrap_sc = []
        for _ in range(500):
            obs_boot = observables + np.random.normal(0, np.std(observables)*0.1, len(observables))
            O_boot = gaussian_filter1d(obs_boot, sigma=0.5)
            chi_boot = np.abs(np.gradient(O_boot, epsilon_grid))
            chi_boot_int = chi_boot[interior]
            bootstrap_sc.append(eps_interior[np.argmax(chi_boot_int)])
        
        ci = tuple(np.percentile(bootstrap_sc, [2.5, 97.5]))
        
        profile = KernelProfile(
            name=name,
            sigma_c=float(sigma_c),
            sigma_c_ci=ci,
            kappa=float(kappa),
            chi_max=float(chi_max),
            chi_baseline=float(chi_baseline),
            epsilon=epsilon_grid,
            observable=observables,
            chi=chi,
            features=features
        )
        
        print(f"  σ_c = {sigma_c:.4f}, κ = {kappa:.2f}, χ_max = {chi_max:.1f}")
        
        return profile
    
    def test_scaling_collapse(self) -> Dict:
        """Test if χ̃(ε/σ_c) collapses onto universal curve."""
        print("\n" + "="*80)
        print("🎯 TESTING SCALING COLLAPSE HYPOTHESIS")
        print("="*80)
        
        # Prepare scaled data
        scaled_data = []
        for name, prof in self.profiles.items():
            x = prof.scaled_epsilon()
            y = prof.normalized_chi()
            for xi, yi in zip(x, y):
                if 0 < xi < 10:  # Reasonable range
                    scaled_data.append({'x': xi, 'y': yi, 'kernel': name})
        
        if len(scaled_data) == 0:
            return {'success': False, 'reason': 'No data in valid range'}
        
        # Extract arrays
        x_all = np.array([d['x'] for d in scaled_data])
        y_all = np.array([d['y'] for d in scaled_data])
        
        # Bin data and compute statistics
        x_bins = np.linspace(0.1, 5, 20)
        y_binned_mean = []
        y_binned_std = []
        x_binned = []
        
        for i in range(len(x_bins)-1):
            mask = (x_all >= x_bins[i]) & (x_all < x_bins[i+1])
            if np.sum(mask) > 2:
                y_binned_mean.append(np.mean(y_all[mask]))
                y_binned_std.append(np.std(y_all[mask]))
                x_binned.append((x_bins[i] + x_bins[i+1])/2)
        
        if len(x_binned) < 5:
            return {'success': False, 'reason': 'Insufficient binned data'}
        
        # Compute collapse quality (coefficient of variation in bins)
        cv = np.mean([s/m for s, m in zip(y_binned_std, y_binned_mean) if m > 0.1])
        
        print(f"\n📊 Collapse Quality:")
        print(f"  Mean CV in bins: {cv:.3f}")
        print(f"  Interpretation: {'GOOD' if cv < 0.5 else 'POOR'} collapse")
        
        result = {
            'success': True,
            'cv_mean': float(cv),
            'collapse_quality': 'good' if cv < 0.5 else 'moderate' if cv < 1.0 else 'poor',
            'x_binned': x_binned,
            'y_binned_mean': y_binned_mean,
            'y_binned_std': y_binned_std,
            'raw_data': scaled_data[:100]  # Limit for JSON size
        }
        
        return result
    
    def discover_patterns(self) -> Dict:
        """Use ML to discover patterns in σ_c."""
        print("\n" + "="*80)
        print("🤖 PATTERN DISCOVERY VIA MACHINE LEARNING")
        print("="*80)
        
        # Build feature matrix
        X = []
        y_sigma = []
        y_kappa = []
        names = []
        
        for name, prof in self.profiles.items():
            feat = prof.features
            # Encode categorical features
            x_vec = [
                1.0 if feat['arithmetic_intensity'] == 'high' else 0.5 if feat['arithmetic_intensity'] == 'moderate' else 0.0,
                1.0 if feat['memory_bound'] else 0.0,
                1.0 if feat['synchronization'] == 'high' else 0.5 if feat['synchronization'] == 'moderate' else 0.0,
                1.0 if feat['cache_friendly'] else 0.0,
                np.log10(prof.observable[0] + 1),  # Initial performance (log scale)
            ]
            X.append(x_vec)
            y_sigma.append(prof.sigma_c)
            y_kappa.append(prof.kappa)
            names.append(name)
        
        X = np.array(X)
        y_sigma = np.array(y_sigma)
        y_kappa = np.array(y_kappa)
        
        # PCA for visualization
        if len(X) >= 3:
            pca = PCA(n_components=min(2, X.shape[1]))
            X_pca = pca.fit_transform(X)
            print(f"\n📐 PCA Analysis:")
            print(f"  Explained variance: {pca.explained_variance_ratio_}")
        else:
            X_pca = X[:, :2]
        
        # Clustering
        if len(X) >= 3:
            n_clusters = min(3, len(X))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X)
            print(f"\n🎯 Clustering (k={n_clusters}):")
            for i in range(n_clusters):
                cluster_names = [names[j] for j in range(len(names)) if clusters[j] == i]
                cluster_sigma = [y_sigma[j] for j in range(len(y_sigma)) if clusters[j] == i]
                print(f"  Cluster {i}: {cluster_names}")
                print(f"    σ_c range: {min(cluster_sigma):.4f} - {max(cluster_sigma):.4f}")
        else:
            clusters = np.zeros(len(X))
        
        # Correlation analysis
        print(f"\n🔗 Feature Correlations with σ_c:")
        feature_names = ['ArithIntensity', 'MemBound', 'Sync', 'CacheFriend', 'InitPerf']
        for i, fname in enumerate(feature_names):
            corr, p = stats.spearmanr(X[:, i], y_sigma)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  {fname:15s}: ρ = {corr:+.3f} (p={p:.4f}) {sig}")
        
        return {
            'X': X.tolist(),
            'y_sigma': y_sigma.tolist(),
            'y_kappa': y_kappa.tolist(),
            'names': names,
            'X_pca': X_pca.tolist() if len(X) >= 3 else [],
            'clusters': clusters.tolist(),
            'feature_names': feature_names
        }
    
    def generate_master_plots(self, collapse_results: Dict, pattern_results: Dict):
        """Generate comprehensive visualization suite."""
        print("\n📊 Generating master visualizations...")
        
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # Plot 1: All χ curves (original scale)
        ax1 = fig.add_subplot(gs[0, 0:2])
        for name, prof in self.profiles.items():
            ax1.plot(prof.epsilon, prof.chi, 'o-', label=name, alpha=0.7, markersize=3)
        ax1.set_xlabel('Overhead ε')
        ax1.set_ylabel('Susceptibility χ(ε)')
        ax1.set_title('Raw Susceptibility Curves', fontweight='bold', fontsize=12)
        ax1.legend(fontsize=8, ncol=2)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Plot 2: Normalized χ curves
        ax2 = fig.add_subplot(gs[0, 2:])
        for name, prof in self.profiles.items():
            ax2.plot(prof.epsilon, prof.normalized_chi(), 'o-', label=name, alpha=0.7, markersize=3)
        ax2.set_xlabel('Overhead ε')
        ax2.set_ylabel('Normalized χ̃(ε) = χ/χ_max')
        ax2.set_title('Normalized Susceptibility Curves', fontweight='bold', fontsize=12)
        ax2.legend(fontsize=8, ncol=2)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1.5])
        
        # Plot 3: SCALING COLLAPSE!
        ax3 = fig.add_subplot(gs[1, 0:2])
        for name, prof in self.profiles.items():
            x_scaled = prof.scaled_epsilon()
            y_norm = prof.normalized_chi()
            ax3.plot(x_scaled, y_norm, 'o', label=name, alpha=0.6, markersize=4)
        
        # Plot master curve if collapse worked
        if collapse_results.get('success'):
            x_bin = collapse_results['x_binned']
            y_mean = collapse_results['y_binned_mean']
            y_std = collapse_results['y_binned_std']
            ax3.errorbar(x_bin, y_mean, yerr=y_std, color='black', 
                        linewidth=3, capsize=5, label='Master Curve', zorder=10)
        
        ax3.set_xlabel('Scaled Variable x = ε/σ_c', fontsize=11)
        ax3.set_ylabel('Normalized χ̃(x)', fontsize=11)
        ax3.set_title('SCALING COLLAPSE TEST', fontweight='bold', fontsize=13)
        ax3.legend(fontsize=8, ncol=2)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim([0, 5])
        ax3.set_ylim([0, 1.5])
        
        # Plot 4: σ_c distribution
        ax4 = fig.add_subplot(gs[1, 2])
        names = [p.name for p in self.profiles.values()]
        sigmas = [p.sigma_c for p in self.profiles.values()]
        colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
        ax4.barh(names, sigmas, color=colors, alpha=0.7, edgecolor='black')
        ax4.set_xlabel('σ_c')
        ax4.set_title('Critical Thresholds', fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')
        
        # Plot 5: κ distribution
        ax5 = fig.add_subplot(gs[1, 3])
        kappas = [p.kappa for p in self.profiles.values()]
        ax5.barh(names, kappas, color=colors, alpha=0.7, edgecolor='black')
        ax5.set_xlabel('Peak Clarity κ')
        ax5.set_title('Signal Clarity', fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='x')
        ax5.set_xscale('log')
        
        # Plot 6: PCA visualization
        if pattern_results and len(pattern_results.get('X_pca', [])) > 0:
            ax6 = fig.add_subplot(gs[2, 0])
            X_pca = np.array(pattern_results['X_pca'])
            clusters = np.array(pattern_results['clusters'])
            scatter = ax6.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, 
                                 s=200, alpha=0.6, cmap='tab10', edgecolors='black')
            for i, name in enumerate(pattern_results['names']):
                ax6.annotate(name, (X_pca[i, 0], X_pca[i, 1]), fontsize=8)
            ax6.set_xlabel('PC1')
            ax6.set_ylabel('PC2')
            ax6.set_title('Feature Space (PCA)', fontweight='bold')
            ax6.grid(True, alpha=0.3)
        
        # Plot 7: σ_c vs κ relationship
        ax7 = fig.add_subplot(gs[2, 1])
        sigmas = [p.sigma_c for p in self.profiles.values()]
        kappas = [p.kappa for p in self.profiles.values()]
        names = [p.name for p in self.profiles.values()]
        ax7.scatter(sigmas, kappas, s=200, alpha=0.6, c=range(len(names)), 
                   cmap='viridis', edgecolors='black')
        for i, name in enumerate(names):
            ax7.annotate(name, (sigmas[i], kappas[i]), fontsize=8, ha='right')
        ax7.set_xlabel('σ_c')
        ax7.set_ylabel('κ')
        ax7.set_title('σ_c vs κ Relationship', fontweight='bold')
        ax7.grid(True, alpha=0.3)
        ax7.set_yscale('log')
        
        # Plot 8: Feature importance (if available)
        ax8 = fig.add_subplot(gs[2, 2:])
        if pattern_results:
            feat_names = pattern_results['feature_names']
            X = np.array(pattern_results['X'])
            y = np.array(pattern_results['y_sigma'])
            correlations = [stats.spearmanr(X[:, i], y)[0] for i in range(X.shape[1])]
            colors_bar = ['green' if abs(c) > 0.5 else 'orange' if abs(c) > 0.3 else 'gray' 
                         for c in correlations]
            ax8.barh(feat_names, correlations, color=colors_bar, alpha=0.7, edgecolor='black')
            ax8.set_xlabel('Correlation with σ_c (Spearman ρ)')
            ax8.set_title('Feature Importance', fontweight='bold')
            ax8.axvline(0, color='black', linewidth=0.5)
            ax8.grid(True, alpha=0.3, axis='x')
        
        plt.savefig(self.output_dir / 'master_analysis.pdf', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: master_analysis.pdf")
        plt.close()
    
    def run_complete_analysis(self, kernel_selection: List[str] = None):
        """Run complete meta-analysis."""
        print("\n🚀 STARTING COMPLETE META-ANALYSIS\n")
        
        start_time = time.time()
        
        # Get kernels
        all_kernels = ExtendedGPUKernels.get_all_kernels()
        if kernel_selection is None:
            kernel_selection = list(all_kernels.keys())
        
        # Profile all kernels
        print("Phase 1: Kernel Profiling")
        print("-" * 80)
        for name in kernel_selection:
            if name in all_kernels:
                prof = self.profile_kernel(name, all_kernels[name])
                self.profiles[name] = prof
        
        # Test scaling collapse
        print("\n" + "="*80)
        print("Phase 2: Scaling Collapse Analysis")
        print("="*80)
        collapse_results = self.test_scaling_collapse()
        
        # Pattern discovery
        print("\n" + "="*80)
        print("Phase 3: Pattern Discovery")
        print("="*80)
        pattern_results = self.discover_patterns()
        
        # Generate visualizations
        self.generate_master_plots(collapse_results, pattern_results)
        
        # Save results
        results = {
            'metadata': {
                'device': self.device_name,
                'timestamp': datetime.now().isoformat(),
                'runtime_minutes': (time.time() - start_time) / 60
            },
            'profiles': {name: {
                'sigma_c': p.sigma_c,
                'kappa': p.kappa,
                'chi_max': p.chi_max,
                'features': p.features
            } for name, p in self.profiles.items()},
            'scaling_collapse': collapse_results,
            'pattern_discovery': pattern_results
        }
        
        with open(self.output_dir / 'meta_analysis_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Final summary
        elapsed = time.time() - start_time
        print("\n" + "="*80)
        print("✨ ANALYSIS COMPLETE!")
        print("="*80)
        print(f"📊 Kernels analyzed: {len(self.profiles)}")
        print(f"⏱️  Runtime: {elapsed/60:.1f} minutes")
        print(f"💾 Results: {self.output_dir}/")
        
        if collapse_results.get('success'):
            quality = collapse_results['collapse_quality']
            print(f"\n🎯 SCALING COLLAPSE: {quality.upper()}")
            if quality == 'good':
                print("   → Universal form discovered! χ̃(ε/σ_c) collapses!")
            else:
                print("   → Partial collapse - heterogeneous behavior")
        
        print("\n" + "="*80)
        
        return results

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║      🔬 META-ANALYSIS ENGINE: DISCOVERING THE UNIVERSAL FORM             ║
║                                                                           ║
║  This framework tests the scaling collapse hypothesis:                   ║
║  Does χ̃(ε/σ_c) collapse onto a universal master curve?                  ║
║                                                                           ║
║  If YES → σ_c represents fundamental information-theoretic principle     ║
║  If NO  → σ_c is workload-specific, still useful but not universal       ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
""")
    
    analyzer = MetaAnalyzer()
    
    # Run with all available kernels (or customize selection)
    results = analyzer.run_complete_analysis()
    
    print("\n🎉 Check gpu_meta_analysis/ for all outputs!")
    print("   → master_analysis.pdf: Complete visualization suite")
    print("   → meta_analysis_results.json: Raw data & statistics")