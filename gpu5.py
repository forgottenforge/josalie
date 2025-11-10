#!/usr/bin/env python3
"""
🎯 GPU CHAOS-CRITICALITY MAPPER v1.0
====================================
Copyright (c) 2025 ForgottenForge.xyz

Multi-dimensional mapping of σ_c emergence vs chaos levels.

Key Insight:
σ_c is a CHAOS-DEPENDENT order parameter. This framework maps:
- WHERE σ_c emerges reliably (low-chaos "islands of order")
- WHERE chaos dominates (high-variance regimes)
- HOW hardware state affects criticality

Output:
- 3D σ_c landscape (size × thermal × memory)
- Chaos quantification for each regime
- Practical optimization recommendations

Runtime: ~12-24 hours for comprehensive mapping

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
import subprocess
from datetime import datetime
from typing import Dict, List, Tuple
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import griddata
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({'font.family': 'serif', 'font.size': 10, 'figure.dpi': 300})


# ============================================================================
# CHAOS QUANTIFICATION
# ============================================================================

class ChaosAnalyzer:
    """Quantify chaos level during measurements."""
    
    @staticmethod
    def compute_chaos_index(thermal_history: List[Dict],
                           performance_data: List[float]) -> Dict:
        """
        Compute multi-dimensional chaos index.
        
        Components:
        1. Clock variance (frequency instability)
        2. Thermal variance (temperature instability)
        3. Performance variance (measurement noise)
        4. Temporal correlation (predictability)
        
        Returns:
            Dict with chaos metrics and combined index
        """
        # Extract time series
        clocks = np.array([h['clock_after'] for h in thermal_history])
        temps = np.array([h['temp_after'] for h in thermal_history])
        perf = np.array(performance_data)
        
        # 1. Clock chaos (CV of clock speeds)
        clock_mean = np.mean(clocks)
        clock_cv = np.std(clocks) / clock_mean if clock_mean > 0 else 0
        
        # 2. Thermal chaos (CV of temperatures)
        temp_mean = np.mean(temps)
        temp_cv = np.std(temps) / temp_mean if temp_mean > 0 else 0
        
        # 3. Performance chaos (CV of measurements)
        perf_mean = np.mean(perf)
        perf_cv = np.std(perf) / perf_mean if perf_mean > 0 else 0
        
        # 4. Temporal predictability (autocorrelation)
        if len(perf) > 2:
            autocorr = np.corrcoef(perf[:-1], perf[1:])[0, 1]
            predictability = max(0, autocorr)  # 1=predictable, 0=random
        else:
            predictability = 0
        
        # Combined chaos index (weighted average)
        # Higher values = more chaos
        chaos_index = (
            0.4 * clock_cv +      # Clock instability (most important)
            0.2 * temp_cv +       # Thermal instability
            0.3 * perf_cv +       # Performance noise
            0.1 * (1 - predictability)  # Unpredictability
        )
        
        # Chaos level classification
        if chaos_index < 0.05:
            chaos_level = "LOW"
        elif chaos_index < 0.15:
            chaos_level = "MEDIUM"
        else:
            chaos_level = "HIGH"
        
        return {
            'chaos_index': float(chaos_index),
            'chaos_level': chaos_level,
            'clock_cv': float(clock_cv),
            'temp_cv': float(temp_cv),
            'perf_cv': float(perf_cv),
            'predictability': float(predictability),
            'clock_range': [float(min(clocks)), float(max(clocks))],
            'temp_range': [float(min(temps)), float(max(temps))]
        }
    
    @staticmethod
    def classify_stability(sigma_c_cv: float) -> str:
        """Classify σ_c stability level."""
        if sigma_c_cv < 0.05:
            return "STABLE"      # CV < 5%: Reliable optimization regime
        elif sigma_c_cv < 0.10:
            return "MARGINAL"    # CV 5-10%: Usable with caution
        else:
            return "CHAOTIC"     # CV > 10%: Avoid for optimization


# ============================================================================
# ENHANCED GPU CONTROLLER
# ============================================================================

class GPUController:
    """Minimal GPU control for chaos mapping."""
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
    
    def set_thermal_state(self, target: str) -> bool:
        """
        Set thermal state: 'cool', 'normal', 'warm'.
        
        Returns True if state achieved.
        """
        target_temps = {
            'cool': 50.0,
            'normal': 55.0,
            'warm': 65.0
        }
        
        max_temp = target_temps.get(target, 55.0)
        print(f"    Setting thermal state: {target} (<{max_temp}°C)")
        
        start = time.time()
        while time.time() - start < 300:  # 5 min max
            temp = self.get_temperature()
            if temp <= max_temp:
                print(f"    ✓ Thermal state achieved: {temp:.1f}°C")
                return True
            time.sleep(2)
        
        print(f"    ⚠️  Timeout, proceeding anyway")
        return False
    
    def get_temperature(self) -> float:
        """Get current GPU temperature."""
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=temperature.gpu',
                '--format=csv,noheader,nounits',
                f'--id={self.device_id}'
            ], capture_output=True, text=True, timeout=2)
            
            if result.returncode == 0:
                return float(result.stdout.strip())
        except:
            pass
        return 50.0
    
    def get_state(self) -> Dict:
        """Get full GPU state."""
        try:
            result = subprocess.run([
                'nvidia-smi',
                '--query-gpu=temperature.gpu,clocks.current.graphics,power.draw',
                '--format=csv,noheader,nounits',
                f'--id={self.device_id}'
            ], capture_output=True, text=True, timeout=2)
            
            if result.returncode == 0:
                temp, clock, power = result.stdout.strip().split(', ')
                return {
                    'temperature': float(temp),
                    'clock_after': float(clock),
                    'power_watts': float(power)
                }
        except:
            pass
        
        return {'temperature': 50.0, 'clock_after': 210.0, 'power_watts': 0.0}


# ============================================================================
# CHAOS-CRITICALITY MAPPER
# ============================================================================

class ChaosCriticalityMapper:
    """Map σ_c emergence across multi-dimensional parameter space."""
    
    def __init__(self, device_id: int = 0):
        cp.cuda.Device(device_id).use()
        props = cp.cuda.runtime.getDeviceProperties(device_id)
        self.device_name = props['name'].decode()
        
        self.gpu_control = GPUController(device_id)
        self.chaos_analyzer = ChaosAnalyzer()
        
        self.results = {
            'device': self.device_name,
            'timestamp': datetime.now().isoformat(),
            'parameter_space': [],
            'chaos_map': []
        }
        
        self.output_dir = Path("gpu_chaos_mapping")
        self.output_dir.mkdir(exist_ok=True)
        
        print("=" * 80)
        print("🎯 GPU CHAOS-CRITICALITY MAPPER")
        print("=" * 80)
        print(f"Device: {self.device_name}")
        print(f"Mission: Map σ_c emergence vs chaos levels")
        print("=" * 80)
    
    def measure_sigma_c_with_chaos(self,
                                   size: int,
                                   thermal_state: str,
                                   memory_alpha: float,
                                   n_passes: int = 5) -> Dict:
        """
        Measure σ_c with full chaos quantification.
        
        Args:
            size: Matrix size
            thermal_state: 'cool', 'normal', 'warm'
            memory_alpha: Memory pressure multiplier
            n_passes: Validation passes (5-10)
        
        Returns:
            Dict with σ_c, stability, chaos metrics
        """
        print(f"\n  📍 Config: size={size}, thermal={thermal_state}, α={memory_alpha:.2f}")
        
        # Set thermal state
        self.gpu_control.set_thermal_state(thermal_state)
        
        # Multi-pass measurement
        sigma_c_values = []
        chaos_metrics = []
        
        for pass_num in range(n_passes):
            # Quick measurement with chaos tracking
            result = self._single_measurement(size, memory_alpha)
            
            sigma_c_values.append(result['sigma_c'])
            chaos_metrics.append(result['chaos_metrics'])
            
            if pass_num < n_passes - 1:
                time.sleep(10)  # Brief cooldown
        
        # Statistical analysis
        sigma_c_array = np.array(sigma_c_values)
        sigma_c_median = np.median(sigma_c_array)
        sigma_c_mad = np.median(np.abs(sigma_c_array - sigma_c_median))
        sigma_c_cv = sigma_c_mad / sigma_c_median if sigma_c_median > 0 else 1.0
        
        # Average chaos metrics
        avg_chaos = {
            key: np.mean([m[key] for m in chaos_metrics])
            for key in ['chaos_index', 'clock_cv', 'temp_cv', 'perf_cv', 'predictability']
        }
        avg_chaos['chaos_level'] = self.chaos_analyzer.classify_stability(sigma_c_cv)
        
        # Stability classification
        stability = self.chaos_analyzer.classify_stability(sigma_c_cv)
        
        result = {
            'size': size,
            'thermal_state': thermal_state,
            'memory_alpha': memory_alpha,
            'sigma_c_median': float(sigma_c_median),
            'sigma_c_mad': float(sigma_c_mad),
            'sigma_c_cv': float(sigma_c_cv),
            'sigma_c_values': [float(x) for x in sigma_c_values],
            'stability': stability,
            'chaos_metrics': avg_chaos,
            'n_passes': n_passes
        }
        
        print(f"    σ_c = {sigma_c_median:.3f} ± {sigma_c_mad:.3f} (CV={sigma_c_cv*100:.1f}%)")
        print(f"    Stability: {stability}, Chaos: {avg_chaos['chaos_level']} (index={avg_chaos['chaos_index']:.3f})")
        
        return result
    
    def _single_measurement(self, size: int, memory_alpha: float) -> Dict:
        """Single quick σ_c measurement with chaos tracking."""
        # Fast grid (11 points)
        epsilon = np.linspace(0.0, 1.0, 11)
        
        # Baseline
        A = cp.random.random((size, size), dtype=cp.float32)
        B = cp.random.random((size, size), dtype=cp.float32)
        
        times_baseline = []
        for _ in range(3):
            cp.cuda.runtime.deviceSynchronize()
            start = time.perf_counter()
            C = cp.dot(A, B)
            cp.cuda.runtime.deviceSynchronize()
            times_baseline.append(time.perf_counter() - start)
        
        flops = 2 * size**3
        gflops_baseline = flops / (np.mean(times_baseline) * 1e9)
        del A, B, C
        
        # Sweep with chaos tracking
        perf_ratios = []
        thermal_history = []
        
        for eps in epsilon:
            state_before = self.gpu_control.get_state()
            
            # Overhead injection
            n_mem = int(eps * 12 * memory_alpha)
            mem_size = max(1, size // 4)
            
            mem = [cp.random.random((mem_size, mem_size), dtype=cp.float32)
                   for _ in range(n_mem)]
            for m in mem:
                _ = cp.sum(m)
            
            A = cp.random.random((size, size), dtype=cp.float32)
            B = cp.random.random((size, size), dtype=cp.float32)
            
            times = []
            for _ in range(3):
                cp.cuda.runtime.deviceSynchronize()
                start = time.perf_counter()
                C = cp.dot(A, B)
                cp.cuda.runtime.deviceSynchronize()
                times.append(time.perf_counter() - start)
            
            gflops = flops / (np.mean(times) * 1e9)
            perf_ratios.append(gflops / gflops_baseline)
            
            state_after = self.gpu_control.get_state()
            thermal_history.append({
                'epsilon': float(eps),
                'temp_after': state_after['temperature'],
                'clock_after': state_after['clock_after']
            })
            
            del mem, A, B, C
        
        # Compute σ_c
        perf_array = np.array(perf_ratios)
        obs_smooth = gaussian_filter1d(perf_array, sigma=0.3)
        chi = np.abs(np.gradient(obs_smooth, epsilon))
        
        idx_max = np.argmax(chi)
        sigma_c = float(epsilon[idx_max])
        
        # Chaos metrics
        chaos_metrics = self.chaos_analyzer.compute_chaos_index(
            thermal_history, perf_ratios
        )
        
        return {
            'sigma_c': sigma_c,
            'chaos_metrics': chaos_metrics
        }
    
    def map_parameter_space(self) -> Dict:
        """
        Systematically map σ_c across parameter space.
        
        Dimensions:
        - Size: [512, 1024, 2048]
        - Thermal: ['cool', 'normal', 'warm']
        - Memory: [0.5, 1.0, 1.5]
        
        Total: 3×3×3 = 27 configurations
        Runtime: ~6-9 hours (15 min per config)
        """
        print("\n" + "=" * 80)
        print("🗺️  MAPPING PARAMETER SPACE")
        print("=" * 80)
        
        # Define parameter grid
        sizes = [512, 1024, 2048]
        thermal_states = ['cool', 'normal', 'warm']
        memory_alphas = [0.5, 1.0, 1.5]
        
        n_configs = len(sizes) * len(thermal_states) * len(memory_alphas)
        print(f"Total configurations: {n_configs}")
        print(f"Estimated runtime: {n_configs * 15 / 60:.1f}-{n_configs * 20 / 60:.1f} hours")
        print("=" * 80)
        
        # Map each configuration
        config_num = 0
        for size in sizes:
            for thermal in thermal_states:
                for memory_alpha in memory_alphas:
                    config_num += 1
                    
                    print(f"\n{'='*80}")
                    print(f"CONFIG {config_num}/{n_configs}")
                    print('='*80)
                    
                    try:
                        result = self.measure_sigma_c_with_chaos(
                            size=size,
                            thermal_state=thermal,
                            memory_alpha=memory_alpha,
                            n_passes=5
                        )
                        
                        self.results['chaos_map'].append(result)
                        
                        # Save intermediate
                        self._save_results()
                        
                    except KeyboardInterrupt:
                        print("\n⚠️  Interrupted by user")
                        raise
                    except Exception as e:
                        print(f"    ❌ Error: {e}")
                        continue
        
        return self.results
    
    def analyze_chaos_correlation(self) -> Dict:
        """Analyze correlation between chaos and σ_c stability."""
        print("\n" + "=" * 80)
        print("📊 CHAOS-STABILITY CORRELATION ANALYSIS")
        print("=" * 80)
        
        chaos_map = self.results['chaos_map']
        
        # Extract data
        chaos_indices = [c['chaos_metrics']['chaos_index'] for c in chaos_map]
        sigma_c_cvs = [c['sigma_c_cv'] for c in chaos_map]
        
        # Correlation
        corr_pearson = np.corrcoef(chaos_indices, sigma_c_cvs)[0, 1]
        corr_spearman = stats.spearmanr(chaos_indices, sigma_c_cvs)[0]
        
        # Classify regimes
        stable_regimes = [c for c in chaos_map if c['stability'] == 'STABLE']
        marginal_regimes = [c for c in chaos_map if c['stability'] == 'MARGINAL']
        chaotic_regimes = [c for c in chaos_map if c['stability'] == 'CHAOTIC']
        
        analysis = {
            'correlation_pearson': float(corr_pearson),
            'correlation_spearman': float(corr_spearman),
            'n_stable': len(stable_regimes),
            'n_marginal': len(marginal_regimes),
            'n_chaotic': len(chaotic_regimes),
            'stable_regimes': stable_regimes,
            'marginal_regimes': marginal_regimes,
            'chaotic_regimes': chaotic_regimes
        }
        
        print(f"  Chaos-Stability Correlation: r={corr_pearson:.3f} (Pearson), ρ={corr_spearman:.3f} (Spearman)")
        print(f"  Stable regimes: {len(stable_regimes)}/{len(chaos_map)}")
        print(f"  Marginal regimes: {len(marginal_regimes)}/{len(chaos_map)}")
        print(f"  Chaotic regimes: {len(chaotic_regimes)}/{len(chaos_map)}")
        
        return analysis
    
    def generate_visualizations(self):
        """Generate publication-quality visualizations."""
        print("\n" + "=" * 80)
        print("📈 GENERATING VISUALIZATIONS")
        print("=" * 80)
        
        chaos_map = self.results['chaos_map']
        if len(chaos_map) == 0:
            print("  ⚠️  No data to visualize")
            return
        
        # Extract data
        sizes = np.array([c['size'] for c in chaos_map])
        chaos_idx = np.array([c['chaos_metrics']['chaos_index'] for c in chaos_map])
        sigma_c = np.array([c['sigma_c_median'] for c in chaos_map])
        sigma_c_cv = np.array([c['sigma_c_cv'] for c in chaos_map])
        stability = [c['stability'] for c in chaos_map]
        
        # Color map
        color_map = {'STABLE': 'green', 'MARGINAL': 'orange', 'CHAOTIC': 'red'}
        colors = [color_map[s] for s in stability]
        
        # Figure 1: Chaos vs Stability
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(chaos_idx, sigma_c_cv * 100, 
                           c=colors, s=100, alpha=0.6, edgecolors='black')
        ax.set_xlabel('Chaos Index', fontsize=12)
        ax.set_ylabel('σ_c Coefficient of Variation (%)', fontsize=12)
        ax.set_title('Chaos-Criticality Correlation', fontsize=14, fontweight='bold')
        ax.axhline(y=5, color='green', linestyle='--', label='Stable threshold (5%)')
        ax.axhline(y=10, color='orange', linestyle='--', label='Marginal threshold (10%)')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'chaos_stability_correlation.png', dpi=300)
        print("  ✓ Saved: chaos_stability_correlation.png")
        plt.close()
        
        # Figure 2: 3D Landscape (Size × Chaos × σ_c)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(sizes, chaos_idx, sigma_c,
                           c=colors, s=100, alpha=0.7, edgecolors='black')
        
        ax.set_xlabel('Matrix Size', fontsize=11)
        ax.set_ylabel('Chaos Index', fontsize=11)
        ax.set_zlabel('σ_c', fontsize=11)
        ax.set_title('σ_c Landscape across Parameter Space', fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'sigma_c_landscape_3d.png', dpi=300)
        print("  ✓ Saved: sigma_c_landscape_3d.png")
        plt.close()
        
        # Figure 3: Islands of Order (2D heatmap)
        # Create grid for interpolation
        if len(np.unique(sizes)) > 1 and len(np.unique(chaos_idx)) > 1:
            grid_size = np.linspace(sizes.min(), sizes.max(), 50)
            grid_chaos = np.linspace(chaos_idx.min(), chaos_idx.max(), 50)
            grid_x, grid_y = np.meshgrid(grid_size, grid_chaos)
            
            # Interpolate σ_c CV (stability)
            grid_cv = griddata((sizes, chaos_idx), sigma_c_cv,
                              (grid_x, grid_y), method='cubic')
            
            fig, ax = plt.subplots(figsize=(10, 6))
            im = ax.contourf(grid_x, grid_y, grid_cv * 100, levels=20, cmap='RdYlGn_r')
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('σ_c CV (%)', fontsize=11)
            
            # Overlay actual measurements
            ax.scatter(sizes, chaos_idx, c='black', s=50, alpha=0.5, edgecolors='white')
            
            ax.set_xlabel('Matrix Size', fontsize=12)
            ax.set_ylabel('Chaos Index', fontsize=12)
            ax.set_title('Islands of Order: Stability Landscape', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'islands_of_order.png', dpi=300)
            print("  ✓ Saved: islands_of_order.png")
            plt.close()
    
    def generate_recommendations(self) -> Dict:
        """Generate practical optimization recommendations."""
        print("\n" + "=" * 80)
        print("💡 PRACTICAL RECOMMENDATIONS")
        print("=" * 80)
        
        analysis = self.analyze_chaos_correlation()
        
        stable = analysis['stable_regimes']
        chaotic = analysis['chaotic_regimes']
        
        recommendations = {
            'optimize_here': [],
            'avoid_here': [],
            'summary': {}
        }
        
        if len(stable) > 0:
            print("\n✅ RELIABLE OPTIMIZATION REGIMES:")
            for regime in stable:
                print(f"  • Size={regime['size']}, Thermal={regime['thermal_state']}, "
                      f"α={regime['memory_alpha']:.1f}")
                print(f"    σ_c = {regime['sigma_c_median']:.3f}, CV = {regime['sigma_c_cv']*100:.1f}%")
                recommendations['optimize_here'].append(regime)
        
        if len(chaotic) > 0:
            print("\n❌ CHAOTIC REGIMES (Avoid for optimization):")
            for regime in chaotic:
                print(f"  • Size={regime['size']}, Thermal={regime['thermal_state']}, "
                      f"α={regime['memory_alpha']:.1f}")
                print(f"    σ_c = {regime['sigma_c_median']:.3f}, CV = {regime['sigma_c_cv']*100:.1f}%")
                recommendations['avoid_here'].append(regime)
        
        # Summary statistics
        recommendations['summary'] = {
            'total_configs': len(self.results['chaos_map']),
            'reliable_fraction': len(stable) / len(self.results['chaos_map']),
            'chaotic_fraction': len(chaotic) / len(self.results['chaos_map']),
            'correlation_strength': analysis['correlation_pearson']
        }
        
        print(f"\n📊 Summary:")
        print(f"  Reliable regimes: {len(stable)}/{len(self.results['chaos_map'])} "
              f"({len(stable)/len(self.results['chaos_map'])*100:.1f}%)")
        print(f"  Chaotic regimes: {len(chaotic)}/{len(self.results['chaos_map'])} "
              f"({len(chaotic)/len(self.results['chaos_map'])*100:.1f}%)")
        
        return recommendations
    
    def _save_results(self):
        """Save current results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"chaos_map_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("""
╔════════════════════════════════════════════════════════════════════════╗
║                                                                        ║
║   🎯 GPU CHAOS-CRITICALITY MAPPER                                     ║
║                                                                        ║
║   Maps σ_c emergence across multi-dimensional parameter space         ║
║   Quantifies chaos and identifies "islands of order"                  ║
║                                                                        ║
║   Output: Publication-quality figures + practical recommendations     ║
║   Runtime: ~6-12 hours for comprehensive mapping                      ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝
    """)
    
    proceed = input("\n🚀 Start chaos-criticality mapping? (y/n): ").lower()
    if proceed != 'y':
        print("Aborted.")
        return
    
    mapper = ChaosCriticalityMapper(device_id=0)
    
    try:
        # Map parameter space
        mapper.map_parameter_space()
        
        # Analyze correlations
        mapper.analyze_chaos_correlation()
        
        # Generate visualizations
        mapper.generate_visualizations()
        
        # Practical recommendations
        recommendations = mapper.generate_recommendations()
        
        # Save everything
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_file = mapper.output_dir / f"final_analysis_{timestamp}.json"
        
        final_results = {
            'chaos_map': mapper.results,
            'recommendations': recommendations
        }
        
        with open(final_file, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        print("\n" + "=" * 80)
        print("🎉 CHAOS-CRITICALITY MAPPING COMPLETE!")
        print("=" * 80)
        print(f"📂 Results: {mapper.output_dir}")
        print(f"📊 Visualizations ready for publication")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        mapper._save_results()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()