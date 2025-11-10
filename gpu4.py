#!/usr/bin/env python3
"""
🎯 ADAPTIVE SELF-OPTIMIZING GPU σ_c FRAMEWORK v2.0
====================================================
Copyright (c) 2025 ForgottenForge.xyz

Enhanced with aggressive GPU control and multi-pass statistics.

Improvements:
- Multiple GPU lock methods with fallbacks
- Multi-pass validation (10-15 passes)
- Statistical robustness (median + MAD)
- Size-specific optimization
- Enhanced thermal management

Runtime: ~6-10 hours per complete experiment
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
import platform
from datetime import datetime
from typing import Dict, Tuple, List, Optional
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'figure.dpi': 300,
})


# ============================================================================
# ENHANCED GPU CONTROL
# ============================================================================

class EnhancedGPUController:
    """Advanced GPU control with multiple lock strategies."""
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.original_state = None
        self.locked = False
        self.lock_method_used = None
        self.is_windows = platform.system() == 'Windows'
        self.is_admin = self._check_admin()
        
    def _check_admin(self) -> bool:
        """Check if running with administrator privileges."""
        if self.is_windows:
            try:
                import ctypes
                return ctypes.windll.shell32.IsUserAnAdmin() != 0
            except:
                return False
        else:
            import os
            return os.geteuid() == 0
    
    def get_available_clocks(self) -> List[int]:
        """Query available fixed clock frequencies."""
        try:
            result = subprocess.run([
                'nvidia-smi',
                '--query-supported-clocks=gr',
                '--format=csv,noheader',
                f'--id={self.device_id}'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0 and result.stdout.strip():
                clocks = [int(c.strip()) for c in result.stdout.strip().split('\n')]
                return sorted(clocks, reverse=True)
        except:
            pass
        return []
    
    def lock_gpu_clock(self, target_clock: Optional[int] = None) -> bool:
        """
        Try multiple methods to lock GPU clock.
        
        Methods attempted:
        1. nvidia-smi clock lock (requires admin/root)
        2. Power limit reduction (more permissive)
        3. Auto-boost disable (if supported)
        4. Persistence mode (fallback)
        
        Args:
            target_clock: Target MHz (None = use 70% of max)
        
        Returns:
            Success status
        """
        if not self.is_admin:
            print("  ⚠️  NOT running as administrator/root - GPU lock will likely fail")
            print("  💡 Windows: Right-click → Run as Administrator")
            print("  💡 Linux: Use sudo")
        
        print("\n  🔒 Attempting GPU lock with multiple methods...")
        
        # Method 1: Direct clock lock
        success = self._method_1_clock_lock(target_clock)
        if success:
            self.lock_method_used = "clock_lock"
            return True
        
        # Method 2: Power limit
        success = self._method_2_power_limit()
        if success:
            self.lock_method_used = "power_limit"
            return True
        
        # Method 3: Auto-boost disable
        success = self._method_3_disable_boost()
        if success:
            self.lock_method_used = "boost_disable"
            return True
        
        # Method 4: Persistence mode only
        success = self._method_4_persistence()
        if success:
            self.lock_method_used = "persistence"
            print("  ⚠️  Only persistence mode enabled - no frequency lock")
            return False
        
        print("  ❌ All GPU lock methods failed")
        return False
    
    def _method_1_clock_lock(self, target_clock: Optional[int]) -> bool:
        """Method 1: Direct clock locking."""
        try:
            print("    Method 1: Direct clock lock...")
            
            # Enable persistence mode first
            subprocess.run([
                'nvidia-smi', '-pm', '1',
                '-i', str(self.device_id)
            ], capture_output=True, timeout=5)
            
            # Get available clocks
            clocks = self.get_available_clocks()
            if not clocks:
                print("      ⚠️  Cannot query available clocks")
                return False
            
            # Select target (70% of max for stability)
            if target_clock is None:
                target_clock = int(max(clocks) * 0.70)
            
            # Find nearest available clock
            nearest_clock = min(clocks, key=lambda x: abs(x - target_clock))
            
            # Lock to this clock
            result = subprocess.run([
                'nvidia-smi', '-lgc', str(nearest_clock),
                '-i', str(self.device_id)
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                self.locked = True
                print(f"      ✅ Clock locked to {nearest_clock} MHz")
                time.sleep(2)
                return True
            else:
                print(f"      ❌ Failed: {result.stderr.strip()}")
                return False
                
        except Exception as e:
            print(f"      ❌ Exception: {e}")
            return False
    
    def _method_2_power_limit(self) -> bool:
        """Method 2: Reduce power limit (FIXED VERSION)."""
        try:
            print("    Method 2: Power limit reduction...")
            
            # Query DEFAULT power limit
            result = subprocess.run([
                'nvidia-smi',
                '--query-gpu=power.default_limit',  # ← DEFAULT, nicht current!
                '--format=csv,noheader,nounits',
                f'--id={self.device_id}'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode != 0:
                print("      ⚠️  Cannot query default power limit")
                return False
            
            default_limit = float(result.stdout.strip())
            target_limit = int(default_limit * 0.85)  # 85% vom DEFAULT
            
            result = subprocess.run([
                'nvidia-smi', '-pl', str(target_limit),
                '-i', str(self.device_id)
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                print(f"      ✅ Power limited to {target_limit}W (was {default_limit:.0f}W)")
                time.sleep(2)
                return True
            else:
                print(f"      ❌ Failed: {result.stderr.strip()}")
                return False
                
        except Exception as e:
            print(f"      ❌ Exception: {e}")
            return False
    
    def _method_3_disable_boost(self) -> bool:
        """Method 3: Disable auto-boost."""
        try:
            print("    Method 3: Disable auto-boost...")
            
            result = subprocess.run([
                'nvidia-smi', '-acp', '0',  # Disable auto-boost
                '-i', str(self.device_id)
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                print("      ✅ Auto-boost disabled")
                time.sleep(2)
                return True
            else:
                # Not all GPUs support this
                print("      ⚠️  Auto-boost control not supported")
                return False
                
        except Exception as e:
            print(f"      ❌ Exception: {e}")
            return False
    
    def _method_4_persistence(self) -> bool:
        """Method 4: Enable persistence mode only."""
        try:
            print("    Method 4: Persistence mode...")
            
            result = subprocess.run([
                'nvidia-smi', '-pm', '1',
                '-i', str(self.device_id)
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                print("      ✅ Persistence mode enabled")
                return True
            else:
                print("      ❌ Failed")
                return False
                
        except Exception as e:
            print(f"      ❌ Exception: {e}")
            return False
    
    def unlock_gpu_clock(self):
        """Reset GPU to default boost behavior."""
        if not self.locked and self.lock_method_used != "power_limit":
            return
        
        try:
            print("\n  🔓 Restoring GPU defaults...")
            
            # Reset clocks
            subprocess.run([
                'nvidia-smi', '-rgc',
                '-i', str(self.device_id)
            ], capture_output=True, timeout=5)
            
            # Reset power limit
            subprocess.run([
                'nvidia-smi', '-rpl',
                '-i', str(self.device_id)
            ], capture_output=True, timeout=5)
            
            # Disable persistence mode
            subprocess.run([
                'nvidia-smi', '-pm', '0',
                '-i', str(self.device_id)
            ], capture_output=True, timeout=5)
            
            self.locked = False
            self.lock_method_used = None
            print("  ✅ GPU restored to defaults")
            
        except Exception as e:
            print(f"  ⚠️  Restore error: {e}")
    
    def get_current_state(self) -> Dict:
        """Get current GPU state with robust error handling."""
        try:
            result = subprocess.run([
                'nvidia-smi',
                '--query-gpu=temperature.gpu,clocks.current.graphics,power.draw',
                '--format=csv,noheader,nounits',
                f'--id={self.device_id}'
            ], capture_output=True, text=True, timeout=3)
            
            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split(', ')
                if len(parts) >= 3:
                    temp, clock, power = parts[0], parts[1], parts[2]
                    return {
                        'temperature': float(temp),
                        'clock_mhz': float(clock),
                        'power_watts': float(power)
                    }
        except subprocess.TimeoutExpired:
            print("      ⚠️  nvidia-smi timeout")
        except Exception as e:
            print(f"      ⚠️  GPU query error: {e}")
        
        # Return default values if query fails
        return {
            'temperature': 49.0,  # Safe default
            'clock_mhz': 210.0,
            'power_watts': 0.0
        }
    
    def wait_thermal_stable(self, max_temp: float = 50.0, 
                           max_wait: float = 120.0) -> bool:
        """Wait for GPU to cool down below threshold."""
        print(f"  ⏳ Waiting for thermal stability (<{max_temp}°C)...")
        start_time = time.time()
        
        stable_count = 0
        required_stable = 3  # Reduced from 5 to 3
        failed_queries = 0
        max_failed_queries = 5
        
        while time.time() - start_time < max_wait:
            state = self.get_current_state()
            temp = state['temperature']
            
            # Check if query succeeded
            if temp == 49.0 and state['clock_mhz'] == 210.0:
                # Likely a default value from failed query
                failed_queries += 1
                if failed_queries >= max_failed_queries:
                    print(f"  ⚠️  nvidia-smi not responding, assuming thermal OK")
                    return True
                time.sleep(2)
                continue
            
            # Reset failed counter on successful query
            failed_queries = 0
            
            if temp <= max_temp:
                stable_count += 1
                if stable_count >= required_stable:
                    print(f"  ✓ Thermally stable: {temp:.1f}°C")
                    return True
            else:
                stable_count = 0
                print(f"    Current: {temp:.1f}°C (waiting...)")
            
            time.sleep(2)
        
        print(f"  ⚠️  Timeout after {max_wait:.0f}s, proceeding anyway")
        return True  # Proceed even on timeout
    
    def __enter__(self):
        """Context manager entry."""
        self.original_state = self.get_current_state()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - always restore."""
        self.unlock_gpu_clock()


# ============================================================================
# ADAPTIVE SAMPLING STRATEGY
# ============================================================================

class AdaptiveSampler:
    """Smart ε-grid generation based on previous results."""
    
    @staticmethod
    def generate_exploration_grid(epsilon_min: float = 0.0,
                                  epsilon_max: float = 1.0,
                                  n_points: int = 21) -> np.ndarray:
        """Pass 1: Coarse exploration grid."""
        return np.linspace(epsilon_min, epsilon_max, n_points)
    
    @staticmethod
    def generate_exploitation_grid(sigma_c: float,
                                   window: float = 0.15,
                                   n_points: int = 31,
                                   epsilon_min: float = 0.0,
                                   epsilon_max: float = 1.0) -> np.ndarray:
        """Pass 2: Dense sampling around σ_c."""
        grid_min = max(epsilon_min, sigma_c - window)
        grid_max = min(epsilon_max, sigma_c + window)
        
        dense_grid = np.linspace(grid_min, grid_max, n_points)
        
        extra_points = []
        if grid_min > epsilon_min:
            extra_points.append(epsilon_min)
        if grid_max < epsilon_max:
            extra_points.append(epsilon_max)
        
        if extra_points:
            all_points = np.sort(np.concatenate([dense_grid, extra_points]))
            return all_points
        
        return dense_grid


# ============================================================================
# MULTI-PASS VALIDATOR WITH STATISTICAL ROBUSTNESS
# ============================================================================

class MultiPassValidator:
    """
    Enhanced adaptive multi-pass GPU validation.
    
    Strategy:
    1. Exploration: Quick coarse sweep
    2. Multi-Pass Exploitation: 10-15 dense measurements
    3. Statistical analysis: Median + MAD instead of mean
    """
    
    def __init__(self, device_id: int = 0):
        cp.cuda.Device(device_id).use()
        props = cp.cuda.runtime.getDeviceProperties(device_id)
        self.device_name = props['name'].decode()
        self.device_id = device_id
        
        np.random.seed(42)
        cp.random.seed(42)
        
        self.gpu_control = EnhancedGPUController(device_id)
        self.sampler = AdaptiveSampler()
        
        self.results = {
            'device': self.device_name,
            'timestamp': datetime.now().isoformat(),
            'gpu_control': {
                'is_admin': self.gpu_control.is_admin,
                'lock_method': None
            },
            'passes': {}
        }
        
        self.output_dir = Path("gpu_adaptive_output_v2")
        self.output_dir.mkdir(exist_ok=True)
        
        print("=" * 80)
        print("🎯 ENHANCED ADAPTIVE GPU σ_c VALIDATION v2.0")
        print("=" * 80)
        print(f"Device: {self.device_name}")
        print(f"Admin/Root: {'✅ YES' if self.gpu_control.is_admin else '⚠️  NO'}")
        print(f"Strategy: Exploration → Multi-Pass (10+) → Statistical Validation")
        print("=" * 80)
    
    def measure_performance_curve(self, 
                                 epsilon_grid: np.ndarray,
                                 size: int = 1024,
                                 n_reps: int = 12,
                                 lock_gpu: bool = False,
                                 warmup: bool = False) -> Dict:
        """
        Measure GFLOPS vs ε with optional GPU locking and warmup.
        
        Args:
            epsilon_grid: ε values to test
            size: Matrix size
            n_reps: Repetitions per point (increased to 12)
            lock_gpu: Whether to lock GPU clock
            warmup: Whether to perform GPU warmup
        
        Returns:
            Dict with epsilon, gflops, std, thermal_history
        """
        if lock_gpu:
            success = self.gpu_control.lock_gpu_clock()
            if success:
                self.results['gpu_control']['lock_method'] = self.gpu_control.lock_method_used
            else:
                print("  ⚠️  GPU lock failed, continuing without lock...")
        
        # Warmup for large sizes
        if warmup and size >= 2048:
            print(f"  🔥 GPU warmup for size={size}...")
            for _ in range(5):
                A = cp.random.random((size, size), dtype=cp.float32)
                B = cp.random.random((size, size), dtype=cp.float32)
                C = cp.dot(A, B)
                cp.cuda.runtime.deviceSynchronize()
                time.sleep(1)
            del A, B, C
            time.sleep(5)
        
        # Baseline
        A = cp.random.random((size, size), dtype=cp.float32)
        B = cp.random.random((size, size), dtype=cp.float32)
        
        times_baseline = []
        for _ in range(n_reps):
            cp.cuda.runtime.deviceSynchronize()
            start = time.perf_counter()
            C = cp.dot(A, B)
            cp.cuda.runtime.deviceSynchronize()
            times_baseline.append(time.perf_counter() - start)
        
        flops = 2 * size**3
        gflops_baseline = flops / (np.mean(times_baseline) * 1e9)
        del A, B, C
        
        # Sweep
        gflops_data = []
        gflops_std = []
        thermal_history = []
        
        for eps in tqdm(epsilon_grid, desc=f"Measuring (size={size})"):
            state_before = self.gpu_control.get_current_state()
            
            # Overhead injection
            n_mem = int(eps * 12)
            mem_size = max(1, size // 4)
            
            mem = [cp.random.random((mem_size, mem_size), dtype=cp.float32)
                   for _ in range(n_mem)]
            for m in mem:
                _ = cp.sum(m)
            
            A = cp.random.random((size, size), dtype=cp.float32)
            B = cp.random.random((size, size), dtype=cp.float32)
            
            times = []
            for _ in range(n_reps):
                cp.cuda.runtime.deviceSynchronize()
                start = time.perf_counter()
                C = cp.dot(A, B)
                cp.cuda.runtime.deviceSynchronize()
                times.append(time.perf_counter() - start)
            
            gflops = flops / (np.mean(times) * 1e9)
            gflops_data.append(gflops)
            gflops_std.append(np.std(times) * gflops / np.mean(times))
            
            state_after = self.gpu_control.get_current_state()
            thermal_history.append({
                'epsilon': float(eps),
                'temp_before': state_before['temperature'],
                'temp_after': state_after['temperature'],
                'clock_before': state_before['clock_mhz'],
                'clock_after': state_after['clock_mhz']
            })
            
            del mem, A, B, C
        
        if lock_gpu:
            self.gpu_control.unlock_gpu_clock()
        
        return {
            'epsilon': epsilon_grid.tolist(),
            'gflops_baseline': float(gflops_baseline),
            'gflops': [float(x) for x in gflops_data],
            'gflops_std': [float(x) for x in gflops_std],
            'performance_ratio': [float(gf / gflops_baseline) for gf in gflops_data],
            'thermal_history': thermal_history
        }
    
    def compute_susceptibility_full(self, 
                                    epsilon: np.ndarray,
                                    observable: np.ndarray,
                                    kernel_sigma: float = 0.5) -> Dict:
        """Full susceptibility analysis with peak detection."""
        obs_smooth = gaussian_filter1d(observable, sigma=kernel_sigma)
        chi = np.gradient(obs_smooth, epsilon)
        chi_abs = np.abs(chi)
        
        # Edge damping
        n = len(chi_abs)
        damping = np.ones(n)
        if n > 4:
            damping[:2] = np.linspace(0.1, 1.0, 2)
            damping[-2:] = np.linspace(1.0, 0.1, 2)
        chi_abs = chi_abs * damping
        
        # Baseline
        if n > 4:
            interior_chi = chi_abs[1:-1]
            interior_positive = interior_chi[interior_chi > 1e-9]
            if len(interior_positive) > 0:
                baseline = float(np.percentile(interior_positive, 10))
            else:
                baseline = 1e-5
        else:
            baseline = 1e-5
        baseline = max(baseline, 1e-5)
        
        # Peak detection
        idx_max = np.argmax(chi_abs)
        sigma_c = float(epsilon[idx_max])
        chi_max = float(chi_abs[idx_max])
        kappa = min(chi_max / baseline, 200.0)
        
        # Interior check
        margin = 0.05 * (epsilon[-1] - epsilon[0])
        interior_range = [epsilon[0] + margin, epsilon[-1] - margin]
        is_interior = interior_range[0] <= sigma_c <= interior_range[1]
        
        return {
            'chi_abs': chi_abs.tolist(),
            'sigma_c': sigma_c,
            'chi_max': chi_max,
            'kappa': kappa,
            'baseline': baseline,
            'interior': is_interior,
            'interior_range': [float(x) for x in interior_range],
            'peak_index': int(idx_max)
        }
    
    def pass_1_exploration(self, size: int = 1024) -> Dict:
        """Pass 1: Coarse exploration."""
        print("\n" + "=" * 80)
        print("PASS 1: EXPLORATION (Coarse Sweep)")
        print("=" * 80)
        
        self.gpu_control.wait_thermal_stable(max_temp=50.0)
        
        epsilon_grid = self.sampler.generate_exploration_grid(
            epsilon_min=0.0,
            epsilon_max=1.0,
            n_points=21
        )
        
        print(f"  Grid: {len(epsilon_grid)} points, range [0, 1.0]")
        print(f"  GPU lock: NO (exploration mode)")
        
        data = self.measure_performance_curve(
            epsilon_grid,
            size=size,
            n_reps=8,
            lock_gpu=False,
            warmup=(size >= 2048)
        )
        
        susc = self.compute_susceptibility_full(
            np.array(data['epsilon']),
            np.array(data['performance_ratio'])
        )
        
        result = {
            'strategy': 'exploration',
            'epsilon_grid': data['epsilon'],
            'n_points': len(epsilon_grid),
            'size': size,
            'gpu_locked': False,
            'data': data,
            'susceptibility': susc
        }
        
        print(f"\n  📊 Exploration Results:")
        print(f"    σ_c ≈ {susc['sigma_c']:.3f}")
        print(f"    κ = {susc['kappa']:.2f}")
        print(f"    Interior: {susc['interior']}")
        
        temps = [h['temp_after'] for h in data['thermal_history']]
        clocks = [h['clock_after'] for h in data['thermal_history']]
        temp_range = max(temps) - min(temps)
        clock_range = max(clocks) - min(clocks)
        
        print(f"    Thermal variance: {temp_range:.1f}°C")
        print(f"    Clock variance: {clock_range:.0f} MHz")
        
        if clock_range > 200:
            print(f"    ⚠️  High clock variance → GPU lock CRITICAL for validation")
        
        return result
    
    def multi_pass_exploitation(self, 
                                sigma_c_hint: float,
                                size: int = 1024,
                                n_passes: int = 10) -> Dict:
        """
        Multi-pass exploitation with statistical analysis.
        
        Args:
            sigma_c_hint: Approximate σ_c from exploration
            size: Matrix size
            n_passes: Number of validation passes (10-15)
        
        Returns:
            Dict with all passes + statistical summary
        """
        print("\n" + "=" * 80)
        print(f"MULTI-PASS EXPLOITATION ({n_passes} passes)")
        print("=" * 80)
        print(f"  Target: σ_c ≈ {sigma_c_hint:.3f}")
        print(f"  Strategy: Dense sampling with statistical validation")
        
        # Generate dense grid around hint
        epsilon_grid = self.sampler.generate_exploitation_grid(
            sigma_c=sigma_c_hint,
            window=0.15,
            n_points=31
        )
        
        print(f"  Grid: {len(epsilon_grid)} points, range [{epsilon_grid[0]:.3f}, {epsilon_grid[-1]:.3f}]")
        print(f"  GPU lock: YES (attempting all methods)")
        print(f"  Passes: {n_passes}")
        
        all_passes = []
        sigma_c_values = []
        kappa_values = []
        
        for pass_num in range(n_passes):
            print(f"\n  🔄 Pass {pass_num+1}/{n_passes}")
            
            # Thermal reset between passes
            if pass_num > 0:
                print("    Thermal reset...")
                time.sleep(120)  # 2 min cooldown
            
            self.gpu_control.wait_thermal_stable(max_temp=50.0)
            
            # Measure
            data = self.measure_performance_curve(
                epsilon_grid,
                size=size,
                n_reps=12,
                lock_gpu=True,
                warmup=(size >= 2048 and pass_num == 0)
            )
            
            # Analyze
            susc = self.compute_susceptibility_full(
                np.array(data['epsilon']),
                np.array(data['performance_ratio'])
            )
            
            # Bootstrap CI for this pass
            sigma_c_samples = []
            noise_std = np.mean(data['gflops_std']) / data['gflops_baseline']
            
            for _ in range(1000):
                perf_noisy = np.array(data['performance_ratio']) + \
                            np.random.normal(0, noise_std, len(data['performance_ratio']))
                susc_boot = self.compute_susceptibility_full(
                    np.array(data['epsilon']),
                    perf_noisy
                )
                sigma_c_samples.append(susc_boot['sigma_c'])
            
            ci_lower = np.percentile(sigma_c_samples, 2.5)
            ci_upper = np.percentile(sigma_c_samples, 97.5)
            
            pass_result = {
                'pass_number': pass_num + 1,
                'data': data,
                'susceptibility': susc,
                'sigma_c_ci': [float(ci_lower), float(ci_upper)]
            }
            
            all_passes.append(pass_result)
            sigma_c_values.append(susc['sigma_c'])
            kappa_values.append(susc['kappa'])
            
            print(f"    σ_c = {susc['sigma_c']:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
            print(f"    κ = {susc['kappa']:.2f}")
            
            # Check thermal/clock stability
            temps = [h['temp_after'] for h in data['thermal_history']]
            clocks = [h['clock_after'] for h in data['thermal_history']]
            temp_cv = np.std(temps) / np.mean(temps) if np.mean(temps) > 0 else 0
            clock_cv = np.std(clocks) / np.mean(clocks) if np.mean(clocks) > 0 else 0
            
            print(f"    Thermal CV: {temp_cv*100:.1f}%")
            print(f"    Clock CV: {clock_cv*100:.1f}%")
        
        # Statistical analysis across passes
        sigma_c_array = np.array(sigma_c_values)
        kappa_array = np.array(kappa_values)
        
        # Use median + MAD (more robust than mean + std)
        sigma_c_median = np.median(sigma_c_array)
        sigma_c_mad = np.median(np.abs(sigma_c_array - sigma_c_median))
        sigma_c_cv = sigma_c_mad / sigma_c_median if sigma_c_median > 0 else np.nan
        
        kappa_median = np.median(kappa_array)
        kappa_mad = np.median(np.abs(kappa_array - kappa_median))
        
        # Reproducibility criterion: MAD-based CV < 10%
        reproducible = sigma_c_cv < 0.10
        
        statistics = {
            'sigma_c_values': [float(x) for x in sigma_c_values],
            'sigma_c_median': float(sigma_c_median),
            'sigma_c_mad': float(sigma_c_mad),
            'sigma_c_cv': float(sigma_c_cv),
            'sigma_c_mean': float(np.mean(sigma_c_array)),
            'sigma_c_std': float(np.std(sigma_c_array)),
            'kappa_median': float(kappa_median),
            'kappa_mad': float(kappa_mad),
            'reproducible': reproducible,
            'n_passes': n_passes
        }
        
        result = {
            'strategy': 'multi_pass_exploitation',
            'epsilon_grid': epsilon_grid.tolist(),
            'n_points': len(epsilon_grid),
            'size': size,
            'gpu_locked': True,
            'all_passes': all_passes,
            'statistics': statistics
        }
        
        print("\n" + "=" * 80)
        print("📊 MULTI-PASS STATISTICAL SUMMARY")
        print("=" * 80)
        print(f"  Median σ_c: {sigma_c_median:.4f}")
        print(f"  MAD: {sigma_c_mad:.4f}")
        print(f"  CV (MAD-based): {sigma_c_cv*100:.1f}%")
        print(f"  Reproducible: {'✅ YES' if reproducible else '⚠️  NO (>10% variance)'}")
        print(f"  Pass range: [{min(sigma_c_values):.4f}, {max(sigma_c_values):.4f}]")
        print("=" * 80)
        
        return result
    
    def run_adaptive_experiment_e1(self, size: int = 1024, n_passes: int = 10) -> Dict:
        """
        Complete E1 with adaptive multi-pass strategy.
        
        Args:
            size: Matrix size
            n_passes: Number of exploitation passes (10-15 recommended)
        
        Returns:
            Dict with exploration + multi-pass results
        """
        print("\n" + "=" * 80)
        print(f"🎯 ADAPTIVE E1: Interior Peak Detection (size={size})")
        print("=" * 80)
        
        # Pass 1: Exploration
        pass1 = self.pass_1_exploration(size=size)
        
        # Multi-Pass: Exploitation with statistics
        sigma_c_hint = pass1['susceptibility']['sigma_c']
        multi_pass = self.multi_pass_exploitation(sigma_c_hint, size=size, n_passes=n_passes)
        
        # Final assessment
        stats = multi_pass['statistics']
        
        result = {
            'size': size,
            'pass_1_exploration': pass1,
            'multi_pass_exploitation': multi_pass,
            'final_statistics': stats,
            'status': 'PASSED' if stats['reproducible'] else 'PARTIAL'
        }
        
        print("\n" + "=" * 80)
        print("🎉 EXPERIMENT COMPLETE")
        print("=" * 80)
        print(f"  Final σ_c (median): {stats['sigma_c_median']:.4f}")
        print(f"  Reproducibility CV: {stats['sigma_c_cv']*100:.1f}%")
        print(f"  Status: {result['status']}")
        print("=" * 80)
        
        return result
    
    def run_complete_adaptive_suite(self, n_passes: int = 10):
        """
        Run adaptive experiments with multi-pass validation.
        
        Args:
            n_passes: Number of passes per size (10-15 recommended)
        """
        print("\n" + "=" * 80)
        print("🚀 STARTING ENHANCED ADAPTIVE VALIDATION SUITE")
        print("=" * 80)
        print(f"  Multi-pass strategy: {n_passes} passes per size")
        print(f"  Estimated runtime: {n_passes * 0.5:.0f}-{n_passes * 0.7:.0f} hours")
        print("=" * 80)
        
        # Prioritize size 1024 (hardware sweet spot)
        sizes = [1024, 512, 2048]
        
        for size in sizes:
            print(f"\n{'='*80}")
            print(f"PROCESSING SIZE {size}")
            print('='*80)
            
            result = self.run_adaptive_experiment_e1(size=size, n_passes=n_passes)
            self.results['passes'][f'e1_size_{size}'] = result
            
            # Save intermediate results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.output_dir / f"adaptive_results_v2_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            
            print(f"\n  💾 Intermediate results saved: {filename}")
        
        # Final summary
        self._print_summary()
        
        return filename
    
    def _print_summary(self):
        """Print final summary."""
        print("\n" + "=" * 80)
        print("📊 ENHANCED ADAPTIVE VALIDATION SUMMARY")
        print("=" * 80)
        
        for key, result in self.results['passes'].items():
            if 'final_statistics' in result:
                stats = result['final_statistics']
                status = result['status']
                symbol = '✅' if status == 'PASSED' else '⚠️'
                
                print(f"\n  {symbol} {key}:")
                print(f"    σ_c (median): {stats['sigma_c_median']:.4f}")
                print(f"    MAD: {stats['sigma_c_mad']:.4f}")
                print(f"    CV: {stats['sigma_c_cv']*100:.1f}%")
                print(f"    Reproducible: {stats['reproducible']}")
                print(f"    Status: {status}")
        
        lock_method = self.results['gpu_control'].get('lock_method')
        if lock_method:
            print(f"\n  🔒 GPU Control Method Used: {lock_method}")
        else:
            print(f"\n  ⚠️  No GPU lock achieved - results may show higher variance")
        
        print("\n" + "=" * 80)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main execution with privilege check."""
    print("""
╔════════════════════════════════════════════════════════════════════════╗
║                                                                        ║
║   🎯 ENHANCED ADAPTIVE GPU σ_c FRAMEWORK v2.0                         ║
║                                                                        ║
║   Strategy: Multi-pass with statistical robustness                    ║
║   • Pass 1: Exploration (coarse sweep)                               ║
║   • Multi-Pass: 10-15 dense measurements with GPU lock               ║
║   • Statistics: Median + MAD for robustness                          ║
║                                                                        ║
║   Runtime: ~6-10 hours | Output: Statistically validated σ_c         ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝

⚠️  CRITICAL: GPU clock locking requires administrator privileges!
   
   Windows: Right-click Python script → "Run as Administrator"
   Linux:   sudo python gpu5_enhanced.py
   
   Without admin rights, GPU lock will fail and results will show
   higher variance due to dynamic frequency scaling.
    """)
    
    n_passes = input("\n🔢 How many validation passes per size? (10-15 recommended, default=10): ").strip()
    if not n_passes or not n_passes.isdigit():
        n_passes = 10
    else:
        n_passes = int(n_passes)
        n_passes = max(5, min(n_passes, 20))  # Clamp to [5, 20]
    
    proceed = input(f"\n🚀 Start enhanced validation with {n_passes} passes? (y/n): ").lower()
    if proceed != 'y':
        print("Aborted.")
        return
    
    validator = MultiPassValidator(device_id=0)
    
    try:
        results_file = validator.run_complete_adaptive_suite(n_passes=n_passes)
        print(f"\n✅ All results saved to: {results_file}")
        print(f"📂 Output directory: {validator.output_dir}")
        print("\n🎉 ENHANCED VALIDATION COMPLETE!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        validator.gpu_control.unlock_gpu_clock()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        validator.gpu_control.unlock_gpu_clock()
        import traceback
        traceback.print_exc()
    finally:
        validator.gpu_control.unlock_gpu_clock()


if __name__ == "__main__":
    main()