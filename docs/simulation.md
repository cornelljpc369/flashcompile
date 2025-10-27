# Hardware Simulation Architecture

## Overview

FlashCompile uses **performance models** and **analytical simulators** to enable multi-device optimization without requiring physical access to GPUs, TPUs, or ASICs.

## Motivation

### The Problem
- Modern ML needs diverse hardware targets
- Each device requires different optimizations
- Testing requires expensive hardware access
- Companies have limited hardware budgets

### The Solution
- Build high-fidelity performance models
- Predict execution time from IR analysis
- Optimize for devices we don't own
- Validate predictions when hardware available

## Simulation Stack
```
┌─────────────────────────────────────────┐
│         Application Code                │
│  (Python API: fc.compile, fc.simulate)  │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│      Performance Prediction Engine      │
│                                          │
│  • Analyze compiled IR                  │
│  • Count FLOPs, memory accesses         │
│  • Apply device model                   │
│  • Predict time, utilization            │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│        Device Model Library             │
│                                          │
│  ┌────────┬────────┬────────┬────────┐  │
│  │  GPU   │  TPU   │  CPU   │  ASIC  │  │
│  │ Models │ Models │ Models │ Models │  │
│  └────────┴────────┴────────┴────────┘  │
│                                          │
│  Each model includes:                   │
│  • Peak performance (FLOPS)             │
│  • Memory bandwidth (GB/s)              │
│  • Cache hierarchy                      │
│  • Special features (Tensor Cores)      │
└─────────────────────────────────────────┘
```

## Performance Model: Roofline

### Core Equation
```
Time = max(Compute_Time, Memory_Time)

Where:
  Compute_Time = FLOPs / Peak_FLOPS
  Memory_Time = Bytes / Bandwidth
  
If Compute_Time > Memory_Time: Compute-bound
If Memory_Time > Compute_Time: Memory-bound
```

### Example: MatMul on A100
```python
# Matrix multiply: 1024×1024 @ 1024×1024
M = K = N = 1024

# Count operations
flops = 2 * M * N * K  # 2 for mul+add
flops = 2 * 1024 * 1024 * 1024 = 2,147,483,648

# Count memory traffic (naive)
bytes_A = M * K * 4  # 4 bytes per float32
bytes_B = K * N * 4
bytes_C = M * N * 4
bytes_total = bytes_A + bytes_B + bytes_C
bytes_total = 3 * 1024 * 1024 * 4 = 12,582,912

# A100 specs
peak_flops = 19.5e12  # 19.5 TFLOPS
bandwidth = 1.935e12  # 1.935 TB/s

# Predict time
compute_time = flops / peak_flops = 0.110 ms
memory_time = bytes_total / bandwidth = 0.006 ms

# Bottleneck
time = max(0.110, 0.006) = 0.110 ms

# Compute-bound! (Good for large matmul)
```

## Implementation

### Device Class
```python
from dataclasses import dataclass

@dataclass
class DeviceSpec:
    """Hardware specification"""
    name: str
    peak_fp32: float  # GFLOPS
    bandwidth: float  # GB/s
    l1_size: int      # bytes
    l2_size: int      # bytes
    l3_size: int      # bytes (0 for GPUs)
    has_tensor_cores: bool = False
    has_bfloat16: bool = False

class DeviceSimulator:
    """Base class for device simulators"""
    
    def __init__(self, spec: DeviceSpec):
        self.spec = spec
        self.ridge_point = spec.peak_fp32 / spec.bandwidth
    
    def simulate(self, ir) -> SimulationResult:
        """Predict performance for given IR"""
        
        # Analyze IR
        analysis = self.analyze_ir(ir)
        
        # Compute times
        compute_time = analysis.flops / (self.spec.peak_fp32 * 1e9)
        memory_time = analysis.bytes / (self.spec.bandwidth * 1e9)
        
        # Determine bottleneck
        is_compute_bound = compute_time > memory_time
        
        # Model additional effects
        cache_factor = self.model_cache_effects(analysis)
        occupancy_factor = self.model_occupancy(analysis)
        
        total_time = max(compute_time, memory_time)
        total_time *= cache_factor * occupancy_factor
        
        return SimulationResult(
            time_seconds=total_time,
            time_ms=total_time * 1000,
            is_compute_bound=is_compute_bound,
            utilization=self.compute_utilization(compute_time, total_time),
            bottleneck="compute" if is_compute_bound else "memory"
        )
```

### NVIDIA A100 Simulator
```python
A100_SPEC = DeviceSpec(
    name="NVIDIA A100",
    peak_fp32=19500,  # GFLOPS
    bandwidth=1935,   # GB/s
    l1_size=192 * 1024,  # 192 KB per SM
    l2_size=40 * 1024 * 1024,  # 40 MB
    l3_size=0,  # No L3 on GPUs
    has_tensor_cores=True,
    has_bfloat16=True
)

class NvidiaA100Simulator(DeviceSimulator):
    def __init__(self):
        super().__init__(A100_SPEC)
        self.sm_count = 108
    
    def model_occupancy(self, analysis):
        """Model GPU occupancy effects"""
        # Simplified: Full occupancy assumed
        # Real implementation would check:
        # - Register usage per thread
        # - Shared memory per block
        # - Thread block dimensions
        return 1.0
    
    def model_cache_effects(self, analysis):
        """Model L2 cache hit rate"""
        working_set = analysis.working_set_bytes
        
        if working_set < self.spec.l2_size:
            # Fits in L2: minimal impact
            return 1.0
        else:
            # Doesn't fit: more HBM traffic
            spillover = working_set / self.spec.l2_size
            return 1.0 + (spillover - 1.0) * 0.3  # 30% penalty
```

## Validation Methodology

### Phase 1: Synthetic Benchmarks
```python
# Create known workloads
benchmarks = [
    ("matmul_small", 128, 128, 128),
    ("matmul_medium", 512, 512, 512),
    ("matmul_large", 2048, 2048, 2048),
]

for name, M, K, N in benchmarks:
    # Predict
    prediction = simulator.predict(matmul(M, K, N))
    
    # When hardware available, measure
    actual = gpu.measure(matmul(M, K, N))
    
    # Validate
    error = abs(prediction - actual) / actual
    print(f"{name}: {error*100:.1f}% error")
```

### Phase 2: Real Models
```python
# Test on actual ML models
models = load_pytorch_models([
    "resnet50",
    "bert-base",
    "gpt2-small"
])

for model in models:
    predicted = simulator.predict(model)
    actual = gpu.measure(model)
    validate(predicted, actual)
```

### Acceptance Criteria
- Prediction error < 15% for compute-bound
- Prediction error < 20% for memory-bound
- Correctly identifies bottleneck >90% of time

## Future Enhancements

### Cycle-Accurate Simulation
- Instruction-level modeling
- Pipeline simulation
- Branch prediction
- Out-of-order execution

### Advanced Memory Model
- Cache coherency
- TLB misses
- Bank conflicts (GPU)
- Prefetching effects

### Multi-Device Simulation
- PCIe transfer modeling
- NVLink/GPU-GPU bandwidth
- Multi-GPU scaling
- Heterogeneous execution

## References

1. Roofline Model: Williams et al., "Roofline: An Insightful Visual Performance Model"
2. GPGPU-Sim: Bakhoda et al., "Analyzing CUDA Workloads Using a Detailed GPU Simulator"
3. gem5: Binkert et al., "The gem5 simulator"
4. Analytical Modeling: Hong & Kim, "An Analytical Model for GPU Architectures"