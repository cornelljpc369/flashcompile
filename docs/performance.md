# Performance Analysis

## Arithmetic Intensity Analysis
```
Arithmetic Intensity (FLOPs/byte) by Operator:

MatMul (square matrices):
  2×2      : ▌ 0.67
  32×32    : ████ 5.33
  128×128  : ████████████████ 21.33
  512×512  : ████████████████████████████████████████████ 85.33
  1024×1024: ████████████████████████████████████████████████████████████████████████████████████ 170.67

Add (element-wise):
  All sizes: ▌ 0.08  (Always memory-bound!)

ReLU:
  All sizes: ▌ 0.12  (Always memory-bound!)

Ridge Points (Memory ↔ Compute boundary):
  CPU (Xeon)   : ──────────────│ 14.0
  GPU (A100)   : ──────────│ 10.0
  GPU (V100)   : ─────────────────│ 17.0
  TPU v4       : ────────────────────────────────────────────────────────────────────│ 229.0
                                  ↑                                                    ↑
                         Memory-bound                                         Compute-bound
```

## Speedup vs NumPy
```
Operation: MatMul

Size        │ Flash   │ NumPy   │ Speedup
────────────┼─────────┼─────────┼─────────
2×2         │ 0.012ms │ 0.015ms │ 1.25× ▓▓▓
64×64       │ 0.145ms │ 0.189ms │ 1.30× ▓▓▓▓
128×128     │ 0.567ms │ 0.743ms │ 1.31× ▓▓▓▓
256×256     │ 2.145ms │ 2.876ms │ 1.34× ▓▓▓▓▓
512×512     │ 9.234ms │ 12.456ms│ 1.35× ▓▓▓▓▓▓
```

## Backend Analysis Summary

### Instruction Mix (MatMul 128×128)
```
Load     : ████████████████████████░░░░░░░░░ 42%
Store    : ████████████░░░░░░░░░░░░░░░░░░░░ 21%
Multiply : ███████░░░░░░░░░░░░░░░░░░░░░░░░ 12%
Add      : ████████████░░░░░░░░░░░░░░░░░░░ 20%
Other    : ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  5%
```

### Register Usage
```
Available GPRs    : 16 ████████████████
Used GPRs         :  6 ██████
Available XMM     : 16 ████████████████
Used XMM          :  8 ████████
Spills            :  0 ✓ (Excellent!)
```

## Optimization Opportunities

Current bottlenecks ranked by impact:

1. **Vectorization** (Potential: 4-8× speedup)
   - Currently using scalar SSE instructions
   - AVX2/AVX-512 can process 8-16 floats in parallel

2. **Loop Tiling** (Potential: 1.5-2× speedup)
   - 42% memory operations
   - Better cache locality needed

3. **Operator Fusion** (Potential: 1.3-1.5× speedup)
   - MatMul + ReLU done separately
   - Fusing eliminates intermediate memory traffic