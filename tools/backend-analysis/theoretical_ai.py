#!/usr/bin/env python3
"""Calculate theoretical arithmetic intensity for ML operators"""

def matmul_ai(M, K, N):
    """Matrix multiply: A(M×K) @ B(K×N) → C(M×N)"""
    flops = 2 * M * N * K
    bytes_moved = 4 * (M*K + K*N + M*N)  # float32
    return flops / bytes_moved

def elementwise_add_ai(N):
    """Element-wise add: C = A + B"""
    flops = N
    bytes_moved = 4 * (N + N + N)  # read A, B, write C
    return flops / bytes_moved

def relu_ai(N):
    """ReLU: B = max(0, A)"""
    flops = N  # approximate
    bytes_moved = 4 * (N + N)  # read A, write B
    return flops / bytes_moved

def conv2d_ai(H, W, C_in, C_out, K):
    """2D Convolution"""
    flops = H * W * C_out * K * K * C_in * 2
    bytes_moved = 4 * (H*W*C_in + K*K*C_in*C_out + H*W*C_out)
    return flops / bytes_moved

print("Theoretical Arithmetic Intensity (FLOPs/byte)")
print("=" * 60)
print()

print("Matrix Multiply (square matrices):")
for n in [2, 8, 32, 128, 512, 1024]:
    ai = matmul_ai(n, n, n)
    print(f"  {n:4}×{n:4}: AI = {ai:8.2f}")
print()

print("Element-wise Add:")
for n in [100, 1000, 10000, 100000]:
    ai = elementwise_add_ai(n)
    print(f"  {n:6} elements: AI = {ai:.4f}")
print()

print("ReLU:")
for n in [100, 1000, 10000, 100000]:
    ai = relu_ai(n)
    print(f"  {n:6} elements: AI = {ai:.4f}")
print()

print("Conv2D (224×224, 3×3 kernel):")
configs = [
    (64, 64, "64→64 channels"),
    (128, 128, "128→128 channels"),
    (256, 256, "256→256 channels"),
]
for c_in, c_out, desc in configs:
    ai = conv2d_ai(224, 224, c_in, c_out, 3)
    print(f"  {desc:20}: AI = {ai:8.2f}")
print()

print("Hardware Ridge Points:")
print("-" * 60)
print("  CPU (Intel/AMD):     ~14 FLOPs/byte")
print("  GPU (NVIDIA A100):   ~10 FLOPs/byte")
print("  GPU (NVIDIA V100):   ~17 FLOPs/byte")
print("  TPU v4:             ~229 FLOPs/byte")
print()

print("Interpretation:")
print("-" * 60)
print("  AI < Ridge Point  → Memory-bound (optimize data movement)")
print("  AI > Ridge Point  → Compute-bound (optimize computation)")
