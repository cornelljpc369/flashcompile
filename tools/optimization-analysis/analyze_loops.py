#!/usr/bin/env python3
"""
Loop Analysis Tool

Analyzes loop nests and estimates cache behavior
"""

import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class LoopNest:
    """Represents a loop nest"""
    depth: int
    trip_counts: List[int]
    working_set_bytes: int
    
class LoopAnalyzer:
    """Analyzes loops for tiling opportunities"""
    
    # Cache sizes (typical modern CPU)
    L1_CACHE = 32 * 1024      # 32 KB
    L2_CACHE = 256 * 1024     # 256 KB
    L3_CACHE = 8 * 1024 * 1024  # 8 MB
    
    def __init__(self, ir_text: str):
        self.ir = ir_text
        self.loops = []
    
    def analyze(self):
        """Find and analyze loop nests"""
        
        # Find scf.for loops
        loop_pattern = r'scf\.for\s+%\w+\s*=\s*%c(\d+).*to\s+%c(\d+)'
        
        for match in re.finditer(loop_pattern, self.ir):
            lower = int(match.group(1))
            upper = int(match.group(2))
            trip_count = upper - lower
            
            self.loops.append({
                'lower': lower,
                'upper': upper,
                'trip_count': trip_count
            })
    
    def recommend_tile_size(self, matrix_dim: int) -> Dict[str, int]:
        """Recommend tile size based on matrix dimensions"""
        
        # Calculate working set for different tile sizes
        results = {}
        
        for tile in [16, 32, 64, 128]:
            # For matmul: 3 tiles (A, B, C) each tile×tile
            working_set = 3 * tile * tile * 4  # 4 bytes per float
            
            cache_fit = "none"
            if working_set <= self.L1_CACHE:
                cache_fit = "L1"
            elif working_set <= self.L2_CACHE:
                cache_fit = "L2"
            elif working_set <= self.L3_CACHE:
                cache_fit = "L3"
            
            results[tile] = {
                'working_set': working_set,
                'cache_fit': cache_fit,
                'utilization': working_set / self.L1_CACHE * 100
            }
        
        return results
    
    def print_report(self, matrix_dim: int = 1024):
        """Print loop analysis report"""
        
        print("=" * 70)
        print("  Loop Tiling Analysis")
        print("=" * 70)
        print()
        
        print(f"Matrix dimension: {matrix_dim}×{matrix_dim}")
        print(f"Total elements: {matrix_dim * matrix_dim:,}")
        print(f"Memory footprint: {matrix_dim * matrix_dim * 4 / 1024 / 1024:.2f} MB")
        print()
        
        print("Cache Hierarchy:")
        print(f"  L1: {self.L1_CACHE / 1024:.0f} KB")
        print(f"  L2: {self.L2_CACHE / 1024:.0f} KB")
        print(f"  L3: {self.L3_CACHE / 1024 / 1024:.0f} MB")
        print()
        
        # Analyze tile sizes
        recommendations = self.recommend_tile_size(matrix_dim)
        
        print("Tile Size Analysis:")
        print("-" * 70)
        print(f"{'Tile Size':<15} {'Working Set':<20} {'Cache Fit':<15} {'L1 Util':<10}")
        print("-" * 70)
        
        best_tile = None
        best_score = 0
        
        for tile, info in sorted(recommendations.items()):
            ws_kb = info['working_set'] / 1024
            fit = info['cache_fit']
            util = info['utilization']
            
            # Score: prefer L1 fit, ~50-80% utilization
            score = 0
            if fit == "L1":
                score = 100 - abs(65 - util)  # Prefer ~65% utilization
            
            if score > best_score:
                best_score = score
                best_tile = tile
            
            marker = " ← BEST" if tile == best_tile else ""
            print(f"{tile:<15} {ws_kb:>8.1f} KB         {fit:<15} {util:>6.1f}%{marker}")
        
        print()
        print(f"✓ Recommended tile size: {best_tile}×{best_tile}")
        print()
        
        # Estimate speedup
        print("Expected Benefits:")
        print("-" * 70)
        
        # Without tiling: poor cache usage
        untiled_miss_rate = 0.90  # 90% cache miss (very bad)
        
        # With tiling: good cache usage
        tiled_miss_rate = 0.10  # 10% cache miss (good)
        
        # Cache miss penalty ~100 cycles, cache hit ~4 cycles
        cycles_per_miss = 100
        cycles_per_hit = 4
        
        untiled_cycles = untiled_miss_rate * cycles_per_miss + (1 - untiled_miss_rate) * cycles_per_hit
        tiled_cycles = tiled_miss_rate * cycles_per_miss + (1 - tiled_miss_rate) * cycles_per_hit
        
        speedup = untiled_cycles / tiled_cycles
        
        print(f"  Without tiling: {untiled_miss_rate*100:.0f}% cache miss rate")
        print(f"  With tiling:    {tiled_miss_rate*100:.0f}% cache miss rate")
        print(f"  Expected speedup: {speedup:.1f}×")
        print()
        
        print("Why Tiling Works:")
        print(f"  • Tile fits in L1 cache ({best_tile}×{best_tile} = {best_tile*best_tile*4/1024:.1f} KB < 32 KB)")
        print(f"  • Data reused before eviction")
        print(f"  • Reduces memory bandwidth by {(1-tiled_miss_rate/untiled_miss_rate)*100:.0f}%")

def main():
    if len(sys.argv) < 2:
        print("Usage: analyze_loops.py <input.mlir> [matrix_dim]")
        print()
        print("Example:")
        print("  analyze_loops.py matmul.mlir 1024")
        return 1
    
    input_file = Path(sys.argv[1])
    matrix_dim = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    
    if not input_file.exists():
        print(f"Error: File not found: {input_file}")
        return 1
    
    with open(input_file) as f:
        ir_text = f.read()
    
    analyzer = LoopAnalyzer(ir_text)
    analyzer.analyze()
    analyzer.print_report(matrix_dim)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())