#!/usr/bin/env python3
"""
Fusion Opportunity Analyzer

Analyzes Flash IR to detect fusion opportunities and estimate speedup.
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict
from dataclasses import dataclass

@dataclass
class FusionOpportunity:
    """Represents a detected fusion opportunity"""
    pattern: str  # e.g., "matmul + relu"
    operations: List[str]
    estimated_speedup: float
    memory_reduction: str
    
class FusionAnalyzer:
    """Analyzes IR for fusion opportunities"""
    
    # Fusion patterns and their benefits
    PATTERNS = {
        ('matmul', 'add'): {
            'name': 'MatMul + Bias',
            'speedup': 1.5,
            'memory_reduction': '33%'
        },
        ('matmul', 'relu'): {
            'name': 'MatMul + ReLU',
            'speedup': 1.8,
            'memory_reduction': '50%'
        },
        ('add', 'relu'): {
            'name': 'Add + ReLU',
            'speedup': 1.9,
            'memory_reduction': '50%'
        },
        ('matmul', 'add', 'relu'): {
            'name': 'MatMul + Bias + ReLU',
            'speedup': 2.5,
            'memory_reduction': '67%'
        },
    }
    
    def __init__(self, ir_text: str):
        self.ir = ir_text
        self.opportunities = []
    
    def analyze(self) -> List[FusionOpportunity]:
        """Find all fusion opportunities"""
        
        # Parse operations
        ops = self._parse_operations()
        
        # Look for fusion patterns
        for i in range(len(ops)):
            # Check 2-op patterns
            if i + 1 < len(ops):
                pattern = (ops[i]['type'], ops[i+1]['type'])
                if pattern in self.PATTERNS:
                    self._add_opportunity(pattern, [ops[i], ops[i+1]])
            
            # Check 3-op patterns
            if i + 2 < len(ops):
                pattern = (ops[i]['type'], ops[i+1]['type'], ops[i+2]['type'])
                if pattern in self.PATTERNS:
                    self._add_opportunity(pattern, [ops[i], ops[i+1], ops[i+2]])
        
        return self.opportunities
    
    def _parse_operations(self) -> List[Dict]:
        """Parse operations from IR"""
        ops = []
        
        # Match Flash operations
        matmul_pattern = r'flash\.matmul'
        add_pattern = r'flash\.add'
        relu_pattern = r'flash\.relu'
        
        for line_no, line in enumerate(self.ir.split('\n'), 1):
            if 'flash.matmul' in line:
                ops.append({'type': 'matmul', 'line': line_no, 'text': line.strip()})
            elif 'flash.add' in line:
                ops.append({'type': 'add', 'line': line_no, 'text': line.strip()})
            elif 'flash.relu' in line:
                ops.append({'type': 'relu', 'line': line_no, 'text': line.strip()})
        
        return ops
    
    def _add_opportunity(self, pattern: Tuple, ops: List[Dict]):
        """Record a fusion opportunity"""
        info = self.PATTERNS[pattern]
        
        op = FusionOpportunity(
            pattern=info['name'],
            operations=[o['text'] for o in ops],
            estimated_speedup=info['speedup'],
            memory_reduction=info['memory_reduction']
        )
        
        self.opportunities.append(op)
    
    def print_report(self):
        """Print analysis report"""
        print("=" * 70)
        print("  Fusion Opportunity Analysis")
        print("=" * 70)
        print()
        
        if not self.opportunities:
            print("✓ No fusion opportunities detected")
            print("  (This might mean the code is already optimized!)")
            return
        
        print(f"Found {len(self.opportunities)} fusion opportunities:")
        print()
        
        total_speedup = 1.0
        for i, opp in enumerate(self.opportunities, 1):
            print(f"Opportunity {i}: {opp.pattern}")
            print(f"  Operations:")
            for op in opp.operations:
                print(f"    • {op}")
            print(f"  Estimated speedup: {opp.estimated_speedup:.1f}×")
            print(f"  Memory reduction: {opp.memory_reduction}")
            print()
            
            total_speedup *= opp.estimated_speedup
        
        print("-" * 70)
        print(f"Total estimated speedup if all fused: {total_speedup:.1f}×")
        print()
        
        # Recommendations
        print("Recommendations:")
        print("  1. Implement fused operations in Flash dialect")
        print("  2. Add fusion patterns to flash-opt")
        print("  3. Lower fused ops directly to optimized code")
        print("  4. Benchmark to validate speedup predictions")

def main():
    if len(sys.argv) < 2:
        print("Usage: analyze_fusion.py <input.mlir>")
        return 1
    
    input_file = Path(sys.argv[1])
    
    if not input_file.exists():
        print(f"Error: File not found: {input_file}")
        return 1
    
    with open(input_file) as f:
        ir_text = f.read()
    
    analyzer = FusionAnalyzer(ir_text)
    analyzer.analyze()
    analyzer.print_report()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())