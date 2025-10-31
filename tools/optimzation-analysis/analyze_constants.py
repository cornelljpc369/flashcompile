#!/usr/bin/env python3
"""
Constant Folding & CSE Opportunity Analyzer

Detects opportunities for compile-time evaluation and redundancy elimination.
"""

import re
import sys
from pathlib import Path
from typing import List, Dict, Set
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class ConstantOp:
    """Represents a constant operation"""
    var: str
    value: str
    line: int

@dataclass
class Operation:
    """Represents a Flash operation"""
    result: str
    op_type: str
    operands: List[str]
    line: int
    text: str

class OptimizationAnalyzer:
    """Analyzes IR for constant folding and CSE opportunities"""
    
    def __init__(self, ir_text: str):
        self.ir = ir_text
        self.constants = {}  # var → value
        self.operations = []
        self.const_fold_opportunities = []
        self.cse_opportunities = []
    
    def analyze(self):
        """Run all analyses"""
        self._parse_ir()
        self._find_constant_folding()
        self._find_cse()
    
    def _parse_ir(self):
        """Parse IR to extract constants and operations"""
        for line_no, line in enumerate(self.ir.split('\n'), 1):
            line = line.strip()
            
            # Parse constants: %c1 = arith.constant 2.0 : f32
            const_match = re.search(r'(%\w+)\s*=\s*arith\.constant\s+([^\s:]+)', line)
            if const_match:
                var = const_match.group(1)
                value = const_match.group(2)
                self.constants[var] = ConstantOp(var, value, line_no)
            
            # Parse Flash operations: %result = flash.add %a, %b
            flash_match = re.search(r'(%\w+)\s*=\s*flash\.(\w+)\s+(.*?):', line)
            if flash_match:
                result = flash_match.group(1)
                op_type = flash_match.group(2)
                operands_str = flash_match.group(3)
                
                # Extract operands
                operands = re.findall(r'%\w+', operands_str)
                
                self.operations.append(Operation(
                    result=result,
                    op_type=op_type,
                    operands=operands,
                    line=line_no,
                    text=line
                ))
    
    def _find_constant_folding(self):
        """Find operations on constants that could be folded"""
        for op in self.operations:
            # Check if all operands are constants
            if all(operand in self.constants for operand in op.operands):
                self.const_fold_opportunities.append({
                    'op': op,
                    'type': 'full_fold',
                    'description': f'{op.op_type} on all constant operands'
                })
            
            # Check for algebraic simplifications
            if op.op_type == 'add' and len(op.operands) == 2:
                lhs, rhs = op.operands
                
                # Check if adding zero
                if rhs in self.constants:
                    val = self.constants[rhs].value
                    if 'dense<0' in val or val == '0.0':
                        self.const_fold_opportunities.append({
                            'op': op,
                            'type': 'simplify',
                            'description': f'{lhs} + 0 → {lhs}'
                        })
    
    def _find_cse(self):
        """Find common subexpressions"""
        # Group operations by (op_type, operands)
        seen_ops = defaultdict(list)
        
        for op in self.operations:
            # Create signature: (op_type, tuple of operands)
            signature = (op.op_type, tuple(sorted(op.operands)))
            seen_ops[signature].append(op)
        
        # Find duplicates
        for signature, ops in seen_ops.items():
            if len(ops) > 1:
                # Found duplicate operations!
                first_op = ops[0]
                duplicate_ops = ops[1:]
                
                self.cse_opportunities.append({
                    'first': first_op,
                    'duplicates': duplicate_ops,
                    'savings': f'{len(duplicate_ops)} redundant computation(s)'
                })
    
    def print_report(self):
        """Print analysis report"""
        print("=" * 70)
        print("  Constant Folding & CSE Analysis")
        print("=" * 70)
        print()
        
        # Constant Folding
        print(f"Constant Folding Opportunities: {len(self.const_fold_opportunities)}")
        print("-" * 70)
        
        if not self.const_fold_opportunities:
            print("✓ No constant folding opportunities (already optimized!)")
        else:
            for i, opp in enumerate(self.const_fold_opportunities, 1):
                op = opp['op']
                print(f"\n{i}. Line {op.line}: {opp['type']}")
                print(f"   Operation: {op.text}")
                print(f"   Opportunity: {opp['description']}")
                
                if opp['type'] == 'full_fold':
                    # Show what the constant values are
                    const_vals = [self.constants[operand].value 
                                 for operand in op.operands]
                    print(f"   Constants: {', '.join(const_vals)}")
                    print(f"   → Can compute at compile time!")
        
        print()
        print()
        
        # CSE
        print(f"Common Subexpression Elimination Opportunities: {len(self.cse_opportunities)}")
        print("-" * 70)
        
        if not self.cse_opportunities:
            print("✓ No duplicate computations found (already optimized!)")
        else:
            for i, opp in enumerate(self.cse_opportunities, 1):
                first = opp['first']
                dups = opp['duplicates']
                
                print(f"\n{i}. Duplicate computation: flash.{first.op_type}")
                print(f"   First occurrence (line {first.line}):")
                print(f"     {first.text}")
                print(f"   Duplicates:")
                for dup in dups:
                    print(f"     Line {dup.line}: {dup.text}")
                print(f"   Savings: {opp['savings']}")
                print(f"   → Can reuse {first.result} instead of recomputing")
        
        print()
        print()
        
        # Summary
        total_opportunities = (len(self.const_fold_opportunities) + 
                              len(self.cse_opportunities))
        
        if total_opportunities > 0:
            print("=" * 70)
            print(f"  TOTAL: {total_opportunities} optimization opportunities found")
            print("=" * 70)
            print()
            print("Recommendations:")
            print("  1. Run --flash-constant-fold pass")
            print("  2. Run --flash-cse pass")
            print("  3. Estimated speedup: 5-15% (eliminates redundant work)")
        else:
            print("=" * 70)
            print("  Code is already well-optimized!")
            print("=" * 70)

def main():
    if len(sys.argv) < 2:
        print("Usage: analyze_constants.py <input.mlir>")
        return 1
    
    input_file = Path(sys.argv[1])
    
    if not input_file.exists():
        print(f"Error: File not found: {input_file}")
        return 1
    
    with open(input_file) as f:
        ir_text = f.read()
    
    analyzer = OptimizationAnalyzer(ir_text)
    analyzer.analyze()
    analyzer.print_report()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())