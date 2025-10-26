#!/usr/bin/env python3
"""
Backend Analysis Tool - Understand LLVM's compilation decisions

Analyzes:
1. Instruction selection (which CPU instructions chosen)
2. Register allocation (registers used, spills)
3. Instruction scheduling (reordering, latency hiding)
4. Assembly code generation
"""

import subprocess
from dataclasses import dataclass
from typing import List, Dict, Optional
import tempfile
import os
import sys
import re
from pathlib import Path

#color output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_header(msg):
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'='*70}{Colors.RESET}")
    print(f"{Colors.BLUE}{Colors.BOLD}  {msg}{Colors.RESET}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'='*70}{Colors.RESET}\n")

def print_section(msg):
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'─'*70}{Colors.RESET}")
    print(f"{Colors.CYAN}{Colors.BOLD}{msg}{Colors.RESET}")
    print(f"{Colors.CYAN}{Colors.BOLD}{'─'*70}{Colors.RESET}")

def print_info(msg):
    print(f"{Colors.GREEN}✓{Colors.RESET} {msg}")

def print_warning(msg):
    print(f"{Colors.YELLOW}⚠{Colors.RESET} {msg}")

def print_detail(label, value):
    print(f"  {Colors.CYAN}{label}:{Colors.RESET} {value}")

#==============================================================================
# Data Structures
#==============================================================================

@dataclass
class InstructionStats:
    """Statistics about instruction selection"""
    total_instructions: int = 0
    load_instructions: int = 0
    store_instructions: int = 0
    arithmetic_ops: int = 0
    float_ops: int = 0
    vector_ops: int = 0
    branches: int = 0
    instruction_types: Dict[str, int] = None
    
    def __post_init__(self):
        if self.instruction_types is None:
            self.instruction_types = {}

@dataclass
class RegisterStats:
    """Statistics about register allocation"""
    total_physical_regs: int = 0
    gpr_used: List[str] = None
    vector_used: List[str] = None
    spills: int = 0
    
    def __post_init__(self):
        if self.gpr_used is None:
            self.gpr_used = []
        if self.vector_used is None:
            self.vector_used = []

#==============================================================================
# Compilation Pipeline
#==============================================================================

def compile_to_llvm_ir(flash_ir_file: Path, project_root: Path) -> Optional[Path]:
    """Compile Flash IR to LLVM IR"""
    print_section("Step 1: Compiling Flash IR → LLVM IR")
    
    flash_opt = project_root / "build" / "tools" / "flash-opt" / "flash-opt"
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as llvm_dialect_file:
        llvm_dialect_path = Path(llvm_dialect_file.name)
    
    try:
        # Run flash-opt to lower to LLVM dialect
        result = subprocess.run(
            [
                str(flash_opt),
                str(flash_ir_file),
                "--convert-flash-to-linalg",
                "--one-shot-bufferize",
                "--convert-linalg-to-loops",
                "--lower-affine",
                "--convert-scf-to-cf",
                "--convert-cf-to-llvm",
                "--convert-arith-to-llvm",
                "--convert-func-to-llvm",
                "--finalize-memref-to-llvm",
                "--reconcile-unrealized-casts",
                "-o", str(llvm_dialect_path)
            ],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode != 0:
            print(f"{Colors.RED}Error lowering to LLVM dialect:{Colors.RESET}")
            print(result.stderr)
            return None
        
        print_info("Lowered to LLVM dialect")
        
        # Translate to LLVM IR
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ll', delete=False) as llvm_ir_file:
            llvm_ir_path = Path(llvm_ir_file.name)
        
        result = subprocess.run(
            ["mlir-translate", "--mlir-to-llvmir", str(llvm_dialect_path), "-o", str(llvm_ir_path)],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode != 0:
            print(f"{Colors.RED}Error translating to LLVM IR:{Colors.RESET}")
            print(result.stderr)
            return None
        
        print_info(f"Generated LLVM IR: {llvm_ir_path}")
        
        return llvm_ir_path
    
    except Exception as e:
        print(f"{Colors.RED}Compilation error: {e}{Colors.RESET}")
        return None
    
#==============================================================================
# Assembly Generation and Analysis
#==============================================================================

def generate_assembly(llvm_ir_path: Path) -> Optional[Path]:
    """Generate assembly code from LLVM IR"""
    print_section("Step 2: Generating Assembly Code")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.s', delete=False) as asm_file:
        asm_path = Path(asm_file.name)
    
    try:
        result = subprocess.run(
            ["llc", str(llvm_ir_path), "-o", str(asm_path)],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode != 0:
            print(f"{Colors.RED}Error generating assembly:{Colors.RESET}")
            print(result.stderr)
            return None
        
        print_info(f"Generated assembly: {asm_path}")
        
        return asm_path
    
    except Exception as e:
        print(f"{Colors.RED}Assembly generation error: {e}{Colors.RESET}")
        return None

def analyze_assembly(asm_path: Path) -> InstructionStats:
    """Analyze assembly code for instruction statistics"""
    print_section("Step 3: Analyzing Assembly Code")
    
    stats = InstructionStats()
    
    try:
        with open(asm_path, 'r') as f:
            asm_code = f.read()
        
        lines = asm_code.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Skip comments and labels
            if not line or line.startswith('.') or line.startswith('#') or line.endswith(':'):
                continue
            
            # Count instruction types
            parts = line.split()
            if not parts:
                continue
            
            instruction = parts[0]
            stats.total_instructions += 1
            
            # Categorize instructions
            if instruction.startswith('ld') or instruction.startswith('ldr') or instruction.startswith('mov') and 'mem' in line.lower():
                stats.load_instructions += 1
            elif instruction.startswith('st') or instruction.startswith('str'):
                stats.store_instructions += 1
            elif instruction in ['add', 'sub', 'mul', 'div', 'fadd', 'fsub', 'fmul', 'fdiv',
                               'addss', 'subss', 'mulss', 'divss', 'addps', 'mulps']:
                stats.arithmetic_ops += 1
                if 'f' in instruction or 'ss' in instruction or 'ps' in instruction or 'sd' in instruction:
                    stats.float_ops += 1
            elif instruction.startswith('v') or 'xmm' in line.lower() or 'ymm' in line.lower():
                stats.vector_ops += 1
            elif instruction in ['b', 'br', 'beq', 'bne', 'jmp', 'je', 'jne', 'jl', 'jg']:
                stats.branches += 1
            
            # Track instruction types
            stats.instruction_types[instruction] = stats.instruction_types.get(instruction, 0) + 1
        
        # Print statistics
        print_detail("Total instructions", stats.total_instructions)
        print_detail("Load instructions", stats.load_instructions)
        print_detail("Store instructions", stats.store_instructions)
        print_detail("Arithmetic operations", stats.arithmetic_ops)
        print_detail("Float operations", stats.float_ops)
        print_detail("Vector operations", stats.vector_ops)
        print_detail("Branch instructions", stats.branches)
        
        print(f"\n{Colors.MAGENTA}Top 10 Instructions:{Colors.RESET}")
        sorted_instructions = sorted(stats.instruction_types.items(), key=lambda x: x[1], reverse=True)
        for inst, count in sorted_instructions[:10]:
            print(f"  {inst:15} : {count:3} times")
        
        return stats
    
    except Exception as e:
        print(f"{Colors.RED}Assembly analysis error: {e}{Colors.RESET}")
        return stats
    
    #==============================================================================
# LLVM IR Analysis
#==============================================================================

def analyze_llvm_ir(llvm_ir_path: Path):
    """Analyze LLVM IR for insights"""
    print_section("Step 4: Analyzing LLVM IR")
    
    try:
        with open(llvm_ir_path, 'r') as f:
            llvm_ir = f.read()
        
        # Count basic blocks
        bb_count = llvm_ir.count('label %')
        print_detail("Basic blocks", bb_count)
        
        # Count load/store
        load_count = llvm_ir.count(' load ')
        store_count = llvm_ir.count(' store ')
        print_detail("LLVM load instructions", load_count)
        print_detail("LLVM store instructions", store_count)
        
        # Count float operations
        fmul_count = llvm_ir.count(' fmul ')
        fadd_count = llvm_ir.count(' fadd ')
        print_detail("Float multiplies (fmul)", fmul_count)
        print_detail("Float additions (fadd)", fadd_count)
        
        # Check for vectorization
        vector_types = re.findall(r'<\d+ x [^>]+>', llvm_ir)
        if vector_types:
            print_warning(f"Vector types found: {len(vector_types)} - potential SIMD opportunity")
        else:
            print_info("No vector types - scalar operations only")
        
        # Function info
        functions = re.findall(r'define [^@]+@(\w+)', llvm_ir)
        print_detail("Functions", ', '.join(functions) if functions else 'main')
        
    except Exception as e:
        print(f"{Colors.RED}LLVM IR analysis error: {e}{Colors.RESET}")

#==============================================================================
# Register Allocation Analysis
#==============================================================================

def analyze_registers(asm_path: Path) -> RegisterStats:
    """Analyze register usage in assembly"""
    print_section("Step 5: Register Allocation Analysis")
    
    stats = RegisterStats()
    
    try:
        with open(asm_path, 'r') as f:
            asm_code = f.read()
        
        # Find register usage (x86/ARM patterns)
        # x86: rax, rbx, rcx, rdx, rsi, rdi, r8-r15, xmm0-xmm15
        # ARM: r0-r15, s0-s31, d0-d31
        
        gpr_pattern = r'\b(rax|rbx|rcx|rdx|rsi|rdi|rbp|rsp|r\d+|x\d+|w\d+)\b'
        vector_pattern = r'\b(xmm\d+|ymm\d+|zmm\d+|v\d+|q\d+|d\d+|s\d+)\b'
        
        gpr_regs = set(re.findall(gpr_pattern, asm_code))
        vector_regs = set(re.findall(vector_pattern, asm_code))
        
        stats.gpr_used = sorted(list(gpr_regs))
        stats.vector_used = sorted(list(vector_regs))
        stats.total_physical_regs = len(stats.gpr_used) + len(stats.vector_used)
        
        # Check for spills (stack operations beyond frame setup)
        spill_pattern = r'(str.*sp|ldr.*sp|\[rsp.*\]|push|pop)'
        potential_spills = len(re.findall(spill_pattern, asm_code))
        # Heuristic: more than 4 stack ops likely indicates spilling
        if potential_spills > 4:
            stats.spills = potential_spills - 4  # Subtract frame setup overhead
        
        print_detail("Total physical registers used", stats.total_physical_regs)
        print_detail("GPRs used", len(stats.gpr_used))
        print_detail("Vector regs used", len(stats.vector_used))
        
        if stats.gpr_used:
            print(f"  {Colors.CYAN}GPR registers:{Colors.RESET} {', '.join(stats.gpr_used[:10])}")
            if len(stats.gpr_used) > 10:
                print(f"    ... and {len(stats.gpr_used) - 10} more")
        
        if stats.vector_used:
            print(f"  {Colors.CYAN}Vector registers:{Colors.RESET} {', '.join(stats.vector_used[:10])}")
            if len(stats.vector_used) > 10:
                print(f"    ... and {len(stats.vector_used) - 10} more")
        
        if stats.spills > 0:
            print_warning(f"Potential register spills detected: ~{stats.spills}")
        else:
            print_info("No register spills detected (good!)")
        
        return stats
    
    except Exception as e:
        print(f"{Colors.RED}Register analysis error: {e}{Colors.RESET}")
        return stats
def calculate_arithmetic_intensity(inst_stats: InstructionStats, operation_type: str = "unknown") -> float:
    """
    Calculate arithmetic intensity (FLOPs/byte) more accurately
    
    This is approximate since we don't have exact FLOP counts from assembly,
    but we can estimate based on float operations and memory operations.
    """
    # Estimate FLOPs from float operations (each float op ≈ 1 FLOP)
    flops = inst_stats.float_ops
    
    # Estimate bytes moved (assuming 4 bytes per float)
    # Each load/store is approximately 4 bytes
    bytes_moved = (inst_stats.load_instructions + inst_stats.store_instructions) * 4
    
    if bytes_moved == 0:
        return 0.0
    
    ai = flops / bytes_moved
    return ai 
#==============================================================================
# Optimization Suggestions
#==============================================================================

def suggest_optimizations(inst_stats: InstructionStats, reg_stats: RegisterStats):
    """Suggest potential optimizations based on analysis"""
    print_section("Step 6: Optimization Opportunities")
    
    suggestions = []
    
    # Calculate arithmetic intensity
    ai = calculate_arithmetic_intensity(inst_stats)
    print_detail("Arithmetic Intensity", f"{ai:.4f} FLOPs/byte")
    
    # Hardware ridge points (approximate)
    ridge_points = {
        "CPU (Intel/AMD)": 14.0,
        "GPU (NVIDIA V100)": 17.0,
        "GPU (NVIDIA A100)": 10.0,
        "Apple M1": 13.0,
        "TPU v4": 229.0
    }
    
    print(f"\n{Colors.CYAN}Ridge Points for Reference:{Colors.RESET}")
    for hw, ridge in ridge_points.items():
        if ai < ridge:
            bound = f"{Colors.RED}Memory-bound{Colors.RESET}"
        else:
            bound = f"{Colors.GREEN}Compute-bound{Colors.RESET}"
        print(f"  {hw:25} : {ridge:6.1f} FLOPs/byte → {bound}")
    
    print()
    
    # Determine if memory or compute bound (use A100 ridge point as default)
    default_ridge = 10.0
    if ai < default_ridge:
        print_warning(f"Memory-bound operation (AI={ai:.2f} < ridge={default_ridge})")
        print(f"  {Colors.YELLOW}→ Optimize data movement, improve cache locality{Colors.RESET}")
    else:
        print_info(f"Compute-bound operation (AI={ai:.2f} >= ridge={default_ridge})")
        print(f"  {Colors.GREEN}→ Optimize computation, consider algorithmic improvements{Colors.RESET}")
    
    print()
    
    # Check for vectorization opportunities
    if inst_stats.float_ops > 4 and inst_stats.vector_ops == 0:
        suggestions.append("⚡ Vectorization: Many scalar float ops detected. SIMD can provide 4-8x speedup.")
    
    # Check memory operations ratio
    mem_ops = inst_stats.load_instructions + inst_stats.store_instructions
    mem_ratio = mem_ops / max(inst_stats.total_instructions, 1)
    if mem_ratio > 0.3:
        suggestions.append(f"⚡ Memory bandwidth: {mem_ratio*100:.1f}% memory ops. Loop tiling or data reuse can help.")
    
    # Check register pressure
    if reg_stats.spills > 0:
        suggestions.append(f"⚡ Register spills: {reg_stats.spills} detected. Reduce live ranges or use register blocking.")
    
    # Low arithmetic intensity
    if ai < 1.0:
        suggestions.append(f"⚡ Very low AI ({ai:.2f}): This operation is memory-bound on ALL hardware.")
        suggestions.append("   → Consider: fusion with other ops, in-place operations, or better cache blocking")
    
    # Branch density
    branch_ratio = inst_stats.branches / max(inst_stats.total_instructions, 1)
    if branch_ratio > 0.1:
        suggestions.append(f"⚡ High branch density: {branch_ratio*100:.1f}%. Loop unrolling or predication may help.")
    
    if suggestions:
        print(f"{Colors.YELLOW}Optimization Suggestions:{Colors.RESET}")
        for suggestion in suggestions:
            print(f"  {suggestion}")
    else:
        print_info("Code looks well-optimized! No obvious opportunities found.")
#==============================================================================
# Main Analysis Function
#==============================================================================

def analyze_backend(flash_ir_file: Path, project_root: Path):
    """Main backend analysis pipeline"""
    
    print_header("LLVM Backend Analysis Tool")
    print(f"{Colors.CYAN}Input:{Colors.RESET} {flash_ir_file}")
    print(f"{Colors.CYAN}Project:{Colors.RESET} {project_root}\n")
    
    # Step 1: Compile to LLVM IR
    llvm_ir_path = compile_to_llvm_ir(flash_ir_file, project_root)
    if not llvm_ir_path:
        print(f"\n{Colors.RED}✗ Analysis failed at compilation stage{Colors.RESET}")
        return 1
    
    # Step 2: Generate assembly
    asm_path = generate_assembly(llvm_ir_path)
    if not asm_path:
        print(f"\n{Colors.RED}✗ Analysis failed at assembly generation{Colors.RESET}")
        return 1
    
    # Step 3: Analyze assembly
    inst_stats = analyze_assembly(asm_path)
    
    # Step 4: Analyze LLVM IR
    analyze_llvm_ir(llvm_ir_path)
    
    # Step 5: Analyze registers
    reg_stats = analyze_registers(asm_path)
    
    # Step 6: Suggest optimizations
    suggest_optimizations(inst_stats, reg_stats)
    
    # Summary
    print_header("Analysis Complete")
    print_info(f"LLVM IR: {llvm_ir_path}")
    print_info(f"Assembly: {asm_path}")
    print(f"\n{Colors.GREEN}{Colors.BOLD}✓ Backend analysis successful!{Colors.RESET}\n")
    
    return 0

#==============================================================================
# CLI
#==============================================================================

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <flash_ir_file.mlir>")
        print(f"\nExample:")
        print(f"  {sys.argv[0]} /tmp/matmul.mlir")
        return 1
    
    flash_ir_file = Path(sys.argv[1])
    
    if not flash_ir_file.exists():
        print(f"{Colors.RED}Error: File not found: {flash_ir_file}{Colors.RESET}")
        return 1
    
    # Find project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    return analyze_backend(flash_ir_file, project_root)

if __name__ == "__main__":
    sys.exit(main())