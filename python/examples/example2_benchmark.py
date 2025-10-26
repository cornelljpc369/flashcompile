#!/usr/bin/env python3
"""
Example 2: Benchmarking Flash compiler performance
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import flashcompile.benchmark as fcbench

# Run comprehensive benchmark
results = fcbench.benchmark_suite()

# Print summary
fcbench.print_summary(results)

# Save results
fcbench.save_results(results, 'benchmark_results.json')