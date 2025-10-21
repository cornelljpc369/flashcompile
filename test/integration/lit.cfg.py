# -*- Python -*-

import os
import lit.formats
import lit.util

# Configuration file for 'lit' test runner.
# This is the integration test suite for FlashCompile.

config.name = 'FlashCompile-Integration'
config.test_format = lit.formats.ShTest(True)

# Suffixes of files to test
config.suffixes = ['.mlir']

# Test source root (where .mlir files are)
config.test_source_root = os.path.dirname(__file__)

# Test execution root (where tests run)
config.test_exec_root = os.path.join(config.flash_obj_root, 'test', 'integration')

# Exclude certain files
config.excludes = ['CMakeLists.txt', 'lit.cfg.py', 'lit.site.cfg.py']

# Add substitutions for tools
config.substitutions.append(('%flash-opt', config.flash_opt_tool))