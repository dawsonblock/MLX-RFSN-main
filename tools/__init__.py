"""
Tools Package
"""

from .repo_extractor import *
from .test_runner import *
from .log_parser import *
from .patch_manager import *
from .proof_regression import *

__all__ = [
    # Repo Extractor
    "RepoInfo",
    "FileInfo",
    "RepoExtractor",
    "extract_repo_info",
    "get_file_tree",
    "find_test_files",
    "find_source_files",
    "get_language_statistics",
    
    # Test Runner
    "TestSuiteResult",
    "TestRunner",
    "run_pytest",
    "run_npm_test",
    "run_compile_check",
    
    # Log Parser
    "LogAnalysis",
    "ParsedFailure",
    "LogParser",
    "parse_test_output",
    "parse_traceback_to_failures",
    "extract_assertion_info",
    
    # Patch Manager
    "SimpleFileChange",
    "generate_unified_diff",
    "apply_simple_patch",
    "create_patch_file",
    "apply_patch_file",
    "create_diff",
    "apply_text_patch",

    # Proof Regression
    "DEFAULT_THRESHOLDS",
    "Breach",
    "merge_thresholds",
    "load_json",
    "load_thresholds_file",
    "compare_section",
    "compare_proof_dirs",
    "report_to_markdown",
]