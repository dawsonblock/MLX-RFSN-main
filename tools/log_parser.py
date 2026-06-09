"""
Log parser for extracting structured information from test failures and logs.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import re
import json

from agent_core.schemas import TestResult, TestResultStatus

logger = __import__('logging').getLogger(__name__)


@dataclass
class ParsedFailure:
    """Structured representation of a test failure."""
    test_name: str
    failure_type: str  # assertion, exception, timeout, etc.
    file_path: Optional[Path] = None
    line_number: Optional[int] = None
    message: str = ""
    traceback: List[str] = field(default_factory=list)
    expected: Optional[str] = None
    actual: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "test_name": self.test_name,
            "failure_type": self.failure_type,
            "file_path": str(self.file_path) if self.file_path else None,
            "line_number": self.line_number,
            "message": self.message,
            "traceback": self.traceback,
            "expected": self.expected,
            "actual": self.actual
        }


@dataclass
class LogAnalysis:
    """Analysis of a log file or test output."""
    log_source: str = ""
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    skipped_tests: int = 0
    error_tests: int = 0
    failures: List[ParsedFailure] = field(default_factory=list)
    error_messages: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    
    def __post_init__(self):
        if not self.log_source:
            raise ValueError("Log source is required")


class LogParser:
    """
    Parses test logs and output to extract structured failure information.
    
    Supports parsing:
    - Pytest output
    - NPM test output
    - Python tracebacks
    - Stack traces
    - Assertion error messages
    """
    
    def __init__(self):
        logger = __import__('logging').getLogger(__name__)
        self.logger = logger
    
    def parse_pytest_output(self, stdout: str, stderr: str) -> LogAnalysis:
        """
        Parse pytest output into structured failure information.
        
        Args:
            stdout: Standard output from pytest
            stderr: Standard error from pytest
            
        Returns:
            LogAnalysis with parsed failures and statistics
        """
        analysis = LogAnalysis(
            log_source="pytest",
            duration_seconds=0.0  # We don't extract timing from this basic parse
        )
        
        # Combine stdout and stderr for parsing
        full_output = stdout + "\n" + stderr
        
        # Parse test summary line
        summary_match = re.search(r'(\d+)\s+passed\s+(\d+)\s+failed\s+(\d+)\s+skipped\s+(\d+)\s+errors', full_output)
        if summary_match:
            analysis.passed_tests = int(summary_match.group(1))
            analysis.failed_tests = int(summary_match.group(2))
            analysis.skipped_tests = int(summary_match.group(3))
            analysis.error_tests = int(summary_match.group(4))
            analysis.total_tests = analysis.passed_tests + analysis.failed_tests + analysis.skipped_tests + analysis.error_tests
        
        # Parse individual failures
        # Look for patterns like:
        # _______ TestClass.test_method _______
        # or
        # FAILED test_file.py::test_function - AssertionError: message
        failure_patterns = [
            # Standard pytest failure format
            r'_{5,}\s+(.*?)\s+_{5,}\n(.*?)\n_{5,}',  # ____.test_name____\n traceback \n ____
            # Alternative format
            r'FAILED\s+(.*?)\s+-\s+(.+?)(?=\n{2,}|$\n\w)',  # FAILED test_name - error
        ]
        
        # More robust parsing: split by test boundaries
        # Pytest separates tests with blank lines or ===
        test_sections = re.split(r'\n={5,}\n|\n_{5,}\n|\n\n(?=\w+\s+::\w+)', full_output)
        
        for section in test_sections:
            section = section.strip()
            if not section or len(section) < 10:
                continue
                
            # Look for test name pattern
            test_match = re.search(r'(\S+::\S+|\S+\.\w+)', section)
            if test_match:
                test_name = test_match.group(1)
                
                # Determine failure type and extract info
                if "FAILED" in section:
                    failure_type = "assertion" if "AssertionError" in section else "exception"
                    analysis.failed_tests += 1  # We'll adjust this based on actual parsing later
                    
                    # Extract traceback
                    traceback_lines = []
                    lines = section.split('\n')
                    in_traceback = False
                    for line in lines:
                        if line.startswith("___") or line.startswith("==="):
                            in_traceback = not in_traceback
                            continue
                        if in_traceback and line.strip():
                            traceback_lines.append(line)
                        elif "AssertionError:" in line or "." in line and "(" in line and ")" in line:
                            # Possible start of traceback
                            pass
                    
                    # Extract assertion details
                    expected = None
                    actual = None
                    assertion_match = re.search(r'AssertionError:?\s*(.*)', section)
                    if assertion_match:
                        assertion_text = assertion_match.group(1)
                        # Try to parse expected vs actual from common patterns
                        # "assert expected == actual" -> expected, actual
                        if "==" in assertion_text:
                            parts = assertion_text.split("==")
                            if len(parts) == 2:
                                expected = parts[0].strip()
                                actual = parts[1].strip()
                        elif "!=" in assertion_text:
                            parts = assertion_text.split("!=")
                            if len(parts) == 2:
                                expected = parts[0].strip()
                                actual = parts[1].strip()
                    
                    failure = ParsedFailure(
                        test_name=test_name,
                        failure_type=failure_type,
                        message=section[:200],  # First 200 chars as message
                        traceback=traceback_lines[:10],  # Limit traceback length
                        expected=expected,
                        actual=actual
                    )
                    
                    # Try to extract file and line number
                    file_match = re.search(r'(\S+:\d+)', section)
                    if file_match:
                        file_line = file_match.group(1)
                        if ':' in file_line:
                            file_part, line_part = file_line.rsplit(':', 1)
                            try:
                                failure.file_path = Path(file_part.strip())
                                failure.line_number = int(line_part.strip())
                            except ValueError:
                                pass  # Keep as None if not a valid integer
                    
                    analysis.failures.append(failure)
                
                elif "PASSED" in section:
                    analysis.passed_tests += 1
                elif "SKIPPED" in section:
                    analysis.skipped_tests += 1
                elif "ERROR" in section:
                    analysis.error_tests += 1
                    failure_type = "error"
                    failure = ParsedFailure(
                        test_name=test_name,
                        failure_type=failure_type,
                        message=section[:200],
                        traceback=[]
                    )
                    analysis.failures.append(failure)
        
        # If we didn't parse any specific failures but we know there were failures,
        # create generic failure entries
        if analysis.failed_tests > 0 and len([f for f in analysis.failures if f.failure_type in ["assertion", "exception"]]) == 0:
            # Create placeholder failures based on the count
            for i in range(analysis.failed_tests):
                failure = ParsedFailure(
                    test_name=f"unknown_failed_test_{i+1}",
                    failure_type="unknown",
                    message="Test failed but details could not be parsed from output"
                )
                analysis.failures.append(failure)
        
        # Extract any remaining error messages
        error_lines = [line.strip() for line in stderr.split('\n') if line.strip() and ('error' in line.lower() or 'exception' in line.lower())]
        analysis.error_messages = error_lines[:5]  # Limit to 5 error messages
        
        # Extract warnings
        warning_lines = [line.strip() for line in (stdout + stderr).split('\n') 
                        if line.strip() and 'warning' in line.lower()]
        analysis.warnings = warning_lines[:5]  # Limit to 5 warnings
        
        # Try to extract duration
        duration_match = re.search(r'(\d+\.\d+)\s+seconds', full_output)
        if duration_match:
            try:
                analysis.duration_seconds = float(duration_match.group(1))
            except ValueError:
                pass
        
        return analysis

    def parse_json_report(self, json_path: Path) -> LogAnalysis:
        """Parse pytest --json-report output into structured analysis."""
        data = json.loads(json_path.read_text(encoding="utf-8"))
        analysis = LogAnalysis(log_source=f"pytest-json:{json_path}")

        summary = data.get("summary", {})
        analysis.passed_tests = int(summary.get("passed", 0))
        analysis.failed_tests = int(summary.get("failed", 0))
        analysis.skipped_tests = int(summary.get("skipped", 0))
        analysis.error_tests = int(summary.get("error", 0))
        analysis.total_tests = (
            analysis.passed_tests
            + analysis.failed_tests
            + analysis.skipped_tests
            + analysis.error_tests
        )
        analysis.duration_seconds = float(summary.get("duration", 0.0) or 0.0)

        for test_item in data.get("tests", []):
            outcome = str(test_item.get("outcome", "")).lower()
            if outcome not in {"failed", "error"}:
                continue

            nodeid = test_item.get("nodeid", "unknown_test")
            call_info = test_item.get("call", {}) or {}
            longrepr = call_info.get("longrepr") or test_item.get("longrepr") or ""
            traceback_lines = str(longrepr).splitlines()
            message = traceback_lines[-1] if traceback_lines else "pytest json failure"

            failure = ParsedFailure(
                test_name=nodeid,
                failure_type="exception" if outcome == "error" else "assertion",
                message=message[:500],
                traceback=traceback_lines[-20:],
            )

            line = int(call_info.get("lineno", 0) or 0)
            if line > 0:
                failure.line_number = line
            path = test_item.get("path")
            if path:
                failure.file_path = Path(path)

            analysis.failures.append(failure)

        return analysis
    
    def parse_traceback(self, traceback_text: str) -> List[ParsedFailure]:
        """
        Parse a Python traceback into structured failures.
        
        Args:
            traceback_text: Python traceback text
            
        Returns:
            List of ParsedFailure objects
        """
        failures = []
        
        # Split by exception boundaries if multiple exceptions
        tb_sections = re.split(r'\n{2,}(?=Traceback)', traceback_text)
        
        for tb_section in tb_sections:
            tb_section = tb_section.strip()
            if not tb_section or not tb_section.startswith("Traceback"):
                continue
            
            # Extract the last frame (where the exception occurred)
            lines = tb_section.split('\n')
            last_frame_line = None
            for i, line in enumerate(lines):
                if line.strip().startswith('File "') and 'line ' in line:
                    last_frame_line = line
            
            if last_frame_line:
                # Parse file and line number
                file_match = re.search(r'File "([^"]+)", line (\d+)', last_frame_line)
                if file_match:
                    file_path = Path(file_match.group(1))
                    line_number = int(file_match.group(2))
                    
                    # Find the exception type and message
                    exception_lines = [line for line in lines if ':' in line and not line.startswith(' ') and not line.startswith('Traceback')]
                    if exception_lines:
                        last_exception_line = exception_lines[-1].strip()
                        if ': ' in last_exception_line:
                            exc_type, exc_message = last_exception_line.split(': ', 1)
                        else:
                            exc_type = last_exception_line
                            exc_message = ""
                        
                        failure = ParsedFailure(
                            test_name="unknown_from_traceback",
                            failure_type="exception",
                            file_path=file_path,
                            line_number=line_number,
                            message=exc_message,
                            traceback=[tb_section],
                            expected=None,
                            actual=None
                        )
                        failures.append(failure)
        
        return failures
    
    def extract_assertion_details(self, assertion_text: str) -> tuple[Optional[str], Optional[str]]:
        """
        Extract expected and actual values from assertion text.
        
        Args:
            assertion_text: Text from an assertion error
            
        Returns:
            Tuple of (expected, actual) strings
        """
        # Common assertion patterns
        patterns = [
            # assert expected == actual
            (r'(.+?)\s*==\s*(.+)', lambda m: (m.group(1).strip(), m.group(2).strip())),
            # assert expected != actual
            (r'(.+?)\s*!=\s*(.+)', lambda m: (m.group(1).strip(), m.group(2).strip())),
            # assert expected < actual
            (r'(.+?)\s*<\s*(.+)', lambda m: (m.group(1).strip(), m.group(2).strip())),
            # assert expected > actual
            (r'(.+?)\s*>\s*(.+)', lambda m: (m.group(1).strip(), m.group(2).strip())),
            # assert expected in actual
            (r'(.+?)\s*in\s+(.+)', lambda m: (m.group(1).strip(), m.group(2).strip())),
            # assert isinstance(obj, type)
            (r'isinstance\s*\(\s*(.+?)\s*,\s*(.+?)\s*\)', lambda m: (m.group(1).strip(), m.group(2).strip())),
        ]
        
        for pattern, extractor in patterns:
            match = re.search(pattern, assertion_text, re.IGNORECASE)
            if match:
                try:
                    return extractor(match)
                except Exception:
                    continue
        
        return None, None


def parse_test_output(stdout: str, stderr: str) -> LogAnalysis:
    """Convenience function to parse test output."""
    parser = LogParser()
    return parser.parse_pytest_output(stdout, stderr)


def parse_traceback_to_failures(traceback_text: str) -> List[ParsedFailure]:
    """Convenience function to parse traceback to failures."""
    parser = LogParser()
    return parser.parse_traceback(traceback_text)


def extract_assertion_info(assertion_text: str) -> tuple[Optional[str], Optional[str]]:
    """Convenience function to extract assertion details."""
    parser = LogParser()
    return parser.extract_assertion_details(assertion_text)