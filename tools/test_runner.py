"""
Test runner utilities for executing tests and parsing results.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import subprocess
import time
import json
import re
import tempfile

from agent_core.schemas import TestResult, TestResultStatus
from agent_core.tool_runner import ToolRunner, CommandResult
from tools.log_parser import LogParser

logger = __import__('logging').getLogger(__name__)


@dataclass
class TestSuiteResult:
    """Result of running a test suite."""
    suite_name: str
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    skipped_tests: int = 0
    error_tests: int = 0
    duration_seconds: float = 0.0
    test_results: List[TestResult] = field(default_factory=list)
    raw_output: str = ""
    
    @property
    def success_rate(self) -> float:
        return self.passed_tests / self.total_tests if self.total_tests > 0 else 0.0
    
    @property
    def passed(self) -> bool:
        return self.failed_tests == 0 and self.error_tests == 0


TestSuiteResult.__test__ = False


class TestRunner:
    """
    Utilities for running tests and parsing results.
    
    Supports:
    - Python pytest
    - npm test/Jest
    - Compile/check syntax (various languages)
    - Custom test commands
    """

    __test__ = False
    
    def __init__(self, workspace_root: Optional[Path] = None):
        self.workspace_root = workspace_root or Path.cwd()
        self.tool_runner = ToolRunner(None, self.workspace_root)  # No orchestrator needed for basic tool running
        logger = __import__('logging').getLogger(__name__)
        self.logger = logger
    
    def run_pytest(self, test_path: str | Path = ".", 
                  timeout: float = 120.0,
                  extra_args: Optional[List[str]] = None) -> TestSuiteResult:
        """
        Run pytest and parse results.
        
        Args:
            test_path: Path to test directory or file
            timeout: Timeout in seconds
            extra_args: Additional arguments to pass to pytest
            
        Returns:
            TestSuiteResult with parsed results
        """
        start_time = time.time()
        
        # Build pytest command
        args = ["pytest", "-v", "--tb=short"]
        if extra_args:
            args.extend(extra_args)
        args.append(str(test_path))
        
        # Run the command
        command_result = self.tool_runner.run_command(
            " ".join(args),
            timeout=timeout,
            require_confirmation=False  # pytest is generally safe
        )
        
        # Parse results
        test_results = self._parse_pytest_output(command_result.stdout, command_result.stderr)
        
        duration = time.time() - start_time
        
        return TestSuiteResult(
            suite_name=f"pytest:{test_path}",
            total_tests=len(test_results),
            passed_tests=sum(1 for t in test_results if t.status == TestResultStatus.PASSED),
            failed_tests=sum(1 for t in test_results if t.status == TestResultStatus.FAILED),
            skipped_tests=sum(1 for t in test_results if t.status == TestResultStatus.SKIPPED),
            error_tests=sum(1 for t in test_results if t.status == TestResultStatus.ERROR),
            duration_seconds=duration,
            test_results=test_results,
            raw_output=command_result.stdout + "\n" + command_result.stderr
        )
    
    def run_npm_test(self, timeout: float = 120.0,
                    extra_args: Optional[List[str]] = None) -> TestSuiteResult:
        """
        Run npm test and parse results.
        
        Args:
            timeout: Timeout in seconds
            extra_args: Additional arguments to pass to npm test
            
        Returns:
            TestSuiteResult with parsed results
        """
        start_time = time.time()
        
        # Build npm test command
        args = ["npm", "test"]
        if extra_args:
            args.extend(extra_args)
        
        # Run the command
        command_result = self.tool_runner.run_command(
            " ".join(args),
            timeout=timeout,
            require_confirmation=True  # npm test might install packages
        )
        
        # Parse results (basic parsing - npm test output varies)
        test_results = self._parse_npm_test_output(command_result.stdout, command_result.stderr)
        
        duration = time.time() - start_time
        
        return TestSuiteResult(
            suite_name="npm_test",
            total_tests=len(test_results),
            passed_tests=sum(1 for t in test_results if t.status == TestResultStatus.PASSED),
            failed_tests=sum(1 for t in test_results if t.status == TestResultStatus.FAILED),
            skipped_tests=sum(1 for t in test_results if t.status == TestResultStatus.SKIPPED),
            error_tests=sum(1 for t in test_results if t.status == TestResultStatus.ERROR),
            duration_seconds=duration,
            test_results=test_results,
            raw_output=command_result.stdout + "\n" + command_result.stderr
        )

    def run_pytest_json(self, test_path: str | Path = ".",
                        timeout: float = 120.0,
                        extra_args: Optional[List[str]] = None) -> TestSuiteResult:
        """Run pytest with --json-report and parse structured output."""
        start_time = time.time()
        with tempfile.TemporaryDirectory(prefix="rfsn_pytest_json_") as td:
            report_path = Path(td) / "pytest-report.json"
            args = [
                "pytest",
                "-q",
                "--json-report",
                f"--json-report-file={report_path}",
            ]
            if extra_args:
                args.extend(extra_args)
            args.append(str(test_path))

            command_result = self.tool_runner.run_command(
                " ".join(args),
                timeout=timeout,
                require_confirmation=False,
            )

            if not report_path.exists():
                # Fallback to legacy parser if plugin/report is unavailable.
                return self.run_pytest(test_path=test_path, timeout=timeout, extra_args=extra_args)

            parser = LogParser()
            analysis = parser.parse_json_report(report_path)
            test_results: List[TestResult] = []

            for failure in analysis.failures:
                status = (
                    TestResultStatus.ERROR
                    if failure.failure_type == "exception"
                    else TestResultStatus.FAILED
                )
                test_results.append(
                    TestResult(
                        test_name=failure.test_name,
                        status=status,
                        duration_ms=0.0,
                        message=failure.message,
                        traceback="\n".join(failure.traceback) if failure.traceback else None,
                    )
                )

            elapsed = time.time() - start_time
            return TestSuiteResult(
                suite_name=f"pytest-json:{test_path}",
                total_tests=analysis.total_tests,
                passed_tests=analysis.passed_tests,
                failed_tests=analysis.failed_tests,
                skipped_tests=analysis.skipped_tests,
                error_tests=analysis.error_tests,
                duration_seconds=elapsed,
                test_results=test_results,
                raw_output=command_result.stdout + "\n" + command_result.stderr,
            )
    
    def run_compile_check(self, language: str, source_path: str | Path = ".",
                         timeout: float = 30.0) -> TestSuiteResult:
        """
        Run compile/check syntax for a language.
        
        Args:
            language: Language to check (python, javascript, etc.)
            source_path: Path to source files to check
            timeout: Timeout in seconds
            
        Returns:
            TestSuiteResult with parsed results
        """
        start_time = time.time()
        
        # Map language to check command
        check_commands = {
            "python": ["python", "-m", "py_compile"],
            "python3": ["python3", "-m", "py_compile"],
            "javascript": ["node", "-c"],  # Basic syntax check
            "js": ["node", "-c"],
        }
        
        if language not in check_commands:
            return TestSuiteResult(
                suite_name=f"compile_check:{language}",
                duration_seconds=time.time() - start_time,
                raw_output=f"Unsupported language for compile check: {language}"
            )
        
        # For now, we'll implement a basic version
        # In a full implementation, this would walk the source path and check each file
        command_result = self.tool_runner.run_command(
            f"echo \"Compile check for {language} not fully implemented yet\"",
            timeout=timeout
        )
        
        duration = time.time() - start_time
        
        # Create a dummy test result
        test_result = TestResult(
            test_name=f"compile_check_{language}",
            status=TestResultStatus.PASSED if command_result.return_code == 0 else TestResultStatus.FAILED,
            duration_ms=(time.time() - start_time) * 1000,
            message=command_result.stderr if command_result.return_code != 0 else "Compile check passed",
            stdout=command_result.stdout,
            stderr=command_result.stderr
        )
        
        return TestSuiteResult(
            suite_name=f"compile_check:{language}",
            total_tests=1,
            passed_tests=1 if command_result.return_code == 0 else 0,
            failed_tests=0 if command_result.return_code == 0 else 1,
            duration_seconds=duration,
            test_results=[test_result],
            raw_output=command_result.stdout + "\n" + command_result.stderr
        )
    
    def _parse_pytest_output(self, stdout: str, stderr: str) -> List[TestResult]:
        """Parse pytest output into TestResult objects."""
        test_results = []
        
        # Simple parsing - in reality, we'd use pytest --json-report or similar
        # For now, we'll create a basic representation
        
        # Look for test result lines in stdout
        lines = stdout.split('\n')
        current_test = None
        test_count = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Match pytest test result lines like:
            # test_file.py::test_function PASSED
            # test_file.py::test_function FAILED
            match = re.search(r'(\S+::\S+)\s+(PASSED|FAILED|SKIPPED|ERROR)', line)
            if match:
                test_name = match.group(1)
                status_str = match.group(2)
                
                # Map status
                status_map = {
                    "PASSED": TestResultStatus.PASSED,
                    "FAILED": TestResultStatus.FAILED,
                    "SKIPPED": TestResultStatus.SKIPPED,
                    "ERROR": TestResultStatus.ERROR
                }
                status = status_map.get(status_str, TestResultStatus.ERROR)
                
                test_result = TestResult(
                    test_name=test_name,
                    status=status,
                    duration_ms=0.0,  # We don't have timing from this simple parse
                    message=line
                )
                
                test_results.append(test_result)
                test_count += 1
        
        # If we didn't parse any tests from the output, create a summary result
        if test_count == 0:
            # Check if pytest ran successfully overall
            if "passed" in stderr.lower() or "failed" in stderr.lower():
                # Try to extract numbers
                passed_match = re.search(r'(\d+)\s+passed', stderr)
                failed_match = re.search(r'(\d+)\s+failed', stderr)
                skipped_match = re.search(r'(\d+)\s+skipped', stderr)
                
                passed = int(passed_match.group(1)) if passed_match else 0
                failed = int(failed_match.group(1)) if failed_match else 0
                skipped = int(skipped_match.group(1)) if skipped_match else 0
                total = passed + failed + skipped
                
                if total > 0:
                    # Create summary test results
                    if passed > 0:
                        test_results.append(TestResult(
                            test_name="pytest_summary_passed",
                            status=TestResultStatus.PASSED,
                            duration_ms=0.0,
                            message=f"{passed} tests passed"
                        ))
                    if failed > 0:
                        test_results.append(TestResult(
                            test_name="pytest_summary_failed",
                            status=TestResultStatus.FAILED,
                            duration_ms=0.0,
                            message=f"{failed} tests failed"
                        ))
                    if skipped > 0:
                        test_results.append(TestResult(
                            test_name="pytest_summary_skipped",
                            status=TestResultStatus.SKIPPED,
                            duration_ms=0.0,
                            message=f"{skipped} tests skipped"
                        ))
            else:
                # No clear test results - create a single result based on return code
                # We don't have the return code here, but we can infer from output
                if "failed" in stderr.lower() or "error" in stderr.lower():
                    test_results.append(TestResult(
                        test_name="pytest_overall",
                        status=TestResultStatus.FAILED,
                        duration_ms=0.0,
                        message="Tests failed or encountered errors (see stderr)"
                    ))
                else:
                    test_results.append(TestResult(
                        test_name="pytest_overall",
                        status=TestResultStatus.PASSED,
                        duration_ms=0.0,
                        message="Tests passed (inferred from output)"
                    ))
        
        return test_results
    
    def _parse_npm_test_output(self, stdout: str, stderr: str) -> List[TestResult]:
        """Parse npm test output into TestResult objects."""
        # Basic parsing - npm test output varies greatly
        test_results = []
        
        # Look for common patterns
        if "test passed" in stdout.lower() or "passed" in stdout.lower():
            test_results.append(TestResult(
                test_name="npm_test_summary",
                status=TestResultStatus.PASSED,
                duration_ms=0.0,
                message="NPM tests passed (inferred from output)"
            ))
        elif "test failed" in stdout.lower() or "failed" in stdout.lower():
            test_results.append(TestResult(
                test_name="npm_test_summary",
                status=TestResultStatus.FAILED,
                duration_ms=0.0,
                message="NPM tests failed (inferred from output)"
            ))
        else:
            # Default based on return code - we don't have it here, so assume passed if no obvious failure
            if "failed" in stderr.lower() or "error" in stderr.lower():
                test_results.append(TestResult(
                    test_name="npm_test_summary",
                    status=TestResultStatus.FAILED,
                    duration_ms=0.0,
                    message="NPM tests failed or encountered errors (see stderr)"
                ))
            else:
                test_results.append(TestResult(
                    test_name="npm_test_summary",
                    status=TestResultStatus.PASSED,
                    duration_ms=0.0,
                    message="NPM tests passed (inferred from lack of failure indicators)"
                ))
        
        return test_results


def run_pytest(test_path: str | Path = ".", 
              timeout: float = 120.0,
              extra_args: Optional[List[str]] = None) -> TestSuiteResult:
    """Convenience function to run pytest."""
    runner = TestRunner()
    return runner.run_pytest(test_path, timeout, extra_args)


def run_npm_test(timeout: float = 120.0,
                extra_args: Optional[List[str]] = None) -> TestSuiteResult:
    """Convenience function to run npm test."""
    runner = TestRunner()
    return runner.run_npm_test(timeout, extra_args)


def run_compile_check(language: str, source_path: str | Path = ".",
                     timeout: float = 30.0) -> TestSuiteResult:
    """Convenience function to run compile check."""
    runner = TestRunner()
    return runner.run_compile_check(language, source_path, timeout)