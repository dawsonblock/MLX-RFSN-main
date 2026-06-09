from __future__ import annotations

import json

from tools.log_parser import LogParser


def test_parse_json_report_extracts_summary_and_failures(tmp_path):
    report_path = tmp_path / "pytest-report.json"
    report = {
        "summary": {
            "passed": 2,
            "failed": 1,
            "skipped": 1,
            "error": 1,
            "duration": 3.14,
        },
        "tests": [
            {
                "nodeid": "tests/test_a.py::test_ok",
                "outcome": "passed",
            },
            {
                "nodeid": "tests/test_b.py::test_fail",
                "outcome": "failed",
                "path": "tests/test_b.py",
                "call": {
                    "lineno": 42,
                    "longrepr": "assert 1 == 2\nE   AssertionError: mismatch",
                },
            },
            {
                "nodeid": "tests/test_c.py::test_error",
                "outcome": "error",
                "call": {
                    "longrepr": "RuntimeError: boom",
                },
            },
        ],
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")

    parser = LogParser()
    analysis = parser.parse_json_report(report_path)

    assert analysis.total_tests == 5
    assert analysis.passed_tests == 2
    assert analysis.failed_tests == 1
    assert analysis.skipped_tests == 1
    assert analysis.error_tests == 1
    assert len(analysis.failures) == 2
    assert analysis.failures[0].test_name == "tests/test_b.py::test_fail"
    assert analysis.failures[0].line_number == 42
