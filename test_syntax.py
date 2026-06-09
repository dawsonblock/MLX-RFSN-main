#!/usr/bin/env python3
import ast
import sys

def check_syntax(filepath):
    """Check Python syntax of a file."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        ast.parse(content)
        return True, "Syntax OK"
    except SyntaxError as e:
        return False, f"Syntax Error: {e}"
    except Exception as e:
        return False, f"Error: {e}"

if __name__ == "__main__":
    files_to_check = [
        "rfsn_v10/runtime/engine.py",
        "rfsn_v10/kv_manager.py", 
        "rfsn_v10/clickhouse_client.py",
        "rfsn_v10/async_writer.py",
        "agent_core/tool_runner.py",
        "agent_core/orchestrator.py",
        "tools/test_runner.py",
        "tools/log_parser.py"
    ]
    
    all_passed = True
    for filepath in files_to_check:
        passed, message = check_syntax(filepath)
        if passed:
            print(f"✓ {filepath}: {message}")
        else:
            print(f"✗ {filepath}: {message}")
            all_passed = False
    
    if all_passed:
        print("\nAll files passed syntax check!")
        sys.exit(0)
    else:
        print("\nSome files failed syntax check!")
        sys.exit(1)