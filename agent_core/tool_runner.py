"""
Command execution sandbox for safely running tools and scripts.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import subprocess
import time
import os
import signal
import threading
from queue import Queue, Empty
import shlex
import re
from .schemas import TaskState
from .orchestrator import Orchestrator
logger = __import__('logging').getLogger(__name__)


@dataclass
class CommandResult:
    """Result of executing a command."""
    command: str
    return_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool = False
    killed_by_signal: Optional[int] = None
    
    @property
    def success(self) -> bool:
        return self.return_code == 0 and not self.timed_out


@dataclass
class ToolPermission:
    """Permission rule for tool execution."""
    tool_pattern: str  # e.g., "pytest", "npm test", "pip install"
    allowed: bool
    requires_confirmation: bool = False
    description: str = ""


class ToolRunner:
    """
    Secure tool execution sandbox.
    
    The tool runner provides a safe environment for executing commands
    with timeout controls, output capturing, and permission restrictions.
    """
    
    def __init__(self, orchestrator: Orchestrator, workspace_root: Optional[Path] = None):
        self.orchestrator = orchestrator
        self.workspace_root = workspace_root or Path.cwd()
        self.permission_rules: List[ToolPermission] = []
        self._setup_default_permissions()
        logger = __import__('logging').getLogger(__name__)
        self.logger = logger
    
    def _setup_default_permissions(self):
        """Set up default permission rules for common tools."""
        # Safe commands - allowed by default
        self.permission_rules.extend([
            ToolPermission("ls", True, False, "List directory contents"),
            ToolPermission("find", True, False, "Find files"),
            ToolPermission("grep", True, False, "Search file contents"),
            ToolPermission("cat", True, False, "Display file contents"),
            ToolPermission("echo", True, False, "Echo text"),
            ToolPermission("python", True, False, "Run Python interpreter"),
            ToolPermission("python3", True, False, "Run Python 3 interpreter"),
            ToolPermission("pip", True, False, "Python package installer"),
            ToolPermission("pytest", True, False, "Python test runner"),
            ToolPermission("git", True, False, "Git version control (read-only ops)"),
            ToolPermission("mkdir", True, False, "Create directories"),
            ToolPermission("rmdir", True, False, "Remove empty directories"),
            ToolPermission("cp", True, False, "Copy files"),
        ])
        
        # Commands requiring confirmation
        self.permission_rules.extend([
            ToolPermission("rm -rf", True, True, "Recursive force delete"),
            ToolPermission("git push", True, True, "Push to remote repository"),
            ToolPermission("git push --force", True, True, "Force push to remote"),
            ToolPermission("pip install", True, True, "Install Python packages"),
            ToolPermission("npm install", True, True, "Install Node.js packages"),
        ])
        
        # Dangerous commands - not allowed by default
        self.permission_rules.extend([
            ToolPermission("mkfs", False, False, "Format filesystem"),
            ToolPermission("dd", False, False, "Low-level disk copying"),
            ToolPermission(">", False, False, "Output redirection (potentially dangerous)"),
            ToolPermission("sudo", False, False, "Superuser privileges"),
            ToolPermission("su", False, False, "Switch user"),
            ToolPermission("chmod 777", False, False, "Dangerous permissions"),
            ToolPermission("chown -R", False, False, "Recursive ownership change"),
        ])
    
    def add_permission_rule(self, pattern: str, allowed: bool, 
                           requires_confirmation: bool = False, 
                           description: str = ""):
        """Add a custom permission rule."""
        self.permission_rules.append(ToolPermission(
            tool_pattern=pattern,
            allowed=allowed,
            requires_confirmation=requires_confirmation,
            description=description
        ))
    
    def _check_permission(self, command: str) -> Tuple[bool, bool]:
        """
        Check if a command is allowed to run.
        
        Returns:
            Tuple of (is_allowed, requires_confirmation)
        """
        command_lower = command.lower().strip()
        try:
            command_tokens = [t.lower() for t in shlex.split(command)]
        except ValueError:
            return (False, False)
        
        # Check each rule - sort by pattern length (descending) for specificity
        # Longer patterns are more specific and should be checked first
        sorted_rules = sorted(self.permission_rules, key=lambda r: len(r.tool_pattern), reverse=True)
        
        for rule in sorted_rules:
            pattern = rule.tool_pattern.lower().strip()
            pattern_tokens = pattern.split()
            if pattern_tokens and len(command_tokens) >= len(pattern_tokens):
                if command_tokens[:len(pattern_tokens)] == pattern_tokens:
                    return (rule.allowed, rule.requires_confirmation)

            # Fallback for single-token patterns and symbolic operators.
            if len(pattern_tokens) == 1:
                token = re.escape(pattern_tokens[0])
                if re.search(rf"(^|\s){token}($|\s)", command_lower):
                    return (rule.allowed, rule.requires_confirmation)

            if pattern in {">", "<", "|", "&&", "||"} and pattern in command:
                return (rule.allowed, rule.requires_confirmation)
        
        # Default: not allowed if no matching rule
        return (False, False)

    def _resolve_workspace_cwd(self, cwd: Optional[Path]) -> Optional[Path]:
        """Resolve cwd and enforce that it stays within workspace_root."""
        try:
            root = self.workspace_root.resolve()
            candidate = (cwd or self.workspace_root).resolve()
        except Exception:
            return None

        try:
            candidate.relative_to(root)
        except ValueError:
            return None

        return candidate if candidate.exists() else root
    
    def run_command(self, command: str, timeout: Optional[float] = None,
                   cwd: Optional[Path] = None, 
                   env: Optional[Dict[str, str]] = None,
                   require_confirmation: bool = False) -> CommandResult:
        """
        Run a command in the sandbox.
        
        Args:
            command: Command to execute
            timeout: Timeout in seconds (None for no timeout)
            cwd: Working directory (defaults to workspace_root)
            env: Environment variables (defaults to current env)
            require_confirmation: Whether to simulate user confirmation
            
        Returns:
            CommandResult with output and status
        """
        # Check permissions
        is_allowed, needs_confirmation = self._check_permission(command)
        
        if not is_allowed:
            return CommandResult(
                command=command,
                return_code=-1,
                stdout="",
                stderr=f"Command not permitted by security policy: {command}",
                duration_seconds=0.0,
                timed_out=False
            )
        
        if needs_confirmation and not require_confirmation:
            return CommandResult(
                command=command,
                return_code=-2,
                stdout="",
                stderr=f"Command requires confirmation: {command}",
                duration_seconds=0.0,
                timed_out=False
            )
        
        # Set up working directory
        work_dir = self._resolve_workspace_cwd(cwd)
        if work_dir is None:
            return CommandResult(
                command=command,
                return_code=-4,
                stdout="",
                stderr="Working directory must resolve inside workspace_root",
                duration_seconds=0.0,
                timed_out=False,
            )
        
        # Set up environment
        run_env = env.copy() if env else os.environ.copy()
        
        # Execute command
        start_time = time.time()
        try:
            # Use subprocess with timeout and output capture
            process = subprocess.Popen(
                shlex.split(command),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=work_dir,
                env=run_env,
                preexec_fn=os.setsid  # Use process group for clean killing
            )
            
            # Wait for completion with timeout
            try:
                stdout, stderr = process.communicate(timeout=timeout)
                return_code = process.returncode
                timed_out = False
            except subprocess.TimeoutExpired:
                # Kill the process group
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    # Give it a moment to terminate gracefully
                    time.sleep(0.5)
                    # Force kill if still alive
                    if process.poll() is None:
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except (OSError, ProcessLookupError):
                    pass  # Process already died
                
                # Get whatever output we can
                try:
                    stdout, stderr = process.communicate(timeout=1.0)
                except subprocess.TimeoutExpired:
                    stdout, stderr = "", ""
                
                return_code = -1
                timed_out = True
                logger.warning(f"Command timed out after {timeout} seconds: {command}")
            
            duration_seconds = time.time() - start_time
            
            # Check if killed by signal
            killed_by_signal = None
            if return_code < 0:
                killed_by_signal = -return_code  # Negative return code indicates signal
            
            result = CommandResult(
                command=command,
                return_code=return_code,
                stdout=ToolRunner._strip_to_reasonable_length(stdout),
                stderr=ToolRunner._strip_to_reasonable_length(stderr),
                duration_seconds=duration_seconds,
                timed_out=timed_out,
                killed_by_signal=killed_by_signal
            )
            
            logger.info(f"Command executed: {command} (return_code={return_code}, duration={duration_seconds:.2f}s)")
            return result
            
        except Exception as e:
            duration_seconds = time.time() - start_time
            logger.error(f"Failed to execute command '{command}': {e}")
            return CommandResult(
                command=command,
                return_code=-3,
                stdout="",
                stderr=f"Failed to execute command: {str(e)}",
                duration_seconds=duration_seconds,
                timed_out=False
            )
    
    def run_pytest(self, test_path: str = ".", timeout: float = 60.0,
                  extra_args: Optional[List[str]] = None) -> CommandResult:
        """Run pytest with standard arguments."""
        args = ["pytest", "-v"]
        if extra_args:
            args.extend(extra_args)
        args.append(test_path)
        command = " ".join(args)
        return self.run_command(command, timeout=timeout)
    
    def run_pip_install(self, package: str, timeout: float = 120.0,
                       upgrade: bool = False) -> CommandResult:
        """Run pip install with confirmation simulation."""
        args = ["pip", "install"]
        if upgrade:
            args.append("--upgrade")
        args.append(package)
        command = " ".join(args)
        return self.run_command(command, timeout=timeout, require_confirmation=True)
    
    def run_npm_install(self, timeout: float = 120.0) -> CommandResult:
        """Run npm install with confirmation simulation."""
        command = "npm install"
        return self.run_command(command, timeout=timeout, require_confirmation=True)
    
    def run_git_command(self, git_args: List[str], timeout: float = 30.0) -> CommandResult:
        """Run a git command."""
        args = ["git"] + git_args
        command = " ".join(args)
        # Most git read operations are safe, but write operations need confirmation
        write_ops = {"push", "commit", "merge", "rebase", "reset"}
        needs_confirmation = any(arg in write_ops for arg in git_args)
        return self.run_command(command, timeout=timeout, require_confirmation=needs_confirmation)
    
    @staticmethod
    def _strip_to_reasonable_length(text: str, max_length: int = 10000) -> str:
        """Strip output to reasonable length to prevent memory issues."""
        if len(text) <= max_length:
            return text
        return text[:max_length] + f"\n... [Output truncated, {len(text) - max_length} more characters]"


# Helper function for easy access
def run_command_safely(command: str, timeout: Optional[float] = None,
                      cwd: Optional[Path] = None,
                      require_confirmation: bool = False) -> CommandResult:
    """
    Convenience function for running a command safely.
    
    Creates a temporary ToolRunner instance for one-off command execution.
    """
    runner = ToolRunner(Orchestrator())
    return runner.run_command(command, timeout=timeout, cwd=cwd, require_confirmation=require_confirmation)