"""
Repository extraction utilities for getting file trees and basic info.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import mimetypes
import os

logger = __import__('logging').getLogger(__name__)


@dataclass
class FileInfo:
    """Information about a file in the repository."""
    path: Path
    size: int
    is_file: bool
    is_dir: bool
    mime_type: Optional[str] = None
    extension: Optional[str] = None
    
    def __post_init__(self):
        if self.is_file:
            self.extension = self.path.suffix.lower()
            if self.extension:
                self.mime_type = mimetypes.guess_type(self.path)[0]


@dataclass
class RepoInfo:
    """Information about a repository."""
    root_path: Path
    total_files: int = 0
    total_dirs: int = 0
    total_size: int = 0
    files: List[FileInfo] = field(default_factory=list)
    directories: List[FileInfo] = field(default_factory=list)
    file_types: Dict[str, int] = field(default_factory=dict)
    
    def __post_init__(self):
        self._analyze()
    
    def _analyze(self):
        """Analyze the repository structure."""
        if not self.root_path.exists():
            logger.warning(f"Repository path does not exist: {self.root_path}")
            return
        
        for item in self.root_path.rglob("*"):
            try:
                # Skip common ignore patterns
                if any(part.startswith('.') and part not in {'.'} for part in item.parts):
                    continue
                
                stat = item.stat()
                is_file = item.is_file()
                is_dir = item.is_dir()
                
                file_info = FileInfo(
                    path=item,
                    size=stat.st_size if is_file else 0,
                    is_file=is_file,
                    is_dir=is_dir
                )
                
                if is_file:
                    self.files.append(file_info)
                    self.total_files += 1
                    self.total_size += stat.st_size
                    
                    ext = file_info.extension or "no_extension"
                    self.file_types[ext] = self.file_types.get(ext, 0) + 1
                else:
                    self.directories.append(file_info)
                    self.total_dirs += 1
                    
            except (OSError, PermissionError) as e:
                logger.warning(f"Could not access {item}: {e}")
                continue


class RepoExtractor:
    """
    Extracts information from a repository.
    
    Provides utilities for:
    - Getting file trees
    - Identifying file types
    - Finding test files
    - Identifying source code files
    """
    
    def __init__(self, workspace_root: Optional[Path] = None):
        self.workspace_root = workspace_root or Path.cwd()
        logger = __import__('logging').getLogger(__name__)
        self.logger = logger
    
    def extract_repo_info(self, repo_path: Path | str) -> RepoInfo:
        """
        Extract comprehensive information about a repository.
        
        Args:
            repo_path: Path to the repository root
            
        Returns:
            RepoInfo object containing repository analysis
        """
        path = Path(repo_path)
        if not path.is_absolute():
            path = self.workspace_root / path
        
        logger.info(f"Extracting repository info from: {path}")
        return RepoInfo(root_path=path)
    
    def get_file_tree(self, repo_path: Path | str, 
                     max_depth: Optional[int] = None,
                     ignore_patterns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get a nested dictionary representing the file tree.
        
        Args:
            repo_path: Path to the repository root
            max_depth: Maximum depth to traverse (None for unlimited)
            ignore_patterns: List of glob patterns to ignore
            
        Returns:
            Nested dictionary representing the file structure
        """
        path = Path(repo_path)
        if not path.is_absolute():
            path = self.workspace_root / path
        
        if ignore_patterns is None:
            ignore_patterns = [
                "*/__pycache__/*",
                "*/.*",
                "*/node_modules/*",
                "*/.git/*",
                "*/dist/*",
                "*/build/*",
                "*/*.pyc",
                "*/*.pyo",
                "*/*.pyd"
            ]
        
        def _should_ignore(item_path: Path) -> bool:
            """Check if an item should be ignored based on patterns."""
            for pattern in ignore_patterns:
                if item_path.match(pattern):
                    return True
            return False
        
        def _build_tree(directory: Path, current_depth: int = 0) -> Dict[str, Any]:
            """Recursively build the tree structure."""
            if max_depth is not None and current_depth > max_depth:
                return {"__truncated__": True}
            
            tree = {
                "type": "directory",
                "name": directory.name,
                "path": str(directory.relative_to(self.workspace_root)),
                "children": {}
            }
            
            try:
                items = sorted(directory.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
                for item in items:
                    if _should_ignore(item):
                        continue
                    
                    if item.is_dir():
                        tree["children"][item.name] = _build_tree(item, current_depth + 1)
                    else:
                        tree["children"][item.name] = {
                            "type": "file",
                            "name": item.name,
                            "path": str(item.relative_to(self.workspace_root)),
                            "size": item.stat().st_size,
                            "extension": item.suffix.lower()
                        }
            except (OSError, PermissionError) as e:
                logger.warning(f"Could not read directory {directory}: {e}")
                tree["error"] = str(e)
            
            return tree
        
        return _build_tree(path)
    
    def find_test_files(self, repo_path: Path | str) -> List[Path]:
        """
        Find test files in a repository.
        
        Args:
            repo_path: Path to the repository root
            
        Returns:
            List of paths to test files
        """
        path = Path(repo_path)
        if not path.is_absolute():
            path = self.workspace_root / path
        
        test_patterns = [
            "**/test_*.py",
            "**/*_test.py",
            "**/tests/**/*.py",
            "**/__tests__/**/*.py",
            "**/*.test.js",
            "**/*.spec.js",
            "**/test/**/*.js",
            "**/__tests__/**/*.js"
        ]
        
        test_files = []
        for pattern in test_patterns:
            try:
                test_files.extend(path.glob(pattern))
            except (OSError, PermissionError) as e:
                logger.warning(f"Error searching for pattern {pattern}: {e}")
        
        # Remove duplicates and sort
        return sorted(list(set(test_files)))
    
    def find_source_files(self, repo_path: Path | str, 
                         extensions: Optional[List[str]] = None) -> List[Path]:
        """
        Find source code files in a repository.
        
        Args:
            repo_path: Path to the repository root
            extensions: List of file extensions to consider as source (None for common ones)
            
        Returns:
            List of paths to source files
        """
        if extensions is None:
            extensions = [
                ".py", ".js", ".ts", ".jsx", ".tsx",
                ".java", ".cpp", ".c", ".h", ".hpp",
                ".cs", ".go", ".rs", ".php", ".rb",
                ".swift", ".kt", ".scala", ".clj", ".hs"
            ]
        
        path = Path(repo_path)
        if not path.is_absolute():
            path = self.workspace_root / path
        
        source_files = []
        for ext in extensions:
            try:
                source_files.extend(path.glob(f"**/*{ext}"))
            except (OSError, PermissionError) as e:
                logger.warning(f"Error searching for extension {ext}: {e}")
        
        # Filter out common non-source directories and sort
        filtered_files = []
        ignore_dirs = {".git", "__pycache__", "node_modules", "dist", "build", ".venv", "venv"}
        
        for file_path in source_files:
            if not any(part in ignore_dirs for part in file_path.parts):
                filtered_files.append(file_path)
        
        return sorted(list(set(filtered_files)))
    
    def get_language_statistics(self, repo_path: Path | str) -> Dict[str, Any]:
        """
        Get statistics about programming languages used in the repository.
        
        Args:
            repo_path: Path to the repository root
            
        Returns:
            Dictionary with language statistics
        """
        repo_info = self.extract_repo_info(repo_path)
        
        # Map extensions to languages
        ext_to_lang = {
            ".py": "Python",
            ".js": "JavaScript",
            ".ts": "TypeScript",
            ".jsx": "JavaScript (React)",
            ".tsx": "TypeScript (React)",
            ".java": "Java",
            ".cpp": "C++",
            ".c": "C",
            ".h": "C Header",
            ".hpp": "C++ Header",
            ".cs": "C#",
            ".go": "Go",
            ".rs": "Rust",
            ".php": "PHP",
            ".rb": "Ruby",
            ".swift": "Swift",
            ".kt": "Kotlin",
            ".scala": "Scala",
            ".clj": "Clojure",
            ".hs": "Haskell",
            ".m": "Objective-C",
            ".mm": "Objective-C++",
            ".dart": "Dart",
            ".lua": "Lua",
            ".r": "R",
            ".mat": "MATLAB",
            ".scm": "Scheme",
            ".ss": "Scheme",
            ".el": "Emacs Lisp",
            ".ex": "Elixir",
            ".exs": "Elixir",
            ".erl": "Erlang",
            ".cl": "Common Lisp",
            ".hs": "Haskell"
        }
        
        lang_counts = {}
        for file_info in repo_info.files:
            if file_info.extension:
                lang = ext_to_lang.get(file_info.extension, file_info.extension.upper())
                lang_counts[lang] = lang_counts.get(lang, 0) + 1
        
        total_files = sum(lang_counts.values())
        lang_percentages = {
            lang: (count / total_files * 100) if total_files > 0 else 0
            for lang, count in lang_counts.items()
        }
        
        return {
            "total_files": repo_info.total_files,
            "total_size_mb": repo_info.total_size / (1024 * 1024),
            "language_counts": lang_counts,
            "language_percentages": lang_percentages,
            "most_common_language": max(lang_counts.items(), key=lambda x: x[1])[0] if lang_counts else None
        }


def extract_repo_info(repo_path: Path | str) -> RepoInfo:
    """Convenience function to extract repo info."""
    extractor = RepoExtractor()
    return extractor.extract_repo_info(repo_path)


def get_file_tree(repo_path: Path | str, **kwargs) -> Dict[str, Any]:
    """Convenience function to get file tree."""
    extractor = RepoExtractor()
    return extractor.get_file_tree(repo_path, **kwargs)


def find_test_files(repo_path: Path | str) -> List[Path]:
    """Convenience function to find test files."""
    extractor = RepoExtractor()
    return extractor.find_test_files(repo_path)


def find_source_files(repo_path: Path | str, **kwargs) -> List[Path]:
    """Convenience function to find source files."""
    extractor = RepoExtractor()
    return extractor.find_source_files(repo_path, **kwargs)


def get_language_statistics(repo_path: Path | str) -> Dict[str, Any]:
    """Convenience function to get language statistics."""
    extractor = RepoExtractor()
    return extractor.get_language_statistics(repo_path)