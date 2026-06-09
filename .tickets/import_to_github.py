#!/usr/bin/env python3
"""Import tickets to GitHub Issues.

Usage:
    python import_to_github.py                    # Import all non-complete
    python import_to_github.py --tickets 4-1,4-2  # Import specific tickets
    python import_to_github.py --dry-run          # Preview only
    python import_to_github.py --status active    # Import only active tickets

Requirements:
    - gh CLI installed and authenticated (run `gh auth login`)
    - Repository has GitHub Issues enabled
"""

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Ticket:
    id: str
    title: str
    owner: str
    estimate: str
    priority: str
    epic: str
    status: str
    created: str
    labels: list[str]
    body: str


def parse_ticket_file(path: Path) -> Ticket:
    """Parse a ticket markdown file with YAML frontmatter."""
    content = path.read_text()

    # Extract frontmatter
    match = re.match(r'^---\s*\n(.*?)\n---\s*\n(.*)', content, re.DOTALL)
    if not match:
        raise ValueError(f"No frontmatter found in {path}")

    frontmatter = match.group(1)
    body = match.group(2).strip()

    # Parse YAML-like frontmatter
    data = {}
    for line in frontmatter.strip().split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()

            # Handle arrays
            if value.startswith('[') and value.endswith(']'):
                value = [
                    v.strip().strip('"\'') for v in value[1:-1].split(',')
                ]

            data[key] = value

    return Ticket(
        id=data.get('id', path.stem),
        title=data.get('title', 'Untitled'),
        owner=data.get('owner', 'Unassigned'),
        estimate=data.get('estimate', 'TBD'),
        priority=data.get('priority', 'P2'),
        epic=data.get('epic', 'Uncategorized'),
        status=data.get('status', 'not-started'),
        created=data.get('created', '2026-06-07'),
        labels=data.get('labels', []),
        body=body,
    )


def format_issue_body(ticket: Ticket) -> str:
    """Format ticket as GitHub issue body."""
    return f"""## Ticket {ticket.id}: {ticket.title}

**Epic**: {ticket.epic}
**Owner**: {ticket.owner}
**Estimate**: {ticket.estimate}
**Priority**: {ticket.priority}
**Status**: {ticket.status}

---

{ticket.body}

---

*Imported from .tickets/{ticket.id}*
"""


def get_repo_from_git() -> Optional[str]:
    """Get owner/repo from git remote."""
    try:
        result = subprocess.run(
            ['git', 'remote', 'get-url', 'origin'],
            capture_output=True,
            text=True,
            check=True
        )
        url = result.stdout.strip()
        # Handle both HTTPS and SSH URLs
        if 'github.com' in url:
            # https://github.com/owner/repo.git -> owner/repo
            # git@github.com:owner/repo.git -> owner/repo
            match = re.search(r'github\.com[/:]([^/]+/[^/]+?)(?:\.git)?$', url)
            if match:
                return match.group(1)
    except subprocess.CalledProcessError:
        pass
    return None


def issue_exists(repo: str, title: str) -> bool:
    """Check if an issue with this title already exists."""
    try:
        result = subprocess.run(
            ['gh', 'issue', 'list', '--repo', repo,
             '--search', title, '--json', 'title'],
            capture_output=True,
            text=True,
            check=True
        )
        issues = json.loads(result.stdout)
        return len(issues) > 0
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return False


def create_issue(
    repo: str, ticket: Ticket, dry_run: bool = False
) -> Optional[str]:
    """Create a GitHub issue from a ticket."""
    title = f"[{ticket.id}] {ticket.title}"
    body = format_issue_body(ticket)
    labels = ','.join(ticket.labels) if ticket.labels else ""

    if dry_run:
        print("[DRY RUN] Would create issue:")
        print(f"  Title: {title}")
        print(f"  Labels: {labels}")
        print(f"  Body length: {len(body)} chars")
        return None

    # Check for duplicates
    if issue_exists(repo, title):
        print(f"[SKIP] Issue already exists: {title}")
        return None

    cmd = [
        'gh', 'issue', 'create',
        '--repo', repo,
        '--title', title,
        '--body', body,
    ]

    if labels:
        cmd.extend(['--label', labels])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        issue_url = result.stdout.strip()
        return issue_url
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to create: {e}")
        print(f"stderr: {e.stderr}")
        return None


def main():

    parser = argparse.ArgumentParser(description='Import tickets to GitHub Issues')
    parser.add_argument(
        '--tickets', type=str,
        help='Comma-separated ticket IDs (e.g., 4-1,4-2)'
    )
    parser.add_argument(
        '--status', type=str, help='Import only tickets with this status'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Preview without creating issues'
    )
    args = parser.parse_args()

    # Check gh CLI
    try:
        subprocess.run(['gh', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: gh CLI not found or not authenticated.")
        print("Install: https://cli.github.com/")
        print("Authenticate: gh auth login")
        sys.exit(1)

    # Get repo
    repo = get_repo_from_git()
    if not repo:
        print("Error: Could not determine GitHub repo from git remote.")
        sys.exit(1)
    print(f"Target repo: {repo}")

    # Find tickets
    tickets_dir = Path(__file__).parent
    ticket_files = list(tickets_dir.glob('*.md'))
    ticket_files = [
        f for f in ticket_files
        if f.name not in ['README.md', 'TEMPLATE.md']
    ]

    # Filter tickets
    selected_tickets = []
    for f in ticket_files:
        try:
            ticket = parse_ticket_file(f)

            # Skip completed tickets unless explicitly specified
            if ticket.status == 'complete' and not args.tickets:
                continue

            # Filter by ticket IDs
            if args.tickets and ticket.id not in args.tickets.split(','):
                continue

            # Filter by status
            if args.status and ticket.status != args.status:
                continue

            selected_tickets.append(ticket)
        except Exception as e:
            print(f"[WARNING] Failed to parse {f}: {e}")

    if not selected_tickets:
        print("No tickets to import.")
        return

    print(f"\nFound {len(selected_tickets)} ticket(s) to import:\n")
    for t in selected_tickets:
        print(f"  {t.id}: {t.title} ({t.priority}, {t.status})")

    if args.dry_run:
        print("\n[DRY RUN] No issues will be created.")
        return

    # Confirm
    print("\nCreate GitHub issues? [y/N]: ", end='')
    response = input().strip().lower()
    if response != 'y':
        print("Aborted.")
        return

    # Create issues
    print()
    success_count = 0
    for ticket in selected_tickets:
        url = create_issue(repo, ticket, dry_run=False)
        if url:
            print(f"[OK] {ticket.id}: {url}")
            success_count += 1
        else:
            print(f"[FAIL] {ticket.id}")

    print(
        f"\n{success_count}/{len(selected_tickets)} issues created successfully."
    )


if __name__ == '__main__':
    main()
