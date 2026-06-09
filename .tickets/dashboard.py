#!/usr/bin/env python3
"""RFSN v10 Ticket Dashboard — Status and Burndown.

Usage:
    python dashboard.py              # Full dashboard
    python dashboard.py --summary    # One-line summary
    python dashboard.py --critical   # Show only P0 tickets
    python dashboard.py --by-owner    # Group by owner
"""

import argparse
from collections import defaultdict
from pathlib import Path


def parse_ticket_file(path: Path) -> dict:
    """Parse ticket frontmatter."""
    content = path.read_text()
    lines = content.split('\n')

    data = {}
    in_frontmatter = False

    for line in lines:
        if line.strip() == '---':
            if in_frontmatter:
                break
            in_frontmatter = True
            continue

        if in_frontmatter and ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()

            if value.startswith('[') and value.endswith(']'):
                value = [
                    v.strip().strip('"\'')
                    for v in value[1:-1].split(',') if v.strip()
                ]

            data[key] = value

    return data


def load_tickets():
    """Load all tickets from .tickets directory."""
    tickets_dir = Path(__file__).parent
    tickets = []

    for f in tickets_dir.glob('*.md'):
        if f.name in ['README.md', 'TEMPLATE.md']:
            continue
        try:
            data = parse_ticket_file(f)
            if 'id' in data:
                tickets.append(data)
        except Exception:
            pass

    return tickets


def get_status_emoji(status):
    """Get emoji for status."""
    return {
        'complete': '🟢',
        'active': '🔴',
        'verify': '🟡',
        'not-started': '⚪',
        'blocked': '⛔',
    }.get(status, '⚪')


def get_priority_emoji(priority):
    """Get emoji for priority."""
    return {
        'P0': '🔴',
        'P1': '🟡',
        'P2': '🟢',
    }.get(priority, '⚪')


def dashboard():
    """Print full dashboard."""
    tickets = load_tickets()

    # Stats
    by_status = defaultdict(list)
    by_priority = defaultdict(list)
    by_owner = defaultdict(list)

    for t in tickets:
        by_status[t.get('status', 'unknown')].append(t)
        by_priority[t.get('priority', 'P2')].append(t)
        by_owner[t.get('owner', 'Unassigned')].append(t)

    total = len(tickets)
    complete = len(by_status.get('complete', []))
    active = len(by_status.get('active', []))
    p0 = len(by_priority.get('P0', []))
    p0_active = len([
        t for t in by_priority.get('P0', [])
        if t.get('status') == 'active'
    ])

    # Header
    print("=" * 70)
    print("           RFSN v10 REPAIR PLAN — TICKET DASHBOARD")
    print("=" * 70)

    # Summary
    print(f"\n  📊  Total: {total} | Complete: {complete} | Active: {active}")
    print(f"  🔥  P0 Tickets: {p0} total, {p0_active} active")

    # Status breakdown
    print("\n" + "-" * 70)
    print("  STATUS BREAKDOWN")
    print("-" * 70)
    for status in ['complete', 'active', 'verify', 'not-started', 'blocked']:
        count = len(by_status.get(status, []))
        emoji = get_status_emoji(status)
        title = status.replace('-', ' ').title()
        print(f"  {emoji} {title:15} {count:3} tickets")

    # Priority breakdown
    print("\n" + "-" * 70)
    print("  PRIORITY BREAKDOWN")
    print("-" * 70)
    for priority in ['P0', 'P1', 'P2']:
        count = len(by_priority.get(priority, []))
        emoji = get_priority_emoji(priority)
        print(f"  {emoji} {priority:15} {count:3} tickets")

    # Active tickets by priority
    print("\n" + "-" * 70)
    print("  ACTIVE P0 TICKETS (Critical)")
    print("-" * 70)
    p0_active_tickets = [
        t for t in by_priority.get('P0', [])
        if t.get('status') == 'active'
    ]
    if p0_active_tickets:
        for t in sorted(p0_active_tickets, key=lambda x: x.get('id', '')):
            tid = t.get('id')
            title = t.get('title', 'Untitled')
            print(f"  🔴 [{tid}] {title}")
            owner = t.get('owner', 'Unassigned')
            est = t.get('estimate', 'TBD')
            print(f"     Owner: {owner} | Estimate: {est}")
    else:
        print("  🎉 No active P0 tickets!")

    # Owner workload
    print("\n" + "-" * 70)
    print("  WORKLOAD BY OWNER")
    print("-" * 70)
    for owner in sorted(by_owner.keys()):
        count = len(by_owner[owner])
        active_count = len([
            t for t in by_owner[owner]
            if t.get('status') == 'active'
        ])
        p0_count = len([
            t for t in by_owner[owner]
            if t.get('priority') == 'P0'
        ])
        print(
            f"  {owner:15} {count:2} total, "
            f"{active_count:2} active, {p0_count:2} P0"
        )

    print("\n" + "=" * 70)


def summary():
    """Print one-line summary."""
    tickets = load_tickets()
    total = len(tickets)
    complete = len([
        t for t in tickets if t.get('status') == 'complete'
    ])
    active = len([t for t in tickets if t.get('status') == 'active'])
    p0_active = len([
        t for t in tickets
        if t.get('priority') == 'P0' and t.get('status') == 'active'
    ])

    print(f"Tickets: {complete}/{total} complete, {active} active, {p0_active} P0 active")


def critical():
    """Print only P0 active tickets."""
    tickets = load_tickets()
    p0_active = [t for t in tickets if t.get('priority') == 'P0' and t.get('status') == 'active']

    print("=" * 70)
    print("           ACTIVE P0 TICKETS (Critical Path)")
    print("=" * 70)

    if not p0_active:
        print("\n  🎉 No active P0 tickets! Critical path is clear.\n")
        return

    for t in sorted(p0_active, key=lambda x: x.get('id', '')):
        print(f"\n  🔴 [{t.get('id')}] {t.get('title', 'Untitled')}")
        print(f"     Epic: {t.get('epic', 'Unknown')}")
        print(f"     Owner: {t.get('owner', 'Unassigned')}")
        print(f"     Estimate: {t.get('estimate', 'TBD')}")
        print(f"     Labels: {', '.join(t.get('labels', []))}")

    def _get_hours(t):
        est = t.get('estimate', '')
        return int(est.replace('h', '')) if est else 0

    total_hours = sum(
        _get_hours(t) for t in p0_active
    )
    print(f"\n  Total P0 effort: ~{total_hours} hours")
    print("=" * 70)


def by_owner():
    """Print tickets grouped by owner."""
    tickets = load_tickets()
    by_owner_dict = defaultdict(list)
    for t in tickets:
        owner = t.get('owner', 'Unassigned')
        by_owner_dict[owner].append(t)

    print("=" * 70)
    print("           TICKETS BY OWNER")
    print("=" * 70)

    for owner in sorted(by_owner_dict.keys()):
        print(f"\n  📌 {owner}")
        print("  " + "-" * 66)
        for t in sorted(by_owner_dict[owner], key=lambda x: x.get('id', '')):
            status_emoji = get_status_emoji(t.get('status', 'unknown'))
            priority_emoji = get_priority_emoji(t.get('priority', 'P2'))
            print(f"  {status_emoji} [{t.get('id')}] {priority_emoji} {t.get('title', 'Untitled')}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description='RFSN v10 Ticket Dashboard')
    parser.add_argument('--summary', action='store_true', help='One-line summary')
    parser.add_argument('--critical', action='store_true', help='Show only P0 active tickets')
    parser.add_argument('--by-owner', action='store_true', help='Group by owner')
    args = parser.parse_args()

    if args.summary:
        summary()
    elif args.critical:
        critical()
    elif args.by_owner:
        by_owner()
    else:
        dashboard()


if __name__ == '__main__':
    main()
