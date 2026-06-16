#!/usr/bin/env python3
"""Sync this fork's branch from the upstream base repo (CoplayDev/unity-mcp).

GitHub's "Sync fork" button only works while the fork can fast-forward. Once this
fork has its own commits that touch the same lines as upstream, the sync must be a
real merge that may need conflict resolution. This script automates the mechanical
half of that merge:

  1. Ensures an `upstream` remote pointing at the base repo (adds it if missing).
  2. Fetches upstream.
  3. Reports how far ahead / behind the branch is.
  4. Runs a no-commit merge of upstream/<branch> into the current branch.
  5. Stops for human review in every case and NEVER pushes:
       - clean merge   -> changes are staged, ready to commit + push.
       - conflicts     -> conflicted files are listed; repo is left mid-merge.

Conflict *resolution* is deliberately left to a human (or an AI assistant), because
it requires judgment about which side of each change to keep.

Usage:
    python tools/sync_upstream.py            # sync the current branch
    python tools/sync_upstream.py --branch beta
    python tools/sync_upstream.py --abort    # abort an in-progress merge
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

UPSTREAM_REMOTE = "upstream"
UPSTREAM_URL = "https://github.com/CoplayDev/unity-mcp.git"
REPO = Path(__file__).resolve().parent.parent


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(REPO), *args],
        capture_output=True, text=True, check=check,
    )


def git_out(*args: str) -> str:
    return git(*args).stdout.strip()


def fail(msg: str) -> "NoReturn":
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def ensure_upstream() -> None:
    remotes = git_out("remote").splitlines()
    if UPSTREAM_REMOTE not in remotes:
        print(f"Adding '{UPSTREAM_REMOTE}' remote -> {UPSTREAM_URL}")
        git("remote", "add", UPSTREAM_REMOTE, UPSTREAM_URL)
        return
    current = git_out("remote", "get-url", UPSTREAM_REMOTE)
    if current != UPSTREAM_URL:
        print(f"WARNING: '{UPSTREAM_REMOTE}' points at {current}, expected {UPSTREAM_URL}")


def current_branch() -> str:
    return git_out("rev-parse", "--abbrev-ref", "HEAD")


def merge_in_progress() -> bool:
    return (REPO / ".git" / "MERGE_HEAD").exists()


def main() -> None:
    ap = argparse.ArgumentParser(description="Sync this fork from upstream.")
    ap.add_argument("--branch", help="Branch to sync (default: current branch).")
    ap.add_argument("--abort", action="store_true", help="Abort an in-progress merge and exit.")
    args = ap.parse_args()

    if args.abort:
        if not merge_in_progress():
            print("No merge in progress.")
            return
        git("merge", "--abort")
        print("Merge aborted; working tree restored.")
        return

    if merge_in_progress():
        fail("A merge is already in progress. Resolve and commit it, or run with --abort.")

    if git_out("status", "--porcelain"):
        fail("Working tree is dirty. Commit or stash your changes before syncing.")

    branch = args.branch or current_branch()
    if branch != current_branch():
        print(f"Checking out '{branch}'")
        git("checkout", branch)

    ensure_upstream()
    print(f"Fetching {UPSTREAM_REMOTE} ...")
    git("fetch", UPSTREAM_REMOTE)

    upstream_ref = f"{UPSTREAM_REMOTE}/{branch}"
    if git("rev-parse", "--verify", upstream_ref, check=False).returncode != 0:
        fail(f"Upstream branch '{upstream_ref}' not found.")

    counts = git_out("rev-list", "--left-right", "--count", f"{branch}...{upstream_ref}")
    ahead, behind = counts.split()
    print(f"'{branch}' is {ahead} ahead, {behind} behind {upstream_ref}.")
    if behind == "0":
        print("Already up to date with upstream. Nothing to merge.")
        return

    print(f"Merging {upstream_ref} into '{branch}' (no auto-commit) ...")
    result = git("merge", "--no-ff", "--no-commit", upstream_ref, check=False)
    print(result.stdout.strip())

    conflicts = git_out("diff", "--name-only", "--diff-filter=U").splitlines()
    if conflicts:
        print("\nCONFLICTS — resolve these, `git add` them, then `git commit` and push:")
        for f in conflicts:
            print(f"  {f}")
        sys.exit(1)

    print(
        "\nClean merge — changes are staged but NOT committed and NOT pushed.\n"
        "Review with `git diff --cached`, then:\n"
        f"    git commit\n"
        f"    git push origin {branch}"
    )


if __name__ == "__main__":
    main()
