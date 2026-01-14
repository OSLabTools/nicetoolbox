"""
Helper functions for git integration of the code
"""

import os
from typing import NamedTuple, Optional

from git import Repo


class GitMetadata(NamedTuple):
    commit_hash: str
    commit_summary: str


def try_get_toolbox_git_metadata(repo_path: str = ".") -> Optional[GitMetadata]:
    """
    Tries to get git commit metadata from the current repo or environment variables.
    If no .git folder is found or environment variables are not set, it will return None.

    Args:
        repo_path (str): Path to look for a .git folder (default: current folder).

    Raises:
        git.exc.InvalidGitRepositoryError: If .git folder exists but is invalid
        git.exc.GitCommandError: If git operations fail

    Returns:
        Optional[GitMetadata]: with fields commit_hash and commit_summary,
        or None if neither is available.
    """
    # Checks if there's a .git folder in repo_path
    git_folder = os.path.join(repo_path, ".git")
    if os.path.isdir(git_folder):
        # Try to get git commit metadata from the repo
        # It will raise an error if the repo is not a valid git repository
        repo = Repo(repo_path)
        # All good, get the commit hash and summary
        sha = repo.head.object.hexsha
        message = repo.head.object.summary
        return GitMetadata(commit_hash=sha, commit_summary=message)

    # If no .git folder, try to get from environment variables
    git_hash = os.environ.get("NICETOOLBOX_GIT_HASH")
    git_summary = os.environ.get("NICETOOLBOX_GIT_SUMMARY")
    # Summary can be empty - but hash should always exist
    if git_hash and git_summary is not None:
        return GitMetadata(commit_hash=git_hash, commit_summary=git_summary)

    # It seems someone just copied the code and is not using git
    # We will gracefully return None
    return None
