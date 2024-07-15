"""
    Helper functions for git integration of the code
"""

import os
import time

from git import Repo
from git.exc import InvalidGitRepositoryError


class CustomRepo(Repo):
    """Customize the Repo class from gitpython for easier access/use.

    This class extends the base class `Repo` from gitpython and provides additional methods
    for easier access and use of a git repository.

    Args:
        repo_path (str): The path to the git repository.

    Raises:
        RuntimeError: If the `repo_path` is not a valid directory or a git repository.

    Attributes:
        None

    Methods:
        get_git_hash: Get the latest commit hash and message of the git repository.
        get_git_branch: Get the name of the currently active branch.
        get_changed_files: Get a list of all files that have been changed since the last commit.
        get_untracked_files: Get a list of all untracked files in the repository.
        commit_current_state: Commit the current state of the repository.

    See Also:
        - Documentation of the base class `Repo`: https://gitpython.readthedocs.io/en/stable/index.html
    """

    def __init__(self, repo_path):
        assert os.path.isdir(repo_path), f"'{repo_path}' is not a directory"
        try:
            super(CustomRepo, self).__init__(repo_path)
        except InvalidGitRepositoryError as error:
            raise RuntimeError(f"'{repo_path}' is not a path of a git repository!") from error

    def get_git_hash(self):
        """
        Get the latest commit hash and message of the git repository.

        Returns:
            tuple: A tuple containing the latest commit hash (string) and the commit message (string).

        Raises:
            RuntimeError: If the repository is not a valid git repository.
        """
        sha = self.head.object.hexsha
        message = self.head.object.summary
        return sha, message

    def get_git_branch(self):
        """
        Get the name of the currently active branch.

        Returns:
            str: The name of the active branch.
        """
        return self.active_branch.name

    def get_changed_files(self):
        """
        Get a list of all files that have been changed since the last commit.

        Returns:
            list: A list of file paths that have been changed since the last commit.
        """
        changed_files = []
        for item in self.index.diff(None):
            changed_files.append(item.a_path)
        return changed_files

    def get_untracked_files(self):
        """
        Get a list of all untracked files in the repository.

        Returns:
            list: A list of file paths that are untracked in the repository.
        """
        return self.untracked_files

    def commit_current_state(self, message=''):
        
        # ? What is the purpose of this function?
        
        if self.is_dirty():
            print("commit current state to git in 3s...")
            time.sleep(3)
            print(f"committing with message '{message}'...")





