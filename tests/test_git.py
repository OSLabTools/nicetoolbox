"""

"""
import os
import pytest
from oslab_utils.git import CustomRepo


def test_repo_path_no_directory():
    test_dir = os.path.join(os.getcwd(), 'zxcbvalkjds')
    with pytest.raises(AssertionError):
        CustomRepo(test_dir)


def test_repo_path_no_git_repo():
    # assumption: the nome directory is never a git repository
    test_dir = os.path.expanduser('~')
    with pytest.raises(RuntimeError):
        CustomRepo(test_dir)


