""" 
    The file to define fixtures for tests.
    conftest.py is the designated file to define fixtures. It does not need to be imported!
    These are functions that are run before tests are executed.
"""

import numpy as np
import pytest


def pytest_addoption(parser):
    """
        Add random seed option to py.test.
        This adds a commandline option to pytest
    """
    parser.addoption('--seed', dest='seed', type=int, action='store',
                     help='set random seed')


def pytest_configure(config):
    """ 
        configure pytest to automatically set the rnd seed if not passed on CLI
    """
    seed = config.getvalue("seed")
    # if seed was not set by the user, we set one now
    if seed is None or seed == ('NO', 'DEFAULT'):
        config.option.seed = int(np.random.randint(2**31-1))


def pytest_report_header(config):
    """ 
        Report option to return the random seed that is used.
    """
    return f'Using random seed: {config.option.seed}'


@pytest.fixture
def random_state(request):
    """ 
        Ficture that returns the set random seed.
    """
    random_state = np.random.RandomState(request.config.option.seed)
    return random_state

