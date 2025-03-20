"""Fixtures shared by tests."""

import logging
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from faker import Faker

# Shut up third-party logs
logging.getLogger("faker").setLevel(logging.WARNING)


class BaseTestCase(TestCase):
    """Base class for all tests."""

    @classmethod
    def setUpClass(cls) -> None:
        # Seed random generator seed for reproducibility
        cls.seed = int(time.time() * 1000)  # current time in ms
        cls.fake = Faker()
        cls.fake.seed_instance(cls.seed)

        # Note class context method only available in Python 3.11
        # cls.temp_directory = cls.enterClassContext(TemporaryDirectory(dir="."))
        cls.temp_directory_object = TemporaryDirectory(dir=".")
        cls.temp_directory = Path(cls.temp_directory_object.name)

    @classmethod
    def tearDownClass(cls):
        # exit the temporary directory
        cls.temp_directory_object.cleanup()

        super().tearDownClass()
