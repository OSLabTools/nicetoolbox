"""Fixtures shared by tests."""

import logging
import sys
import time
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

        # Logging important stuff to stdout
        cls.logger = logging.getLogger(__name__)
        formatter = logging.Formatter(
            "*** [%(levelname)s][%(asctime)s][%(module)s] %(message)s"
        )
        sh = logging.StreamHandler(stream=sys.stdout)
        sh.setLevel(logging.DEBUG)
        sh.setFormatter(formatter)
        cls.logger.addHandler(sh)

        # Note class context method only available in Python 3.11
        # cls.temp_directory = cls.enterClassContext(TemporaryDirectory(dir="."))
        cls.temp_directory_object = TemporaryDirectory(dir=".")
        cls.temp_directory = cls.temp_directory_object.name

    @classmethod
    def tearDownClass(cls):
        # exit the temporary directory
        cls.temp_directory_object.cleanup()

        super().tearDownClass()

    def run(self, *args, **kwargs):
        results = super().run(*args, **kwargs)

        # When test is skipped, results will be None.
        if results:
            # Can't do ID checking for subTests
            errors_and_failures = [*results.errors, *results.failures]

            # If errors, show seed
            if len(errors_and_failures) > 0:
                self.logger.warning(
                    "%s", "\n".join([_[0].id() for _ in errors_and_failures])
                )
                self.logger.warning("Seed for RNG: %s", self.seed)

        return results
