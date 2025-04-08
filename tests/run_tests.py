import logging
import unittest
from pathlib import Path

from nicetoolbox.utils import logging_utils as lu


def main(logfile, verbosity=2):
    # For custom testing logs additional to unittest's logging to text
    tmp_logfile = Path("tests.log")
    lu.setup_custom_logging(tmp_logfile, "testing", logging.INFO)

    # loader for testing
    loader = unittest.TestLoader()
    # find all test classes in directory 'tests'
    suite = loader.discover(Path("tests"))
    # run testing, collect results in logfile as text
    unittest.TextTestRunner(logfile, verbosity=verbosity).run(suite)

    # copy all temporal logs into the given logfile
    with open(tmp_logfile) as f:
        log_lines = f.readlines()
        logfile.writelines(["\n\n\nTESTS RUN DETAILS:\n\n"] + log_lines)
        f.close()
    Path.unlink(tmp_logfile)


if __name__ == "__main__":
    with open("testing.log", "w") as f:
        main(f)
