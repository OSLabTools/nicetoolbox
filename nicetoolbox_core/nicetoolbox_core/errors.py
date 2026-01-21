from enum import Enum

error_level_to_int = {"STRICT": 0, "VIDEO": 10, "DETECTOR": 20}


class ErrorLevel(str, Enum):
    """
    Defines hierarchical error tolerance levels for the inference loop.

    The levels are ordered such that a higher level includes the tolerances of the lower levels:
    - STRICT (0): No errors are tolerated; any exception will crash the program.
    - VIDEO (10): Video-level errors are tolerated; if an error occurs during video processing,
      the program will skip to the next video.
    - DETECTOR (20): Detector-level errors are tolerated; if an error occurs during algorithm execution,
      the program will skip to the next algorithm for the same video.

    Due to the hierarchical nature, setting the level to DETECTOR also implies tolerating VIDEO-level errors.

    In the future, we plan to extend this enumeration with a FRAME (30) level for even finer error handling during
    frame-by-frame processing. This level will then also imply tolerance of VIDEO and DETECTOR levels.
    """

    STRICT = "STRICT"  # Crash on everything
    VIDEO = "VIDEO"  # Suppress Video-scope errors, continue with next video
    DETECTOR = "DETECTOR"  # Suppress Detector-scope errors, continue with next detector

    def __str__(self):
        # Prettier string representation for e.g. logging
        return self.name
