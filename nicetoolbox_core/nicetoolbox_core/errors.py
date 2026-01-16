import traceback


class SubprocessError(Exception):
    """
    A transport-safe wrapper for exceptions occurring in isolated environments.

    It preserves the original error details as strings to avoid ImportErrors
    during unpickling in the main process (e.g. when facing MMPose exceptions).
    """

    def __init__(self, original_exc: Exception):
        self.original_type = type(original_exc).__name__
        self.message = str(original_exc)
        self.traceback = traceback.format_exc()

        super().__init__(f"[{self.original_type}] {self.message}")

    def __str__(self):
        return (
            f"Subprocess failed with {self.original_type}:\n"
            f"Message: {self.message}\n"
            f"Remote Traceback:\n{self.traceback}"
        )
