# regression test for detectors or evaluation outputs to see the difference in structure and values
# it takes golden data (usually previous release) and new release candidate data
# golden data is considered ground truth and compared against new candidate data
# it shows changes in structure (missing files, npz keys, etc.) and values (basic aggregated statistics)
# keep in mind that it doesn't show if new release has worse performance, just that they are different

import argparse
import os
from pathlib import Path
from pprint import pformat

from nicetoolbox.utils.comparison import compare_npz_files

DEFAULT_GOLDEN_PATH = "../functional_tests/outputs/communication-multiview-0-2-1"
DEFAULT_CANDIDATE_PATH = "../functional_tests/outputs/communication-multiview-0-2-2"
DEFAULT_REPORT_PATH = "regression_output_report.txt"
IGNORE_FILES = ["config_", ".jpg", ".png", ".json", ".mp4"]


def compare_outputs(golden_data: Path, candidate_data: Path):
    # before doing anything - check if the data exist
    assert os.path.isdir(golden_data), f"No folder found at {golden_data}"
    # filter out directories, keeping only files
    golden_files = list(golden_data.rglob("*"))
    golden_files = [f for f in golden_files if f.is_file()]
    assert len(golden_files) > 0, f"No files found inside {golden_data}"

    errors = []
    for golden_file in golden_files:
        # get the path to generated file
        relative_path = golden_file.relative_to(golden_data)
        generated_file = candidate_data / relative_path
        # skip ignore files
        if any(ignore in str(relative_path) for ignore in IGNORE_FILES):
            continue

        # check that the file was actually created in the new run
        if not generated_file.exists():
            errors.append(FileNotFoundError("File not found", str(generated_file)))
            continue

        # compare the files based on their type
        if golden_file.suffix == ".npz":
            npz_errors = compare_npz_files(generated_file, golden_file)
            errors.extend(npz_errors)

    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--golden_data",
        default=DEFAULT_GOLDEN_PATH,
        help="Path to golden (ground-truth) output from detectors or evaluation",
    )
    parser.add_argument(
        "--candidate_data",
        default=DEFAULT_CANDIDATE_PATH,
        help="Path to newer version (release candidate) data",
    )
    args = parser.parse_args()

    errors = compare_outputs(Path(args.golden_data), Path(args.candidate_data))
    if errors:
        report = pformat(errors)
        print("Errors list:\n" + f"{report}")


if __name__ == "__main__":
    main()
