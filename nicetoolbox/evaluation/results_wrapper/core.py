"""
A Pandas-backed API for querying, aggregating, and exporting evaluation results.
"""

import io
from pathlib import Path
from typing import Dict, Iterable, List, Union

import pandas as pd

from .initialize import build_index


class EvaluationResults:
    """
    A Pandas-backed API for querying, aggregating, and exporting evaluation results.

    This class wraps a Pandas DataFrame constructed from a filesystem index of npz files
    and provides a small API for:
    - filtering pandas rows by index levels, by meta data
      (e.g. dataset, algorithm, ...) and npz dimensions (person, camera, ...)
    - filtering pandas columns by metrics
    - performing aggregations with selected aggregation methods
    - exporting the current view to CSV
    - returning the current view as a pandas DataFrame with optional flattening
        and NaN dropping

    The object is stateful: mutating operations (query, aggregate, reset)
    modify the instance in-place and return self to allow method chaining. The
    original unmodified DataFrame built at initialization is retained in
    self._full_df and can be restored via reset().

    Examples:

        >>> from nicetoolbox.evaluation.results_wrapper.core import EvaluationResults
        >>> results = EvaluationResults(root=Path("/path/to/evaluation/results"))
        >>> # Query for specific dataset and metric, aggregate by camera and export
        >>> csv_path = (
        ...     results.query(dataset="my_dataset", metrics=["jpe", "pck"])
        ...     .aggregate(group_by=["camera"])
        ...     .to_csv(output_dir=Path("/path/to/output"), base_name="camera_summary")
        ... )
        >>> print(f"Exported summary to: {csv_path}")

    """

    def __init__(self, root: Path) -> None:
        """
        Builds the pandas DataFrame index from the provided root folder.

        Args:
            root: The root folder containing evaluation results.
        """
        self.root: Path = root
        self._full_df: pd.DataFrame = build_index(root)
        self.df: pd.DataFrame = self._full_df.copy(deep=True)

    def __repr__(self) -> str:
        """Easy printing of the EvaluationResults with pandas .info()."""
        buf = io.StringIO()
        self.df.info(buf=buf)
        info_str = buf.getvalue()
        return f"<EvaluationResults (Pandas backed)> \n\n{info_str}"

    @property
    def available_metrics(self) -> List[str]:
        """
        Returns a list of available metric names in the current DataFrame.

        Returns:
            List[str]: A list of metric names.
        """
        return self.df.index.get_level_values("metric_name").unique().tolist()

    def reset(self) -> "EvaluationResults":
        """
        Restore the view to the originally loaded evaluation results.

        Returns:
            EvaluationResults: Returns self after resetting to allow method chaining.
        """
        self.df = self._full_df.copy(deep=True)
        return self

    def query(self, **filters: Dict[str, str | List[str]]) -> "EvaluationResults":
        """
        Filtering of NICE Toolbox evaluation results based on index levels and metrics.

        Args:
            **filters: Dict[str, str | List[str]]
                Keyword arguments mapping DataFrame index levels to selection values.
                Each keyword must match one of the DataFrame index level names.
                The provided value may be:
                - a single value (e.g., person='p1'),
                - an iterable of values (e.g., dataset=['dataset_A', 'dataset_B'])

        Returns:
            EvaluationResults: Returns self after applying the requested row/column
                indexing to allow for method chaining.

        Examples:
            >>> results = EvaluationResults(root=Path("/path/to/evaluation/results"))
            >>> results.query(dataset="my_dataset", algorithm=["alg1", "alg2"])
            # This will filter the results to only include rows where the 'dataset'
            # is 'my_dataset' and the 'algorithm' is either 'alg1' or 'alg2'.

            >>> results.query(metric_name="jpe", label=["left_knee", "right_knee"])
            # This will filter the results to only include rows where the 'metric_name'
            # is 'jpe' and the 'label'is either 'left_knee' or 'right_knee'.
        """
        # Row filtering based on MultiIndex levels
        if not filters:
            return self

        slicer = [slice(None)] * self.df.index.nlevels
        for dimension, filter_values in filters.items():
            try:
                idx = self.df.index.names.index(dimension)
            except ValueError as e:
                # Error 1: The requested level (e.g., 'datasets') doesn't exist at all.
                raise KeyError(
                    f"'{dimension}' is not a valid index level. "
                    f"Available levels are: {self.df.index.names.to_list()}"
                ) from e

            slicer[idx] = filter_values

        try:
            self.df = self.df.loc[tuple(slicer), :]
        except KeyError as e:
            # Error 2: The requested level exists, but the provided values don't match
            # any entries (e.g., 'dataset' exists, but 'non_existent_dataset' doesn't).
            raise KeyError(
                f"Filtering with {filters} resulted in no matching entries. "
                "Please check that the provided values exist in the data."
            ) from e

        return self

    def aggregate(
        self, group_by: Iterable[str], agg_funcs: Union[str, List[str]] = "mean"
    ) -> "EvaluationResults":
        """
        Performs a flexible aggregation using pandas groupby. Any index levels not
        included in `group_by` are automatically aggregated.

        Args:
            group_by (Iterable[str]): An iterable of index level names to group by.
            agg_funcs (Union[str, List[str]], optional): Aggregation function(s) to
                apply. Defaults to "mean". Can be any valid pandas aggregation function
                name or a list of such names.

        Returns:
            EvaluationResults: Returns self after applying the aggregation to allow
                for method chaining. Note that applying multiple aggregation functions
                results in a MultiIndex on the columns.

        Examples:
            >>> results = EvaluationResults(root=Path("/path/to/evaluation/results"))
            >>> results.aggregate(
            ...     group_by=["dataset", "algorithm"], agg_funcs=["mean", "std"]
            >>> )
            # This will group the results by 'dataset' and 'algorithm', computing a
            # summary statistic for each metric for each dataset-algorithm pair.

            >>> results.reset()
            ...     .aggregate(
            ...         group_by=["dataset", "sequence", "algorithm", "camera"],
            ...         agg_funcs="mean"
            >>> )
            # This will group the results by 'dataset', 'sequence', 'algorithm', and
            # 'camera', computing the mean for each metric for each unique combination
            # of these dimensions, a breakdown per camera within each sequence.

        """
        # Any level not in 'group_by' is automatically aggregated.
        agg_df = self.df.groupby(level=list(group_by), observed=True)["value"].agg(
            agg_funcs
        )

        self.df = agg_df

        return self

    def to_dataframe(self, flatten: bool = False, dropna: bool = False) -> pd.DataFrame:
        """
        Returns current view as DataFrame.

        Args:
            flatten: If True, reset index to flat structure
            dropna: If True, drop rows with all NaN values

        Returns:
            pandas DataFrame (copy of current view)
        """
        df = self.df.copy()

        if dropna:
            df = df.dropna()  # Drops columns that contain NaNs
        if flatten:
            df = df.reset_index()  # Collapses MultiIndex rows
        return df

    def to_csv(self, output_dir: Path, file_name: str = "summary") -> Path:
        """
        Exports the current state to a CSV with a meaningful name.

        Args:
            output_dir (Path): The output directory to save the CSV file.
            base_name (str, optional): The base name for the CSV file.
                Defaults to "summary".

        Returns:
            Path: The path to the exported CSV file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"{file_name}.csv"
        self.to_dataframe(flatten=True).to_csv(output_path, index=False)

        return output_path
