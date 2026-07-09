import pytest

from nicetoolbox.configs.named_wildcards import discover_sequences_for_dataset


@pytest.fixture
def dataset_tree(tmp_path):
    """
    Build a fake filesystem tree under `tmp_path`.

    Pass a list of relative paths. Trailing `/` marks a directory; anything else
    is created as an empty file (parent dirs are created automatically). Returns
    the `tmp_path` root so tests can compose absolute discovery patterns.
    """

    def _build(paths):
        for p in paths:
            full = tmp_path / p
            full.parent.mkdir(parents=True, exist_ok=True)
            if p.endswith("/"):
                full.mkdir(exist_ok=True)
            else:
                full.touch()
        return tmp_path

    return _build


# --- discover_sequences_for_dataset: capture ---


def test_single_named_wildcard(dataset_tree):
    """
    Given: A pattern with one `[name]` token and one matching directory.
    When:  discover_sequences_for_dataset is called.
    Then:  It returns a single dict keyed by the token name.
    """
    root = dataset_tree(["S1/"])
    assert discover_sequences_for_dataset(f"{root}/[session]") == [{"session": "S1"}]


def test_multiple_named_wildcards(dataset_tree):
    """
    Given: A pattern with several `[name]` tokens and matching directories.
    When:  discover_sequences_for_dataset is called.
    Then:  Each dict contains every token as a key.
    """
    root = dataset_tree(["S1/Seq1/", "S1/Seq2/", "S2/Seq1/", "S2/Seq2/"])
    result = discover_sequences_for_dataset(f"{root}/[session]/[recording]")
    assert result == [
        {"session": "S1", "recording": "Seq1"},
        {"session": "S1", "recording": "Seq2"},
        {"session": "S2", "recording": "Seq1"},
        {"session": "S2", "recording": "Seq2"},
    ]


def test_underscore_in_wildcard_name(dataset_tree):
    """
    Given: A `[name]` token whose name contains underscores and digits.
    When:  discover_sequences_for_dataset is called.
    Then:  The result uses that exact name as the dict key.
    """
    root = dataset_tree(["S1/"])
    assert discover_sequences_for_dataset(f"{root}/[my_session_1]") == [{"my_session_1": "S1"}]


# --- discover_sequences_for_dataset: ordering ---


def test_output_is_sorted(dataset_tree):
    """
    Given: Multiple matching directories.
    When:  discover_sequences_for_dataset is called.
    Then:  Results are sorted for deterministic output regardless of glob order.
    """
    root = dataset_tree(["S2/Seq1/", "S1/Seq2/", "S1/Seq1/"])
    result = discover_sequences_for_dataset(f"{root}/[session]/[recording]")
    assert result == [
        {"session": "S1", "recording": "Seq1"},
        {"session": "S1", "recording": "Seq2"},
        {"session": "S2", "recording": "Seq1"},
    ]


# --- discover_sequences_for_dataset: pattern rejection (pre-glob) ---


def test_reject_unresolved_placeholder():
    """
    Given: A pattern containing an unresolved `<placeholder>` token.
    When:  discover_sequences_for_dataset is called.
    Then:  A ValueError is raised naming the unresolved placeholder.
    """
    with pytest.raises(ValueError, match="unresolved placeholders"):
        discover_sequences_for_dataset("/data/<datasets_folder_path>/[session]")


def test_reject_no_named_wildcards():
    """
    Given: A pattern with no `[name]` tokens.
    When:  discover_sequences_for_dataset is called.
    Then:  A ValueError is raised — nothing to discover.
    """
    with pytest.raises(ValueError, match=r"no \[name\] tokens"):
        discover_sequences_for_dataset("/data/static/path")


def test_reject_duplicate_named_wildcards():
    """
    Given: A pattern that reuses the same `[name]` token twice.
    When:  discover_sequences_for_dataset is called.
    Then:  A ValueError is raised naming the duplicate.
    """
    with pytest.raises(ValueError, match="reuses capture names"):
        discover_sequences_for_dataset("/data/[x]/[x]")


# --- discover_sequences_for_dataset: filesystem outcomes ---


def test_zero_matches_raises_file_not_found(tmp_path):
    """
    Given: An empty directory tree.
    When:  discover_sequences_for_dataset is called with a pattern that finds nothing.
    Then:  FileNotFoundError is raised — an empty discovery is treated as a mistake.
    """
    with pytest.raises(FileNotFoundError, match="matched zero paths"):
        discover_sequences_for_dataset(f"{tmp_path}/[session]")


def test_relative_pattern_resolved_against_cwd(dataset_tree, monkeypatch):
    """
    Given: A relative pattern like `[session]/[recording]` and a cwd containing matches.
    When:  discover_sequences_for_dataset is called.
    Then:  `_normalize`'s `.absolute()` makes the relative pattern work against cwd.
    """
    root = dataset_tree(["S1/Seq1/", "S2/Seq1/"])
    monkeypatch.chdir(root)

    result = discover_sequences_for_dataset("[session]/[recording]")
    assert result == [
        {"session": "S1", "recording": "Seq1"},
        {"session": "S2", "recording": "Seq1"},
    ]


# --- discover_sequences_for_dataset: capture edge cases ---


def test_regex_metacharacters_in_literal_are_escaped(dataset_tree):
    """
    Given: A pattern with a literal `.` and two directories that differ only
           in whether that `.` is literal (`data.v1`) or any-char (`dataXv1`).
    When:  discover_sequences_for_dataset is called.
    Then:  Only the literal-dot directory matches — `.` is not treated as any-char.
    """
    root = dataset_tree(["data.v1/S1/", "dataXv1/S1/"])
    result = discover_sequences_for_dataset(f"{root}/data.v1/[x]")
    assert result == [{"x": "S1"}]


def test_partial_segment_capture(dataset_tree):
    """
    Given: A pattern where `[name]` is only part of a segment (e.g. `[name].mp4`).
    When:  discover_sequences_for_dataset is called on file candidates.
    Then:  The capture is the segment prefix, not the whole segment.
    """
    root = dataset_tree(["S1.mp4", "S2.mp4"])
    result = discover_sequences_for_dataset(f"{root}/[name].mp4")
    assert result == [{"name": "S1"}, {"name": "S2"}]


def test_captured_value_allows_dots_dashes_and_spaces(dataset_tree):
    """
    Given: Directories whose names contain dots, dashes, and spaces.
    When:  discover_sequences_for_dataset is called.
    Then:  `[^/]+` accepts any non-slash characters in the capture.
    """
    root = dataset_tree(["Seq-1.take2/", "my recording/"])
    result = discover_sequences_for_dataset(f"{root}/[name]")
    assert result == [{"name": "Seq-1.take2"}, {"name": "my recording"}]


def test_reject_adjacent_wildcards():
    """
    Given: A pattern with two adjacent `[name]` tokens like `[a][b]` and no
           separator between them.
    When:  discover_sequences_for_dataset is called.
    Then:  A ValueError is raised — the capture would be ambiguous.
    """
    with pytest.raises(ValueError, match="adjacent \\[name\\] tokens"):
        discover_sequences_for_dataset("/data/[a][b]/end")


def test_parent_dir_reference_in_pattern(dataset_tree, monkeypatch):
    """
    Given: A pattern using `..` (e.g. `../datasets/comm_multiview/[seq]`) run
           from a cwd sibling to the target directory.
    When:  discover_sequences_for_dataset is called.
    Then:  Discovery works — `..` is preserved literally on both pattern and
           candidate sides (glob passes it through, `.absolute()` doesn't
           collapse it), so the regex matches.
    """
    root = dataset_tree(["datasets/comm_multiview/S1/", "datasets/comm_multiview/S2/", "subproject/"])
    monkeypatch.chdir(root / "subproject")

    result = discover_sequences_for_dataset("../datasets/comm_multiview/[seq]")
    assert result == [{"seq": "S1"}, {"seq": "S2"}]


def test_trailing_slash_in_pattern(dataset_tree):
    """
    Given: A pattern with a trailing slash like `{root}/[session]/`.
    When:  discover_sequences_for_dataset is called.
    Then:  Discovery works — `_normalize` (via `Path`) strips the trailing
           slash, so pattern and candidate shapes stay symmetric.
    """
    root = dataset_tree(["S1/", "S2/"])
    result = discover_sequences_for_dataset(f"{root}/[session]/")
    assert result == [{"session": "S1"}, {"session": "S2"}]


def test_wildcards_separated_by_literal_char(dataset_tree):
    """
    Given: A pattern like `[a]_[b]` with a single literal char between wildcards.
    When:  discover_sequences_for_dataset is called.
    Then:  Both wildcards capture cleanly around the separator.
    """
    root = dataset_tree(["S1_Seq1/", "S2_Seq2/"])
    result = discover_sequences_for_dataset(f"{root}/[session]_[recording]")
    assert result == [
        {"session": "S1", "recording": "Seq1"},
        {"session": "S2", "recording": "Seq2"},
    ]


def test_shallow_pattern_ignores_nested_content(dataset_tree):
    """
    Given: A nested tree `foo/bar/` and a single-wildcard pattern `[x]`.
    When:  discover_sequences_for_dataset is called.
    Then:  Only the top-level `foo` is captured — glob does not recurse, and
           the regex refuses to cross `/`, so `bar` stays invisible.
    """
    root = dataset_tree(["foo/bar/"])
    result = discover_sequences_for_dataset(f"{root}/[x]")
    assert result == [{"x": "foo"}]


# --- discover_sequences_for_dataset: malformed patterns ---


def test_reject_unclosed_bracket():
    """
    Given: A pattern with an unclosed bracket like `/data/[session`.
    When:  discover_sequences_for_dataset is called.
    Then:  A ValueError is raised — no valid `[name]` tokens found.
    """
    with pytest.raises(ValueError, match=r"no \[name\] tokens"):
        discover_sequences_for_dataset("/data/[session")


def test_reject_unnamed_single_wildcard():
    """
    Given: A pattern with a raw `*` glob wildcard alongside `[name]` tokens.
    When:  discover_sequences_for_dataset is called.
    Then:  A ValueError is raised — only named wildcards are supported.
    """
    with pytest.raises(ValueError, match="unnamed glob wildcards"):
        discover_sequences_for_dataset("/data/[session]/*.mp4")


def test_reject_unnamed_recursive_wildcard():
    """
    Given: A pattern with a raw `**` recursive glob wildcard.
    When:  discover_sequences_for_dataset is called.
    Then:  A ValueError is raised — only named wildcards are supported.
    """
    with pytest.raises(ValueError, match="unnamed glob wildcards"):
        discover_sequences_for_dataset("/data/[foo]/**/[bar]")


def test_reject_empty_bracket():
    """
    Given: A pattern with an empty `[]` token and no other named wildcards.
    When:  discover_sequences_for_dataset is called.
    Then:  A ValueError is raised — `[]` is not a valid named wildcard.
    """
    with pytest.raises(ValueError, match=r"no \[name\] tokens"):
        discover_sequences_for_dataset("/data/[]")
