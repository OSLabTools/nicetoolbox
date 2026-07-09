import pytest
import toml

from nicetoolbox.configs.config_loader import ConfigLoader
from nicetoolbox.configs.schemas.dataset_properties import DatasetProperties


@pytest.fixture
def load_dataset_props(tmp_path):
    """
    Load a dict as `DatasetProperties` through the real `ConfigLoader`.

    Serializes the dict to a TOML file so `pre_placeholder_resolve` runs inside
    the normal pipeline (raw → pre_resolve → full resolve → pydantic).
    """

    def _load(raw, placeholders=None, runtime=None):
        path = tmp_path / "dataset_properties.toml"
        path.write_text(toml.dumps(raw))
        loader = ConfigLoader(auto=placeholders or {}, runtime=runtime or set())
        return loader.load_config(path, DatasetProperties)

    return _load


@pytest.fixture
def dataset_tree(tmp_path):
    """Build a fake filesystem tree under `tmp_path/tree` (dirs marked by trailing `/`)."""

    root = tmp_path / "tree"
    root.mkdir()

    def _build(paths):
        for rel in paths:
            full = root / rel
            full.parent.mkdir(parents=True, exist_ok=True)
            if rel.endswith("/"):
                full.mkdir(exist_ok=True)
            else:
                full.touch()
        return root

    return _build


# --- template merge ---


def test_template_fields_merged_into_each_sequence(load_dataset_props):
    """
    Given: A template supplying `subjects_descr` and sequences that omit it.
    When:  DatasetProperties is loaded.
    Then:  Every sequence carries the template's `subjects_descr`.
    """
    result = load_dataset_props(
        {
            "ds": {
                "sequences": [
                    {"sequence_id": "s1", "path_to_calibrations": "/tmp/s1"},
                    {"sequence_id": "s2", "path_to_calibrations": "/tmp/s2"},
                ],
                "template": {"subjects_descr": ["alice", "bob"]},
            }
        }
    )
    seqs = result["ds"].sequences
    assert [s.subjects_descr for s in seqs] == [["alice", "bob"], ["alice", "bob"]]


def test_sequence_value_overrides_template_value(load_dataset_props):
    """
    Given: Template and sequence both define `subjects_descr`.
    When:  DatasetProperties is loaded.
    Then:  The sequence-level value wins on a per-key basis.
    """
    result = load_dataset_props(
        {
            "ds": {
                "sequences": [
                    {
                        "sequence_id": "s1",
                        "path_to_calibrations": "/tmp/s1",
                        "subjects_descr": ["only_me"],
                    },
                ],
                "template": {"subjects_descr": ["alice", "bob"]},
            }
        }
    )
    assert result["ds"].sequences[0].subjects_descr == ["only_me"]


def test_video_block_replaces_not_deep_merged(load_dataset_props):
    """
    Given: Template defines two cameras; sequence overrides `video` with only one camera.
    When:  DatasetProperties is loaded.
    Then:  The sequence's `video` block wholesale replaces the template's — other
           cameras disappear (shallow merge, not a deep dict merge).
    """
    result = load_dataset_props(
        {
            "ds": {
                "sequences": [
                    {
                        "sequence_id": "s1",
                        "path_to_calibrations": "/tmp/s1",
                        "subjects_descr": ["a"],
                        "video": {
                            "cameras": {
                                "only_cam": {"path": "/tmp/s1/only.mp4", "sees_subjects": [0]},
                            },
                        },
                    },
                ],
                "template": {
                    "video": {
                        "cameras": {
                            "cam_a": {"path": "/tmp/s1/a.mp4", "sees_subjects": [0]},
                            "cam_b": {"path": "/tmp/s1/b.mp4", "sees_subjects": [0]},
                        },
                    },
                },
            }
        }
    )
    cameras = result["ds"].sequences[0].video.cameras
    assert set(cameras.keys()) == {"only_cam"}


def test_template_sibling_placeholders_resolve_per_sequence(load_dataset_props):
    """
    Given: Template fields reference each other and `<sequence_id>` (a sibling
           of the merged sequence dict).
    When:  DatasetProperties is loaded.
    Then:  Each sequence's `path_to_calibrations` resolves using its own id.
    """
    result = load_dataset_props(
        {
            "ds": {
                "sequences": [
                    {"sequence_id": "s1"},
                    {"sequence_id": "s2"},
                ],
                "template": {
                    "dataset_root": "/data/ds",
                    "path_to_calibrations": "<dataset_root>/<sequence_id>",
                    "subjects_descr": ["a"],
                },
            }
        }
    )
    seqs = result["ds"].sequences
    assert str(seqs[0].path_to_calibrations).replace("\\", "/") == "/data/ds/s1"
    assert str(seqs[1].path_to_calibrations).replace("\\", "/") == "/data/ds/s2"


# --- discovery ---


def test_discovery_appends_to_explicit_sequences(load_dataset_props, dataset_tree):
    """
    Given: An explicit sequence plus a `discover_sequences` pattern matching two dirs.
    When:  DatasetProperties is loaded.
    Then:  The explicit sequence and the two discovered ones all appear.
    """
    root = dataset_tree(["disc_a/", "disc_b/"])
    result = load_dataset_props(
        {
            "ds": {
                "discover_sequences": f"{root.as_posix()}/[sequence_id]",
                "sequences": [
                    {"sequence_id": "explicit", "path_to_calibrations": "/tmp/explicit"},
                ],
                "template": {
                    "path_to_calibrations": "/tmp/<sequence_id>",
                    "subjects_descr": ["a"],
                },
            }
        }
    )
    ids = [s.sequence_id for s in result["ds"].sequences]
    assert set(ids) == {"explicit", "disc_a", "disc_b"}


def test_discovery_uses_root_level_placeholders(load_dataset_props, dataset_tree):
    """
    Given: `discover_sequences` contains `<datasets_folder_path>`, provided as
           an auto placeholder.
    When:  DatasetProperties is loaded.
    Then:  The shallow root-level resolve inside `pre_placeholder_resolve` fills
           the placeholder before discovery runs against the filesystem.
    """
    root = dataset_tree(["s1/", "s2/"])
    result = load_dataset_props(
        {
            "ds": {
                "discover_sequences": "<datasets_folder_path>/[sequence_id]",
                "sequences": [],
                "template": {
                    "path_to_calibrations": "/tmp/<sequence_id>",
                    "subjects_descr": ["a"],
                },
            }
        },
        placeholders={"datasets_folder_path": root.as_posix()},
    )
    ids = sorted(s.sequence_id for s in result["ds"].sequences)
    assert ids == ["s1", "s2"]


def test_discovered_sequences_get_template_merged(load_dataset_props, dataset_tree):
    """
    Given: Discovery yields raw sequence dicts (only capture keys) and a template
           supplies the required schema fields.
    When:  DatasetProperties is loaded.
    Then:  Discovered sequences validate against `SequenceConfig` because the
           template's fields land on them via the merge.
    """
    root = dataset_tree(["seq1/", "seq2/"])
    result = load_dataset_props(
        {
            "ds": {
                "discover_sequences": f"{root.as_posix()}/[sequence_id]",
                "sequences": [],
                "template": {
                    "path_to_calibrations": "/tmp/<sequence_id>",
                    "subjects_descr": ["a", "b"],
                },
            }
        }
    )
    for seq in result["ds"].sequences:
        assert seq.subjects_descr == ["a", "b"]
        assert seq.sequence_id in str(seq.path_to_calibrations)


def test_discovery_with_sequence_id_capture_needs_no_template_sequence_id(load_dataset_props, dataset_tree):
    """
    Given: Discovery pattern captures `[sequence_id]` directly and the template
           does NOT define `sequence_id`.
    When:  DatasetProperties is loaded.
    Then:  Each sequence uses the capture as its id — no composition step
           needed. Template's `.get("sequence_id")` returning None short-circuits
           the compose loop cleanly.
    """
    root = dataset_tree(["s1/", "s2/"])
    result = load_dataset_props(
        {
            "ds": {
                "discover_sequences": f"{root.as_posix()}/[sequence_id]",
                "sequences": [],
                "template": {
                    "subjects_descr": ["a"],
                    # Note: no `sequence_id` here — capture supplies it.
                },
            }
        }
    )
    ids = sorted(s.sequence_id for s in result["ds"].sequences)
    assert ids == ["s1", "s2"]


def test_template_sequence_id_referencing_outer_scope_breaks_override_match(load_dataset_props, dataset_tree):
    """
    Given: Template `sequence_id` references a dataset-root sibling (e.g.
           `<dataset_root>`) that's not a discovery capture. Strict resolution
           in step 3b can't complete (only the capture dict is in scope), so
           the discovered entry gets no `sequence_id` and override matching
           silently skips it. The user's explicit override then lands as a
           second entry with the same final id → duplicate raises.
    When:  DatasetProperties is loaded.
    Then:  Duplicate sequence_id error is raised (documents the edge case).

    If the pre-resolve later passes outer placeholders into the strict call,
    this test will flip — override matching would succeed and no duplicate
    would fire.
    """
    root = dataset_tree(["PIS_ID_000/", "PIS_ID_02/"])
    root_posix = root.as_posix()
    with pytest.raises(Exception, match=r"[Dd]uplicate sequence_id"):
        load_dataset_props(
            {
                "ds": {
                    "dataset_root": root_posix,
                    "discover_sequences": "<dataset_root>/PIS_ID_[pis_id]",
                    "sequences": [
                        {
                            "sequence_id": f"{root_posix}_PIS_ID_000",
                            "audio": {"tracks": {"room": {"path": "/tmp/room.wav", "hears_subjects": [0]}}},
                        },
                    ],
                    "template": {
                        "sequence_id": "<dataset_root>_PIS_ID_<pis_id>",
                        "subjects_descr": ["a"],
                    },
                }
            }
        )


def test_discovery_without_sequence_id_capture_or_template_composition_raises(load_dataset_props, dataset_tree):
    """
    Given: Discovery captures `[pis_id]` (not `[sequence_id]`) and the template
           does NOT define `sequence_id`.
    When:  DatasetProperties is loaded.
    Then:  Pydantic validation fails — no path produces a `sequence_id` for the
           discovered entries. Documents the "broken config" case so future
           changes to the pre-resolve logic don't silently start accepting it.
    """
    root = dataset_tree(["PIS_ID_000/", "PIS_ID_02/"])
    with pytest.raises(Exception, match="sequence_id"):
        load_dataset_props(
            {
                "ds": {
                    "discover_sequences": f"{root.as_posix()}/PIS_ID_[pis_id]",
                    "sequences": [],
                    "template": {
                        "subjects_descr": ["a"],
                    },
                }
            }
        )


def test_discovery_multi_capture_composed_via_template(load_dataset_props, dataset_tree):
    """
    Given: A multi-capture pattern `[session]/[recording]` and a template that
           composes `sequence_id = "<session>_<recording>"`.
    When:  DatasetProperties is loaded.
    Then:  Each discovered pair lands in the merged dict as sibling values,
           and the placeholder resolver composes them into `sequence_id`.
    """
    root = dataset_tree(["session_a/rec_1/", "session_a/rec_2/", "session_b/rec_1/"])
    result = load_dataset_props(
        {
            "ds": {
                "discover_sequences": f"{root.as_posix()}/[session]/[recording]",
                "sequences": [],
                "template": {
                    "sequence_id": "<session>_<recording>",
                    "path_to_calibrations": "/tmp/<session>/<recording>",
                    "subjects_descr": ["a"],
                },
            }
        }
    )
    ids = sorted(s.sequence_id for s in result["ds"].sequences)
    assert ids == ["session_a_rec_1", "session_a_rec_2", "session_b_rec_1"]


def test_discovery_capture_overrides_template_value(load_dataset_props, dataset_tree):
    """
    Given: Template sets `sequence_id = "template_default"` and discovery yields
           its own `sequence_id` via a `[sequence_id]` capture.
    When:  DatasetProperties is loaded.
    Then:  The discovered value wins — sequence-side keys override template on
           a per-key basis, and discovered dicts are treated as sequence dicts.
    """
    root = dataset_tree(["s1/", "s2/"])
    result = load_dataset_props(
        {
            "ds": {
                "discover_sequences": f"{root.as_posix()}/[sequence_id]",
                "sequences": [],
                "template": {
                    "sequence_id": "template_default",
                    "path_to_calibrations": "/tmp/<sequence_id>",
                    "subjects_descr": ["a"],
                },
            }
        }
    )
    ids = sorted(s.sequence_id for s in result["ds"].sequences)
    assert ids == ["s1", "s2"]


def test_explicit_sequence_overrides_discovered_by_id(load_dataset_props, dataset_tree):
    """
    Given: Discovery finds sequences `PIS_ID_000`, `PIS_ID_02` (via `[pis_id]`
           capture + template `sequence_id = "PIS_ID_<pis_id>"`) and an explicit
           sequence with `sequence_id = "PIS_ID_000"` supplying audio config.
    When:  DatasetProperties is loaded.
    Then:  Two sequences result — the explicit one merges *on top of* the
           matching discovered entry (adding audio while preserving captures),
           and the non-matching discovered sequence is untouched.
    """
    root = dataset_tree(["PIS_ID_000/", "PIS_ID_02/"])
    result = load_dataset_props(
        {
            "ds": {
                "discover_sequences": f"{root.as_posix()}/PIS_ID_[pis_id]",
                "sequences": [
                    {
                        "sequence_id": "PIS_ID_000",
                        "audio": {
                            "tracks": {
                                "room": {"path": "/tmp/room.wav", "hears_subjects": [0]},
                            },
                        },
                    },
                ],
                "template": {
                    "sequence_id": "PIS_ID_<pis_id>",
                    "path_to_calibrations": "/tmp/<sequence_id>",
                    "subjects_descr": ["a"],
                },
            }
        }
    )
    seqs = {s.sequence_id: s for s in result["ds"].sequences}
    assert set(seqs) == {"PIS_ID_000", "PIS_ID_02"}
    # Override applied: audio present on the matched sequence
    assert "room" in seqs["PIS_ID_000"].audio.tracks
    # Non-matching discovered sequence keeps only template-derived fields
    assert seqs["PIS_ID_02"].audio.tracks == {}


def test_explicit_sequence_no_match_still_appended(load_dataset_props, dataset_tree):
    """
    Given: Discovery yields one sequence and an explicit sequence with a
           `sequence_id` that matches nothing discovered.
    When:  DatasetProperties is loaded.
    Then:  The explicit sequence is appended as a standalone entry — override
           semantics don't swallow it.
    """
    root = dataset_tree(["PIS_ID_000/"])
    result = load_dataset_props(
        {
            "ds": {
                "discover_sequences": f"{root.as_posix()}/PIS_ID_[pis_id]",
                "sequences": [
                    {"sequence_id": "extra", "path_to_calibrations": "/tmp/extra"},
                ],
                "template": {
                    "sequence_id": "PIS_ID_<pis_id>",
                    "path_to_calibrations": "/tmp/<sequence_id>",
                    "subjects_descr": ["a"],
                },
            }
        }
    )
    ids = sorted(s.sequence_id for s in result["ds"].sequences)
    assert ids == ["PIS_ID_000", "extra"]


def test_duplicate_explicit_sequence_ids_raise(load_dataset_props):
    """
    Given: Two explicit sequences share the same `sequence_id`.
    When:  DatasetProperties is loaded.
    Then:  `_check_unique_sequence_ids` raises. (Explicit+discovered matches are
           merged as overrides, so uniqueness violations only come from repeated
           explicit ids or repeated discovery captures.)
    """
    with pytest.raises(Exception, match=r"[Dd]uplicate sequence_id.*s1"):
        load_dataset_props(
            {
                "ds": {
                    "sequences": [
                        {"sequence_id": "s1", "path_to_calibrations": "/tmp/s1"},
                        {"sequence_id": "s1", "path_to_calibrations": "/tmp/other"},
                    ],
                    "template": {"subjects_descr": ["a"]},
                }
            }
        )


def test_discovery_zero_matches_raises(load_dataset_props, tmp_path):
    """
    Given: A `discover_sequences` pattern that finds nothing on disk.
    When:  DatasetProperties is loaded.
    Then:  `FileNotFoundError` from `discover_sequences_for_dataset` bubbles
           through `load_config` unchanged.
    """
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="matched zero paths"):
        load_dataset_props(
            {
                "ds": {
                    "discover_sequences": f"{empty.as_posix()}/[sequence_id]",
                    "sequences": [],
                    "template": {
                        "path_to_calibrations": "/tmp/<sequence_id>",
                        "subjects_descr": ["a"],
                    },
                }
            }
        )


# --- dataset-root placeholder scope ---


def test_dataset_root_sibling_placeholders_resolve_before_discovery(load_dataset_props, dataset_tree):
    """
    Given: A dataset root defines `dataset_root` and uses it inside `discover_sequences`.
    When:  DatasetProperties is loaded.
    Then:  `pre_placeholder_resolve`'s shallow root-level self-reference resolution
           fills `<dataset_root>` before discovery runs against the filesystem.
    """
    root = dataset_tree(["a/", "b/"])
    result = load_dataset_props(
        {
            "ds": {
                "dataset_root": root.as_posix(),
                "discover_sequences": "<dataset_root>/[sequence_id]",
                "sequences": [],
                "template": {
                    "path_to_calibrations": "/tmp/<sequence_id>",
                    "subjects_descr": ["x"],
                },
            }
        }
    )
    ids = sorted(s.sequence_id for s in result["ds"].sequences)
    assert ids == ["a", "b"]


def test_runtime_placeholder_in_dataset_root_sibling_raises(load_dataset_props):
    """
    Given: A runtime placeholder (`<cur_thing>`) inside a dataset-root sibling
           field (not `discover_sequences`).
    When:  DatasetProperties is loaded.
    Then:  `pre_placeholder_resolve` raises — the shallow root-level resolver
           processes every top-level string and discards `unreachable`, so any
           deferred placeholder at that scope fails, regardless of which key
           holds it.
    """
    with pytest.raises(ValueError, match="Could not resolve placeholders"):
        load_dataset_props(
            {
                "ds": {
                    "dataset_root": "/data/<cur_thing>",
                    "sequences": [{"sequence_id": "s1", "path_to_calibrations": "/tmp/s1"}],
                    "template": {"subjects_descr": ["a"]},
                }
            },
            runtime={"cur_thing"},
        )


def test_runtime_placeholder_inside_template_is_deferred(load_dataset_props):
    """
    Given: A runtime placeholder (`<cur_thing>`) buried inside `template`.
    When:  DatasetProperties is loaded.
    Then:  It survives pre-resolve (shallow scope skips nested dicts) and is
           left unresolved by the full-resolve pass (runtime placeholders are
           in `unreachable`), so it lands in the final model verbatim.
    """
    result = load_dataset_props(
        {
            "ds": {
                "sequences": [{"sequence_id": "s1"}],
                "template": {
                    "path_to_calibrations": "/tmp/<cur_thing>/s1",
                    "subjects_descr": ["a"],
                },
            }
        },
        runtime={"cur_thing"},
    )
    assert "<cur_thing>" in str(result["ds"].sequences[0].path_to_calibrations)


def test_runtime_placeholder_inside_sequence_is_deferred(load_dataset_props):
    """
    Given: A runtime placeholder (`<cur_thing>`) inside a sequence entry.
    When:  DatasetProperties is loaded.
    Then:  Same as the template case — the shallow root-scope resolver never
           descends into `sequences[]`, and full-resolve defers runtime names.
    """
    result = load_dataset_props(
        {
            "ds": {
                "sequences": [
                    {
                        "sequence_id": "s1",
                        "path_to_calibrations": "/tmp/<cur_thing>/s1",
                        "subjects_descr": ["a"],
                    },
                ],
            }
        },
        runtime={"cur_thing"},
    )
    assert "<cur_thing>" in str(result["ds"].sequences[0].path_to_calibrations)


# --- pre-validation errors (before pydantic) ---


def test_template_wrong_type_raises_with_dataset_name(load_dataset_props):
    """
    Given: A `template` field that is not a table/dict.
    When:  DatasetProperties is loaded.
    Then:  ValueError is raised, tagged with the offending dataset name.
    """
    with pytest.raises(ValueError, match=r"Dataset 'ds'"):
        load_dataset_props({"ds": {"sequences": [], "template": "not a dict"}})


def test_sequences_wrong_type_raises_with_dataset_name(load_dataset_props):
    """
    Given: A `sequences` field that is not a list.
    When:  DatasetProperties is loaded.
    Then:  ValueError is raised, tagged with the offending dataset name.
    """
    with pytest.raises(ValueError, match=r"Dataset 'ds'"):
        load_dataset_props({"ds": {"sequences": 42}})


def test_missing_required_field_after_merge_raises(load_dataset_props):
    """
    Given: Neither template nor sequence supplies `subjects_descr`.
    When:  DatasetProperties is loaded.
    Then:  Pydantic validation fails — the merge doesn't invent fields, so
           gaps between template and sequence surface as schema errors.
    """
    with pytest.raises(Exception, match="subjects_descr"):
        load_dataset_props(
            {
                "ds": {
                    "sequences": [{"sequence_id": "s1", "path_to_calibrations": "/tmp/s1"}],
                }
            }
        )


# --- isolation across datasets ---


def test_templates_do_not_leak_across_datasets(load_dataset_props):
    """
    Given: Two datasets in the same file, each with its own template.
    When:  DatasetProperties is loaded.
    Then:  Each dataset's sequences see only their own template's fields.
    """
    result = load_dataset_props(
        {
            "ds_a": {
                "sequences": [{"sequence_id": "a1", "path_to_calibrations": "/tmp/a1"}],
                "template": {"subjects_descr": ["alice"]},
            },
            "ds_b": {
                "sequences": [{"sequence_id": "b1", "path_to_calibrations": "/tmp/b1"}],
                "template": {"subjects_descr": ["bob"]},
            },
        }
    )
    assert result["ds_a"].sequences[0].subjects_descr == ["alice"]
    assert result["ds_b"].sequences[0].subjects_descr == ["bob"]
