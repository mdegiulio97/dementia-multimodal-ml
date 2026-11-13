from dementia_ml.eeg import io


def test_participants_maps_group_codes(tmp_path):
    (tmp_path / "participants.tsv").write_text(
        "participant_id\tGender\tAge\tGroup\tMMSE\nsub-001\tF\t57\tA\t16\nsub-037\tM\t57\tC\t30\nsub-066\tM\t73\tF\t20\n"
    )
    p = io.participants(tmp_path)
    assert p["group"].tolist() == ["AD", "CN", "FTD"]
    assert p["subject"].tolist() == ["001", "037", "066"]
    assert {"age", "mmse"} <= set(p.columns)


def test_subject_path_uses_derivatives(tmp_path):
    path = io.subject_path(tmp_path, "001", use_derivatives=True)
    assert path.as_posix().endswith("derivatives/sub-001/eeg/sub-001_task-eyesclosed_eeg.set")


def test_feature_cache_key_changes_with_config(tmp_path):
    from dementia_ml.eeg import run

    base = {"epoch_s": 4.0, "overlap": 0.5, "use_derivatives": True}
    a = run.cache_path(tmp_path, base)
    b = run.cache_path(tmp_path, {**base, "epoch_s": 2.0})
    c = run.cache_path(tmp_path, {**base, "fmin": 1.0})
    assert len({a, b, c}) == 3
    assert run.cache_path(tmp_path, dict(base)) == a
