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
