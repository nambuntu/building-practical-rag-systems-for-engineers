from workflow.state import build_run_paths, ensure_run_dirs, load_manifest, save_manifest


def test_manifest_roundtrip(tmp_path):
    paths = build_run_paths(tmp_path, "run-x")
    ensure_run_dirs(paths)

    manifest = load_manifest(paths)
    manifest["stages"]["ingest"] = {"status": "completed"}
    save_manifest(paths, manifest)

    loaded = load_manifest(paths)
    assert loaded["run_id"] == "run-x"
    assert loaded["stages"]["ingest"]["status"] == "completed"
