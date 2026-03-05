from typer.testing import CliRunner

from workflow.app import app


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "ingest" in result.stdout
    assert "evaluate" in result.stdout


def test_ingest_local_requires_input_dir(tmp_path):
    runner = CliRunner()
    result = runner.invoke(app, ["ingest", "--source", "local", "--run-id", "r1"])
    assert result.exit_code != 0
