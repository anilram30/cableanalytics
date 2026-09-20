import json

from cableanalytics.cli import main


def test_cli_dataset_correlate_predict_attribute(tmp_path, capsys):
    assert main(["dataset", "--out", str(tmp_path), "--n", "60"]) == 0
    assert main(["correlate", str(tmp_path / "production_dataset.csv"), "--out", str(tmp_path / "corr")]) == 0
    assert "P+ physics+correction" in capsys.readouterr().out
    assert main(["predict", str(tmp_path / "production_dataset.csv"), "--sample", "S0003", "--temperature", "105"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert "p_pass" in d and "measured" in d
    assert main(["attribute", "0.99", "--line-speed", "80", "--screw-rpm", "55", "--capstan-d", "0.315"]) == 0
    assert json.loads(capsys.readouterr().out)["best"]["element"] == "capstan"
