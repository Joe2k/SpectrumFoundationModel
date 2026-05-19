import json
import subprocess
import sys
from pathlib import Path


def test_stage_script(tmp_path):
    cfs = tmp_path / "cfs" / "spectro/redux/iron/healpix/sv3/bright/000/10016"
    cfs.mkdir(parents=True)
    coadd = cfs / "coadd-sv3-bright-10016.fits"
    rr = cfs / "redrock-sv3-bright-10016.fits"
    coadd.write_bytes(b"x" * 100)
    rr.write_bytes(b"y" * 50)
    src_m = tmp_path / "src.jsonl"
    src_m.write_text(
        json.dumps({"coadd": str(coadd), "redrock": str(rr), "healpix": 10016, "n_rows": 1}) + "\n"
    )
    dst_m = tmp_path / "dst_scratch.jsonl"
    script = Path(__file__).resolve().parents[1] / "scripts" / "stage_data.py"
    subprocess.check_call(
        [
            sys.executable,
            str(script),
            "--src-manifest",
            str(src_m),
            "--dst-root",
            str(tmp_path / "staged"),
            "--dst-manifest",
            str(dst_m),
            "--src-prefix",
            str(tmp_path / "cfs") + "/",
        ]
    )
    rec = json.loads(dst_m.read_text().strip())
    assert "staged" in rec["coadd"]
