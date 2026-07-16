"""Provenance chain tests: verify, chain-link, sign.

Exercises orchestration/provenance.py at the file level (no CAD/FEA needed):

  1. a clean 4-stage chain records and audits green,
  2. tampering with a produced artifact is caught by the end-of-run audit,
  3. tampering with an upstream BETWEEN stages is caught AT THE SEAM,
     when the consuming stage records (not hours later at audit time),
  4. the HMAC signature validates, and any edit to the chain file itself
     (e.g. rewriting a recorded hash to match tampered bytes) invalidates it,
  5. unsigned chains pass by default but fail under --require-signature,
  6. the CLI exits 0 on a clean run and 1 on a tampered one.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from biomimetic_pipeline.orchestration.provenance import (  # noqa: E402
    KEY_ENV_VAR,
    ProvenanceChain,
    ProvenanceError,
    verify_chain,
    verify_file,
)
from biomimetic_pipeline.orchestration.provenance import (
    main as provenance_cli,
)


def _build_run(run_dir: Path) -> ProvenanceChain:
    """Fabricate a minimal 4-stage run: morpho -> cad -> fea -> metrics."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "morphometrics.json").write_text(json.dumps({"rod_diameter_um": 4.2}))
    (run_dir / "cad_params.json").write_text(json.dumps({"ROD_DIAMETER": 2.1}))
    (run_dir / "element_results.csv").write_text("cx_mm,p1_MPa\n0.0,1.0\n")
    (run_dir / "metrics.json").write_text(json.dumps({"tortuosity": 1.03}))

    chain = ProvenanceChain(run_dir=run_dir)
    chain.record("morphometrics", run_dir / "morphometrics.json")
    chain.record(
        "cad_parameters",
        run_dir / "cad_params.json",
        upstream={"morphometrics": run_dir / "morphometrics.json"},
    )
    chain.record(
        "fea_element_results",
        run_dir / "element_results.csv",
        upstream={"cad_parameters": run_dir / "cad_params.json"},
    )
    chain.record(
        "metrics",
        run_dir / "metrics.json",
        upstream={"fea_element_results": run_dir / "element_results.csv"},
    )
    return chain


def test_clean_chain_audits_green(tmp_path):
    chain = _build_run(tmp_path / "run")
    chain.sign()  # no key in env -> recorded as unsigned; still auditable
    report = verify_chain(tmp_path / "run")
    assert report["ok"] is True
    assert report["artifacts_verified"] == 4
    assert report["links_verified"] == 3
    assert report["stages"] == ["morphometrics", "cad_parameters", "fea_element_results", "metrics"]


def test_artifact_tamper_caught_by_audit(tmp_path):
    run = tmp_path / "run"
    _build_run(run)
    # Flip bytes AFTER the run completed: the audit must name the file.
    (run / "element_results.csv").write_text("cx_mm,p1_MPa\n0.0,9.9\n")
    with pytest.raises(ProvenanceError, match="element_results.csv"):
        verify_chain(run)


def test_upstream_tamper_caught_at_the_seam(tmp_path):
    run = tmp_path / "run"
    run.mkdir(parents=True)
    (run / "morphometrics.json").write_text(json.dumps({"rod_diameter_um": 4.2}))
    (run / "cad_params.json").write_text(json.dumps({"ROD_DIAMETER": 2.1}))

    chain = ProvenanceChain(run_dir=run)
    chain.record("morphometrics", run / "morphometrics.json")
    # The upstream mutates BETWEEN stage 1 and stage 2 (mid-run swap):
    (run / "morphometrics.json").write_text(json.dumps({"rod_diameter_um": 999.0}))
    # ...so stage 2's record() must refuse to seal the contract.
    with pytest.raises(ProvenanceError, match="morphometrics"):
        chain.record(
            "cad_parameters",
            run / "cad_params.json",
            upstream={"morphometrics": run / "morphometrics.json"},
        )


def test_signature_roundtrip_and_chainfile_tamper(tmp_path, monkeypatch):
    monkeypatch.setenv(KEY_ENV_VAR, "test-key-of-suitable-entropy")
    run = tmp_path / "run"
    chain = _build_run(run)
    sig = chain.sign()
    assert sig is not None and sig["algo"] == "HMAC-SHA256"

    report = verify_chain(run, require_signature=True)
    assert report["signature"] == "valid"

    # Rewrite one recorded hash inside the chain FILE (what an attacker
    # would do to make tampered bytes look legitimate): MAC must fail first.
    doc = json.loads((run / "provenance_chain.json").read_text())
    doc["entries"][2]["sha256"] = "0" * 64
    (run / "provenance_chain.json").write_text(json.dumps(doc, indent=2))
    with pytest.raises(ProvenanceError, match="signature INVALID"):
        verify_chain(run)


def test_unsigned_chain_fails_only_when_required(tmp_path, monkeypatch):
    monkeypatch.delenv(KEY_ENV_VAR, raising=False)
    run = tmp_path / "run"
    chain = _build_run(run)
    assert chain.sign() is None  # no key -> unsigned
    assert verify_chain(run)["signature"] == "absent"  # default: pass, reported
    with pytest.raises(ProvenanceError, match="UNSIGNED"):
        verify_chain(run, require_signature=True)


def test_verify_file_reports_both_hashes(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("original")
    with pytest.raises(ProvenanceError) as exc:
        verify_file(f, "f" * 64)
    msg = str(exc.value)
    assert "expected sha256" in msg and "actual   sha256" in msg


def test_cli_exit_codes(tmp_path, capsys):
    run = tmp_path / "run"
    _build_run(run)
    assert provenance_cli([str(run)]) == 0
    assert "PROVENANCE OK" in capsys.readouterr().out

    (run / "metrics.json").write_text("{}")  # tamper
    assert provenance_cli([str(run)]) == 1
    assert "PROVENANCE FAIL" in capsys.readouterr().out
