"""Artifact provenance for pipeline runs: verify, chain, sign.

The pipeline's stages communicate through JSON contracts and file artifacts
(morphometrics -> cad_params -> CAD/sidecar -> mesh -> FEA element results ->
metrics/score). ``run_context`` has always been able to *record* SHA-256
hashes; nothing ever *checked* them. This module closes that loop with three
distinct guarantees, in V&V terms:

VERIFY   (are these the bytes we measured?)
    ``verify_file(path, expected)`` re-hashes an artifact from disk and hard-
    fails with BOTH hashes on mismatch. Every stage boundary that consumes an
    upstream artifact verifies it against the hash recorded when the artifact
    was produced -- a mid-run swap or corruption stops the pipeline at the
    seam where it happened, not three stages later in a confusing way.

CHAIN    (is the lineage unbroken?)
    ``ProvenanceChain`` appends one entry per stage to
    ``<run_dir>/provenance_chain.json``: the artifact's hash plus the hashes
    of the upstream artifacts it consumed. Because each stage's upstream
    hashes must equal the earlier stage's recorded artifact hash, the entries
    form a hash-linked lineage from raw morphometrics to final metrics.
    ``verify_chain(run_dir)`` is the end-of-run audit: it re-hashes every
    artifact from disk and re-checks every link. Also exposed as a CLI:

        python -m biomimetic_pipeline.orchestration.provenance <run_dir> \\
            [--require-signature]

VALIDATE / SIGN  (says who?)
    ``sign()`` appends an HMAC-SHA256 over the canonical JSON of the entries,
    keyed by the ``BLP_PROVENANCE_KEY`` environment variable (never stored).
    A verifier holding the key detects ANY edit to the chain file itself --
    including a rewrite of the recorded hashes to match tampered artifacts.
    ``key_id`` (a hash prefix of the key) lets rotated keys be told apart.
    HMAC is a shared-secret scheme chosen for zero dependencies; the upgrade
    path to asymmetric signatures (minisign / Sigstore, as used for the CI
    SBOM) only replaces ``_mac`` and ``verify``'s comparison.

Failure model: every check raises ``ProvenanceError`` naming the artifact,
the stage, and the expected/actual hashes. Nothing is silently skipped: a
missing artifact is a failure, an unsigned chain is reported (and fails under
``--require-signature``).
"""

from __future__ import annotations

import argparse
import hashlib
import hmac as _hmac
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from biomimetic_pipeline.orchestration.run_context import now_iso, sha256_file

CHAIN_FILENAME = "provenance_chain.json"
KEY_ENV_VAR = "BLP_PROVENANCE_KEY"


class ProvenanceError(RuntimeError):
    """An artifact or chain failed hash verification / validation."""


def verify_file(path: Path, expected_sha256: str, *, what: str = "artifact") -> str:
    """Re-hash ``path`` and compare against ``expected_sha256``.

    Returns the (matching) hash so callers can thread it onward. Raises
    ProvenanceError naming the file and BOTH hashes on mismatch -- the two
    hex strings in the message are exactly what a human needs to decide
    whether the file changed or the expectation was stale.
    """
    p = Path(path)
    if not p.exists():
        raise ProvenanceError(f"{what} missing, cannot verify: {p}")
    actual = sha256_file(p)
    if actual != expected_sha256:
        raise ProvenanceError(
            f"{what} hash mismatch for {p}:\n"
            f"  expected sha256 {expected_sha256}\n"
            f"  actual   sha256 {actual}\n"
            f"The bytes on disk are not the bytes that were recorded."
        )
    return actual


@dataclass
class ProvenanceChain:
    """Hash-linked record of every stage artifact in one pipeline run."""

    run_dir: Path
    entries: List[Dict[str, Any]] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Recording (called by the pipeline at each stage seam)              #
    # ------------------------------------------------------------------ #
    def record(
        self,
        stage: str,
        artifact_path: Path,
        upstream: Optional[Dict[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Hash ``artifact_path`` and append a chain entry for ``stage``.

        Every ``upstream`` file is (a) re-hashed from disk NOW, and (b) if it
        already appears as an earlier entry's artifact, VERIFIED against that
        recorded hash -- so a stage can never silently consume an upstream
        artifact that changed after it was produced. This is the per-boundary
        "verify" guarantee.
        """
        artifact_path = Path(artifact_path)
        up_records: Dict[str, Dict[str, str]] = {}
        for name, upath in (upstream or {}).items():
            upath = Path(upath)
            recorded = self._recorded_hash_for(upath)
            if recorded is not None:
                # Upstream was produced earlier in this run: bytes must still
                # match what the producing stage recorded.
                actual = verify_file(upath, recorded, what=f"upstream '{name}' of stage '{stage}'")
            else:
                # External input (e.g. the measured morphometrics): first
                # sighting -- hash it so everything downstream is anchored.
                if not upath.exists():
                    raise ProvenanceError(f"upstream '{name}' of stage '{stage}' missing: {upath}")
                actual = sha256_file(upath)
            up_records[name] = {"path": str(upath), "sha256": actual}

        if not artifact_path.exists():
            raise ProvenanceError(f"stage '{stage}' artifact missing: {artifact_path}")
        entry = {
            "stage": stage,
            "artifact": str(artifact_path),
            "sha256": sha256_file(artifact_path),
            "upstream": up_records,
            "recorded_at": now_iso(),
        }
        self.entries.append(entry)
        self._write()
        return entry

    def _recorded_hash_for(self, path: Path) -> Optional[str]:
        """Hash recorded for ``path`` as a produced artifact, if any."""
        spath = str(Path(path))
        for e in self.entries:
            if e["artifact"] == spath:
                return e["sha256"]
        return None

    # ------------------------------------------------------------------ #
    # Signing (end of run)                                               #
    # ------------------------------------------------------------------ #
    def sign(self) -> Optional[Dict[str, str]]:
        """HMAC-sign the canonical entries with ``BLP_PROVENANCE_KEY``.

        Without the key the chain is written unsigned and ``None`` returns --
        the absence is itself recorded, and ``verify_chain`` reports it
        (fatally, under require_signature).
        """
        key = os.environ.get(KEY_ENV_VAR)
        if not key:
            self._write(signature=None)
            return None
        signature = {
            "algo": "HMAC-SHA256",
            "key_id": hashlib.sha256(key.encode()).hexdigest()[:12],
            "mac": _mac(key, self.entries),
        }
        self._write(signature=signature)
        return signature

    def _write(self, signature: Optional[Dict[str, str]] = "__keep__") -> None:
        """Persist the chain file; ``signature='__keep__'`` preserves any existing one."""
        path = Path(self.run_dir) / CHAIN_FILENAME
        doc: Dict[str, Any] = {"entries": self.entries}
        if signature == "__keep__":
            if path.exists():
                old = json.loads(path.read_text())
                doc["signature"] = old.get("signature")
            else:
                doc["signature"] = None
        else:
            doc["signature"] = signature
        path.write_text(json.dumps(doc, indent=2))


def _mac(key: str, entries: List[Dict[str, Any]]) -> str:
    """HMAC-SHA256 over the canonical (sorted-keys, tight-separator) JSON."""
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return _hmac.new(key.encode(), canonical, hashlib.sha256).hexdigest()


# ---------------------------------------------------------------------- #
# End-of-run audit ("validate")                                          #
# ---------------------------------------------------------------------- #
def verify_chain(run_dir: Path, require_signature: bool = False) -> Dict[str, Any]:
    """Full audit of a completed run's provenance chain.

    Re-hashes every recorded artifact from disk, re-checks every upstream
    link against its producing stage, and validates the HMAC signature when
    a key is available. Returns a report dict; raises ProvenanceError on the
    first failure (fail-fast, with a precise message).
    """
    chain_path = Path(run_dir) / CHAIN_FILENAME
    if not chain_path.exists():
        raise ProvenanceError(f"no provenance chain found: {chain_path}")
    doc = json.loads(chain_path.read_text())
    entries: List[Dict[str, Any]] = doc.get("entries", [])
    if not entries:
        raise ProvenanceError(f"provenance chain is empty: {chain_path}")

    # 1) SIGNATURE first: if the chain file itself was edited, every later
    #    check would be checking attacker-chosen numbers.
    signature = doc.get("signature")
    key = os.environ.get(KEY_ENV_VAR)
    sig_status = "absent"
    if signature:
        if key:
            expected = _mac(key, entries)
            if not _hmac.compare_digest(expected, signature.get("mac", "")):
                raise ProvenanceError(
                    f"chain signature INVALID for {chain_path} "
                    f"(key_id {signature.get('key_id')}): the chain file was "
                    f"modified after signing, or the key differs."
                )
            sig_status = "valid"
        else:
            sig_status = "present-but-unverifiable (set " + KEY_ENV_VAR + ")"
            if require_signature:
                raise ProvenanceError(
                    f"chain is signed but {KEY_ENV_VAR} is not set; cannot validate signature."
                )
    elif require_signature:
        raise ProvenanceError(f"chain is UNSIGNED and --require-signature was given: {chain_path}")

    # 2) ARTIFACTS: every recorded artifact must still hash to its record.
    produced: Dict[str, str] = {}
    for e in entries:
        verify_file(Path(e["artifact"]), e["sha256"], what=f"stage '{e['stage']}' artifact")
        produced[e["artifact"]] = e["sha256"]

    # 3) LINKS: every upstream reference must match the producing stage's
    #    record (when produced in-run) and the bytes on disk (always).
    for e in entries:
        for name, u in e.get("upstream", {}).items():
            verify_file(Path(u["path"]), u["sha256"], what=f"upstream '{name}' of '{e['stage']}'")
            if u["path"] in produced and produced[u["path"]] != u["sha256"]:
                raise ProvenanceError(
                    f"chain link broken: stage '{e['stage']}' consumed '{name}' "
                    f"({u['path']}) with sha256 {u['sha256']}, but the producing "
                    f"stage recorded {produced[u['path']]}."
                )

    return {
        "chain": str(chain_path),
        "stages": [e["stage"] for e in entries],
        "artifacts_verified": len(entries),
        "links_verified": sum(len(e.get("upstream", {})) for e in entries),
        "signature": sig_status,
        "ok": True,
    }


def main(argv: Optional[List[str]] = None) -> int:
    """CLI: audit a run directory's provenance chain. Exit 0 pass / 1 fail."""
    parser = argparse.ArgumentParser(
        description="Verify a pipeline run's provenance chain (hashes, links, signature)."
    )
    parser.add_argument("run_dir", type=Path, help="run directory containing provenance_chain.json")
    parser.add_argument(
        "--require-signature",
        action="store_true",
        help="fail unless the chain carries a signature valid under " + KEY_ENV_VAR,
    )
    args = parser.parse_args(argv)
    try:
        report = verify_chain(args.run_dir, require_signature=args.require_signature)
    except ProvenanceError as exc:
        print(f"PROVENANCE FAIL: {exc}")
        return 1
    print(
        "PROVENANCE OK: "
        f"{report['artifacts_verified']} artifacts, {report['links_verified']} links, "
        f"signature {report['signature']} ({' -> '.join(report['stages'])})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
