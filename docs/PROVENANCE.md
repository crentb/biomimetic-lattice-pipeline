# Run provenance: verify, chain, sign

Every pipeline run writes `provenance_chain.json` in its run directory: one entry per stage
(`morphometrics → cad_parameters → cad_geometry → mesh → fea_element_results → metrics`), each carrying the
artifact's SHA-256 and the hashes of the upstream artifacts it consumed.

Three guarantees, in V&V terms:

| Guarantee | Mechanism | When it fires |
|---|---|---|
| **Verify** — are these the bytes we measured? | Each stage re-hashes its upstream artifacts and compares against the hash recorded when they were produced | **At the stage seam, during the run** — a mid-run swap or corruption stops the pipeline where it happened |
| **Chain** — is the lineage unbroken? | Entries are hash-linked (each upstream hash must equal the producing stage's artifact hash); the audit re-hashes everything from disk | End of run, or any time later via the CLI |
| **Sign** — says who? | HMAC-SHA256 over the canonical entries, keyed by `BLP_PROVENANCE_KEY` (never stored). Any edit to the chain file itself — including rewriting recorded hashes — invalidates the MAC | Sealed at run end; checked first during audit |

## Auditing a run

```bash
python -m biomimetic_pipeline.orchestration.provenance runs/<run_name>
python -m biomimetic_pipeline.orchestration.provenance runs/<run_name> --require-signature
```

Exit `0` = every artifact re-hashed clean, every link intact, signature valid (when required).
Exit `1` prints exactly which artifact, seam, or signature failed, with both hashes.

## Signing

```bash
export BLP_PROVENANCE_KEY="<shared secret>"   # e.g. from your secret manager
```

Set before running the pipeline to seal chains, and before auditing to validate them. The recorded `key_id`
(a hash prefix of the key) distinguishes rotated keys. HMAC is a zero-dependency shared-secret scheme; the
upgrade path to asymmetric signatures (minisign / Sigstore, as already used for the CI SBOM) replaces only
the MAC computation and comparison.
