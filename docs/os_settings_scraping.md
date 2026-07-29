# OS settings documentation snapshots

`scripts/scrape_os_settings.py` creates the normalized ORM corpus consumed by
`train_on_policy_ollama.py`.

```bash
python scripts/scrape_os_settings.py \
  --sources config/document_sources.json \
  --all-enabled-os \
  --snapshot-dir artifacts/document-snapshots \
  --output artifacts/normalized-documents.json \
  --max-evidence-tier 2 \
  --require-all-os \
  --fail-on-stale-source
```

The output is one JSON array. Each document includes its source key, OS family,
canonical locator, content SHA-256, source revision, license identifier,
sections and evidence-linked `setting_variant` claims.

## Boundaries

- Network access is restricted to each source's `allowed_hosts`.
- Non-loopback plain HTTP is rejected.
- `robots.txt`, redirect host validation, request delay, page depth, response
  size and per-source page limits are enforced.
- Raw bytes are written content-addressably under
  `artifacts/document-snapshots/raw/`.
- `manifest.json` records immutable source revisions and snapshot paths.
- With `--fail-on-stale-source`, a changed revision stops the run before the
  normalized output or manifest is replaced.
- Missing OS settings or required document classes fail closed by default.
- Documentation is normalized as evidence only; the script never executes
  commands found in source text.

For a controlled migration to a new upstream revision, inspect the new raw
snapshots and run once without `--fail-on-stale-source`. Commit or archive the
resulting manifest before restoring the stale-source gate.
