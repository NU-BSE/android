# OS settings dataset quality

## Training-claim quality gate

Evidence collection and training-claim extraction are separate. API references,
Android source code, resource XML, JSON and fenced code blocks remain in the
normalized evidence corpus, but they cannot emit `setting_variant` claims.
Claims require either an explicit Android breadcrumb using `Settings > ...`
or an ordered user-guide procedure that explicitly starts by opening the
Settings app. Each visible label is validated before output. Action language
such as `tap`, `disable`, `7 times`, reboot instructions, placeholders and
XML/controller syntax are not stored as UI labels.

The final corpus is globally deduplicated by OS family and normalized UI path.
Section `claim_indexes` are remapped after rejection and deduplication. The
manifest contains a `claim_quality` object with input, emitted, invalid,
duplicate and structural-error counts.

For the strict production scrape, add the rejection gate:

```bash
python scripts/scrape_os_settings.py \
  --sources config/document_sources.json \
  --all-enabled-os \
  --snapshot-dir artifacts/document-snapshots \
  --output artifacts/normalized-documents.json \
  --max-evidence-tier 2 \
  --require-all-os \
  --require-claim-os grapheneos \
  --require-claim-os calyxos \
  --require-claim-os eos \
  --fail-on-stale-source \
  --fail-on-rejected-claim
```

Audit the generated artifact before training:

```bash
python scripts/audit_normalized_documents.py artifacts/normalized-documents.json
```

The audit exits non-zero for malformed claims, duplicate OS/path pairs, an empty
claim corpus, mismatched `display_name`, technical/XML labels, non-Android paths
or broken section indexes. A legacy artifact must be regenerated rather than
edited in place.


## Evidence coverage and claim coverage

`--require-all-os` requires normalized evidence for every selected OS family.
It does not force API references or source code to manufacture UI claims.
Use repeatable `--require-claim-os <family>` only for families that have an
eligible official UI guide. The manifest records `claim_coverage`, including
per-OS document and claim counts and `zero_claim_os`.

## Instruction, placeholder and duplicate-document rejection

The claim gate rejects prose actions, user-specific placeholders and UI-chrome
instructions instead of guessing that they are visible labels. Examples include
`refresh using top-right arrow`, `your account`, `the SIM card you are
troubleshooting`, `3 dots` and generic terminals such as `profile`.

Evidence documents are deduplicated by normalized OS family and canonical URL
before claim validation. The deterministic winner prefers the copy containing
more valid claims, then the lower confidence tier and stable source/document
identifiers. The audit also requires section IDs to be globally unique and
reports `duplicate_document_count` separately from duplicate claims.
