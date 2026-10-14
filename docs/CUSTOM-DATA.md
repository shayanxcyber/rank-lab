# Run an experiment on your own data

Rank Lab v0.2 accepts a document corpus, queries and relevance judgments. It fine-tunes the
pinned MiniLM encoder, compares it with BM25 and pretrained MiniLM, then exports a result you
can open in the explorer. Training runs locally; the browser inspects completed results.

Start with `examples/helpdesk/experiment.toml`. Its twelve documents and twenty-six queries are
original fictional support material. The fixture demonstrates the workflow, not general model
quality. There are sixteen training queries, four validation queries and six reserved test queries.

## Install

Use Python 3.11 or newer. From the repository root, create an environment and install the CPU
version of Torch, then the project:

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/Scripts/python.exe -e ".[dev]"
```

On Linux/macOS, replace `.venv/Scripts/python.exe` with `.venv/bin/python`. A conventional venv
and pip work too. For a compatible local NVIDIA GPU, use the `cu128` wheel index instead of
`cpu`, and select `--device cuda` during preparation. No hosted API or paid compute is required.

The supported base model is the revision-pinned `sentence-transformers/all-MiniLM-L6-v2`.
Preparing an experiment downloads and verifies its public files. This release does not accept
arbitrary model architectures. Model files, checkpoints and run data stay outside Git.

## Run the complete example

```powershell
.venv/Scripts/python.exe -m rank_lab.cli prepare --config examples/helpdesk/experiment.toml
.venv/Scripts/python.exe -m rank_lab.cli train --workspace runs/helpdesk --stop-after-steps 2
.venv/Scripts/python.exe -m rank_lab.cli train --workspace runs/helpdesk --resume
.venv/Scripts/python.exe -m rank_lab.cli compare --workspace runs/helpdesk
.venv/Scripts/python.exe -m rank_lab.cli export --workspace runs/helpdesk --resumed-after-step 2
```

The interruption is optional. For an uninterrupted run, omit `--stop-after-steps`, skip the resume
command, and omit the restart annotation when exporting. Only annotate a restart that actually
happened. Configuration overrides such as `--device cuda`, `--epochs 4`, `--batch-size 8`,
`--learning-rate 0.00002`, `--seed 42` and `--max-length 256` are supported. Prefer setting them
during preparation; later commands can recover the saved configuration using only `--workspace`.
Resume rejects changes to the training/data/split contract.

Export defaults to `WORKSPACE/export`. Open the [live explorer](https://shayanxcyber.github.io/rank-lab/),
choose **Open experiment**, then select `runs/helpdesk/export/rank-lab-study.json`. The bundle
contains the result and exact document text. Files are read in your browser, without uploading
them. Alternatively run `npm ci` and `npm run dev` in `web/` and use the same control locally.

The original SciFact demo remains available through **SciFact demo**. Importing another result
changes its counts, labels, metrics, plots, rankings and downloads. It does not edit the bundled
demo or run another training job. Browser imports currently support bundles up to 50 MB.

## Prepare your own files

Create three UTF-8 files. IDs are nonempty strings without surrounding whitespace; they can be
alphanumeric. Document and query IDs must each be unique within their respective files.

**documents.jsonl** — one JSON object per line. `title` is optional; `text` must be nonempty.

```json
{"id":"retry-guide","title":"Delivery retries","text":"Retry failed deliveries using a stable operation ID."}
{"id":"restore-guide","title":"Restoring data","text":"Verify a completed snapshot before restoring it."}
```

**queries.jsonl** — one query per line, with an explicit `train`, `validation` or `test` split.

```json
{"id":"train-1","text":"How should failed deliveries be retried?","split":"train"}
{"id":"validation-1","text":"Check a snapshot before recovery","split":"validation"}
{"id":"test-1","text":"Prevent duplicate actions during redelivery","split":"test"}
```

These fragments explain the format, rather than constituting a sufficient training dataset.
Use the complete example or supply enough independent training queries and positive pairs.

**judgments.tsv** — tab-separated columns, in this order:

```text
query_id	document_id	relevance
train-1	retry-guide	2
validation-1	restore-guide	2
test-1	retry-guide	2
test-1	restore-guide	0
```

Relevance is an integer from 0 through 30. Zero means judged non-relevant; positive values are
graded relevance. Every query needs at least one positive judgment. Omitted documents are
unjudged, not confirmed irrelevant, and receive zero gain in the reported metrics. Duplicate
query/document judgments and references to unknown IDs are rejected.

Documents alone are insufficient for this supervised workflow. Write representative queries and
label their relevant documents. Keep evaluation queries separate from the examples used to train.

## Configure the experiment

```toml
[experiment]
name = "Support manual experiment"
workspace = "runs/manuals-01"

[data]
documents = "documents.jsonl"
queries = "queries.jsonl"
judgments = "judgments.tsv"
validation_fraction = 0.2

[training]
epochs = 3
batch_size = 8
learning_rate = 0.00002
temperature = 0.05
max_length = 256
seed = 42
device = "cpu"
mixed_precision = false
```

Every path in the TOML file resolves relative to that file. `--workspace` overrides its workspace
for an independent run. Keep experiment names under 161 characters. Unknown settings are errors.
The sequence limit is 1–512 wordpieces for the supported model; shorter limits use less memory.

If any queries explicitly use `validation`, their declared split is retained. Otherwise, Rank Lab
derives validation from the training queries using `validation_fraction` and the seed. Exact
normalized text groups stay together. Explicit train, validation and test splits must not share
normalized query text. All supported paths reserve the test set for the final comparison.
Training needs at least two positive pairs after splitting, and batch size must be at least two.

Preparation snapshots the validated data and records hashes, so subsequent training does not
need the original input files. The saved snapshot is checked before use. Choose a new workspace
for changed data or a new completed experiment; do not edit a prepared run to replace its inputs.

## Understand the output

| Artifact | Purpose |
| --- | --- |
| `preparation.json`, `split.json` | Input identity, counts and exact query membership |
| `run-config.json` | Resolved local settings, including input/workspace locations |
| `training/training-state.pt` | Model, optimizer, scheduler, cursor and RNG state for recovery |
| `training/selection.json` | Validation-selected epoch and its tensor fingerprint |
| `comparison.json` | Final held-out rankings and per-query/aggregate metrics |
| `export/rank-lab-study.json` | Portable browser bundle, including document text |
| `export/study.json`, `document-text.json`, `queries.csv` | Separate-file inspection/export assets |

The selected checkpoint can precede the final training epoch. The explorer distinguishes its
selected fingerprint from the final training-state fingerprint. nDCG rewards graded relevance,
Recall measures relevant-document coverage, and MRR measures the first relevant result, all at ten.
Smaller corpora retain their available results rather than padding rankings to ten documents.

Exports recompute scores from rankings/judgments. The browser independently checks scores,
aggregate means, training metadata and document SHA-256. A bundle preserves the document file's
exact bytes, including Windows newlines. These are consistency checks, not a claim that labels
are correct or that an experiment proves broad retrieval quality.

## Troubleshooting

- **Custom data is not prepared:** run `prepare --config ...` first, then use its workspace.
- **No positive judgment:** provide at least one positive document for each query, including
  validation and test. A query with only zero labels cannot be scored by these metrics.
- **Text groups overlap:** move normalized duplicate queries into one split. Do not merely rename IDs.
- **Checkpoint contract differs:** repeat the original settings or use a new workspace. If preparation
  used CLI overrides, resume through `--workspace` to recover them instead of rereading an older config.
- **CUDA unavailable/out of memory:** prepare a new run with CPU or a smaller batch/sequence limit.
  Do not change device halfway through a checkpointed run; that changes its recovery contract.
- **Wrong browser file:** choose `rank-lab-study.json`, not the separate `study.json` metadata file.
- **Document identity differs:** re-export from an intact workspace. Do not remove the hash check.
- **Destination contains another experiment:** choose a different export directory. The original
  committed demo is protected from accidental replacement by a different comparison.

The fixture's scores and tiny size are useful for checking installation and the complete workflow.
For a real study, use representative data and labels and inspect regressions as well as averages.
