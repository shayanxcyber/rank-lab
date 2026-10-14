# Fictional helpdesk example

Twelve original fictional support documents, sixteen training queries, four validation queries
and six held-out test queries. Relevance 2 is the primary document; relevance 1 supplies an
additional known positive for webhook queries. This tiny fixture demonstrates the workflow;
it is not a representative product benchmark or evidence of general model quality.

From the repository root, prepare with `python -m rank_lab.cli prepare --config examples/helpdesk/experiment.toml`.
The configuration uses CPU by default. Add `--device cuda` during preparation to use a local GPU.
Run `train --workspace runs/helpdesk --stop-after-steps 2`, then `train --workspace runs/helpdesk --resume`,
`compare --workspace runs/helpdesk`, and `export --workspace runs/helpdesk --destination runs/helpdesk/export`.
Each command uses the same `python -m rank_lab.cli` prefix. Open the exported `rank-lab-study.json`
with the explorer's Open experiment control. Inputs stay local; no hosted training service is involved.

All paths in the TOML file resolve relative to that file. Use a new workspace for another
completed experiment; retain train/validation/test boundaries when creating your own labels.
