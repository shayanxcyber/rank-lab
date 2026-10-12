# From training loop to an inspectable ranking

Fine-tuning can move an average while hiding the cases that got worse. Rank Lab makes that
connection visible: a run has a data contract, a changing checkpoint, a selection decision and
individual query outcomes. The implementation combines an original Python training/evaluation
workflow with a static React explorer of the completed experiment.

## The result that shaped the interface

The fine-tuned encoder's reserved-test nDCG@10 is 0.675072, compared with 0.645082 for its
frozen version. That 3.00-percentage-point improvement comes from 53 gains, 27 regressions and
220 unchanged query scores. BM25's MRR@10 remains slightly higher than the selected checkpoint.

The opening field represents all 300 queries, ordered from regressions to gains. Each bar is a
query, rather than a decorative progress mark. The aggregate table keeps all three metrics and
both baselines visible. Changing the selected metric or baseline changes the field, counts,
comparison table and query inspector together.

Query 729 is a useful regression to inspect. Its judged relevant document was first for the
frozen encoder and second for BM25, but outside the fine-tuned top ten. Its nDCG therefore fell
from 1 to 0 despite the run's average improvement. Query 742 shows the other direction: nDCG
rose from 0.301030 to 1. Those cases are observed retrieval outcomes; the benchmark's scientific
claim text is not a factual conclusion made by the lab.

## Data identity before model work

The input downloader pins repository revisions and verifies each file's Git blob or LFS SHA-256
identity before loading it. Partial downloads use a temporary file and atomic replacement.
The loader validates IDs, judgment rows and positive judgments before splitting.

Preparation found two normalized-text collisions across the official train and test sets.
An explicit operation excludes those training IDs, preserving the raw input and every test
query. A seeded split groups the remaining training queries by casefolded, whitespace-normalized
text. This controls exact duplicates in this experiment; it does not prove semantic isolation
or that the pretrained model never encountered related material.

## A restart with state behind it

The training loop updates all 22,713,216 encoder parameters. It masks other known relevant
documents out of the in-batch negative columns and logs loss, learning rate and pre-clipping
gradient norm at every optimizer step.

Checkpoint state includes weights, optimizer, scheduler, batch cursor and CPU/CUDA RNG. Resume
rejects incompatible configuration, data, base revision or Torch version. The measured run
stopped after step two and resumed in a new process to step 78. A controlled CPU test also
compares the uninterrupted and interrupted/resumed weights and losses exactly.

That CPU test and the observed GPU restart establish different things. The project does not
claim a second uninterrupted GPU comparison. The GPU allocation figure is peak allocated
memory in the resumed invocation, rather than total device occupancy. Its timer includes
training and validation in that invocation, excluding preparation, the baseline and first two steps.

## Selection before the reserved comparison

Validation retrieves against the full corpus after each epoch. The predefined selection metric
is validation nDCG@10; exact ties retain the earlier epoch. Epoch three won with 0.700883.
Selected model files and tensor fingerprints are checked before the final test comparison.
The comparison refuses to overwrite an existing result, keeping the completed run intact.

Metrics use graded nDCG, positive-document Recall and reciprocal rank at ten, averaged equally
over queries. Stable document-ID order breaks exact ties. Unjudged results get zero gain but
are displayed as unjudged, rather than being relabeled as confirmed irrelevant.

## Presentation as another verifiable artifact

The exporter reads recorded results offline and recomputes the scores from rankings and
judgments. It rejects missing reserved queries and identity or metric drift. The browser also
checks that the displayed aggregate means agree with all per-query values.

The initial study file contains query outcomes and document titles. Abstracts load separately
when the inspector opens, and their bytes must match the SHA-256 recorded in the study file.
This reduces the initial data payload from about 5.38 MB to 0.89 MB without removing detailed
inspection. Neither browser filtering nor export changes the training run or test results.

The interface uses Base UI for focus management and control behavior, an adapted Select
composition with full-width options, TanStack for sorting/filtering/pagination and Observable
Plot for the genuine training traces. Mobile rows put the comparison values beneath the query
instead of requiring horizontal scrolling. Reduced motion disables decorative transitions,
and the query field supports arrow-key navigation with one entry in the tab sequence.

## Verified scope and remaining research

Twenty-two backend tests and six explorer data tests pass. They cover artifact identity, split
isolation, graded metrics, known-positive masking, checkpoint behavior, offline export, score
consistency, query search and rejection of malformed result records. The local app build passes.
Selected desktop/mobile journeys have been inspected; this is not an exhaustive browser or
assistive-technology audit.

The first study remains one seed, split and benchmark. BM25 and the dense methods have different
effective text lengths. Latency under load, deployment behavior, broader-domain results and
repeatability across multiple training seeds remain unmeasured. The public explorer displays
the completed study; it does not serve the trained model or launch a GPU job.
