# First retrieval study

The question is whether a small domain fine-tuning run improves a useful pretrained encoder,
and whether that improvement holds up against a lexical baseline. This study keeps the
training changes, checkpoint selection and reserved-test comparison inspectable.

## Data before gradients

SciFact supplies 5,183 documents, 1,109 queries, 809 official training queries and 300 test
queries. Documents use title plus abstract. The files are pinned to exact repository
revisions, with verified LFS SHA-256 or ordinary Git blob identities.

The preparation check found two official training queries with the same casefolded,
whitespace-normalized text as test queries: training 1291/test 1292 and training 871/test 870.
The source files remain intact. Those two training IDs are excluded from this experiment;
all test IDs and judgments remain. The remaining 807 queries are divided by normalized-text
group into 727 training and 80 validation queries with seed 1729. The loader rejects unknown
IDs, duplicate judgment rows, empty positive judgments and cross-split query-group leakage.

That grouping catches exact normalized duplicates, not semantic paraphrases. Corpus documents
are shared across retrieval splits. The experiment separates its own fine-tuning query inputs;
it does not establish the pretrained model's full historical exposure.

## Training contract

The base is `sentence-transformers/all-MiniLM-L6-v2`, revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. The encoder uses the model's mean-pooling recipe,
L2 normalization and a 256-wordpiece limit. All 22,713,216 parameters remain trainable.
This is fine-tuning a pretrained model, rather than training the architecture from scratch.

One configuration was used: three epochs, batch size 32, AdamW learning rate 0.00002,
temperature 0.05, seed 1729, gradient-norm clipping at 1.0, linear warmup/decay and CUDA
bfloat16 autocast. The contrastive objective compares each query with documents in its batch.
Other documents already judged relevant to that query are masked from the negative set.
Unjudged documents can still be relevant; the mask covers only the supplied judgments.

Each update writes an atomic checkpoint containing model, optimizer, scheduler, batch cursor
and CPU/CUDA RNG state. Resume checks configuration, data, model revision and runtime contracts.
The first invocation intentionally ended after step two. A new process resumed that checkpoint
and completed step 78. The final weight fingerprint differs from the frozen base.
Controlled CPU tests confirm identical final weights and loss history after interruption;
an uninterrupted second GPU run has not been compared.

## Hardware and software

The local GPU study and CPU test suite used an NVIDIA GeForce RTX 5080 with **16 GB GDDR7
VRAM**, an AMD Ryzen 7 9800X3D (8 cores / 16 threads), 32 GB RAM (two 16 GB modules) and
Windows 11 Pro (10.0.26200). The training environment was Python 3.11.15 with PyTorch
2.7.1+cu128 and CUDA 12.8. The installed GPU reported 16,303 MiB through nvidia-smi;
its rated memory capacity is listed in [NVIDIA's specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5080/).
Total card capacity and peak allocated tensor memory describe different quantities.

## Choose on validation

All epochs use the full 5,183-document corpus for validation retrieval. The frozen encoder's
validation nDCG@10 was 0.65995. Epochs one, two and three yielded 0.67146, 0.70022 and 0.70088.
Epoch three won the predefined validation metric; an earlier epoch would win an exact tie.
Checkpoint bytes and tensor fingerprints are recorded at selection and checked again before
the final comparison. The test comparison ran only after training and selection completed.

## Reserved-test results

| Method | nDCG@10 | Recall@10 | MRR@10 |
| --- | ---: | ---: | ---: |
| BM25, k1=1.5/b=0.75 | 0.664617 | 0.781611 | 0.633999 |
| Frozen MiniLM | 0.645082 | 0.783333 | 0.604725 |
| Selected fine-tuned MiniLM | 0.675072 | 0.834778 | 0.628598 |

All 300 judged queries are included. nDCG uses graded gains, Recall counts all judged positive
documents and reciprocal rank uses the first judged positive, each at cutoff ten. Metrics
are averaged equally across query IDs; document-ID order breaks exact score ties.
Unjudged documents receive zero evaluation gain. BM25 sees the full text while the encoder
applies its wordpiece limit, so the input-content source is shared but effective context differs.

The fine-tuned encoder improved all three metrics over its frozen version. Against BM25, it
improved nDCG and Recall but slightly reduced reciprocal rank. A broad best-model claim would
miss that tradeoff. This is one seed, one split and one benchmark; repeatability across runs,
other domains, latency under load and deployment behavior remain unmeasured.

## Performed checks and remaining work

Twenty-two backend tests pass for artifact identity/atomic writes, split isolation, graded metrics,
duplicate-ranking rejection, known-positive masking and checkpoint-resume behavior.
Ruff lint and formatting pass. A real CUDA matrix operation, public-asset preparation,
GPU gradients, separate-process resume, all three validation passes and final full-corpus
comparison completed. Peak allocated CUDA memory in the resumed training invocation was
2,943,213,568 bytes. Its measured training/validation duration was 54.85 seconds;
that excludes the earlier preparation, baseline and two-step invocation.

The recorded comparison explorer is implemented with a locked web
toolchain and six passing data-contract/search tests. It exposes every query's three rankings,
judgments, on-demand hash-checked abstracts and training/validation traces. The offline exporter
recomputes scores from the saved rankings and rejects identity or query-coverage drift.
The release includes the recorded study and CPU regression checks; repeating GPU training is a separate operation.

Sources: [SciFact corpus and queries](https://huggingface.co/datasets/BeIR/scifact),
[official judgments](https://huggingface.co/datasets/BeIR/scifact-qrels),
[MiniLM recipe](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
[contrastive-loss overview](https://sbert.net/docs/sentence_transformer/loss_overview.html).
