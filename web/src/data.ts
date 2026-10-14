import {
  outcome,
  delta,
  type Study,
  type Query,
  type Metric,
  type Method,
  methods,
} from "./types";

export function parseStudy(value: unknown): Study {
  if (!value || typeof value !== "object")
    throw new Error("Missing study data");
  const study = value as Study;
  if (
    study.schema_version !== 1 ||
    study.kind !== "recorded-study" ||
    !Array.isArray(study.queries) ||
    !study.queries.length ||
    study.queries.length !== study.summary?.test_queries ||
    !study.documents ||
    typeof study.summary.dataset !== "string" ||
    !study.summary.dataset.trim() ||
    !Number.isInteger(study.summary.documents) ||
    study.summary.documents < 1 ||
    Object.keys(study.documents).length > study.summary.documents ||
    !Array.isArray(study.training?.history) ||
    !/^[a-f0-9]{64}$/.test(study.document_text_sha256)
  )
    throw new Error("Unsupported or incomplete recorded study");
  const ids = new Set<string>();
  for (const query of study.queries) {
    if (
      typeof query.id !== "string" ||
      !query.id ||
      ids.has(query.id) ||
      typeof query.text !== "string" ||
      !query.judgments ||
      !Object.values(query.judgments).some(
        (v) => Number.isInteger(v) && v > 0 && v <= 30,
      ) ||
      Object.entries(query.judgments).some(
        ([id, relevance]) =>
          !Object.hasOwn(study.documents, id) ||
          !Number.isInteger(relevance) ||
          relevance < 0 ||
          relevance > 30,
      )
    )
      throw new Error("Invalid query identity or judgments");
    ids.add(query.id);
    for (const method of methods) {
      const ranking = query.rankings?.[method];
      if (
        !Array.isArray(ranking) ||
        ranking.length !== Math.min(10, study.summary.documents) ||
        new Set(ranking).size !== ranking.length ||
        ranking.some(
          (id) => typeof id !== "string" || !Object.hasOwn(study.documents, id),
        )
      )
        throw new Error("Incomplete query rankings");
      for (const metric of ["ndcg@10", "recall@10", "mrr@10"] as Metric[]) {
        const score = query.metrics?.[method]?.[metric];
        if (!Number.isFinite(score) || score < 0 || score > 1)
          throw new Error("Invalid metric value");
        if (
          Math.abs(score - rankingMetrics(ranking, query.judgments)[metric]) >
          1e-12
        )
          throw new Error("Metric differs from rankings and judgments");
      }
    }
  }
  for (const method of methods)
    for (const metric of ["ndcg@10", "recall@10", "mrr@10"] as Metric[]) {
      const mean =
        study.queries.reduce((sum, q) => sum + q.metrics[method][metric], 0) /
        study.queries.length;
      if (
        !Number.isFinite(study.summary.metrics?.[method]?.[metric]) ||
        Math.abs(mean - study.summary.metrics[method][metric]) > 1e-12
      )
        throw new Error("Summary differs from the complete query results");
    }
  if (
    !study.summary.config ||
    !Number.isInteger(study.summary.config.epochs) ||
    study.summary.config.epochs < 1 ||
    !Number.isInteger(study.summary.optimizer_steps) ||
    study.summary.optimizer_steps < 1 ||
    study.training.history.length !== study.summary.optimizer_steps ||
    !Array.isArray(study.validation) ||
    study.validation.length !== study.summary.config.epochs + 1 ||
    study.validation.some(
      (row, index) =>
        row.epoch !== index ||
        !Number.isFinite(row.ndcg) ||
        row.ndcg < 0 ||
        row.ndcg > 1,
    ) ||
    study.training.history.some(
      (row, index) =>
        row.step !== index + 1 ||
        !Number.isFinite(row.loss) ||
        !Number.isFinite(row.gradient_norm) ||
        !Number.isFinite(row.learning_rate),
    ) ||
    Object.values(study.documents).some(
      (row) => !row || typeof row.title !== "string",
    )
  )
    throw new Error("Incomplete training record or document metadata");
  const summary = study.summary;
  if (
    [
      summary.train_queries,
      summary.validation_queries,
      summary.trainable_parameters,
    ].some((value) => !Number.isInteger(value) || value < 1) ||
    !Number.isInteger(summary.selected_epoch) ||
    summary.selected_epoch < 1 ||
    summary.selected_epoch > summary.config.epochs ||
    !Number.isFinite(study.training.elapsed_seconds_resumed_invocation) ||
    study.training.elapsed_seconds_resumed_invocation < 0 ||
    (study.training.resumed_after_step !== null &&
      (!Number.isInteger(study.training.resumed_after_step) ||
        study.training.resumed_after_step < 1 ||
        study.training.resumed_after_step >= summary.optimizer_steps)) ||
    !Number.isInteger(summary.config.batch_size) ||
    summary.config.batch_size < 2 ||
    !Number.isInteger(summary.config.max_length) ||
    summary.config.max_length < 1 ||
    summary.config.max_length > 512 ||
    !Number.isInteger(summary.config.seed) ||
    !Number.isFinite(summary.config.learning_rate) ||
    summary.config.learning_rate <= 0 ||
    !Number.isFinite(summary.config.temperature) ||
    summary.config.temperature <= 0 ||
    typeof summary.gpu !== "string" ||
    typeof summary.base_model !== "string" ||
    !Array.isArray(study.exclusions) ||
    [
      study.comparison_sha256,
      study.dataset_fingerprint,
      study.training.initial_weights,
      study.training.final_weights,
      summary.weights_fingerprint,
    ].some((value) => !/^[a-f0-9]{64}$/.test(value))
  )
    throw new Error("Invalid experiment configuration or identity record");
  const bestValidation = Math.max(
    ...study.validation.slice(1).map((row) => row.ndcg),
  );
  if (
    study.validation.find((row) => row.epoch > 0 && row.ndcg === bestValidation)
      ?.epoch !== summary.selected_epoch
  )
    throw new Error("Selected checkpoint differs from validation results");
  return study;
}
export function selectQueries(
  queries: Query[],
  search: string,
  filter: string,
  metric: Metric,
  baseline: Method,
) {
  const term = search.trim().toLocaleLowerCase();
  const id = /^q?\s*\d+$/.test(term)
    ? String(Number(term.replace(/^q\s*/, "")))
    : null;
  return queries.filter(
    (q) =>
      (!term ||
        q.text.toLocaleLowerCase().includes(term) ||
        q.id === id ||
        q.id.toLocaleLowerCase().includes(term)) &&
      (filter === "all" || outcome(delta(q, metric, baseline)) === filter),
  );
}
export function outcomes(queries: Query[], metric: Metric, baseline: Method) {
  return queries.reduce(
    (a, q) => {
      a[outcome(delta(q, metric, baseline))]++;
      return a;
    },
    { gain: 0, same: 0, loss: 0 },
  );
}

export function rankingMetrics(
  ranking: string[],
  judgments: Record<string, number>,
) {
  const positive = Object.values(judgments)
    .filter((value) => value > 0)
    .sort((a, b) => b - a);
  const ideal = positive
    .slice(0, 10)
    .reduce(
      (sum, value, index) => sum + (2 ** value - 1) / Math.log2(index + 2),
      0,
    );
  const gain = ranking.reduce(
    (sum, id, index) =>
      sum + (2 ** (judgments[id] ?? 0) - 1) / Math.log2(index + 2),
    0,
  );
  const first = ranking.findIndex((id) => (judgments[id] ?? 0) > 0);
  return {
    "ndcg@10": gain / ideal,
    "recall@10":
      ranking.filter((id) => (judgments[id] ?? 0) > 0).length / positive.length,
    "mrr@10": first < 0 ? 0 : 1 / (first + 1),
  };
}

export async function verifyDocumentTexts(
  study: Study,
  bytes: ArrayBuffer,
): Promise<Record<string, string>> {
  const digest = Array.from(
    new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)),
  )
    .map((value) => value.toString(16).padStart(2, "0"))
    .join("");
  if (digest !== study.document_text_sha256)
    throw new Error("Document file identity differs from this study");
  const value = JSON.parse(new TextDecoder().decode(bytes));
  if (
    !value ||
    typeof value !== "object" ||
    Array.isArray(value) ||
    Object.keys(value).length !== Object.keys(study.documents).length ||
    Object.keys(study.documents).some((id) => typeof value[id] !== "string")
  )
    throw new Error("Document texts are incomplete");
  return value;
}

export async function parseBundle(text: string): Promise<Study> {
  const study = parseStudy(JSON.parse(text));
  if (typeof study.document_text_json !== "string")
    throw new Error(
      "Choose rank-lab-study.json from the export folder; study.json needs a separate document file.",
    );
  await verifyDocumentTexts(
    study,
    new TextEncoder().encode(study.document_text_json).buffer,
  );
  return study;
}

export function queriesCsv(queries: Query[]): string {
  const metrics: Metric[] = ["ndcg@10", "recall@10", "mrr@10"];
  const rows = [
    [
      "query_id",
      "query",
      ...methods.flatMap((method) =>
        metrics.map((metric) => `${method}/${metric}`),
      ),
    ],
    ...queries.map((query) => [
      query.id,
      query.text,
      ...methods.flatMap((method) =>
        metrics.map((metric) => String(query.metrics[method][metric])),
      ),
    ]),
  ];
  return (
    rows
      .map((row) =>
        row.map((cell) => `"${cell.replaceAll('"', '""')}"`).join(","),
      )
      .join("\r\n") + "\r\n"
  );
}
