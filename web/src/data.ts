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
    !Array.isArray(study.training?.history) ||
    !/^[a-f0-9]{64}$/.test(study.document_text_sha256)
  )
    throw new Error("Unsupported or incomplete recorded study");
  const ids = new Set<string>();
  for (const query of study.queries) {
    if (
      !query.id ||
      ids.has(query.id) ||
      typeof query.text !== "string" ||
      !query.judgments ||
      !Object.values(query.judgments).some(
        (v) => Number.isFinite(v) && v > 0,
      ) ||
      Object.keys(query.judgments).some((id) => !study.documents[id])
    )
      throw new Error("Invalid query identity or judgments");
    ids.add(query.id);
    for (const method of methods) {
      const ranking = query.rankings?.[method];
      if (
        !Array.isArray(ranking) ||
        ranking.length !== 10 ||
        new Set(ranking).size !== ranking.length ||
        ranking.some((id) => !study.documents[id])
      )
        throw new Error("Incomplete query rankings");
      for (const metric of ["ndcg@10", "recall@10", "mrr@10"] as Metric[]) {
        const score = query.metrics?.[method]?.[metric];
        if (!Number.isFinite(score) || score < 0 || score > 1)
          throw new Error("Invalid metric value");
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
      (!term || q.text.toLocaleLowerCase().includes(term) || q.id === id) &&
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
