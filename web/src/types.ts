export type Method = "lexical" | "pretrained" | "fine_tuned";
export type Metric = "ndcg@10" | "recall@10" | "mrr@10";
export type Metrics = Record<Metric, number>;
export type Query = {
  id: string;
  text: string;
  judgments: Record<string, number>;
  rankings: Record<Method, string[]>;
  metrics: Record<Method, Metrics>;
};
export type Study = {
  schema_version: number;
  kind: string;
  comparison_sha256: string;
  dataset_fingerprint: string;
  summary: {
    documents: number;
    train_queries: number;
    validation_queries: number;
    test_queries: number;
    selected_epoch: number;
    optimizer_steps: number;
    trainable_parameters: number;
    metrics: Record<Method, Metrics>;
    base_model: string;
    base_revision: string;
    gpu: string;
    peak_cuda_allocated_bytes: number;
    config: {
      batch_size: number;
      epochs: number;
      learning_rate: number;
      max_length: number;
      seed: number;
      temperature: number;
    };
    limits: string[];
  };
  queries: Query[];
  documents: Record<string, { title: string }>;
  document_text_sha256: string;
  training: {
    history: {
      epoch: number;
      step: number;
      loss: number;
      gradient_norm: number;
      learning_rate: number;
    }[];
    initial_weights: string;
    final_weights: string;
    resumed_after_step: number;
    elapsed_seconds_resumed_invocation: number;
  };
  validation: { epoch: number; ndcg: number }[];
  exclusions: { train_id: string; test_ids: string[] }[];
};
export const methods: Method[] = ["lexical", "pretrained", "fine_tuned"];
export const methodName: Record<Method, string> = {
  lexical: "BM25",
  pretrained: "Pretrained",
  fine_tuned: "Fine-tuned",
};
export const metricName: Record<Metric, string> = {
  "ndcg@10": "nDCG@10",
  "recall@10": "Recall@10",
  "mrr@10": "MRR@10",
};
export const metricDescription: Record<Metric, string> = {
  "ndcg@10":
    "Rewards relevant documents near the top. Normalized against the ideal ordering.",
  "recall@10":
    "Fraction of judged relevant documents found in the first ten results.",
  "mrr@10":
    "Reciprocal position of the first judged relevant result, or zero if absent from the top ten.",
};
export const delta = (q: Query, metric: Metric, baseline: Method) =>
  q.metrics.fine_tuned[metric] - q.metrics[baseline][metric];
export const outcome = (value: number) =>
  value > 1e-12 ? "gain" : value < -1e-12 ? "loss" : "same";
export const score = (value: number) => value.toFixed(4);
export const points = (value: number) =>
  `${value > 0 ? "+" : ""}${(value * 100).toFixed(2)}`;
