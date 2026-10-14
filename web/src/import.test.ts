import { describe, expect, it } from "vitest";
import original from "../public/data/study.json";
import {
  parseBundle,
  parseStudy,
  queriesCsv,
  rankingMetrics,
  selectQueries,
} from "./data";
import type { Study } from "./types";

async function fixture(): Promise<Study> {
  const study = parseStudy(structuredClone(original));
  const metrics = { "ndcg@10": 1, "recall@10": 1, "mrr@10": 1 };
  study.summary = {
    ...study.summary,
    dataset: "A custom support experiment",
    documents: 2,
    train_queries: 2,
    validation_queries: 1,
    test_queries: 1,
    optimizer_steps: 1,
    selected_epoch: 1,
    config: { ...study.summary.config, epochs: 1 },
    metrics: {
      lexical: { ...metrics },
      pretrained: { ...metrics },
      fine_tuned: { ...metrics },
    },
  };
  study.queries = [
    {
      id: "ticket-1",
      text: 'A "quoted"\nquery',
      judgments: { a: 2, b: 0 },
      rankings: {
        lexical: ["a", "b"],
        pretrained: ["a", "b"],
        fine_tuned: ["a", "b"],
      },
      metrics: {
        lexical: { ...metrics },
        pretrained: { ...metrics },
        fine_tuned: { ...metrics },
      },
    },
  ];
  study.documents = { a: { title: "Retries" }, b: { title: "Other material" } };
  study.training.history = [
    { step: 1, epoch: 1, loss: 0.5, gradient_norm: 0.2, learning_rate: 0.001 },
  ];
  study.training.resumed_after_step = null;
  study.validation = [
    { epoch: 0, ndcg: 0.4 },
    { epoch: 1, ndcg: 0.8 },
  ];
  study.document_text_json =
    '{\r\n  "a": "Retry ↻ safely",\r\n  "b": "Other text"\r\n}\r\n';
  study.document_text_sha256 = Array.from(
    new Uint8Array(
      await crypto.subtle.digest(
        "SHA-256",
        new TextEncoder().encode(study.document_text_json),
      ),
    ),
  )
    .map((value) => value.toString(16).padStart(2, "0"))
    .join("");
  return study;
}

describe("portable experiments", () => {
  it("loads nonnumeric IDs, a short corpus and exact UTF-8/CRLF document bytes", async () => {
    const source = await fixture();
    const study = await parseBundle(JSON.stringify(source));
    expect(study.queries).toHaveLength(1);
    expect(study.summary.documents).toBe(2);
    expect(study.document_text_json).toBe(source.document_text_json);
    expect(
      selectQueries(study.queries, "TICKET-1", "all", "ndcg@10", "pretrained"),
    ).toHaveLength(1);
  });
  it("rejects an altered document sidecar", async () => {
    const study = await fixture();
    study.document_text_json = study.document_text_json!.replace(
      "safely",
      "differently",
    );
    await expect(parseBundle(JSON.stringify(study))).rejects.toThrow(
      /identity/,
    );
  });
  it("explains when the separate-file export was chosen instead of the bundle", async () => {
    const study = await fixture();
    delete study.document_text_json;
    await expect(parseBundle(JSON.stringify(study))).rejects.toThrow(
      /rank-lab-study.json/,
    );
  });
  it("rejects per-query scores that disagree with their own rankings", async () => {
    const study = await fixture();
    study.queries[0].rankings.fine_tuned.reverse();
    expect(() => parseStudy(study)).toThrow(/Metric differs/);
  });
  it("keeps graded gains and reciprocal rank distinct", () => {
    const result = rankingMetrics(["b", "a"], { a: 2, b: 1 });
    expect(result["ndcg@10"]).toBeCloseTo(0.79670758, 7);
    expect(result["recall@10"]).toBe(1);
    expect(result["mrr@10"]).toBe(1);
  });
  it("rejects incomplete training records", async () => {
    const study = await fixture();
    study.training.history = [];
    expect(() => parseStudy(study)).toThrow(/training record/);
  });
  it("downloads the loaded experiment's actual rows with CSV escaping", async () => {
    const study = await fixture();
    const csv = queriesCsv(study.queries);
    expect(csv).toContain('"ticket-1","A ""quoted""\nquery"');
    expect(csv).toContain('"fine_tuned/ndcg@10"');
    expect(csv).not.toContain("742");
  });
});
