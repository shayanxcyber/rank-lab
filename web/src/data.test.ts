import { describe, it, expect } from "vitest";
import raw from "../public/data/study.json";
import { parseStudy, selectQueries, outcomes } from "./data";
import { outcome } from "./types";

describe("recorded study contract", () => {
  it("covers the complete reserved set and reproduces the exported aggregate means", () => {
    const study = parseStudy(raw);
    expect(study.queries).toHaveLength(300);
    expect(outcomes(study.queries, "ndcg@10", "pretrained")).toEqual({
      gain: 53,
      same: 220,
      loss: 27,
    });
  });
  it("rejects a duplicated query rather than counting it twice", () => {
    const broken = structuredClone(raw);
    broken.queries[1] = broken.queries[0];
    expect(() => parseStudy(broken)).toThrow(/identity/);
  });
  it("rejects summary numbers detached from the detailed results", () => {
    const broken = structuredClone(raw);
    broken.summary.metrics.fine_tuned["ndcg@10"] = 0.99;
    expect(() => parseStudy(broken)).toThrow(/Summary/);
  });
  it("rejects missing documents, nonfinite metrics and partial rankings", () => {
    const broken = structuredClone(raw);
    broken.queries[0].rankings.fine_tuned[0] = "missing";
    expect(() => parseStudy(broken)).toThrow(/rankings/);
    broken.queries[0].rankings.fine_tuned = [
      ...raw.queries[0].rankings.fine_tuned,
    ];
    broken.queries[0].metrics.fine_tuned["ndcg@10"] = NaN;
    expect(() => parseStudy(broken)).toThrow(/metric/);
    broken.queries[0].metrics.fine_tuned["ndcg@10"] =
      raw.queries[0].metrics.fine_tuned["ndcg@10"];
    broken.queries[0].rankings.fine_tuned.pop();
    expect(() => parseStudy(broken)).toThrow(/rankings/);
  });
  it("searches IDs and text without modifying the underlying study", () => {
    const before = JSON.stringify(raw);
    const study = parseStudy(raw);
    expect(
      selectQueries(
        study.queries,
        " Q0742 ",
        "all",
        "ndcg@10",
        "pretrained",
      ).map((q) => q.id),
    ).toEqual(["742"]);
    expect(
      selectQueries(
        study.queries,
        "MACROLIDES",
        "all",
        "ndcg@10",
        "pretrained",
      ),
    ).toHaveLength(2);
    expect(
      selectQueries(
        study.queries,
        "no-such-phrase",
        "all",
        "ndcg@10",
        "pretrained",
      ),
    ).toEqual([]);
    expect(JSON.stringify(raw)).toBe(before);
  });
  it("filters regressions and ignores insignificant rounding differences", () => {
    const study = parseStudy(raw);
    expect(
      selectQueries(study.queries, "", "loss", "ndcg@10", "pretrained"),
    ).toHaveLength(27);
    expect(outcome(-1e-15)).toBe("same");
    expect(outcome(1e-15)).toBe("same");
    expect(outcome(-0.1)).toBe("loss");
  });
});
