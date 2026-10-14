import * as Plot from "@observablehq/plot";
import { useEffect, useRef, useState } from "react";
import { ArrowUpRight } from "lucide-react";
import { Segments } from "./ui";
import type { Study } from "./types";

function usePlot(
  draw: (width: number) => SVGSVGElement | HTMLElement,
  deps: unknown[],
) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const host = ref.current;
    if (!host) return;
    const render = () =>
      host.replaceChildren(draw(Math.max(260, host.clientWidth)));
    const observer = new ResizeObserver(render);
    observer.observe(host);
    render();
    return () => {
      observer.disconnect();
      host.replaceChildren();
    };
  }, deps);
  return ref;
}

export function TrainingCharts({ study }: { study: Study }) {
  const [chart, setChart] = useState("loss");
  const history = study.training.history;
  const lossRef = usePlot(
    (width) =>
      Plot.plot({
        width,
        height: 290,
        marginLeft: 48,
        marginTop: 20,
        marginRight: 20,
        marginBottom: 40,
        style: {
          background: "transparent",
          color: "#555e77",
          fontFamily: "Manrope",
          fontSize: "11px",
        },
        x: {
          label: "Optimizer step",
          domain: [1, study.summary.optimizer_steps],
          ticks: Math.min(5, study.summary.optimizer_steps),
        },
        y: {
          label:
            chart === "loss" ? "Batch loss" : "Gradient norm (before clipping)",
          grid: true,
          zero: true,
        },
        marks: [
          ...(study.training.resumed_after_step === null
            ? []
            : [
                Plot.ruleX([study.training.resumed_after_step], {
                  stroke: "#bb4d33",
                  strokeDasharray: "4,4",
                }),
              ]),
          Plot.lineY(history, {
            x: "step",
            y: chart === "loss" ? "loss" : "gradient_norm",
            stroke: "#2446ed",
            strokeWidth: 2,
          }),
          Plot.dot(history, {
            x: "step",
            y: chart === "loss" ? "loss" : "gradient_norm",
            r: 2,
            fill: "#2446ed",
            tip: true,
            title: (d) =>
              `Step ${d.step}\n${chart === "loss" ? "Loss" : "Gradient norm"}: ${(chart === "loss" ? d.loss : d.gradient_norm).toFixed(4)}`,
          }),
        ],
      }),
    [
      history,
      chart,
      study.summary.optimizer_steps,
      study.training.resumed_after_step,
    ],
  );
  const validationRef = usePlot(
    (width) =>
      Plot.plot({
        width,
        height: 240,
        marginLeft: 48,
        marginTop: 24,
        marginRight: 35,
        marginBottom: 45,
        style: {
          background: "transparent",
          color: "#555e77",
          fontFamily: "Manrope",
          fontSize: "11px",
        },
        x: {
          label: "Checkpoint (0 = frozen)",
          domain: [0, study.summary.config.epochs],
          ticks: study.validation.map((row) => row.epoch),
        },
        y: {
          label: "Validation nDCG@10",
          domain: [
            Math.max(
              0,
              Math.min(...study.validation.map((row) => row.ndcg)) - 0.03,
            ),
            Math.min(
              1,
              Math.max(...study.validation.map((row) => row.ndcg)) + 0.03,
            ),
          ],
          grid: true,
          ticks: 4,
        },
        marks: [
          Plot.lineY(study.validation, {
            x: "epoch",
            y: "ndcg",
            stroke: "#2446ed",
            strokeWidth: 2,
          }),
          Plot.dot(study.validation, {
            x: "epoch",
            y: "ndcg",
            r: 5,
            fill: (d) =>
              d.epoch === study.summary.selected_epoch ? "#2446ed" : "#fff",
            stroke: "#2446ed",
            strokeWidth: 2,
            tip: true,
            title: (d) =>
              `${d.epoch === 0 ? "Frozen" : `Epoch ${d.epoch}`}\nnDCG@10: ${d.ndcg.toFixed(4)}`,
          }),
        ],
      }),
    [study.validation, study.summary.selected_epoch],
  );
  return (
    <div className="training-charts">
      <div className="loss-chart">
        <div className="chart-heading">
          <h3>The work, step by step.</h3>
          <Segments
            label="Training measurement"
            value={chart}
            onChange={setChart}
            items={[
              { value: "loss", label: "Loss" },
              { value: "gradient", label: "Gradients" },
            ]}
          />
        </div>
        <div
          ref={lossRef}
          role="img"
          aria-label={`Recorded ${chart} for ${study.summary.optimizer_steps} optimizer steps${study.training.resumed_after_step ? `; restart after step ${study.training.resumed_after_step} marked by a dashed line` : ""}`}
        />
        <p className="chart-caption">
          {study.training.resumed_after_step ? (
            <>
              <span className="dash-line" /> Restart after step{" "}
              {study.training.resumed_after_step} ·{" "}
            </>
          ) : null}{" "}
          One batch per point. Loss is noisy, rather than a smoothed improvement
          claim.
        </p>
      </div>
      <div className="validation-chart">
        <div className="chart-heading">
          <h3>Choose on validation.</h3>
          <ArrowUpRight size={18} />
        </div>
        <div
          ref={validationRef}
          role="img"
          aria-label={`Validation nDCG for the frozen encoder and ${study.summary.config.epochs} epochs; epoch ${study.summary.selected_epoch} selected`}
        />
        <p className="chart-caption">
          Epoch {study.summary.selected_epoch} selected before any reserved-test
          results. The earlier epoch wins an exact tie.
        </p>
      </div>
    </div>
  );
}
