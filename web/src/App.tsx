import {
  lazy,
  Suspense,
  useCallback,
  useEffect,
  useMemo,
  useState,
} from "react";
import { Tooltip } from "@base-ui/react/tooltip";
import { Accordion } from "@base-ui/react/accordion";
import { AnimatePresence, motion, MotionConfig } from "motion/react";
import {
  ArrowDown,
  ArrowUpRight,
  ArrowRight,
  Check,
  ChevronDown,
  Copy,
  Info,
  RotateCcw,
} from "lucide-react";
import { Inspector } from "./Inspector";
import { ExperimentLoader } from "./ExperimentLoader";
import { DownloadData } from "./DownloadData";
import { QueryTable } from "./QueryTable";
import { Choice, Hint, Segments } from "./ui";
import { parseStudy, outcomes } from "./data";
import { RankMark as Mark, QueryGlyph } from "./presentation";
import {
  delta,
  queryLabel,
  metricName,
  metricDescription,
  methodName,
  methods,
  outcome,
  points,
  score,
  type Metric,
  type Method,
  type Study,
  type Query,
} from "./types";
const TrainingCharts = lazy(() =>
  import("./charts").then((module) => ({ default: module.TrainingCharts })),
);

function Heading({
  number,
  title,
  subtitle,
}: {
  number: string;
  title: string;
  subtitle: string;
}) {
  return (
    <div className="section-heading">
      <span className="chapter">{number} /</span>
      <div>
        <h2>{title}</h2>
        <p>{subtitle}</p>
      </div>
    </div>
  );
}

function QueryField({
  queries,
  metric,
  baseline,
  onOpen,
}: {
  queries: Query[];
  metric: Metric;
  baseline: Method;
  onOpen: (q: Query) => void;
}) {
  const ordered = useMemo(
    () =>
      [...queries].sort(
        (a, b) =>
          delta(a, metric, baseline) - delta(b, metric, baseline) ||
          a.id.localeCompare(b.id, undefined, { numeric: true }),
      ),
    [queries, metric, baseline],
  );
  const [focus, setFocus] = useState(0);
  return (
    <div className="query-field">
      <div className="field-label">
        <span>{queries.length.toLocaleString()} reserved queries</span>
        <span>Fine-tuned vs {methodName[baseline]}</span>
      </div>
      <div
        className="query-grid"
        style={
          ordered.length < 12
            ? {
                gridTemplateColumns: `repeat(${ordered.length}, minmax(0, 1fr))`,
              }
            : undefined
        }
        aria-label="Query outcomes ordered from regressions to gains"
      >
        {ordered.map((q, index) => {
          const change = delta(q, metric, baseline);
          return (
            <Tooltip.Root key={q.id}>
              <Tooltip.Trigger
                className={`query-tile ${outcome(change)}`}
                aria-label={`Query ${q.id}: ${points(change)} percentage points. Inspect rankings.`}
                tabIndex={index === focus ? 0 : -1}
                data-query-trigger={q.id}
                onFocus={() => setFocus(index)}
                onClick={() => onOpen(q)}
                onKeyDown={(event) => {
                  const node = event.currentTarget;
                  const parent = node.closest(".query-grid");
                  const columns = parent
                    ? getComputedStyle(parent).gridTemplateColumns.split(" ")
                        .length
                    : 30;
                  const offset =
                    event.key === "ArrowRight"
                      ? 1
                      : event.key === "ArrowLeft"
                        ? -1
                        : event.key === "ArrowDown"
                          ? columns
                          : event.key === "ArrowUp"
                            ? -columns
                            : 0;
                  const next =
                    event.key === "Home"
                      ? 0
                      : event.key === "End"
                        ? ordered.length - 1
                        : Math.max(
                            0,
                            Math.min(ordered.length - 1, index + offset),
                          );
                  if (offset || event.key === "Home" || event.key === "End") {
                    event.preventDefault();
                    setFocus(next);
                    parent
                      ?.querySelectorAll<HTMLButtonElement>(".query-tile")
                      [next]?.focus();
                  }
                }}
              >
                <QueryGlyph change={change} />
              </Tooltip.Trigger>
              <Tooltip.Portal>
                <Tooltip.Positioner sideOffset={8}>
                  <Tooltip.Popup className="tooltip query-tooltip">
                    <b>
                      {queryLabel(q.id)} · {points(change)} pp
                    </b>
                    <span>{q.text}</span>
                    <small>Click to inspect all three rankings</small>
                  </Tooltip.Popup>
                </Tooltip.Positioner>
              </Tooltip.Portal>
            </Tooltip.Root>
          );
        })}
      </div>
      <div className="field-scale">
        <span>← Regressions</span>
        <span>Unchanged</span>
        <span>Gains →</span>
      </div>
      <p className="field-instruction">
        Every tile is a query. Select one to follow its ranking.{" "}
        <span>Keyboard: arrow keys, then Enter.</span>
      </p>
    </div>
  );
}

function Reproduce() {
  const [target, setTarget] = useState("cpu");
  const [dataset, setDataset] = useState("custom");
  const [copied, setCopied] = useState(false);
  const workspace = dataset === "custom" ? "runs/helpdesk" : "runs/scifact";
  const prepareArgs =
    dataset === "custom" ? " --config examples/helpdesk/experiment.toml" : "";
  const command = `uv venv --python 3.11 .venv\nuv pip install --python .venv/Scripts/python.exe torch==2.7.1 --index-url https://download.pytorch.org/whl/${target === "cuda" ? "cu128" : "cpu"}\nuv pip install --python .venv/Scripts/python.exe -e ".[dev]\"\n.venv/Scripts/python.exe -m rank_lab.cli prepare${prepareArgs} --device ${target}\n.venv/Scripts/python.exe -m rank_lab.cli train --workspace ${workspace} --device ${target} --stop-after-steps 2\n.venv/Scripts/python.exe -m rank_lab.cli train --workspace ${workspace} --device ${target} --resume\n.venv/Scripts/python.exe -m rank_lab.cli compare --workspace ${workspace} --device ${target}\n.venv/Scripts/python.exe -m rank_lab.cli export --workspace ${workspace} --destination ${workspace}/export --resumed-after-step 2`;

  async function copy() {
    try {
      await navigator.clipboard.writeText(command);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      setCopied(false);
    }
  }
  return (
    <div className="reproduce">
      <div className="reproduce-top">
        <span className="eyebrow">Run an experiment / Windows</span>
        <Choice
          label="Example data"
          value={dataset}
          onChange={setDataset}
          items={[
            { value: "custom", label: "Helpdesk example" },
            { value: "scifact", label: "SciFact benchmark" },
          ]}
        />
        <Segments
          label="Training device"
          value={target}
          onChange={setTarget}
          items={[
            { value: "cuda", label: "CUDA" },
            { value: "cpu", label: "CPU" },
          ]}
        />
        <button
          className="icon-button"
          onClick={copy}
          aria-label={copied ? "Commands copied" : "Copy commands"}
        >
          {copied ? <Check size={18} /> : <Copy size={18} />}
        </button>
      </div>
      <pre>
        <code>{command}</code>
      </pre>
      <p>
        The helpdesk fixture is fictional; SciFact downloads public benchmark
        data. Use a new workspace for another completed experiment. This
        explorer displays the recorded run; it does not run these commands.
      </p>
    </div>
  );
}

function Masthead({
  study,
  onLoad,
  onReset,
}: {
  study: Study;
  onLoad: (study: Study) => void;
  onReset: () => void;
}) {
  const imported = typeof study.document_text_json === "string";
  return (
    <>
      <a className="skip-link" href="#queries">
        Skip to query results
      </a>
      <header className="masthead">
        <a className="brand" href="#" aria-label="Rank Lab home">
          <Mark />
          <span>
            RANK LAB<span className="brand-dot">.</span>
          </span>
        </a>
        <nav aria-label="Main navigation">
          <a href="#results">Results</a>
          <a href="#queries">Queries</a>
          <a href="#training">The run</a>
          <a href="#protocol">Protocol</a>
        </nav>
        <span className="run-badge">
          <span /> {imported ? "Local experiment" : "SciFact demo"}
        </span>
        <ExperimentLoader
          onLoad={onLoad}
          onReset={onReset}
          imported={imported}
        />
      </header>
    </>
  );
}

function Experience({ study }: { study: Study }) {
  const imported = typeof study.document_text_json === "string";
  const restarted = study.training.resumed_after_step;
  const [metric, setMetric] = useState<Metric>("ndcg@10");
  const [baseline, setBaseline] = useState<Method>("pretrained");
  const [query, setQuery] = useState<Query | null>(
    () =>
      study.queries.find(
        (q) => q.id === new URLSearchParams(location.search).get("q"),
      ) ?? null,
  );
  const counts = useMemo(
    () => outcomes(study.queries, metric, baseline),
    [study.queries, metric, baseline],
  );
  const gain =
    study.summary.metrics.fine_tuned[metric] -
    study.summary.metrics[baseline][metric];
  const open = useCallback((q: Query) => {
    const active = document.activeElement;
    if (active instanceof HTMLElement && !active.closest(".inspector")) {
      document
        .querySelector("[data-last-query]")
        ?.removeAttribute("data-last-query");
      active.setAttribute("data-last-query", "");
    }
    const url = new URL(location.href);
    url.searchParams.set("q", q.id);
    history.replaceState(null, "", url);
    setQuery(q);
  }, []);
  function close() {
    const url = new URL(location.href);
    url.searchParams.delete("q");
    history.replaceState(null, "", url);
    setQuery(null);
  }
  const regression = useMemo(
    () =>
      [...study.queries].sort(
        (a, b) => delta(a, metric, baseline) - delta(b, metric, baseline),
      )[0],
    [study.queries, metric, baseline],
  );
  const missingQuery =
    new URLSearchParams(location.search).has("q") &&
    !query &&
    !study.queries.some(
      (q) => q.id === new URLSearchParams(location.search).get("q"),
    );
  return (
    <Tooltip.Provider delay={220}>
      <main>
        {missingQuery && (
          <div className="query-notice" role="status">
            That query is not in this reserved split. Choose one of the{" "}
            {study.queries.length} recorded queries below.
            <button onClick={close}>Dismiss</button>
          </div>
        )}
        <section className="hero">
          <div className="hero-meta">
            <span>Retrieval / Training / Evaluation</span>
            <span>
              {study.summary.dataset} · seed {study.summary.config.seed} ·{" "}
              {study.summary.config.epochs} epochs
            </span>
          </div>
          <div className="hero-layout">
            <div className="hero-copy">
              <p className="eyebrow">An experiment you can inspect</p>
              <h1>
                FOLLOW
                <br />
                THE RANK<span>.</span>
              </h1>
              <p className="hero-intro">
                A recorded fine-tuning run.{" "}
                {restarted
                  ? "A checkpoint restart."
                  : "Validation-led selection."}
                <br />
                Three methods, {study.queries.length.toLocaleString()} queries,
                every result in view.
              </p>
              <div className="hero-action">
                <a className="primary-action" href="#queries">
                  Explore the results <ArrowDown size={19} />
                </a>
                <button
                  className="text-button"
                  onClick={() => open(regression)}
                >
                  {delta(regression, metric, baseline) < -1e-12
                    ? "Inspect a regression"
                    : "Inspect a query"}{" "}
                  <ArrowUpRight size={17} />
                </button>
              </div>
            </div>
            <div className="hero-evidence">
              <div className="evidence-top">
                <span className="eyebrow">Query outcomes</span>
                <div className="field-controls">
                  <Choice
                    label="Metric"
                    value={metric}
                    onChange={setMetric}
                    items={(["ndcg@10", "recall@10", "mrr@10"] as Metric[]).map(
                      (v) => ({ value: v, label: metricName[v] }),
                    )}
                  />
                  <Choice
                    label="Compare against"
                    value={baseline}
                    onChange={setBaseline}
                    items={[
                      { value: "pretrained", label: "Pretrained" },
                      { value: "lexical", label: "BM25" },
                    ]}
                  />
                </div>
              </div>
              <QueryField
                queries={study.queries}
                metric={metric}
                baseline={baseline}
                onOpen={open}
              />
              <div className="outcome-strip">
                <span>
                  <b>{counts.loss}</b> regressions
                </span>
                <span>
                  <b>{counts.same}</b> unchanged
                </span>
                <span>
                  <b>{counts.gain}</b> gains
                </span>
              </div>
            </div>
          </div>
          <div className="study-strip">
            <span>
              <b>{study.summary.documents.toLocaleString()}</b> documents
            </span>
            <span>
              <b>
                {study.summary.train_queries} /{" "}
                {study.summary.validation_queries} /{" "}
                {study.summary.test_queries}
              </b>{" "}
              train / validation / test
            </span>
            <span>
              <b>{(study.summary.trainable_parameters / 1e6).toFixed(1)}M</b>{" "}
              trainable parameters
            </span>
            <a href="#protocol">
              Read the experiment contract <ArrowUpRight size={16} />
            </a>
          </div>
        </section>
        <section id="results" className="results-section">
          <Heading
            number="01"
            title="The result isn’t one number."
            subtitle="Compare the selected checkpoint with both baselines. Inspect the gains and regressions behind each average."
          />
          <div className="results-layout">
            <div className="result-callout">
              <span className="eyebrow">
                Fine-tuned vs {methodName[baseline]}
              </span>
              <AnimatePresence mode="wait">
                <motion.div
                  key={`${metric}-${baseline}`}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -8 }}
                  transition={{ duration: 0.2 }}
                  className={`big-result ${gain < 0 ? "negative" : ""}`}
                >
                  {points(gain)}
                  <span>pp</span>
                </motion.div>
              </AnimatePresence>
              <p>{metricName[metric]} on the complete reserved test set.</p>
              <Hint text={metricDescription[metric]}>
                <Info size={14} /> What this metric measures
              </Hint>
              <p className="result-footnote">
                One seed and dataset. Judgments may be incomplete.
                <br />
                No claim that fine-tuning always wins.
              </p>
            </div>
            <div className="metrics-table-wrap">
              <table className="metrics-table">
                <caption className="sr-only">
                  Reserved test results for all three methods
                </caption>
                <thead>
                  <tr>
                    <th scope="col">Method</th>
                    {(["ndcg@10", "recall@10", "mrr@10"] as Metric[]).map(
                      (m) => (
                        <th key={m} scope="col">
                          <Hint text={metricDescription[m]}>
                            {metricName[m]}
                          </Hint>
                        </th>
                      ),
                    )}
                  </tr>
                </thead>
                <tbody>
                  {methods.map((method, i) => (
                    <tr
                      key={method}
                      className={method === "fine_tuned" ? "fine-row" : ""}
                    >
                      <th scope="row">
                        <span className="method-index">0{i + 1}</span>
                        {methodName[method]}
                        <small>
                          {method === "lexical"
                            ? "Full-text lexical ranking"
                            : method === "pretrained"
                              ? "Pinned MiniLM, frozen weights"
                              : `Epoch ${study.summary.selected_epoch}, validation selected`}
                        </small>
                      </th>
                      {(["ndcg@10", "recall@10", "mrr@10"] as Metric[]).map(
                        (m) => (
                          <td
                            key={m}
                            className={
                              study.summary.metrics[method][m] ===
                              Math.max(
                                ...methods.map(
                                  (n) => study.summary.metrics[n][m],
                                ),
                              )
                                ? "best"
                                : ""
                            }
                          >
                            {score(study.summary.metrics[method][m])}
                            {study.summary.metrics[method][m] ===
                              Math.max(
                                ...methods.map(
                                  (n) => study.summary.metrics[n][m],
                                ),
                              ) && <span aria-label="Best observed value" />}
                          </td>
                        ),
                      )}
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="table-note">
                Higher is better for all three metrics. Dense inputs truncate at
                {study.summary.config.max_length} wordpieces; BM25 receives full
                document text.
              </p>
            </div>
          </div>
        </section>
        <section id="queries" className="queries-section">
          <Heading
            number="02"
            title="Start with a query. Follow the evidence."
            subtitle={`Search all ${study.queries.length} outcomes, look at what moved, and open the rankings behind each score.`}
          />
          <div className="section-controls">
            <Choice
              label="Result metric"
              value={metric}
              onChange={setMetric}
              items={(["ndcg@10", "recall@10", "mrr@10"] as Metric[]).map(
                (v) => ({ value: v, label: metricName[v] }),
              )}
            />
            <Choice
              label="Result baseline"
              value={baseline}
              onChange={setBaseline}
              items={[
                { value: "pretrained", label: "Pretrained" },
                { value: "lexical", label: "BM25" },
              ]}
            />
          </div>
          <QueryTable
            queries={study.queries}
            metric={metric}
            baseline={baseline}
            onOpen={open}
          />
        </section>
        <section id="training" className="training-section">
          <Heading
            number="03"
            title="Training and checkpoint recovery."
            subtitle={
              restarted
                ? `Stopped after step ${restarted}, resumed from its checkpoint and completed ${study.summary.optimizer_steps} updates.`
                : `Completed ${study.summary.optimizer_steps} updates with validation before the reserved test comparison.`
            }
          />
          <div className="checkpoint-track">
            <div>
              <span className="checkpoint-symbol">01</span>
              <b>Initialize</b>
              <small>Pinned base weights</small>
            </div>
            <div className="checkpoint-restart">
              <span className="checkpoint-symbol">
                <RotateCcw size={22} />
              </span>
              <b>
                {restarted ? `Stop at step ${restarted}` : "Save checkpoint"}
              </b>
              <small>Model · optimizer · scheduler · RNG</small>
            </div>
            <div>
              <span className="checkpoint-symbol">03</span>
              <b>
                {restarted ? "Resume" : "Train"} → step{" "}
                {study.summary.optimizer_steps}
              </b>
              <small>
                {study.summary.config.epochs} validation checkpoints
              </small>
            </div>
            <div>
              <span className="checkpoint-symbol">
                <Check size={23} />
              </span>
              <b>Freeze the choice</b>
              <small>
                Epoch {study.summary.selected_epoch} → reserved test
              </small>
            </div>
          </div>
          <Suspense
            fallback={
              <p className="chart-caption" role="status">
                Loading recorded training charts…
              </p>
            }
          >
            <TrainingCharts study={study} />
          </Suspense>
          <div className="run-details">
            <span>
              <b>
                {study.summary.peak_cuda_allocated_bytes === null
                  ? "—"
                  : `${(study.summary.peak_cuda_allocated_bytes / 1024 ** 3).toFixed(2)} GiB`}
              </b>{" "}
              {study.summary.peak_cuda_allocated_bytes === null
                ? "CUDA memory not measured on CPU"
                : "peak allocated CUDA memory"}
            </span>
            <span>
              <b>
                {study.training.elapsed_seconds_resumed_invocation.toFixed(2)} s
              </b>{" "}
              {restarted
                ? "resumed invocation + validation"
                : "training invocation + validation"}
            </span>
            <span>
              <b>{study.summary.gpu}</b> {study.summary.platform} · Torch{" "}
              {study.summary.torch}
            </span>
          </div>
          <p className="run-limit">
            The timer covers the recorded training invocation and validation,
            excluding downloads and baseline preparation.{" "}
            {restarted
              ? `Earlier work before the restart at step ${restarted} is not included.`
              : "No restart is annotated for this run."}{" "}
            This record does not establish equivalence to an uninterrupted GPU
            run.
          </p>
        </section>
        <section id="protocol" className="protocol-section">
          <Heading
            number="04"
            title="Know what the run can say."
            subtitle="Pinned inputs, explicit splits, selection before test. The limits are part of the result."
          />
          <div className="protocol-layout">
            <Accordion.Root
              multiple
              defaultValue={["split"]}
              className="protocol-accordion"
            >
              {[
                {
                  id: "split",
                  title: "Split by query text. Keep test intact.",
                  body: (
                    <>
                      <p>
                        Train, validation and test query groups are kept
                        separate. This run retains {study.summary.test_queries}{" "}
                        reserved test queries.{" "}
                        {study.exclusions.length > 0
                          ? `${study.exclusions.length} training queries with exact normalized test text were excluded.`
                          : "No training exclusions were needed."}
                      </p>
                      <div className="split-bar">
                        <span style={{ flex: study.summary.train_queries }}>
                          {study.summary.train_queries} train
                        </span>
                        <span
                          style={{ flex: study.summary.validation_queries }}
                        >
                          {study.summary.validation_queries} val
                        </span>
                        <span style={{ flex: study.summary.test_queries }}>
                          {study.summary.test_queries} test
                        </span>
                      </div>
                      <p className="small-copy">
                        Casefold + collapsed whitespace groups, seed{" "}
                        {study.summary.config.seed}. This checks exact
                        duplicates, rather than semantic paraphrases or the base
                        model’s pretraining exposure.
                      </p>
                      {study.exclusions.map((e) => (
                        <div key={e.train_id} className="exclusion">
                          Excluded train {e.train_id} <ArrowRight size={14} />{" "}
                          preserved test {e.test_ids.join(", ")}
                        </div>
                      ))}
                    </>
                  ),
                },
                {
                  id: "learning",
                  title: "Train against known positives.",
                  body: (
                    <>
                      <p>
                        All{" "}
                        {study.summary.trainable_parameters.toLocaleString()}{" "}
                        encoder parameters are trainable. Mean pooling,
                        normalized vectors, in-batch contrastive loss, and
                        masking of other known relevant documents from negative
                        columns.
                      </p>
                      <dl className="config-grid">
                        <div>
                          <dt>Batch</dt>
                          <dd>{study.summary.config.batch_size}</dd>
                        </div>
                        <div>
                          <dt>AdamW rate</dt>
                          <dd>{study.summary.config.learning_rate}</dd>
                        </div>
                        <div>
                          <dt>Temperature</dt>
                          <dd>{study.summary.config.temperature}</dd>
                        </div>
                        <div>
                          <dt>Token limit</dt>
                          <dd>{study.summary.config.max_length}</dd>
                        </div>
                      </dl>
                      <p className="small-copy">
                        Unjudged documents can still be relevant. Training uses
                        gradient clipping at 1, linear warmup and decay;
                        precision follows the recorded configuration.
                      </p>
                    </>
                  ),
                },
                {
                  id: "identity",
                  title: "Connect the result to the checkpoint.",
                  body: (
                    <>
                      <p>
                        Downloads verify pinned Git/LFS file identities. Resume
                        rejects changes in data, configuration or Torch version.
                        Evaluation checks the chosen model’s file and tensor
                        hashes.
                      </p>
                      <dl className="hash-list">
                        <div>
                          <dt>Base revision</dt>
                          <dd>{study.summary.base_revision}</dd>
                        </div>
                        <div>
                          <dt>Clean data fingerprint</dt>
                          <dd>{study.dataset_fingerprint}</dd>
                        </div>
                        <div>
                          <dt>Selected tensor fingerprint</dt>
                          <dd>{study.summary.weights_fingerprint}</dd>
                        </div>
                        <div>
                          <dt>Recorded comparison SHA-256</dt>
                          <dd>{study.comparison_sha256}</dd>
                        </div>
                      </dl>
                    </>
                  ),
                },
                {
                  id: "limits",
                  title: "One experiment, with trade-offs.",
                  body: (
                    <>
                      <p>
                        Metrics are equal query means over the full
                        {study.summary.documents.toLocaleString()}-document
                        corpus. Stable document-ID ordering resolves exact score
                        ties. Unjudged documents receive zero gain.
                      </p>
                      <p>
                        BM25 uses full text while the encoders truncate. One
                        seed and dataset are an experiment rather than a broad
                        quality or latency claim. Relevance judgments may be
                        incomplete.
                      </p>
                    </>
                  ),
                },
              ].map((item) => (
                <Accordion.Item value={item.id} key={item.id}>
                  <Accordion.Header>
                    <Accordion.Trigger>
                      <span>{item.title}</span>
                      <ChevronDown size={20} />
                    </Accordion.Trigger>
                  </Accordion.Header>
                  <Accordion.Panel className="protocol-panel">
                    {item.body}
                  </Accordion.Panel>
                </Accordion.Item>
              ))}
            </Accordion.Root>
            <Reproduce />
          </div>
        </section>
        <section className="closing">
          <div className="closing-top">
            <span className="eyebrow">Built to be inspected.</span>
            <DownloadData
              content={JSON.stringify(study, null, 2)}
              filename={
                imported ? "rank-lab-study.json" : "recorded-study.json"
              }
              label={imported ? "Download experiment" : "Recorded study JSON"}
            />
          </div>
          <p>
            BETTER RANKINGS.
            <br />
            <span>VISIBLE TRADE-OFFS.</span>
          </p>
          <div className="closing-bottom">
            <a
              href={
                imported
                  ? "https://github.com/shayanxcyber/rank-lab/blob/main/docs/CUSTOM-DATA.md"
                  : "https://huggingface.co/datasets/BeIR/scifact"
              }
              target="_blank"
              rel="noreferrer"
            >
              {imported ? "Custom data guide ↗" : "SciFact / BEIR ↗"}
            </a>
            <a
              href="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2"
              target="_blank"
              rel="noreferrer"
            >
              MiniLM base ↗
            </a>
            <span>Original training + evaluation engineering</span>
            <a href="#">Back to start ↑</a>
          </div>
        </section>
      </main>
      <Inspector
        study={study}
        query={query}
        metric={metric}
        baseline={baseline}
        onClose={close}
        onNext={() => {
          if (query)
            open(
              study.queries[
                (study.queries.findIndex((q) => q.id === query.id) + 1) %
                  study.queries.length
              ],
            );
        }}
      />
    </Tooltip.Provider>
  );
}

export default function App() {
  const [study, setStudy] = useState<Study | null>(null);
  const [error, setError] = useState("");
  const [retry, setRetry] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    setError("");
    fetch("./data/study.json", { signal: controller.signal })
      .then((r) => {
        if (!r.ok) throw new Error(`Study file unavailable (${r.status})`);
        return r.json();
      })
      .then((data) => setStudy(parseStudy(data)))
      .catch((e) => {
        if (e.name !== "AbortError") setError(e.message);
      });
    return () => controller.abort();
  }, [retry]);
  return (
    <MotionConfig reducedMotion="user">
      {study ? (
        <>
          <Masthead
            study={study}
            onLoad={(next) => {
              const url = new URL(location.href);
              url.searchParams.delete("q");
              history.replaceState(null, "", url);
              setStudy(next);
            }}
            onReset={() => {
              setStudy(null);
              setRetry((value) => value + 1);
            }}
          />
          <Experience key={study.comparison_sha256} study={study} />
        </>
      ) : (
        <div className="load-screen">
          <div className="brand">
            <Mark />
            <span>RANK LAB.</span>
          </div>
          <span className="eyebrow">Recorded experiment / 001</span>
          {error ? (
            <>
              <h1>The study couldn’t load.</h1>
              <p role="alert">{error}</p>
              <button
                className="primary-action"
                onClick={() => setRetry((v) => v + 1)}
              >
                Retry loading <RotateCcw size={17} />
              </button>
            </>
          ) : (
            <>
              <h1>Follow the ranking.</h1>
              <div className="loading-rule" />
              <p role="status">Loading recorded query results…</p>
            </>
          )}
        </div>
      )}
    </MotionConfig>
  );
}
