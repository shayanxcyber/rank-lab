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
  Download,
  Info,
  RotateCcw,
} from "lucide-react";
import { Inspector } from "./Inspector";
import { QueryTable } from "./QueryTable";
import { Choice, Hint, Segments } from "./ui";
import { parseStudy, outcomes } from "./data";
import { RankMark as Mark, QueryGlyph } from "./presentation";
import {
  delta,
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
          Number(a.id) - Number(b.id),
      ),
    [queries, metric, baseline],
  );
  const [focus, setFocus] = useState(0);
  return (
    <div className="query-field">
      <div className="field-label">
        <span>300 reserved queries</span>
        <span>Fine-tuned vs {methodName[baseline]}</span>
      </div>
      <div
        className="query-grid"
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
                      Q{q.id} · {points(change)} pp
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
  const [target, setTarget] = useState("cuda");
  const [copied, setCopied] = useState(false);
  const command = `uv venv --python 3.11 .venv\nuv pip install --python .venv/Scripts/python.exe torch==2.7.1 --index-url https://download.pytorch.org/whl/${target === "cuda" ? "cu128" : "cpu"}\nuv pip install --python .venv/Scripts/python.exe -e ".[dev]"\n.venv/Scripts/python.exe -m rank_lab.cli prepare\n.venv/Scripts/python.exe -m rank_lab.cli train --device ${target} --stop-after-steps 2\n.venv/Scripts/python.exe -m rank_lab.cli train --device ${target} --resume\n.venv/Scripts/python.exe -m rank_lab.cli compare --device ${target}\n.venv/Scripts/python.exe -m rank_lab.cli export --resumed-after-step 2`;
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
        <span className="eyebrow">Run a fresh study / Windows</span>
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
        Downloads pinned public inputs. Use a new workspace for another study.
        This explorer displays the recorded run; it does not run these commands.
      </p>
    </div>
  );
}

function Experience({ study }: { study: Study }) {
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
          <span /> Recorded / 001
        </span>
      </header>
      <main>
        {missingQuery && (
          <div className="query-notice" role="status">
            That query is not in this reserved split. Choose one of the 300
            recorded queries below.<button onClick={close}>Dismiss</button>
          </div>
        )}
        <section className="hero">
          <div className="hero-meta">
            <span>Retrieval / Training / Evaluation</span>
            <span>SciFact · seed 1729 · 3 epochs</span>
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
                A real fine-tuning run. A checkpoint restart.
                <br />
                Three methods, 300 queries, every result in view.
              </p>
              <div className="hero-action">
                <a className="primary-action" href="#queries">
                  Explore the results <ArrowDown size={19} />
                </a>
                <button
                  className="text-button"
                  onClick={() => open(regression)}
                >
                  Inspect a regression <ArrowUpRight size={17} />
                </button>
              </div>
            </div>
            <div className="hero-evidence">
              <div className="evidence-top">
                <span className="eyebrow">One run. Mixed outcomes.</span>
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
              <b>5,183</b> documents
            </span>
            <span>
              <b>727 / 80 / 300</b> train / validation / test
            </span>
            <span>
              <b>22.7M</b> trainable parameters
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
            subtitle="The selected checkpoint improves average retrieval quality. The lexical baseline still wins on reciprocal rank."
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
                One seed and benchmark. Incomplete judgments.
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
                              : "Epoch 3, validation selected"}
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
                256 wordpieces; BM25 receives full title + abstract.
              </p>
            </div>
          </div>
        </section>
        <section id="queries" className="queries-section">
          <Heading
            number="02"
            title="Start with a query. Follow the evidence."
            subtitle="Search all 300 outcomes, look at what moved, and open the rankings behind each score."
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
            title="A stop is part of the story."
            subtitle="The process stopped after two optimizer steps. A fresh process loaded the checkpoint and completed all 78."
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
              <b>Stop at step 2</b>
              <small>Model · optimizer · scheduler · RNG</small>
            </div>
            <div>
              <span className="checkpoint-symbol">03</span>
              <b>Resume → step 78</b>
              <small>Three validation checkpoints</small>
            </div>
            <div>
              <span className="checkpoint-symbol">
                <Check size={23} />
              </span>
              <b>Freeze the choice</b>
              <small>Epoch 3 → reserved test</small>
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
                {(study.summary.peak_cuda_allocated_bytes / 1024 ** 3).toFixed(
                  2,
                )}{" "}
                GiB
              </b>{" "}
              peak allocated CUDA memory
            </span>
            <span>
              <b>
                {study.training.elapsed_seconds_resumed_invocation.toFixed(2)} s
              </b>{" "}
              resumed invocation + validation
            </span>
            <span>
              <b>RTX 5080</b> Windows · Torch 2.7.1 + cu128
            </span>
          </div>
          <p className="run-limit">
            The timer excludes downloads, baseline evaluation and the first two
            steps. Exact restart equivalence passes on a controlled CPU fixture;
            a second uninterrupted GPU run was not compared.
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
                        The official training split contained two queries with
                        exactly the same normalized text as test queries. They
                        were removed only from training. The raw files and all
                        300 test queries were preserved.
                      </p>
                      <div className="split-bar">
                        <span style={{ flex: 727 }}>727 train</span>
                        <span style={{ flex: 80 }}>80 val</span>
                        <span style={{ flex: 300 }}>300 test</span>
                      </div>
                      <p className="small-copy">
                        Casefold + collapsed whitespace groups, seed 1729. This
                        checks exact duplicates, rather than semantic
                        paraphrases or the base model’s pretraining exposure.
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
                        All 22,713,216 MiniLM parameters are trainable. Mean
                        pooling, normalized vectors, in-batch contrastive loss,
                        and masking of other known relevant documents from
                        negative columns.
                      </p>
                      <dl className="config-grid">
                        <div>
                          <dt>Batch</dt>
                          <dd>32</dd>
                        </div>
                        <div>
                          <dt>AdamW rate</dt>
                          <dd>2 × 10⁻⁵</dd>
                        </div>
                        <div>
                          <dt>Temperature</dt>
                          <dd>0.05</dd>
                        </div>
                        <div>
                          <dt>Token limit</dt>
                          <dd>256</dd>
                        </div>
                      </dl>
                      <p className="small-copy">
                        Unjudged documents can still be relevant. Bfloat16
                        autocast, gradient clipping at 1, linear warmup and
                        decay.
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
                          <dd>{study.training.final_weights}</dd>
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
                  title: "A focused benchmark, with trade-offs.",
                  body: (
                    <>
                      <p>
                        Metrics are equal query means over the full
                        5,183-document corpus. Stable document-ID ordering
                        resolves exact score ties. Unjudged documents receive
                        zero gain.
                      </p>
                      <p>
                        BM25 uses full text while the encoders truncate. One
                        seed, one small scientific-retrieval benchmark, and no
                        latency competition or broad accuracy claim. The
                        scientific claims are benchmark queries, not advice.
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
            <a href="./data/study.json" download className="text-button">
              Recorded study JSON <Download size={16} />
            </a>
          </div>
          <p>
            BETTER RANKINGS.
            <br />
            <span>VISIBLE TRADE-OFFS.</span>
          </p>
          <div className="closing-bottom">
            <a
              href="https://huggingface.co/datasets/BeIR/scifact"
              target="_blank"
              rel="noreferrer"
            >
              SciFact / BEIR ↗
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
        <Experience study={study} />
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
