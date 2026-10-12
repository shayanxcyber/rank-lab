import { Dialog } from "@base-ui/react/dialog";
import { Tabs } from "@base-ui/react/tabs";
import { Accordion } from "@base-ui/react/accordion";
import { ArrowDown, Check, ChevronDown, Link2, X } from "lucide-react";
import { useEffect, useState } from "react";
import { DocumentIdentity } from "./presentation";
import {
  methods,
  methodName,
  score,
  metricName,
  points,
  delta,
  type Study,
  type Query,
  type Metric,
  type Method,
} from "./types";

export function Inspector({
  study,
  query,
  metric,
  baseline,
  onClose,
  onNext,
}: {
  study: Study;
  query: Query | null;
  metric: Metric;
  baseline: Method;
  onClose: () => void;
  onNext: () => void;
}) {
  const [copied, setCopied] = useState(false);
  const [texts, setTexts] = useState<Record<string, string> | null>(null);
  const [textError, setTextError] = useState("");
  const [retry, setRetry] = useState(0);
  useEffect(() => {
    if (!query || texts) return;
    const controller = new AbortController();
    setTextError("");
    fetch("./data/document-text.json", { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error("Document abstracts could not load");
        const bytes = await response.arrayBuffer();
        const digest = Array.from(
          new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)),
        )
          .map((v) => v.toString(16).padStart(2, "0"))
          .join("");
        if (digest !== study.document_text_sha256)
          throw new Error("Document file identity differs from this study");
        const value = JSON.parse(new TextDecoder().decode(bytes));
        if (
          Object.keys(value).length !== Object.keys(study.documents).length ||
          Object.keys(study.documents).some(
            (id) => typeof value[id] !== "string",
          )
        )
          throw new Error("Document abstracts are incomplete");
        setTexts(value);
      })
      .catch((error) => {
        if (error.name !== "AbortError") setTextError(error.message);
      });
    return () => controller.abort();
  }, [!!query, texts, retry, study]);
  const abstract = (id: string) =>
    texts?.[id] ??
    (textError
      ? "Abstract unavailable. Retry the document file above."
      : "Loading recorded abstract…");
  async function copyLink() {
    try {
      await navigator.clipboard.writeText(location.href);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      setCopied(false);
    }
  }
  return (
    <Dialog.Root
      open={!!query}
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <Dialog.Portal>
        <Dialog.Backdrop className="dialog-backdrop" />
        <Dialog.Popup
          className="inspector"
          finalFocus={() =>
            document.querySelector<HTMLButtonElement>("[data-last-query]")
          }
        >
          {query && (
            <>
              <div className="inspector-header">
                <span className="eyebrow">
                  Reserved query / {query.id.padStart(4, "0")}
                </span>
                <div>
                  <button
                    className="icon-button"
                    aria-label={copied ? "Link copied" : "Copy query link"}
                    onClick={copyLink}
                  >
                    {copied ? <Check size={18} /> : <Link2 size={18} />}
                  </button>
                  <Dialog.Close
                    className="icon-button"
                    aria-label="Close query inspector"
                  >
                    <X size={22} />
                  </Dialog.Close>
                </div>
              </div>
              <Dialog.Title className="query-title">{query.text}</Dialog.Title>
              <Dialog.Description className="inspector-description">
                Recorded top ten for the same query and corpus. Relevance uses
                the benchmark judgments; an unjudged result is not a confirmed
                irrelevant result.
              </Dialog.Description>
              <div
                className={`query-summary ${delta(query, metric, baseline) < 0 ? "loss" : ""}`}
              >
                <span>
                  <b>{points(delta(query, metric, baseline))}</b> pp{" "}
                  {metricName[metric]} vs {methodName[baseline]}
                </span>
                <span>
                  {Object.values(query.judgments).filter((v) => v > 0).length}{" "}
                  judged relevant{" "}
                  {Object.values(query.judgments).filter((v) => v > 0)
                    .length === 1
                    ? "document"
                    : "documents"}
                </span>
                <button className="text-button" onClick={onNext}>
                  Next query <ArrowDown size={16} />
                </button>
              </div>
              {textError && (
                <div className="query-notice" role="alert">
                  {textError}
                  <button onClick={() => setRetry((v) => v + 1)}>
                    Retry abstracts
                  </button>
                </div>
              )}
              <Tabs.Root defaultValue="rankings">
                <Tabs.List className="inspector-tabs" aria-label="Query detail">
                  <Tabs.Tab value="rankings">Rankings</Tabs.Tab>
                  <Tabs.Tab value="judgments">Judgments</Tabs.Tab>
                </Tabs.List>
                <Tabs.Panel value="rankings">
                  <div className="ranking-columns">
                    {methods.map((method) => (
                      <section
                        key={method}
                        className={
                          method === "fine_tuned"
                            ? "ranking-column selected-method"
                            : "ranking-column"
                        }
                      >
                        <div className="ranking-heading">
                          <h3>{methodName[method]}</h3>
                          <span>{score(query.metrics[method][metric])}</span>
                        </div>
                        <Accordion.Root multiple className="document-list">
                          {query.rankings[method].map((id, index) => {
                            const doc = study.documents[id];
                            const relevant = (query.judgments[id] ?? 0) > 0;
                            return (
                              <Accordion.Item
                                value={id}
                                key={id}
                                className={
                                  relevant ? "document relevant" : "document"
                                }
                              >
                                <Accordion.Header>
                                  <Accordion.Trigger className="document-trigger">
                                    <span className="rank-number">
                                      {String(index + 1).padStart(2, "0")}
                                    </span>
                                    <DocumentIdentity title={doc.title} judgment={query.judgments[id]} />
                                    <ChevronDown size={14} />
                                  </Accordion.Trigger>
                                </Accordion.Header>
                                <Accordion.Panel className="abstract-panel">
                                  <p>{abstract(id)}</p>
                                  <span>
                                    Corpus ID {id} · judgment{" "}
                                    {query.judgments[id] ?? "unavailable"}
                                  </span>
                                </Accordion.Panel>
                              </Accordion.Item>
                            );
                          })}
                        </Accordion.Root>
                      </section>
                    ))}
                  </div>
                </Tabs.Panel>
                <Tabs.Panel value="judgments">
                  <div className="judgments">
                    <p>
                      The complete judgment set for this query, including
                      relevant documents outside a method’s top ten.
                    </p>
                    {Object.entries(query.judgments).map(([id, value]) => (
                      <article key={id}>
                        <span className="eyebrow">
                          Corpus {id} / relevance {value}
                        </span>
                        <h3>{study.documents[id].title}</h3>
                        <p>{abstract(id)}</p>
                        <dl>
                          {methods.map((method) => (
                            <div key={method}>
                              <dt>{methodName[method]}</dt>
                              <dd>
                                {query.rankings[method].includes(id)
                                  ? `Rank ${query.rankings[method].indexOf(id) + 1}`
                                  : "Outside top ten"}
                              </dd>
                            </div>
                          ))}
                        </dl>
                      </article>
                    ))}
                  </div>
                </Tabs.Panel>
              </Tabs.Root>
              <div className="inspector-foot">
                <span>Public SciFact benchmark material · recorded study</span>
                <a
                  href="https://huggingface.co/datasets/BeIR/scifact"
                  target="_blank"
                  rel="noreferrer"
                >
                  Dataset source ↗
                </a>
              </div>
            </>
          )}
        </Dialog.Popup>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
