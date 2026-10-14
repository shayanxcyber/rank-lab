import { Dialog } from "@base-ui/react/dialog";
import { Tabs } from "@base-ui/react/tabs";
import { Accordion } from "@base-ui/react/accordion";
import { ArrowDown, Check, ChevronDown, Link2, X } from "lucide-react";
import { useEffect, useState } from "react";
import { verifyDocumentTexts } from "./data";
import { DocumentIdentity } from "./presentation";
import {
  methods,
  methodName,
  score,
  metricName,
  points,
  delta,
  queryLabel,
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
    const bytes =
      study.document_text_json !== undefined
        ? Promise.resolve(
            new TextEncoder().encode(study.document_text_json).buffer,
          )
        : fetch("./data/document-text.json", {
            signal: controller.signal,
          }).then((response) => {
            if (!response.ok) throw new Error("Document texts could not load");
            return response.arrayBuffer();
          });
    bytes
      .then((value) => verifyDocumentTexts(study, value))
      .then((value) => {
        if (!controller.signal.aborted) setTexts(value);
      })
      .catch((error) => {
        if (error.name !== "AbortError") setTextError(error.message);
      });
    return () => controller.abort();
  }, [!!query, texts, retry, study]);
  const abstract = (id: string) =>
    texts?.[id] ??
    (textError
      ? "Document text unavailable. Retry the file above."
      : "Loading recorded document text…");
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
                  Reserved query / {queryLabel(query.id)}
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
                Recorded rankings for the same query and corpus, up to ten
                results per method. Relevance uses the recorded judgments; an
                unjudged result is not a confirmed irrelevant result.
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
                    Retry document text
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
                                    <DocumentIdentity
                                      title={doc.title}
                                      judgment={query.judgments[id]}
                                    />
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
                <span>{study.summary.dataset} · recorded study</span>
                <a
                  href={
                    study.summary.dataset.toLowerCase() === "beir/scifact"
                      ? "https://huggingface.co/datasets/BeIR/scifact"
                      : "https://github.com/shayanxcyber/rank-lab/blob/main/docs/CUSTOM-DATA.md"
                  }
                  target="_blank"
                  rel="noreferrer"
                >
                  {study.summary.dataset.toLowerCase() === "beir/scifact"
                    ? "Dataset source ↗"
                    : "Custom data guide ↗"}
                </a>
              </div>
            </>
          )}
        </Dialog.Popup>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
