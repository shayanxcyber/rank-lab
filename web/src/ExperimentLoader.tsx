import { Dialog } from "@base-ui/react/dialog";
import { ArrowUpRight, FileJson, FolderOpen, Upload, X } from "lucide-react";
import { useRef, useState } from "react";
import { parseBundle } from "./data";
import type { Study } from "./types";

export function ExperimentLoader({
  onLoad,
  onReset,
  imported,
}: {
  onLoad: (study: Study) => void;
  onReset: () => void;
  imported: boolean;
}) {
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState("");
  const input = useRef<HTMLInputElement>(null);
  const generation = useRef(0);
  async function read(file?: File) {
    if (!file) return;
    const current = ++generation.current;
    setBusy(true);
    setError("");
    try {
      if (file.size > 50 * 1024 * 1024)
        throw new Error("Choose an experiment bundle smaller than 50 MB.");
      const study = await parseBundle(await file.text());
      if (current !== generation.current) return;
      onLoad(study);
      setOpen(false);
    } catch (reason) {
      if (current === generation.current)
        setError(
          reason instanceof Error
            ? reason.message
            : "The experiment could not load.",
        );
    } finally {
      if (current === generation.current) setBusy(false);
      if (input.current) input.current.value = "";
    }
  }
  return (
    <div className="experiment-actions">
      {imported && (
        <button className="text-button demo-reset" onClick={onReset}>
          SciFact demo
        </button>
      )}
      <Dialog.Root
        open={open}
        onOpenChange={(value) => {
          setOpen(value);
          if (!value) {
            generation.current++;
            setBusy(false);
            setDragging(false);
          } else setError("");
        }}
      >
        <Dialog.Trigger className="load-experiment" data-experiment-trigger>
          <FolderOpen size={16} /> Open experiment
        </Dialog.Trigger>
        <Dialog.Portal>
          <Dialog.Backdrop className="dialog-backdrop" />
          <Dialog.Popup
            className="experiment-dialog"
            finalFocus={() =>
              document.querySelector<HTMLButtonElement>(
                "[data-experiment-trigger]",
              )
            }
          >
            <div className="experiment-dialog-heading">
              <span className="eyebrow">Local experiment</span>
              <Dialog.Close
                className="icon-button"
                aria-label="Close experiment loader"
              >
                <X size={21} />
              </Dialog.Close>
            </div>
            <Dialog.Title>Inspect your own experiment.</Dialog.Title>
            <Dialog.Description>
              Open a completed Rank Lab export. The rankings, judgments and
              training record travel together.
            </Dialog.Description>
            <div
              className="experiment-drop"
              data-dragging={dragging || undefined}
              aria-busy={busy}
              onDragOver={(event) => {
                event.preventDefault();
                setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={(event) => {
                event.preventDefault();
                setDragging(false);
                void read(event.dataTransfer.files[0]);
              }}
            >
              <FileJson size={36} strokeWidth={1.2} />
              <strong>
                {busy
                  ? "Checking your experiment…"
                  : "Drop rank-lab-study.json here"}
              </strong>
              <span>
                or choose the bundle from your export folder · up to 50 MB
              </span>
              <input
                ref={input}
                hidden
                type="file"
                accept=".json,application/json"
                aria-label="Experiment bundle"
                onChange={(event) => void read(event.target.files?.[0])}
              />
              <button
                className="primary-action"
                disabled={busy}
                onClick={() => input.current?.click()}
              >
                <Upload size={16} /> Choose experiment
              </button>
            </div>
            {error && (
              <div className="experiment-error" role="alert">
                <b>The file couldn’t open.</b>
                <span>{error}</span>
                <small>
                  Your current experiment is still available. Choose another
                  export to retry.
                </small>
              </div>
            )}
            <div className="experiment-instructions">
              <span className="control-label">
                Create an export in your terminal
              </span>
              <code>
                python -m rank_lab.cli export --workspace runs/my-study
                --destination runs/my-study/export
              </code>
              <p>
                Files stay in this browser. Nothing is uploaded, and opening a
                result does not start training.
              </p>
              <a
                href="https://github.com/shayanxcyber/rank-lab/blob/main/docs/CUSTOM-DATA.md"
                target="_blank"
                rel="noreferrer"
              >
                Set up a custom experiment <ArrowUpRight size={14} />
              </a>
            </div>
          </Dialog.Popup>
        </Dialog.Portal>
      </Dialog.Root>
    </div>
  );
}
