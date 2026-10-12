import { Check } from "lucide-react";

export function RankMark() {
  return (
    <svg viewBox="0 0 48 48" aria-hidden="true">
      <path className="mark-bar" d="M9 33h8V15h-8z" fill="currentColor" />
      <path className="mark-bar" d="M20 33h8V9h-8z" fill="currentColor" />
      <path className="mark-bar" d="M31 33h8V21h-8z" fill="currentColor" />
    </svg>
  );
}

export function QueryGlyph({ change }: { change: number }) {
  return <span className="query-glyph" style={{ height: 5 + Math.abs(change) * 20 }} />;
}

export function DocumentIdentity({ title, judgment }: { title: string; judgment: number | undefined }) {
  return (
    <span className="document-title">
      {title}
      <small>
        {judgment !== undefined && judgment > 0 ? <><Check size={11} /> Judged relevant</> : judgment === 0 ? "Judged non-relevant" : "Unjudged"}
      </small>
    </span>
  );
}
