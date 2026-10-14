import { useEffect, useState } from "react";
import { Download } from "lucide-react";

export function DownloadData({
  content,
  filename,
  label,
  type = "application/json",
}: {
  content: string;
  filename: string;
  label: string;
  type?: string;
}) {
  const [url, setUrl] = useState("");
  useEffect(() => {
    const next = URL.createObjectURL(new Blob([content], { type }));
    setUrl(next);
    return () => URL.revokeObjectURL(next);
  }, [content, type]);
  return (
    <a
      className="text-button"
      href={url || undefined}
      download={filename}
      aria-disabled={!url}
    >
      {label} <Download size={16} />
    </a>
  );
}
