import { useMemo, useState } from "react";
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  getPaginationRowModel,
  flexRender,
  type ColumnDef,
  type SortingState,
} from "@tanstack/react-table";
import {
  ArrowDownUp,
  ArrowDown,
  ArrowUp,
  ArrowUpRight,
  ChevronLeft,
  ChevronRight,
  Search,
  X,
} from "lucide-react";
import { Segments } from "./ui";
import { selectQueries } from "./data";
import {
  delta,
  score,
  points,
  outcome,
  methodName,
  type Query,
  type Metric,
  type Method,
} from "./types";

export function QueryTable({
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
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState("all");
  const [sorting, setSorting] = useState<SortingState>([
    { id: "change", desc: true },
  ]);
  const data = useMemo(
    () => selectQueries(queries, search, filter, metric, baseline),
    [queries, search, filter, metric, baseline],
  );
  const columns = useMemo<ColumnDef<Query>[]>(
    () => [
      {
        id: "query",
        header: "Reserved query",
        accessorFn: (q) => q.text,
        cell: ({ row }) => (
          <button
            className="table-query"
            onClick={() => onOpen(row.original)}
            data-query-trigger={row.original.id}
          >
            <span className="query-id">
              Q{row.original.id.padStart(4, "0")}
            </span>
            <span>{row.original.text}</span>
            <ArrowUpRight size={18} />
          </button>
        ),
      },
      {
        id: "baseline",
        header: methodName[baseline],
        accessorFn: (q) => q.metrics[baseline][metric],
        cell: ({ getValue }) => (
          <span className="numeric">{score(getValue<number>())}</span>
        ),
      },
      {
        id: "fine",
        header: "Fine-tuned",
        accessorFn: (q) => q.metrics.fine_tuned[metric],
        cell: ({ getValue }) => (
          <span className="numeric">{score(getValue<number>())}</span>
        ),
      },
      {
        id: "change",
        header: "Change (pp)",
        accessorFn: (q) => delta(q, metric, baseline),
        cell: ({ getValue }) => (
          <span className={`numeric change ${outcome(getValue<number>())}`}>
            {points(getValue<number>())}
          </span>
        ),
      },
    ],
    [metric, baseline, onOpen],
  );
  const table = useReactTable({
    data,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    initialState: { pagination: { pageSize: 10 } },
    autoResetPageIndex: true,
  });
  return (
    <>
      <div className="query-toolbar">
        <label className="query-search">
          <Search size={18} />
          <input
            aria-label="Search test queries"
            placeholder="Find a query, phrase or ID…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
          {search && (
            <button
              aria-label="Clear query search"
              onClick={() => setSearch("")}
            >
              <X size={16} />
            </button>
          )}
        </label>
        <Segments
          label="Query outcome"
          value={filter}
          onChange={(value) => {
            setFilter(value);
            if (value === "loss" || value === "gain")
              setSorting([{ id: "change", desc: value === "gain" }]);
          }}
          className="filter-tabs"
          items={[
            { value: "all", label: "All" },
            { value: "gain", label: "Gains" },
            { value: "loss", label: "Regressions" },
            { value: "same", label: "Unchanged" },
          ]}
        />
      </div>
      <div className="table-scroll">
        <table className="query-table">
          <thead>
            {table.getHeaderGroups().map((group) => (
              <tr key={group.id}>
                {group.headers.map((header) => (
                  <th
                    key={header.id}
                    scope="col"
                    aria-sort={
                      header.column.getIsSorted() === "asc"
                        ? "ascending"
                        : header.column.getIsSorted() === "desc"
                          ? "descending"
                          : "none"
                    }
                  >
                    <button onClick={header.column.getToggleSortingHandler()}>
                      {flexRender(
                        header.column.columnDef.header,
                        header.getContext(),
                      )}
                      {header.column.getIsSorted() === "asc" ? (
                        <ArrowUp size={13} />
                      ) : header.column.getIsSorted() === "desc" ? (
                        <ArrowDown size={13} />
                      ) : (
                        <ArrowDownUp size={13} />
                      )}
                    </button>
                  </th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody>
            {table.getRowModel().rows.map((row) => (
              <tr key={row.original.id}>
                {row.getVisibleCells().map((cell) => (
                  <td
                    key={cell.id}
                    data-label={
                      typeof cell.column.columnDef.header === "string"
                        ? cell.column.columnDef.header
                        : ""
                    }
                  >
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {!data.length && (
        <div className="empty-state">
          <span className="eyebrow">No matching queries</span>
          <h3>Try a different phrase.</h3>
          <p>
            Your filters only affect the view. The recorded study is unchanged.
          </p>
          <button
            className="text-button"
            onClick={() => {
              setSearch("");
              setFilter("all");
            }}
          >
            Reset search and filters <ArrowUpRight size={18} />
          </button>
        </div>
      )}
      <div className="table-footer">
        <span role="status" aria-live="polite">
          {data.length
            ? `${table.getState().pagination.pageIndex * 10 + 1}–${Math.min((table.getState().pagination.pageIndex + 1) * 10, data.length)} of ${data.length} queries`
            : "0 queries"}
        </span>
        <a href="./data/queries.csv" download>
          Download results CSV ↗
        </a>
        <div className="pagination">
          <button
            aria-label="Previous page"
            disabled={!table.getCanPreviousPage()}
            onClick={() => table.previousPage()}
          >
            <ChevronLeft size={18} />
          </button>
          <span>
            {data.length ? table.getState().pagination.pageIndex + 1 : 0} /{" "}
            {table.getPageCount()}
          </span>
          <button
            aria-label="Next page"
            disabled={!table.getCanNextPage()}
            onClick={() => table.nextPage()}
          >
            <ChevronRight size={18} />
          </button>
        </div>
      </div>
    </>
  );
}
