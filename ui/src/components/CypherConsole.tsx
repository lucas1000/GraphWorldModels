import { useState, useRef } from "react";
import type { CypherResult, PresetQuery } from "../lib/grid";

interface Props {
  onRunQuery: (query: string) => void;
  result: CypherResult | null;
  presets: PresetQuery[];
}

export function CypherConsole({ onRunQuery, result, presets }: Props) {
  const [query, setQuery] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleRun = () => {
    const q = query.trim();
    if (q) onRunQuery(q);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault();
      handleRun();
    }
  };

  return (
    <div className="cypher-panel">
      <h3>Cypher Console</h3>

      <div className="cypher-presets">
        {presets.map((p) => (
          <button
            key={p.label}
            onClick={() => {
              setQuery(p.query);
              textareaRef.current?.focus();
            }}
          >
            {p.label}
          </button>
        ))}
      </div>

      <div className="cypher-input-row">
        <textarea
          ref={textareaRef}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Enter Cypher query... (Cmd+Enter to run)"
          rows={2}
        />
        <button onClick={handleRun}>Run</button>
      </div>

      <div className="cypher-results">
        {result === null ? (
          <div className="cypher-empty">Run a query to see results</div>
        ) : result.error ? (
          <div className="cypher-error">{result.error}</div>
        ) : result.rows.length === 0 ? (
          <div className="cypher-empty">No results</div>
        ) : (
          <table>
            <thead>
              <tr>
                {result.columns.map((col) => (
                  <th key={col}>{col}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {result.rows.map((row, i) => (
                <tr key={i}>
                  {row.map((cell, j) => (
                    <td key={j} title={String(cell)}>
                      {typeof cell === "object" ? JSON.stringify(cell) : String(cell ?? "")}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
