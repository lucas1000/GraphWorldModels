import type { StateField } from "../lib/grid";

interface Props {
  fields: StateField[];
  values: Record<string, unknown>;
}

function formatCell(kind: string, value: unknown): { text: string; className?: string } {
  if (value === null || value === undefined) return { text: "—" };
  if (kind === "list") {
    const arr = Array.isArray(value) ? value : [];
    return { text: arr.length ? arr.join(", ") : "(empty)" };
  }
  if (kind === "reward" && typeof value === "number") {
    const cls = value >= 0 ? "reward-pos" : "reward-neg";
    const txt = `${value >= 0 ? "+" : ""}${value.toFixed(1)}`;
    return { text: txt, className: cls };
  }
  if (kind === "number") {
    return { text: typeof value === "number" ? String(value) : String(value) };
  }
  // text / default
  return { text: String(value) };
}

export function StatusPanel({ fields, values }: Props) {
  return (
    <div className="status-panel">
      <h3>State</h3>
      {fields.length === 0 ? (
        <div className="status-empty">No state fields defined.</div>
      ) : (
        fields.map((f) => {
          const { text, className } = formatCell(f.kind, values[f.key]);
          return (
            <div className="status-row" key={f.key}>
              <span className="label">{f.label}</span>
              <span className={`value ${className ?? ""}`}>{text}</span>
            </div>
          );
        })
      )}
    </div>
  );
}
