const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface VerdictInput {
  court_name: string;
  case_number: string;
  judge_name: string;
  plaintiff: string;
  defendant: string;
  claim_description: string;
  established_facts: string;
  decision: string;
  verdict_date: string;
}

export interface ModelInfo {
  key: string;
  provider: string;
  label: string;
}

export async function fetchModels(): Promise<ModelInfo[]> {
  const res = await fetch(`${API_BASE}/api/models`);
  if (!res.ok) throw new Error("Failed to load models");
  return res.json();
}

export async function generateVerdict(
  input: VerdictInput,
  model: string
): Promise<string> {
  const res = await fetch(`${API_BASE}/api/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ input, model }),
  });
  if (!res.ok) throw new Error(`Generation failed (${res.status})`);
  const data = await res.json();
  return data.text;
}

export async function downloadPdf(
  input: VerdictInput,
  verdictText: string
): Promise<void> {
  const res = await fetch(`${API_BASE}/api/pdf`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ input, verdict_text: verdictText }),
  });
  if (!res.ok) throw new Error(`PDF generation failed (${res.status})`);
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `presuda_${input.case_number.replace(/[/\s]/g, "-")}.pdf`;
  a.click();
  URL.revokeObjectURL(url);
}
