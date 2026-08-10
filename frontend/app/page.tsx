"use client";

import { useEffect, useState } from "react";
import {
  VerdictInput,
  ModelInfo,
  fetchModels,
  generateVerdict,
  downloadPdf,
} from "@/lib/api";

const EMPTY_INPUT: VerdictInput = {
  court_name: "",
  case_number: "",
  judge_name: "",
  plaintiff: "",
  defendant: "",
  claim_description: "",
  established_facts: "",
  decision: "",
  verdict_date: "",
};

// Field definitions drive the form — extend here (and in backend
// schemas.py + prompt.py) when the final set of sections is decided.
const TEXT_FIELDS: { key: keyof VerdictInput; label: string }[] = [
  { key: "court_name", label: "Naziv suda" },
  { key: "case_number", label: "Broj predmeta" },
  { key: "judge_name", label: "Sudija" },
  { key: "plaintiff", label: "Tužilac" },
  { key: "defendant", label: "Tuženi" },
  { key: "verdict_date", label: "Datum presude" },
];

const TEXTAREA_FIELDS: { key: keyof VerdictInput; label: string }[] = [
  { key: "claim_description", label: "Tužbeni zahtev" },
  { key: "established_facts", label: "Utvrđeno činjenično stanje" },
  { key: "decision", label: "Odluka suda" },
];

export default function Home() {
  const [input, setInput] = useState<VerdictInput>(EMPTY_INPUT);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [model, setModel] = useState<string>("");
  const [verdictText, setVerdictText] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchModels()
      .then((m) => {
        setModels(m);
        if (m.length > 0) setModel(m[0].key);
      })
      .catch(() =>
        setError("Ne mogu da učitam listu modela — da li je backend pokrenut?")
      );
  }, []);

  const setField = (key: keyof VerdictInput, value: string) =>
    setInput((prev) => ({ ...prev, [key]: value }));

  const handleGenerate = async () => {
    setLoading(true);
    setError(null);
    try {
      setVerdictText(await generateVerdict(input, model));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Greška pri generisanju");
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = async () => {
    setError(null);
    try {
      await downloadPdf(input, verdictText);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Greška pri izradi PDF-a");
    }
  };

  return (
    <main className="mx-auto max-w-5xl p-8">
      <h1 className="mb-6 text-2xl font-bold">Generator presuda</h1>

      <div className="grid gap-8 md:grid-cols-2">
        {/* Input form */}
        <section className="space-y-4">
          <h2 className="text-lg font-semibold">Podaci o presudi</h2>

          {TEXT_FIELDS.map(({ key, label }) => (
            <label key={key} className="block">
              <span className="mb-1 block text-sm font-medium">{label}</span>
              <input
                className="w-full rounded border border-gray-300 px-3 py-2"
                value={input[key]}
                onChange={(e) => setField(key, e.target.value)}
              />
            </label>
          ))}

          {TEXTAREA_FIELDS.map(({ key, label }) => (
            <label key={key} className="block">
              <span className="mb-1 block text-sm font-medium">{label}</span>
              <textarea
                className="w-full rounded border border-gray-300 px-3 py-2"
                rows={4}
                value={input[key]}
                onChange={(e) => setField(key, e.target.value)}
              />
            </label>
          ))}

          <label className="block">
            <span className="mb-1 block text-sm font-medium">Model</span>
            <select
              className="w-full rounded border border-gray-300 px-3 py-2"
              value={model}
              onChange={(e) => setModel(e.target.value)}
            >
              {models.map((m) => (
                <option key={m.key} value={m.key}>
                  {m.label}
                </option>
              ))}
            </select>
          </label>

          <button
            className="rounded bg-blue-600 px-4 py-2 font-medium text-white disabled:opacity-50"
            onClick={handleGenerate}
            disabled={loading || !model}
          >
            {loading ? "Generisanje..." : "Generiši presudu"}
          </button>
        </section>

        {/* Result: review, edit, download */}
        <section className="space-y-4">
          <h2 className="text-lg font-semibold">
            Generisana presuda (proverite i izmenite pre preuzimanja)
          </h2>
          <textarea
            className="h-[32rem] w-full rounded border border-gray-300 px-3 py-2 font-serif"
            value={verdictText}
            onChange={(e) => setVerdictText(e.target.value)}
            placeholder="Ovde će se prikazati generisana presuda..."
          />
          <button
            className="rounded bg-green-700 px-4 py-2 font-medium text-white disabled:opacity-50"
            onClick={handleDownload}
            disabled={!verdictText}
          >
            Preuzmi PDF
          </button>
        </section>
      </div>

      {error && <p className="mt-4 text-red-600">{error}</p>}
    </main>
  );
}
