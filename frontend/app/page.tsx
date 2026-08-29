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
  court_reporter_name: "",
  verdict_date: "",
  defendant_name: "",
  defendant_jmbg: "",
  defendant_residence: "",
  defendant_additional_info: "",
  criminal_offense: "",
  charge_description: "",
  established_facts: "",
  decision_type: "",
  sentence: "",
};

type FieldDef = {
  key: keyof VerdictInput;
  label: string;
  type: "text" | "date" | "textarea" | "select";
  pattern?: RegExp;
  errorMessage?: string;
  options?: { value: string; label: string }[];
};

const DECISION_TYPES = [
  { value: "osuđujuća", label: "Osuđujuća" },
  { value: "oslobađajuća", label: "Oslobađajuća" },
  { value: "odbijajuća", label: "Odbijajuća" },
];

// Fields grouped by meaning and rendered as cards — extend a group here
// (and in backend schemas.py + prompt.py) when the final set of sections
// is decided. `pattern`/`errorMessage` add inline validation for fields
// with a fixed format.
const SECTIONS: { title: string; fields: FieldDef[] }[] = [
  {
    title: "Sud i predmet",
    fields: [
      { key: "court_name", label: "Naziv suda", type: "text" },
      { key: "case_number", label: "Broj predmeta", type: "text" },
      { key: "judge_name", label: "Sudija", type: "text" },
      { key: "court_reporter_name", label: "Zapisničar", type: "text" },
      { key: "verdict_date", label: "Datum donošenja presude", type: "date" },
    ],
  },
  {
    title: "Okrivljeni",
    fields: [
      { key: "defendant_name", label: "Okrivljeni", type: "text" },
      {
        key: "defendant_jmbg",
        label: "JMBG okrivljenog",
        type: "text",
        pattern: /^\d{13}$/,
        errorMessage: "JMBG mora imati tačno 13 cifara",
      },
      { key: "defendant_residence", label: "Prebivalište okrivljenog", type: "text" },
      {
        key: "defendant_additional_info",
        label:
          "Dodatni lični podaci (zanimanje, mesto rođenja, porodično stanje, ranija osuđivanost...)",
        type: "textarea",
      },
    ],
  },
  {
    title: "Krivično delo i odluka",
    fields: [
      { key: "criminal_offense", label: "Krivično delo", type: "text" },
      { key: "decision_type", label: "Vrsta odluke", type: "select", options: DECISION_TYPES },
      { key: "charge_description", label: "Opis dela iz optužnog akta", type: "textarea" },
      { key: "established_facts", label: "Utvrđeno činjenično stanje", type: "textarea" },
      { key: "sentence", label: "Kazna/mera ili obrazloženje odluke", type: "textarea" },
    ],
  },
];

function fieldError(
  field: { pattern?: RegExp; errorMessage?: string },
  value: string
): string | null {
  if (field.pattern && value && !field.pattern.test(value)) {
    return field.errorMessage ?? "Neispravan format";
  }
  return null;
}

function FormField({
  field,
  value,
  onChange,
}: {
  field: FieldDef;
  value: string;
  onChange: (value: string) => void;
}) {
  const err = fieldError(field, value);
  const className = `w-full rounded border px-3 py-2 ${
    err ? "border-red-500" : "border-gray-300"
  }`;

  return (
    <label className="block">
      <span className="mb-1 block text-sm font-medium">{field.label}</span>
      {field.type === "date" ? (
        <input
          type="date"
          className={className}
          value={value}
          onChange={(e) => onChange(e.target.value)}
        />
      ) : field.type === "textarea" ? (
        <textarea
          className={className}
          rows={3}
          value={value}
          onChange={(e) => onChange(e.target.value)}
        />
      ) : field.type === "select" ? (
        <select className={className} value={value} onChange={(e) => onChange(e.target.value)}>
          <option value="">Izaberite...</option>
          {field.options?.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
        </select>
      ) : (
        <input className={className} value={value} onChange={(e) => onChange(e.target.value)} />
      )}
      {err && <span className="mt-1 block text-sm text-red-600">{err}</span>}
    </label>
  );
}

export default function Home() {
  const [input, setInput] = useState<VerdictInput>(EMPTY_INPUT);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [model, setModel] = useState<string>("");
  const [useRag, setUseRag] = useState(false);
  const [numExamples, setNumExamples] = useState(3);
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

  const hasErrors = SECTIONS.some((section) =>
    section.fields.some((field) => fieldError(field, input[field.key]) !== null)
  );

  const handleGenerate = async () => {
    setLoading(true);
    setError(null);
    try {
      setVerdictText(await generateVerdict(input, model, useRag, numExamples));
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
    <main className="mx-auto max-w-6xl p-8">
      <h1 className="mb-6 text-2xl font-bold">Generator presuda</h1>

      {/* Input form, grouped by meaning */}
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {SECTIONS.map((section) => (
          <section key={section.title} className="space-y-3 rounded-lg border border-gray-200 p-4">
            <h2 className="text-lg font-semibold">{section.title}</h2>
            {section.fields.map((field) => (
              <FormField
                key={field.key}
                field={field}
                value={input[field.key]}
                onChange={(v) => setField(field.key, v)}
              />
            ))}
          </section>
        ))}
      </div>

      <div className="mt-6 flex flex-wrap items-end gap-4">
        <label className="block">
          <span className="mb-1 block text-sm font-medium">Model</span>
          <select
            className="rounded border border-gray-300 px-3 py-2"
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

        <label className="flex items-center gap-2 pb-2">
          <input
            type="checkbox"
            checked={useRag}
            onChange={(e) => setUseRag(e.target.checked)}
          />
          <span className="text-sm font-medium">
            Koristi RAG (primeri po sličnosti umesto fiksnog seta)
          </span>
        </label>

        <label className="block">
          <span className="mb-1 block text-sm font-medium">Broj primera</span>
          <input
            type="number"
            min={1}
            max={10}
            disabled={!useRag}
            className="w-20 rounded border border-gray-300 px-3 py-2 disabled:opacity-50"
            value={numExamples}
            onChange={(e) => setNumExamples(Number(e.target.value))}
          />
        </label>

        <button
          className="rounded bg-blue-600 px-4 py-2 font-medium text-white disabled:opacity-50"
          onClick={handleGenerate}
          disabled={loading || !model || hasErrors}
        >
          {loading ? "Generisanje..." : "Generiši presudu"}
        </button>
      </div>

      {error && <p className="mt-4 text-red-600">{error}</p>}

      {/* Result: review, edit, download */}
      <section className="mt-8 space-y-4">
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
    </main>
  );
}
