"""Prompt building for verdict generation.

This is where most of the experimentation for the thesis will happen:
system prompt wording, few-shot examples from real (anonymized) verdicts,
and later possibly retrieved similar verdicts (RAG).
"""

from datetime import date
from pathlib import Path

from ..schemas import VerdictInput

SYSTEM_PROMPT = """\
Ti si iskusan sudija u Republici Srbiji koji sastavlja presude u krivičnom postupku.
Pišeš na srpskom jeziku, formalnim pravnim stilom koji se koristi u srpskim sudovima.

Presuda mora da sadrži sledeće delove, ovim redosledom:
1. Uvod (naziv suda, sudija, podaci o okrivljenom, krivično delo, datum)
2. IZREKU (osuđujuća/oslobađajuća/odbijajuća odluka, jasno i precizno formulisana,
   uključujući izrečenu kaznu ili meru ako je osuđujuća)
3. Obrazloženje (tok postupka, utvrđeno činjenično stanje, ocena dokaza,
   pravna kvalifikacija dela, razlozi za odluku o kazni ili oslobađanju)
4. Pouku o pravnom leku

Koristi isključivo podatke koje ti je sudija dostavio. Ne izmišljaj činjenice,
imena, datume ni iznose koji nisu navedeni. Ako neki podatak nedostaje,
označi mesto sa [NEDOSTAJE PODATAK].
"""

# Up to a few hand-reviewed, anonymized real verdicts used as few-shot style
# examples. Loaded from disk (never hardcoded here) so real verdict text never
# ends up committed to git — see ml/data/fewshot/ (git-ignored) and the
# README section on working with real verdicts.
_FEWSHOT_DIR = Path(__file__).resolve().parents[3] / "ml" / "data" / "fewshot"
_MAX_FEWSHOT_EXAMPLES = 3


def _load_fewshot_examples() -> list[str]:
    if not _FEWSHOT_DIR.exists():
        return []
    files = sorted(_FEWSHOT_DIR.glob("*.txt"))[:_MAX_FEWSHOT_EXAMPLES]
    return [f.read_text(encoding="utf-8") for f in files]


def _format_serbian_date(value: str) -> str:
    """The frontend's date picker submits ISO 'YYYY-MM-DD'; verdicts are
    written in the Serbian 'DD.MM.GGGG.' style. Values not in ISO form
    (e.g. hand-typed dates from older callers) pass through unchanged."""
    try:
        d = date.fromisoformat(value)
    except ValueError:
        return value
    return f"{d.day:02d}.{d.month:02d}.{d.year}."


def build_user_prompt(data: VerdictInput) -> str:
    additional_info_line = (
        f"Dodatni podaci o okrivljenom: {data.defendant_additional_info}\n"
        if data.defendant_additional_info
        else ""
    )
    examples = _load_fewshot_examples()
    examples_block = ""
    if examples:
        joined = "\n\n---\n\n".join(examples)
        examples_block = f"""\
Primeri stvarnih presuda (anonimizovani, koristi ih samo kao uzor za stil,
strukturu i formalni jezik — ne za činjenice). Oznake u zagradama poput
[IME] ili [JMBG] su artefakti anonimizacije originalnih dokumenata, ne deo
uobičajenog stila presude — nikad ih ne preslikavaj u svoju presudu. Pominji
samo osobe i uloge (npr. sudije porotnike, zapisničara) koje su izričito
navedene u podacima ispod; ne izmišljaj dodatne osobe da bi pratio strukturu
primera:

{joined}

---

"""

    return f"""\
{examples_block}Sastavi presudu na osnovu sledećih podataka:

Sud: {data.court_name}
Broj predmeta: {data.case_number}
Sudija: {data.judge_name}
Zapisničar: {data.court_reporter_name}
Datum donošenja presude: {_format_serbian_date(data.verdict_date)}
Okrivljeni: {data.defendant_name}
JMBG: {data.defendant_jmbg}
Prebivalište: {data.defendant_residence}
{additional_info_line}Krivično delo: {data.criminal_offense}

Opis dela iz optužnog akta:
{data.charge_description}

Utvrđeno činjenično stanje:
{data.established_facts}

Vrsta odluke: {data.decision_type}

Kazna/mera ili obrazloženje odluke:
{data.sentence}
"""
