"""Prompt building for verdict generation.

This is where most of the experimentation for the thesis will happen:
system prompt wording, few-shot examples from real (anonymized) verdicts,
and later possibly retrieved similar verdicts (RAG).
"""

from ..schemas import VerdictInput

SYSTEM_PROMPT = """\
Ti si iskusan sudija u Republici Srbiji koji sastavlja presude u parničnom postupku.
Pišeš na srpskom jeziku, formalnim pravnim stilom koji se koristi u srpskim sudovima.

Presuda mora da sadrži sledeće delove, ovim redosledom:
1. Uvod (naziv suda, sudija, stranke, predmet spora, datum)
2. IZREKU (odluka suda, jasno i precizno formulisana)
3. Obrazloženje (tok postupka, utvrđeno činjenično stanje, ocena dokaza,
   primena materijalnog prava)
4. Pouku o pravnom leku

Koristi isključivo podatke koje ti je sudija dostavio. Ne izmišljaj činjenice,
imena, datume ni iznose koji nisu navedeni. Ako neki podatak nedostaje,
označi mesto sa [NEDOSTAJE PODATAK].
"""


def build_user_prompt(data: VerdictInput) -> str:
    return f"""\
Sastavi presudu na osnovu sledećih podataka:

Sud: {data.court_name}
Broj predmeta: {data.case_number}
Sudija: {data.judge_name}
Tužilac: {data.plaintiff}
Tuženi: {data.defendant}
Datum presude: {data.verdict_date}

Tužbeni zahtev:
{data.claim_description}

Utvrđeno činjenično stanje:
{data.established_facts}

Odluka suda:
{data.decision}
"""
