"""Pydantic schemas shared by the API.

VerdictInput holds the structured data the judge enters in the UI.
The exact set of sections is still being decided — extend this model
as the requirements become clear; the prompt builder and the PDF
template read from it.
"""

from pydantic import BaseModel, Field


class VerdictInput(BaseModel):
    court_name: str = Field(..., description="Naziv suda, npr. 'Osnovni sud u Novom Sadu'")
    case_number: str = Field(..., description="Broj predmeta, npr. 'P 123/2026'")
    judge_name: str = Field(..., description="Ime i prezime sudije")
    plaintiff: str = Field(..., description="Tužilac")
    defendant: str = Field(..., description="Tuženi")
    claim_description: str = Field(..., description="Opis tužbenog zahteva")
    established_facts: str = Field(..., description="Utvrđeno činjenično stanje")
    decision: str = Field(..., description="Odluka suda (osnov za izreku)")
    verdict_date: str = Field(..., description="Datum donošenja presude")


class GenerateRequest(BaseModel):
    input: VerdictInput
    model: str = Field(..., description="Model key from /api/models")


class GenerateResponse(BaseModel):
    model: str
    text: str


class ModelInfo(BaseModel):
    key: str
    provider: str
    label: str


class PdfRequest(BaseModel):
    input: VerdictInput
    # The generated (and possibly judge-edited) verdict body text.
    verdict_text: str
