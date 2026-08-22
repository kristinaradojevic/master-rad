"""Pydantic schemas shared by the API.

VerdictInput holds the structured data the judge enters in the UI.
The exact set of sections is still being decided — extend this model
as the requirements become clear; the prompt builder and the PDF
template read from it.
"""

from pydantic import BaseModel, Field


class VerdictInput(BaseModel):
    court_name: str = Field(..., description="Naziv suda, npr. 'Osnovni sud u Novom Sadu'")
    case_number: str = Field(..., description="Broj predmeta, npr. 'K 45/2026'")
    judge_name: str = Field(..., description="Ime i prezime sudije")
    court_reporter_name: str = Field(..., description="Ime i prezime zapisničara")
    verdict_date: str = Field(..., description="Datum donošenja presude")
    defendant_name: str = Field(..., description="Ime i prezime okrivljenog")
    defendant_jmbg: str = Field(
        ..., pattern=r"^\d{13}$", description="JMBG okrivljenog (13 cifara)"
    )
    defendant_residence: str = Field(..., description="Prebivalište okrivljenog")
    defendant_additional_info: str = Field(
        default="",
        description=(
            "Dodatni lični podaci o okrivljenom (npr. zanimanje, mesto rođenja, "
            "porodično stanje, ranija osuđivanost) — po potrebi"
        ),
    )
    criminal_offense: str = Field(
        ..., description="Krivično delo i zakonska kvalifikacija, npr. 'čl. 246a st. 1 KZ'"
    )
    charge_description: str = Field(..., description="Činjenični opis dela iz optužnog akta")
    established_facts: str = Field(..., description="Utvrđeno činjenično stanje i ocena dokaza")
    decision_type: str = Field(
        ..., description="Vrsta odluke: 'osuđujuća', 'oslobađajuća' ili 'odbijajuća'"
    )
    sentence: str = Field(
        ..., description="Izrečena kazna/mera, ili obrazloženje oslobađanja/odbijanja"
    )


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
