from fastapi import APIRouter, Response

from ..schemas import PdfRequest
from ..services.pdf import render_verdict_pdf

router = APIRouter(prefix="/api", tags=["pdf"])


@router.post("/pdf")
def download_pdf(request: PdfRequest):
    pdf_bytes = render_verdict_pdf(request.input, request.verdict_text)
    filename = f"presuda_{request.input.case_number.replace('/', '-').replace(' ', '_')}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
