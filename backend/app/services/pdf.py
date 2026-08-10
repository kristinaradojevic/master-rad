from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from weasyprint import HTML

from ..schemas import VerdictInput

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"

_env = Environment(
    loader=FileSystemLoader(_TEMPLATES_DIR),
    autoescape=select_autoescape(["html"]),
)


def render_verdict_pdf(data: VerdictInput, verdict_text: str) -> bytes:
    template = _env.get_template("verdict.html")
    html = template.render(input=data, verdict_text=verdict_text)
    return HTML(string=html).write_pdf()
