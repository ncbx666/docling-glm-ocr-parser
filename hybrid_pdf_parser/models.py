from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ParseSettings:
    images_scale: float = 2.0
    low_text_threshold: int = 80
    glm_max_new_tokens: int = 8192
    glm_model: str = "zai-org/GLM-OCR"
    glm_backend: str = "transformers"
    continue_on_glm_error: bool = False


@dataclass
class TargetRegion:
    region_id: str
    page_no: int
    order: int
    kind: str
    reason: str
    image_path: Path
    docling_ref: str | None = None
    label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["image_path"] = str(self.image_path)
        return data


@dataclass
class OcrResult:
    region_id: str
    backend: str
    prompt_kind: str
    text: str
    seconds: float
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DocumentArtifacts:
    root: Path
    pages_dir: Path
    regions_dir: Path
    tables_dir: Path


@dataclass
class DoclingParseResult:
    pdf_path: Path
    artifacts: DocumentArtifacts
    page_markdown: dict[int, str]
    page_text: dict[int, str]
    targets: list[TargetRegion]
    docling_seconds: float
    page_count: int

