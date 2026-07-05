#!/usr/bin/env python3
"""Generate production RAG embedding decision tree as Excalidraw JSON."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "docs" / "production-embedding-decision-tree.excalidraw"

# LinkedIn-style palette
C_TITLE_BG = "#1e3a5f"
C_TITLE_FG = "#ffffff"
C_DECISION = "#fff3cd"
C_DECISION_STROKE = "#856404"
C_OUTCOME = "#d4edda"
C_OUTCOME_STROKE = "#155724"
C_API = "#cce5ff"
C_API_STROKE = "#004085"
C_HYBRID = "#e2d5f5"
C_HYBRID_STROKE = "#5a3d8a"
C_WARN = "#f8d7da"
C_WARN_STROKE = "#721c24"
C_RULE = "#f5f5f5"
C_ARROW = "#495057"
C_LABEL = "#212529"


def _nid() -> str:
    return format(random.getrandbits(64), "x")[:16]


def _base(el_id: str, etype: str, x: float, y: float, w: float, h: float, **kw) -> dict:
    return {
        "id": el_id,
        "type": etype,
        "x": x,
        "y": y,
        "width": w,
        "height": h,
        "angle": 0,
        "strokeColor": kw.get("stroke", "#1e1e1e"),
        "backgroundColor": kw.get("bg", "transparent"),
        "fillStyle": "solid",
        "strokeWidth": kw.get("sw", 2),
        "strokeStyle": "solid",
        "roughness": 0,
        "opacity": 100,
        "groupIds": [],
        "frameId": None,
        "roundness": {"type": 3} if etype == "rectangle" else None,
        "seed": random.randint(1, 2**31),
        "version": 1,
        "versionNonce": random.randint(1, 2**31),
        "isDeleted": False,
        "boundElements": [],
        "updated": int(time.time() * 1000),
        "link": None,
        "locked": False,
    }


def rect(x, y, w, h, label: str, bg: str, stroke: str, fs: int = 16) -> list[dict]:
    rid = _nid()
    tid = _nid()
    box = _base(rid, "rectangle", x, y, w, h, bg=bg, stroke=stroke)
    box["boundElements"] = [{"id": tid, "type": "text"}]
    text = _base(tid, "text", x + 8, y + h / 2 - fs / 2, w - 16, fs + 8, stroke=C_LABEL)
    text.update(
        {
            "text": label,
            "fontSize": fs,
            "fontFamily": 1,
            "textAlign": "center",
            "verticalAlign": "middle",
            "containerId": rid,
            "originalText": label,
            "autoResize": True,
            "lineHeight": 1.25,
        }
    )
    return [box, text]


def diamond(x, y, size: float, label: str, fs: int = 15) -> list[dict]:
    did = _nid()
    tid = _nid()
    d = _base(did, "diamond", x, y, size, size, bg=C_DECISION, stroke=C_DECISION_STROKE)
    d["boundElements"] = [{"id": tid, "type": "text"}]
    text = _base(tid, "text", x + size * 0.12, y + size * 0.38, size * 0.76, size * 0.24, stroke=C_LABEL)
    text.update(
        {
            "text": label,
            "fontSize": fs,
            "fontFamily": 1,
            "textAlign": "center",
            "verticalAlign": "middle",
            "containerId": did,
            "originalText": label,
            "autoResize": True,
            "lineHeight": 1.2,
        }
    )
    return [d, text]


def arrow(x1, y1, x2, y2, label: str = "") -> list[dict]:
    aid = _nid()
    w = x2 - x1
    h = y2 - y1
    a = _base(aid, "arrow", x1, y1, w, h, stroke=C_ARROW, sw=2)
    a.update(
        {
            "points": [[0, 0], [w, h]],
            "lastCommittedPoint": None,
            "startBinding": None,
            "endBinding": None,
            "startArrowhead": None,
            "endArrowhead": "arrow",
            "elbowed": False,
        }
    )
    els = [a]
    if label:
        tid = _nid()
        tx = x1 + w / 2 - 40
        ty = y1 + h / 2 - 20
        text = _base(tid, "text", tx, ty, 80, 20, stroke="#6c757d")
        text.update(
            {
                "text": label,
                "fontSize": 13,
                "fontFamily": 1,
                "textAlign": "center",
                "verticalAlign": "middle",
                "containerId": None,
                "originalText": label,
                "autoResize": True,
            }
        )
        els.append(text)
    return els


def build() -> dict:
    els: list[dict] = []

    # Title banner
    els += rect(40, 20, 1120, 56, "Production RAG — Embedding Decision Tree", C_TITLE_BG, C_TITLE_BG, fs=22)
    els[-1]["strokeColor"] = C_TITLE_FG
    els[-2]["boundElements"] = [{"id": els[-1]["id"], "type": "text"}]
    els[-1].update({"strokeColor": C_TITLE_FG, "text": "Production RAG — Embedding Decision Tree (no fine-tuning path)"})

    els += rect(40, 86, 1120, 36, "Golden rule: same embedder for index + query  •  one index = one recipe  •  dim is a storage knob, not quality", C_RULE, "#adb5bd", fs=14)

    # Level 0 — modality
    els += rect(460, 140, 280, 48, "START: What content are you embedding?", "#e9ecef", "#495057", fs=15)
    d_size = 130

    # Modality diamond
    els += diamond(535, 210, d_size, "Content\ntype?", fs=14)

    # Branch outcomes row 1
    els += rect(80, 400, 200, 70, "TEXT\n(general RAG)", C_OUTCOME, C_OUTCOME_STROKE, fs=15)
    els += rect(340, 400, 200, 70, "IMAGES / PDF\ncharts & figures", C_API, C_API_STROKE, fs=14)
    els += rect(600, 400, 200, 70, "SOURCE CODE\nrepos, APIs, SQL", C_API, C_API_STROKE, fs=14)
    els += rect(860, 400, 200, 70, "MULTILINGUAL\ntext", C_OUTCOME, C_OUTCOME_STROKE, fs=14)

    els += arrow(535, 340, 180, 400, "text")
    els += arrow(600, 340, 440, 400, "visual")
    els += arrow(665, 340, 700, 400, "code")
    els += arrow(730, 340, 960, 400, "multi-L")

    # Text path — domain diamond
    els += diamond(115, 510, 120, "Domain\njargon?", fs=13)
    els += rect(20, 680, 210, 80, "Domain API or\npre-trained domain model\nVoyage-law, PubMedBERT", C_API, C_API_STROKE, fs=13)
    els += rect(250, 680, 130, 56, "Strong general +\nreranker", C_OUTCOME, C_OUTCOME_STROKE, fs=12)

    els += arrow(180, 400, 175, 510)
    els += arrow(115, 630, 115, 680, "yes")
    els += arrow(235, 570, 315, 680, "no")

    # Budget diamond
    els += diamond(400, 510, 130, "Latency &\ncost?", fs=13)
    els += rect(300, 680, 155, 64, "CPU / offline\nMiniLM, bge-small\n384d", C_OUTCOME, C_OUTCOME_STROKE, fs=12)
    els += rect(480, 680, 155, 64, "Balanced local\ne5-base, bge-base\n768d", C_OUTCOME, C_OUTCOME_STROKE, fs=12)
    els += rect(660, 680, 155, 64, "Cloud API\nOpenAI, Cohere\nVoyage", C_API, C_API_STROKE, fs=12)

    els += arrow(380, 680, 465, 640)
    els += arrow(465, 640, 557, 680)
    els += arrow(557, 640, 737, 680)

    # Query style diamond
    els += diamond(400, 780, 130, "Query\nstyle?", fs=13)
    els += rect(200, 940, 175, 72, "Paraphrased NL\nquestions\n→ Instruct E5/BGE\nquery: / passage:", C_OUTCOME, C_OUTCOME_STROKE, fs=12)
    els += rect(400, 940, 175, 72, "Exact IDs / SKUs\nproduct codes\n→ SPLADE + dense\nor hybrid index", C_HYBRID, C_HYBRID_STROKE, fs=12)
    els += rect(600, 940, 175, 72, "Mixed signals\n→ dual-index\nsparse + dense + RRF", C_HYBRID, C_HYBRID_STROKE, fs=12)

    els += arrow(465, 744, 465, 780)
    els += arrow(400, 910, 287, 940, "NL Q")
    els += arrow(465, 910, 487, 940, "exact")
    els += arrow(530, 910, 687, 940, "both")

    # Index size
    els += diamond(400, 1060, 120, "Index\nsize?", fs=13)
    els += rect(180, 1220, 150, 56, "< 100K chunks\nany dim OK", C_OUTCOME, C_OUTCOME_STROKE, fs=12)
    els += rect(360, 1220, 150, 56, "1M+ chunks\n≤768d or\nMatryoshka", C_OUTCOME, C_OUTCOME_STROKE, fs=12)
    els += rect(540, 1220, 150, 56, "10M+ chunks\n≤384d + HNSW\n+ quantization", C_OUTCOME, C_OUTCOME_STROKE, fs=12)

    els += arrow(465, 1012, 465, 1060)
    els += arrow(400, 1180, 255, 1220)
    els += arrow(460, 1180, 435, 1220)
    els += arrow(520, 1180, 615, 1220)

    # Modality-specific outcomes
    els += rect(340, 500, 200, 64, "CLIP / SigLIP\ntext + image same space\n512d", C_API, C_API_STROKE, fs=12)
    els += rect(600, 500, 200, 64, "jina-code\nunixcoder\n768d", C_API, C_API_STROKE, fs=12)
    els += rect(860, 500, 200, 64, "multilingual-e5\nbge-m3\n768–1024d", C_API, C_API_STROKE, fs=12)

    els += arrow(440, 470, 440, 500)
    els += arrow(700, 470, 700, 500)
    els += arrow(960, 470, 960, 500)

    # Reranker gate
    els += diamond(400, 1320, 140, "Top-3 still\nwrong?", fs=13)
    els += rect(300, 1500, 180, 64, "Ship bi-encoder\nRecall@5 good enough", C_OUTCOME, C_OUTCOME_STROKE, fs=13)
    els += rect(520, 1500, 220, 64, "Add cross-encoder reranker\nbge-reranker on top-20", C_WARN, C_WARN_STROKE, fs=12)

    els += arrow(465, 1276, 465, 1320)
    els += arrow(400, 1460, 390, 1500, "no")
    els += arrow(530, 1460, 630, 1500, "yes")

    # Legend
    els += rect(900, 680, 260, 200, "", "#fafafa", "#dee2e6", fs=12)
    legend_id = els[-2]["id"]
    legend_items = [
        ("◆ Decision gate", C_DECISION),
        ("▢ Dense / local outcome", C_OUTCOME),
        ("▢ API / multimodal", C_API),
        ("▢ Hybrid / sparse+dense", C_HYBRID),
        ("▢ Reranker layer", C_WARN),
    ]
    ly = 700
    for label, color in legend_items:
        lid = _nid()
        box = _base(lid, "rectangle", 920, ly, 24, 24, bg=color, stroke="#6c757d", sw=1)
        els.append(box)
        tid = _nid()
        t = _base(tid, "text", 952, ly, 190, 24, stroke=C_LABEL)
        t.update(
            {
                "text": label,
                "fontSize": 13,
                "fontFamily": 1,
                "textAlign": "left",
                "verticalAlign": "middle",
                "containerId": None,
                "originalText": label,
                "autoResize": True,
            }
        )
        els.append(t)
        ly += 32

    ltid = _nid()
    lt = _base(ltid, "text", 920, 668, 200, 24, stroke=C_LABEL)
    lt.update(
        {
            "text": "LEGEND",
            "fontSize": 14,
            "fontFamily": 1,
            "textAlign": "left",
            "verticalAlign": "middle",
            "containerId": legend_id,
            "originalText": "LEGEND",
            "autoResize": True,
        }
    )
    els.append(lt)

    # Eval footer
    els += rect(40, 1600, 1120, 72,
        "Before shipping: 20–50 real queries → Recall@5 / MRR → A/B 2–3 candidates → pick faster tie-breaker",
        "#fff9e6", "#ffc107", fs=14)

    return {
        "type": "excalidraw",
        "version": 2,
        "source": "https://excalidraw.com",
        "elements": els,
        "appState": {
            "gridSize": 20,
            "viewBackgroundColor": "#ffffff",
            "scrollX": 0,
            "scrollY": 0,
            "zoom": {"value": 0.75},
        },
        "files": {},
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data = build()
    OUT.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
