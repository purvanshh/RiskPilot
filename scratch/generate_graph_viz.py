"""Phase 9 — Generate a visualisation of the LangGraph for the demo (Member D).

Produces, into docs/:
  - graph.png   (rendered PNG; needs mermaid.ink network access OR graphviz)
  - graph.mmd   (Mermaid source; always written, renders in any Mermaid viewer)
  - graph_ascii.txt (ASCII fallback; always written)

Run:  python scratch/generate_graph_viz.py
"""

import os
import sys

# Allow running from repo root: `python scratch/generate_graph_viz.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph.graph import graph  # noqa: E402

DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")


def main():
    os.makedirs(DOCS_DIR, exist_ok=True)
    g = graph.get_graph()

    # 1. Mermaid source — always works, no dependencies.
    mmd_path = os.path.join(DOCS_DIR, "graph.mmd")
    try:
        mermaid_src = g.draw_mermaid()
        with open(mmd_path, "w", encoding="utf-8") as f:
            f.write(mermaid_src)
        print(f"[ok] Mermaid source -> {mmd_path}")
    except Exception as e:
        print(f"[warn] could not write Mermaid source: {e}")

    # 2. ASCII fallback — always works.
    ascii_path = os.path.join(DOCS_DIR, "graph_ascii.txt")
    try:
        ascii_art = g.draw_ascii()
        with open(ascii_path, "w", encoding="utf-8") as f:
            f.write(ascii_art)
        print(f"[ok] ASCII diagram -> {ascii_path}")
    except Exception as e:
        print(f"[warn] could not write ASCII diagram: {e}")

    # 3. PNG — best for slides, but may need network or graphviz.
    png_path = os.path.join(DOCS_DIR, "graph.png")
    try:
        png_bytes = g.draw_mermaid_png()
        with open(png_path, "wb") as f:
            f.write(png_bytes)
        print(f"[ok] PNG -> {png_path}")
    except Exception as e:
        print(f"[warn] PNG render failed ({e}).")
        print("       Use docs/graph.mmd in a Mermaid viewer (e.g. mermaid.live) "
              "to export a PNG manually.")


if __name__ == "__main__":
    main()
