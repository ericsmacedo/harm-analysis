# MIT License
#
# Copyright (c) 2025-2026 ericsmacedo
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Generate llms.txt and llms-full.txt (https://llmstxt.org) from the Sphinx text build.

Usage: python docs/make_llms_txt.py TEXT_DIR OUT_DIR

TEXT_DIR is the output of ``sphinx-build -b text docs TEXT_DIR``; OUT_DIR is the HTML
output directory, where both files are written next to the HTML pages.
"""

import sys
from itertools import pairwise
from pathlib import Path

TITLE = "harm-analysis"
SUMMARY = (
    "Python package that uses the FFT to estimate signal metrics: SNR, THD, SINAD and THD+N "
    "for signals with an injected tone (harm_analysis), and DC, tones and noise for signals "
    "without one (spec_analysis)."
)
DETAILS = (
    "Install with `pip install harm-analysis`. Both functions take a 1-D real signal `x` and a "
    "sampling frequency `fs` and return a result object (`HarmResult` / `SpecResult`) whose "
    "attributes hold the metrics; `to_dict()` returns them as a dictionary. The command line "
    "tools `harm-analysis` and `spec-analysis` run the same analyses on a text file."
)

# Pages in the order of the docs navigation, grouped as in the llms.txt sections.
SECTIONS = {
    "Docs": {
        "index": "Overview and installation",
        "usage": "How to use harm_analysis and spec_analysis, with examples and their output",
    },
    "API": {
        "api": "Reference for harm_analysis, spec_analysis, HarmResult and SpecResult",
        "cli": "Reference for the harm-analysis and spec-analysis command line tools",
    },
    "Optional": {
        "changelog": "Changes per release",
        "contributing": "How to set up the project and contribute",
    },
}


def _title(text: str, default: str) -> str:
    """Return the first heading of a Sphinx text page (a line underlined with ``*``, ``=``, ...)."""
    for line, underline in pairwise(text.splitlines()):
        marker = underline.strip()
        if line.strip() and len(marker) >= 3 and len(set(marker)) == 1 and marker[0] in "*=-~^#":  # noqa: PLR2004
            return line.strip()
    return default


def main(text_dir: Path, out_dir: Path) -> None:
    """Write llms.txt and llms-full.txt into out_dir."""
    index = [f"# {TITLE}", "", f"> {SUMMARY}", "", DETAILS, ""]
    full = [f"# {TITLE}", "", f"> {SUMMARY}", ""]

    for section, pages in SECTIONS.items():
        index += [f"## {section}", ""]
        for page, description in pages.items():
            text = (text_dir / f"{page}.txt").read_text(encoding="utf-8").strip()
            index.append(f"- [{_title(text, page)}]({page}.html): {description}")
            full += ["", "-" * 80, f"Source: {page}.html", "-" * 80, "", text, ""]
        index.append("")

    index += ["## Full documentation", "", "- [llms-full.txt](llms-full.txt): All pages above in one file", ""]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "llms.txt").write_text("\n".join(index), encoding="utf-8")
    (out_dir / "llms-full.txt").write_text("\n".join(full), encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 3:  # noqa: PLR2004
        sys.exit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]))
