#!/usr/bin/env python3
"""Read one case's expansion out of a MiniMax-H3-Image-PE evidence document (SW-7104, AT9).

The document is `core_llm::minimax_h3_image_pe::quality::runs_to_json`'s output, written by the
`minimax-h3-image-pe` profile's PE jobs. Writes the case's `expanded_caption` to OUT as UTF-8 and
prints the document's tier label. Exits non-zero when the case is missing or did not parse.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print(f'usage: {argv[0]} EVIDENCE.json CASE_ID OUT.txt', file=sys.stderr)
        return 2
    evidence, case_id, out = Path(argv[1]), argv[2], Path(argv[3])
    doc = json.loads(evidence.read_text(encoding='utf-8'))
    cases = [case for case in doc.get('cases', []) if case.get('id') == case_id]
    if len(cases) != 1:
        print(f'{evidence}: expected one case {case_id!r}, found {len(cases)}', file=sys.stderr)
        return 1
    caption = cases[0].get('expanded_caption')
    if not isinstance(caption, str) or not caption.strip():
        print(f'{evidence}: case {case_id!r} has no expansion: {cases[0].get("error")!r}',
              file=sys.stderr)
        return 1
    out.write_text(caption, encoding='utf-8')
    print(doc.get('label', '?'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
