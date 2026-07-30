"""Long-term memory store.

Append-only records with a dependency-free keyword retriever. The interface is
deliberately small so a local vector backend (FAISS/Chroma with local embeddings)
can be dropped in later without touching agents — but the default stays pure
stdlib so nothing needs installing in an airgap.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class MemoryRecord:
    id: str
    kind: str          # "fact" | "result" | "decision" | ...
    text: str
    source: str = ""
    tags: List[str] = field(default_factory=list)


def _tokens(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


class MemoryStore:
    """Keyword-scored long-term store persisted as JSONL."""

    def __init__(self, path: Optional[str] = None) -> None:
        self._path = Path(path) if path else None
        self._records: List[MemoryRecord] = []
        if self._path and self._path.exists():
            for line in self._path.read_text().splitlines():
                if line.strip():
                    self._records.append(MemoryRecord(**json.loads(line)))

    def add(self, record: MemoryRecord) -> None:
        self._records.append(record)
        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a") as fh:
                fh.write(json.dumps(asdict(record)) + "\n")

    def search(self, query: str, k: int = 5) -> List[MemoryRecord]:
        q = _tokens(query)
        if not q:
            return []
        scored = []
        for rec in self._records:
            overlap = len(q & _tokens(rec.text))
            if overlap:
                scored.append((overlap, rec))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [rec for _, rec in scored[:k]]

    def all(self) -> List[MemoryRecord]:
        return list(self._records)
