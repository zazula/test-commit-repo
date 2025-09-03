"""Simple memory logging stubs."""

from dataclasses import dataclass, field


@dataclass
class MemoryLog:
    """Store interaction traces."""

    entries: list[str] = field(default_factory=list)

    def append(self, entry: str) -> None:
        self.entries.append(entry)
