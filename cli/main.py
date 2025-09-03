"""Command-line interface stubs."""

from core.llm import LLMInterface


def main() -> None:
    """Entry point for the CLI."""
    llm = LLMInterface()
    print(llm.generate("Hello"))


if __name__ == "__main__":  # pragma: no cover - manual invocation
    main()
