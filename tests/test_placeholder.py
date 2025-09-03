"""Placeholder test suite."""

from cli.main import main


def test_main_output(capsys):
    """CLI main should print mocked LLM response."""
    main()
    captured = capsys.readouterr()
    assert "LLM response" in captured.out
