import pytest

from common.refs import COMMA, build_text, read_reference_traces


def test_char_mode(tmp_path):
    f = tmp_path / "t.txt"
    f.write_text("abc\nabd\n", encoding="utf-8")
    assert read_reference_traces(f) == [["a", "b", "c"], ["a", "b", "d"]]


def test_whitespace_mode(tmp_path):
    f = tmp_path / "t.txt"
    f.write_text("check_in hand_hygiene out\n", encoding="utf-8")
    assert read_reference_traces(f) == [["check_in", "hand_hygiene", "out"]]


def test_delimiter_mode(tmp_path):
    f = tmp_path / "t.txt"
    f.write_text("a,b,c\n", encoding="utf-8")
    assert read_reference_traces(f, delimiter=",") == [["a", "b", "c"]]


def test_blank_lines_skipped(tmp_path):
    f = tmp_path / "t.txt"
    f.write_text("abc\n\n\nabd\n", encoding="utf-8")
    assert read_reference_traces(f) == [["a", "b", "c"], ["a", "b", "d"]]


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        read_reference_traces("does-not-exist-xyz.txt")


def test_reserved_separator_rejected(tmp_path):
    f = tmp_path / "t.txt"
    f.write_text("a,b\n", encoding="utf-8")   # char mode keeps the comma
    with pytest.raises(ValueError):
        read_reference_traces(f)


def test_empty_file(tmp_path):
    f = tmp_path / "t.txt"
    f.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_reference_traces(f)


def test_build_text_layout():
    text = build_text([["a", "b"], ["c"]])
    assert text == [COMMA, "a", "b", COMMA, "c", COMMA]
    assert text[0] == COMMA and text[-1] == COMMA
