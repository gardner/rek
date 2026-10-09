"""Enforce the project's 500-line limit without extra dependencies."""

from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = [root / "main.py", *root.glob("src/**/*.py"), *root.glob("scripts/*.py")]
    too_long = [path for path in paths if len(path.read_text().splitlines()) > 500]
    if too_long:
        raise SystemExit("Python files exceed 500 lines: " + ", ".join(map(str, too_long)))
    print("All Python source files are within 500 lines.")


if __name__ == "__main__":
    main()
