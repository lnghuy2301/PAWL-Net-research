"""Create a source-only ZIP without local data, weights, backups or Git history."""

import argparse
import hashlib
import zipfile
from pathlib import Path


def source_files(root):
    root_files = (
        ".gitignore",
        ".gitattributes",
        ".env.example",
        "README.md",
        "CONTRIBUTING.md",
        "CITATION.cff",
        "pyproject.toml",
        "requirements.txt",
    )
    for name in root_files:
        yield root / name
    allowed = {".py", ".ipynb", ".md", ".yaml", ".yml", ".txt", ".csv", ".json", ".png", ".svg"}
    for directory in (
        "src",
        "scripts",
        "tests",
        "configs",
        "notebooks",
        "docs",
        "reports",
        "requirements",
        ".github",
    ):
        for path in sorted((root / directory).rglob("*")):
            if path.is_file() and path.suffix in allowed and not path.name.endswith(".local.yaml"):
                if "__pycache__" not in path.parts and not any(
                    p.endswith(".egg-info") for p in path.parts
                ):
                    yield path
    for directory in ("data", "outputs"):
        yield from sorted((root / directory).rglob("README.md"))


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "dist/pawl-net-source.zip")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"Archive already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = []
    with zipfile.ZipFile(args.output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in source_files(root):
            relative = path.relative_to(root).as_posix()
            archive.write(path, f"pawl-net/{relative}")
            manifest.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {relative}")
        archive.writestr("pawl-net/SOURCE_MANIFEST.sha256", "\n".join(manifest) + "\n")
    print(f"Prepared {len(manifest)} source files: {args.output}")


if __name__ == "__main__":
    main()
