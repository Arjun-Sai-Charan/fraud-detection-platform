from pathlib import Path


def _pins(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", "--")):
            continue
        package, version = line.split("==", maxsplit=1)
        pins[package.lower().replace("_", "-")] = version
    return pins


def test_serving_requirements_are_identically_pinned_to_primary_requirements():
    root = Path(__file__).resolve().parents[1]
    primary = _pins(root / "requirements.txt")
    serving = _pins(root / "requirements-serving.txt")

    assert {"pytest", "matplotlib", "seaborn", "httpx"}.isdisjoint(serving)
    assert serving
    assert {name: primary.get(name) for name in serving} == serving
