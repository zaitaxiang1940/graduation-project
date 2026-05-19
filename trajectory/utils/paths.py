from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DATA_DIR = PROJECT_ROOT / "data"
RUNS_DIR = PROJECT_ROOT / "runs"
LOGS_DIR = PROJECT_ROOT / "logs"


def build_project_path(*parts):
    return PROJECT_ROOT.joinpath(*parts)


def resolve_runtime_path(path, search_dirs=None):
    if path is None:
        return None

    candidate = Path(path).expanduser()
    if candidate.exists():
        return candidate.resolve()

    bases = [PROJECT_ROOT, CONFIGS_DIR, SCRIPTS_DIR, DATA_DIR, RUNS_DIR, LOGS_DIR]
    if search_dirs:
        bases = [Path(base) for base in search_dirs] + bases

    relatives = []
    if not candidate.is_absolute():
        relatives.append(candidate)
    relatives.append(Path(candidate.name))

    for base in bases:
        for relative in relatives:
            resolved = (base / relative).resolve()
            if resolved.exists():
                return resolved

    if candidate.is_absolute():
        return candidate

    return (PROJECT_ROOT / candidate).resolve()
