"""
Paper Configuration Loader — Single Source of Truth

Loads the frozen paper_v1.yaml configuration and provides typed access to
all thresholds, evaluation parameters, and paths. Every final-paper script
MUST use this loader instead of hard-coding values.

Usage:
    from config_loader import load_paper_config
    cfg = load_paper_config()
    print(cfg['thresholds']['confidence'])  # 0.4743
"""

import yaml
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DEFAULT_CONFIG_PATH = BASE_DIR / "configs" / "paper_v1.yaml"


def load_paper_config(config_path: str = None) -> dict:
    """Load the frozen paper configuration from YAML.

    Args:
        config_path: Optional override path. Defaults to configs/paper_v1.yaml.

    Returns:
        Parsed configuration dictionary with all thresholds, evaluation
        parameters, model paths, and calibration metadata.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the config is missing required sections.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.is_absolute():
        path = BASE_DIR / path

    if not path.exists():
        raise FileNotFoundError(
            f"Paper config not found: {path}\n"
            f"Expected at: {DEFAULT_CONFIG_PATH}"
        )

    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Validate required sections
    required_sections = ['thresholds', 'evaluation', 'model']
    missing = [s for s in required_sections if s not in cfg]
    if missing:
        raise ValueError(
            f"Paper config is missing required sections: {missing}\n"
            f"Config file: {path}"
        )

    version = cfg.get('version', 'unknown')
    frozen_date = cfg.get('frozen_date', 'unknown')
    print(f"[CONFIG] Using {path.name} (version: {version}, frozen: {frozen_date})")

    return cfg


def get_provenance_metadata(config_path: str = None) -> dict:
    """Generate standardized provenance metadata for benchmark result files.

    Includes config version, git commit hash, branch, timestamp, python version,
    and platform info to ensure paper results are fully auditable.
    """
    import subprocess
    import sys
    from datetime import datetime, timezone

    cfg = load_paper_config(config_path)

    # Git commit and branch
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(BASE_DIR)
        ).decode().strip()
    except Exception:
        git_commit = "unknown"

    try:
        git_branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(BASE_DIR)
        ).decode().strip()
    except Exception:
        git_branch = "unknown"

    return {
        "config_version": cfg.get("version", "paper_v1"),
        "config_frozen_date": cfg.get("frozen_date", "2026-09-11"),
        "git_commit": git_commit,
        "git_branch": git_branch,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": sys.platform,
    }

