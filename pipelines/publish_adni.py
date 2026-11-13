"""Copy ADNI aggregate results (metrics JSON and figures) to the public folders.

    uv run python pipelines/publish_adni.py

Only summary files are copied: models, predictions and the ADNI table stay local
(ADNI Data Use Agreement: summary data may be published, participant-level data may not).
"""

import json
import shutil

from dementia_ml.paths import REPO_ROOT

FILES = [
    "conversion_metrics.json",
    "trajectories.json",
    "omics_summary.json",
    "omics_proteins.csv",
    "omics_metabolites.csv",
]


def main() -> None:
    src_res, src_fig = REPO_ROOT / "results" / "adni", REPO_ROOT / "figures" / "adni"
    dst_res, dst_fig = REPO_ROOT / "results" / "adni_public", REPO_ROOT / "figures" / "adni_public"
    dst_res.mkdir(parents=True, exist_ok=True)
    dst_fig.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        if name.endswith(".csv"):  # one row per molecule: summary statistics only
            shutil.copy2(src_res / name, dst_res / name)
            continue
        payload = json.loads((src_res / name).read_text())
        payload["publication"] = "ADNI summary data (Data Use Agreement term 3); see README"
        (dst_res / name).write_text(json.dumps(payload, indent=2))
    for png in sorted(src_fig.glob("*.png")):
        shutil.copy2(png, dst_fig / png.name)
    print(sorted(p.name for p in dst_res.iterdir()), sorted(p.name for p in dst_fig.iterdir()))


if __name__ == "__main__":
    main()
