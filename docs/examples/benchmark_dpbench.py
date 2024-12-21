import json
import logging
import os
import subprocess
from pathlib import Path

from tabulate import tabulate  # type: ignore

from docling_eval.benchmarks.constants import BenchMarkNames, EvaluationModality
from docling_eval.benchmarks.dpbench.create import (
    create_dpbench_layout_dataset,
    create_dpbench_tableformer_dataset,
)
from docling_eval.evaluators.layout_evaluator import (
    DatasetLayoutEvaluation,
    LayoutEvaluator,
)
from docling_eval.evaluators.table_evaluator import (
    DatasetTableEvaluation,
    TableEvaluator,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def is_git_lfs_installed():
    """
    Check if Git LFS is installed.
    """
    try:
        result = subprocess.run(
            ["git", "lfs", "version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode == 0:
            logger.info("Git LFS is installed.")
            return True
        else:
            logger.warning("Git LFS is not installed.")
            return False
    except FileNotFoundError:
        logger.error("Git is not installed.")
        return False


def clone_repository(repo_url, target_directory):
    """
    Clone a Git repository to the specified target directory.
    """
    if os.path.exists(target_directory):
        logger.warning(
            f"Target directory '{target_directory}' already exists. Skipping clone."
        )
        return

    try:
        subprocess.run(
            ["git", "clone", repo_url, target_directory],
            check=True,
        )
        logger.info(f"Repository cloned into '{target_directory}'.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to clone repository: {e}")
        raise


def main():

    REPO_URL = "https://huggingface.co/datasets/upstage/dp-bench.git"  # Replace with your repo URL

    idir = Path("./benchmarks/dp-bench-original")

    if is_git_lfs_installed():
        clone_repository(REPO_URL, idir)
    else:
        logger.error("Please install Git LFS and try again.")

    odir = Path("./benchmarks/dp-bench-dataset")

    odir_lay = Path(odir) / "layout"
    odir_tab = Path(odir) / "tableformer"

    for _ in [odir, odir_lay, odir_tab]:
        os.makedirs(_, exist_ok=True)

    image_scale = 1.0

    create_dpbench_layout_dataset(
        dpbench_dir=idir, output_dir=odir_lay, image_scale=image_scale
    )

    create_dpbench_tableformer_dataset(
        dpbench_dir=idir, output_dir=odir_tab, image_scale=image_scale
    )

    if True:
        save_fn = (
            odir
            / f"evaluation_{BenchMarkNames.DPBENCH.value}_{EvaluationModality.LAYOUT.value}.json"
        )

        layout_evaluator = LayoutEvaluator()
        layout_evaluation = layout_evaluator(odir_lay, split="test")

        logging.info(f"writing results to {save_fn}")
        with open(save_fn, "w") as fd:
            json.dump(layout_evaluation.model_dump(), fd, indent=2, sort_keys=True)

        results = layout_evaluation.to_table()
        logging.info(f"mAP results for layout:\n\n{tabulate(results)}")

    if True:
        save_fn = (
            odir
            / f"evaluation_{BenchMarkNames.DPBENCH.value}_{EvaluationModality.TABLEFORMER.value}.json"
        )

        table_evaluator = TableEvaluator()
        table_evaluation = table_evaluator(odir_tab, split="test")

        logging.info(f"writing results to {save_fn}")
        with open(save_fn, "w") as fd:
            json.dump(table_evaluation.model_dump(), fd, indent=2, sort_keys=True)

        results = table_evaluation.TEDS.to_table()
        md = tabulate(results, headers=["x0 <= TEDS", "TEDS <= x1", "%", "count"])
        logging.info(f"TEDS results for TableFormer:\n\n{md}")


if __name__ == "__main__":
    main()
