"""Command-line entry point for rebuilding the thesis option-price datasets."""

from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.option_dataset import OptionSelectionConfig, create_option_price_dataset

DATASETS = {
    "atm": ("option_price_atm.csv", OptionSelectionConfig(target_moneyness=1.00)),
    "otm_90": ("option_price_otm_90.csv", OptionSelectionConfig(target_moneyness=0.90)),
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "datasets",
        nargs="*",
        choices=sorted(DATASETS),
        help="Datasets to rebuild (default: all).",
    )
    args = parser.parse_args()
    args.datasets = args.datasets or sorted(DATASETS)

    repo_root = Path(__file__).resolve().parents[1]
    for name in args.datasets:
        output_name, config = DATASETS[name]
        create_option_price_dataset(
            base_file=repo_root / "data" / "raw" / "WRDSspx_rf.csv",
            options_csv=repo_root / "data" / "raw" / "WRDSoptions.csv",
            output_file=repo_root / "data" / "processed" / output_name,
            config=config,
        )
        print(f"Wrote data/processed/{output_name}")
