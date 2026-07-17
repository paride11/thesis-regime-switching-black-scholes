"""Command-line entry point for rebuilding the thesis option-price dataset."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.option_dataset import OptionSelectionConfig, create_option_price_dataset


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    create_option_price_dataset(
        base_file=repo_root / "data" / "raw" / "WRDSspx_rf.csv",
        options_csv=repo_root / "data" / "raw" / "WRDSoptions.csv",
        output_file=repo_root / "data" / "processed" / "option_price.csv",
        config=OptionSelectionConfig(target_moneyness=1.00),
    )
