import argparse
import logging
import sys
from pathlib import Path

from hydra.compose import compose
from hydra.initialize import initialize
from omegaconf import DictConfig

from threedgrut.model.model import MixtureOfGaussians

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ingp_to_ply")


def load_config(config_name="apps/colmap_3dgut.yaml", config_path="../../../configs") -> DictConfig:
    with initialize(version_base=None, config_path=config_path):
        return compose(config_name=config_name)


def main():
    parser = argparse.ArgumentParser(description="Convert ingp to PLY")
    parser.add_argument("input_file", type=str, help="Input ingp file path")
    parser.add_argument("--output_file", type=str, help="Output PLY file path")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    if input_path.suffix.lower() != ".ingp":
        logger.error("Input must be an ingp file")
        sys.exit(1)

    output_path = Path(args.output_file) if args.output_file else input_path.with_suffix(".ply")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Reading ingp: {input_path}")
    conf = load_config()
    model = MixtureOfGaussians(conf)
    model.init_from_ingp(str(input_path), init_model=False)

    logger.info(f"Exporting to PLY: {output_path}")
    model.export_vol_ply(str(output_path))
    logger.info("Export complete")


if __name__ == "__main__":
    main()