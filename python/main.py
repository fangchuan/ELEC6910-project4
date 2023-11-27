import os
import sys
import argparse
import logging
from DataLoader import create_dataloader
from Detectors import create_detector


def main(args):
    # Load config file
    config = None
    if args.config is not None:
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        logging.error("Config file is not specified.")
        sys.exit(-1)

    # Create data loaders
    dataloader = create_dataloader(config["dataset"])
    # create detector
    detector = create_detector(config["detector"])
    # create matcher
    # matcher = create_matcher(config["matcher"])

    # Run the pipeline
    for i, data in enumerate(dataloader):
        logging.info(f"Data {i} loaded.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='python_sfm_pipeline')
    parser.add_argument('--config', type=str, default='configs/templering_superpoint_superglu.yaml',
                        help='config file')
    parser.add_argument('--logging', type=str, default='INFO',
                        help='logging level: NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL')

    args = parser.parse_args()

    logging.basicConfig(level=logging._nameToLevel[args.logging])

    main(args)