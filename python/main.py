import os
import sys
import argparse
import logging
from DataLoader import create_dataloader
from Detectors import create_detector
from Matchers import create_matcher
from SFM import sfm

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
    # save ground truth trajectory
    dataloader.save_gt_trajectory(os.path.join(config["dataset"]["root_path"], "gt_trajectory.txt"))

    # create detector
    detector = create_detector(config["detector"])
    # create matcher
    matcher = create_matcher(config["matcher"])

    pipeline = sfm.SFM(dataloader, detector, matcher, camera=dataloader.cam, config=config["sfm"])
    # Run the pipeline
    pipeline.reconstruct()
    
    # bundle adjustment
    pipeline.bundle_adjustment(max_iter=30)

    # save camera poses in the format of TUM trajectory
    pipeline.save_trajectory(os.path.join(config["dataset"]["root_path"], "est_trajectory.txt"))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='python_sfm_pipeline')
    parser.add_argument('--config', type=str, default='configs/llff_trex_superpoint_superglu.yaml',
                        help='config file')
    parser.add_argument('--logging', type=str, default='INFO',
                        help='logging level: NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL')

    args = parser.parse_args()

    logging.basicConfig(level=logging._nameToLevel[args.logging])

    main(args)