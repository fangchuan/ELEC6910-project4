from .TempleRingImageDataLoader import TempleRingImageLoader
from .LLFFImageDataLoader import LLFFImageLoader


def create_dataloader(conf):
    dataset_name = conf['name']
    if dataset_name == "templeRing":
        dataloader = TempleRingImageLoader(conf)
    elif dataset_name == "llff":
        dataloader = LLFFImageLoader(conf)
    else:
        raise NotImplementedError(f"{conf['name']} is not implemented yet.")

    return dataloader