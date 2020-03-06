from dataloaders.datasets import underwater_region
from torch.utils.data import DataLoader


def make_data_loader(opt, mode="train"):

    if opt.dataset in ['underwater', 'UnderWater', 'Underwater']:
        dataset = underwater_region.UnderWaterRegion(opt, mode)
        dataloader = DataLoader(dataset,
                                batch_size=opt.batch_size,
                                num_workers=opt.workers,
                                shuffle=True if mode == "train" else False,
                                pin_memory=True)

        return dataset, dataloader

    else:
        raise NotImplementedError
