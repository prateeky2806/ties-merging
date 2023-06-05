import torch
import logging
from torch.utils import data

from src.utils.distributed_utils import is_distributedSetup

from torch.utils.data.distributed import DistributedSampler

logger = logging.getLogger("root")


class Batcher(object):
    """
    Batcher is responsible for returning batches of data
    """

    def __init__(
        self,
        dataset_reader,
        createPytorchDataset_fn,
        train_batchSize,
        eval_batchSize,
        world_size,
        device,
    ):
        """


        Args:
            dataset_reader:
            createPytorchDataset_fn: function to create_dataset rather than the actual dataset since
                              data is only instantiated inside the batcher
            train_batchSize:
            eval_batchSize:
            world_size:
            device:
        """
        self.dataset_reader = dataset_reader
        self.createPytorchDataset_fn = createPytorchDataset_fn

        self.train_batchSize = train_batchSize
        self.eval_batchSize = eval_batchSize
        self.world_size = world_size
        self.device = device
        self.current_epoch = 0

    def get_metricsForDataset(self):
        return self.dataset_reader.get_metricsForDataset()

    def create_data_loader(self, pytorch_dataset, batch_size, shuffle):

        if is_distributedSetup(self.world_size):
            sampler = DistributedSampler(
                pytorch_dataset,
                num_replicas=self.world_size,
                rank=self.device,
                shuffle=shuffle,
            )

            data_loader = data.DataLoader(
                pytorch_dataset,
                batch_size=batch_size,
                num_workers=0,
                shuffle=False,
                sampler=sampler,
                collate_fn=pytorch_dataset.collate_fn,
            )
            return sampler, data_loader
        else:
            data_loader = data.DataLoader(
                pytorch_dataset,
                batch_size=batch_size,
                num_workers=0,
                shuffle=shuffle,
                collate_fn=pytorch_dataset.collate_fn,
            )

            return None, data_loader

    def get_trainBatches(self, split, template_idx):
        dataset = self.dataset_reader.get_dataset(
            split, template_idx, is_evaluation=False
        )
        logger.info(f"\tTotal Train Examples along with Templates: {len(dataset)}")
        pytorch_dataset = self.createPytorchDataset_fn(dataset)
        sampler, data_loader = self.create_data_loader(
            pytorch_dataset, self.train_batchSize, True
        )

        while True:
            if is_distributedSetup(self.world_size):
                sampler.set_epoch(self.current_epoch)

            for x in data_loader:
                yield x

            self.current_epoch += 1

    def get_splitOfBatches(self, split, template_idx, is_evaluation):
        assert split.lower() in [
            "validation",
            "validation_full",
            "train",
            "test",
        ], f"Evaluation Split {split} not defined"

        dataset = self.dataset_reader.get_dataset(
            split, template_idx, is_evaluation
        )
        pytorch_dataset = self.createPytorchDataset_fn(dataset)
        _, data_loader = self.create_data_loader(
            pytorch_dataset, self.eval_batchSize, False
        )

        for x in data_loader:
            yield x

    def get_evalBatches(self, split, template_idx):
        assert split.lower() in [
            "validation",
            "validation_full",
            "train",
            "test",
        ], f"Evaluation Split {split} not defined"

        dataset = self.dataset_reader.get_dataset(
            split, template_idx, is_evaluation=True
        )
        pytorch_dataset = self.createPytorchDataset_fn(dataset)
        _, data_loader = self.create_data_loader(
            pytorch_dataset, self.eval_batchSize, False
        )

        for x in data_loader:
            yield x
