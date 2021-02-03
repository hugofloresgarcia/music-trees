###############################################################################
# Test data
###############################################################################


def test_dataset(dataset):
    """Shape test for the dataset __getitem__ function

    Arguments
        dataset - NAME.Dataset
            The validation dataset
    """
    # Get an item from the dataset
    item = dataset[0]

    # TODO - check that the shape of the output of __getitem__ is as expected
    raise NotImplementedError


def test_datamodule(datamodule):
    """Shape test for the datamodule loader

    Arguments
        datamodule - PyTorch Lightning DataModule
            The datamodule to test
    """
    # Load a batch from the dataloader
    batch = next(datamodule.val_dataloader())

    # TODO - check that the shape of the batch is as expected
    raise NotImplementedError
