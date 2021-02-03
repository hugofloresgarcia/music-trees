###############################################################################
# Test model
###############################################################################


def test_model(model, datamodule):
    """Shape test for the forward pass

    Arguments
        model - NAME.Model
            The model to test
        datamodule - PyTorch Lightning DataModule
            The data to use for the test
    """
    # TODO - replace expected shape
    expected_shape = (4, 1, 1)
    assert model(*next(datamodule.val_dataloader())).shape == expected_shape
