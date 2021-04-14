from music_trees.models.container import ModelContainer
from music_trees.models.protonet import HierarchicalProtoNet

MODEL_DELIM = '_'


def load_model(model_name: str):
    # TODO: this is overly complicated lol.
    """
    Load a container model with a swappable classification head

    example:
        model-128*_*hproto-1 

    will load a ModelContainer with root dimension 128, 
    and a hierarchical protonet with height 1. 

    format: 
        model-ROOT_DIM*_*<CLASSIFICATION_HEAD>

    """
    container_spec, head_spec = model_name.split(MODEL_DELIM)

    # choose from a variety of classification heads
    if 'hproto' in head_spec:
        height = int(head_spec.split('-')[-1])
        head = HierarchicalProtoNet(height)
    else:
        raise ValueError(f"wrong model name")

    # Finally, load the model container
    assert 'model' in container_spec
    d_root = int(container_spec.split('-')[-1])
    model = ModelContainer(d_root, head)

    return model
