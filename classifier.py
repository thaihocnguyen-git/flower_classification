from typing import List

import torch
from torch.nn import Linear, Module, ModuleList
from torch.nn import functional as F
from torch.nn import Sequential
from torchvision.models import MobileNet_V3_Small_Weights
from torchvision.models.mobilenetv3 import mobilenet_v3_small
from torchvision.models.resnet import resnet50, resnet101

def _feature_extractor(name: str) -> (Module, int):
    """Get CNN backbone for model

    Args:
        name (str): Name of the model (Available options: "resnet50", "resnet101", "mobilenet_v3_small")
        int (_type_): _description_

    Returns:
        _type_: extractor model, output size of the features
    """
    if name == "resnet50":
        _resnet = resnet50(weights="DEFAULT")
        return Sequential(*list(_resnet.children())[:-1]), 2048
    elif name == "resnet101":
        _resnet = resnet101(weights="DEFAULT")
        return Sequential(*list(_resnet.children())[:-1]), 2048
    else:
        mobile_net = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        return Sequential(*list(mobile_net.children())[:-1]), 576


class ImageClassifier(Module):    
    """Image classifier

    Args:
        Module (_type_): _description_
    """
    def __init__(self, **config):
        super().__init__()

        self.dropout_rate = config.get("dropout_rate", 0.5)
        self.backbone = config.get("backbone", "resnet50")
        self.hidden_sizes = config.get("hiddens", [])
        self.class_to_idx = config.get("class_to_idx")
        self.extractor, self.feature_size = _feature_extractor(self.backbone)

        for param in self.extractor.parameters():
            param.requires_grad = False
        self._init_hidden(self.feature_size, self.hidden_sizes, len(self.class_to_idx.keys()))

    @property
    def num_classes(self):
        return len(self.class_to_idx.keys())

    def _init_hidden(self, feature_size: int, hidden_size: List[int], num_class: int):

        input_size = [feature_size] + hidden_size
        output_size = hidden_size + [num_class]
        hidden_shapes = zip(input_size, output_size)
        hidden_layers = [Linear(*shape) for shape in hidden_shapes]
        self.hiddens = ModuleList(hidden_layers)


    def forward(self, x, training=False):
        """Model forward

        Args:
            x (_type_): tensor input
            training (bool, optional): If training. Defaults to False.

        Returns:
            _type_: output score
        """
        x = self.extractor(x)
        x = x.view(x.shape[0], -1)
        for hidden in self.hiddens[:-1]:
            x = F.dropout(F.relu(hidden(x)), p=self.dropout_rate, training=training)

        x = self.hiddens[-1](x)
        return x


    @classmethod
    def save(cls, model, path: str) -> None:
        """Save model and params"""
        params = {
            "dropout_rate": model.dropout_rate,
            "backbone": model.backbone,
            "hiddens": model.hidden_sizes,
            "class_to_idx": model.class_to_idx
        }

        torch.save({
            "params": params,
            "model_state_dict": model.state_dict()
        }, path)


    @classmethod
    def from_path(cls, path: str, gpu: bool = True):
        """Create and load model from save path.

        Args:
            path (str): path of save model

        Returns:
            ImageClassifier: ImageClassifier with loaded weights
        """
        checkpoint = torch.load(path)
        model = ImageClassifier(**checkpoint["params"])

        model.load_state_dict(checkpoint["model_state_dict"])
        if gpu:
            model.cuda()
        model.eval()
        return model
