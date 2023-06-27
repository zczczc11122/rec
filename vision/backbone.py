import torch.nn as nn
import torchvision


class Backbone(object):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def prepare_backbone(arch):
        if arch in ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
            base_model = getattr(torchvision.models, arch)(True)
            feature_dim = base_model.fc.in_features
            base_model.fc = nn.Identity()
            input_size = 224
            input_mean = [0.485, 0.456, 0.406]
            input_std = [0.229, 0.224, 0.225]
        elif arch in ['Inception3', 'Inception_v3']:
            base_model = getattr(torchvision.models, arch)(True)
            feature_dim = base_model.fc.in_features
            base_model.aux_logits = False
            base_model.fc = nn.Identity()
            input_size = 299
            input_mean = [0.485, 0.456, 0.406]
            input_std = [0.229, 0.224, 0.225]
        elif arch in ['efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2',
                      'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5',
                      'efficientnet-b6', 'efficientnet-b7']:
            from efficientnet_pytorch import EfficientNet
            # base_model = EfficientNet.from_pretrained(arch, weights_path="../efficientnet-b3-5fb5a3c3.pth")
            # base_model.set_swish(memory_efficient=False)
            # feature_dim = base_model._fc.in_features
            # base_model._fc = nn.Identity()
            base_model = EfficientNet.from_pretrained(arch,
                                                      weights_path="../efficientnet-b3-5fb5a3c3.pth", include_top=False)
            base_model.set_swish(memory_efficient=False)
            feature_dim = base_model._conv_head.out_channels
            input_size = 224
            #input_size = 300
            input_mean = [0.485, 0.456, 0.406]
            input_std = [0.229, 0.224, 0.225]
        else:
            raise ValueError('Unknown base model: {}'.format(arch))
        return base_model, feature_dim, input_size, input_mean, input_std
