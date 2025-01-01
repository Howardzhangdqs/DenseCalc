import timm
import torch.nn as nn
from rich import print


class Model(nn.Module):
    def __init__(self, input_size=(256, 256), num_classes=2):
        super(Model, self).__init__()
        self.model = timm.create_model(
            "repvit_m2_3",
            # pretrained=True,
            num_classes=num_classes
        )

        self.model.default_cfg['input_size'] = (3, *input_size)

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == "__main__":
    import torch
    model = Model()
    model.eval()
    x = torch.randn(1, 3, 256, 256)  # 修改输入大小为256
    y = model(x)
    print(y.shape)
