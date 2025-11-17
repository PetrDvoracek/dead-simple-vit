import torch
import torchvision
import torchvision.transforms as T

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingLayer(torch.nn.Module):
    def __init__(self, embed_dim, patch_size):
        super(EmbeddingLayer, self).__init__()
        # Patchification is done using convolution with
        # big kernel_size and stride.
        self.conv = torch.nn.Conv2d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        x = self.conv(x)
        # merge `height` and `width` dimensions into one `tokens` dimension
        # batch x tokens x features
        x = x.reshape(x.shape[0], -1, x.shape[1])
        return x


class TransformerBlock(torch.nn.Module):
    def __init__(self, embed, embed_inner):
        super().__init__()
        self.q = torch.nn.Linear(embed, embed_inner)
        self.k = torch.nn.Linear(embed, embed_inner)
        self.v = torch.nn.Linear(embed, embed_inner)
        self.fc = torch.nn.Linear(embed_inner, embed)
        self.lanorm1 = torch.nn.LayerNorm(embed)
        self.lanorm2 = torch.nn.LayerNorm(embed_inner)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x_skip_1 = x.clone()
        x = self.lanorm1(x)

        # self attention
        q, k, v = self.q(x), self.k(x), self.v(x)
        qk = torch.matmul(q, k.transpose(-2, -1))
        qk = self.softmax(qk)
        qkv = torch.matmul(qk, v)

        x = self.lanorm2(qkv)
        x = self.fc(x)
        x = x + x_skip_1
        x_skip_2 = x.clone()
        x = x + x_skip_2
        return x


class Transformer(torch.nn.Module):
    def __init__(self, embed_dim, patch_size, num_classes):
        super(Transformer, self).__init__()
        self.embedding = EmbeddingLayer(embed_dim, patch_size)
        self.cls_token = torch.nn.Parameter(torch.rand(embed_dim))
        self.blocks = torch.nn.Sequential(
            TransformerBlock(embed_dim, int(embed_dim * 1.5)),
            TransformerBlock(embed_dim, int(embed_dim * 1.5)),
            TransformerBlock(embed_dim, int(embed_dim * 1.5)),
        )
        self.cls_layer = torch.nn.Linear(embed_dim, out_features=num_classes)

    def forward(self, x):
        bs = x.shape[0]
        cls_token = torch.stack([self.cls_token] * bs).unsqueeze(1)

        x = self.embedding(x)
        x = torch.cat([cls_token, x], dim=1)

        x = self.blocks(x)

        cls_token = x[:, 0, :]

        out = self.cls_layer(cls_token)

        return out


if __name__ == "__main__":
    device = "cpu"
    dataset = torchvision.datasets.CIFAR10(
        root="/Volumes/External/data/torch_datasets",
        transform=T.Compose([T.ToTensor()]),
        download=True,
    )
    model = Transformer(512, 16, 10)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-5)
    lossfn = torch.nn.CrossEntropyLoss()

    dataloader_train = torch.utils.data.DataLoader(dataset, batch_size=128)
    for x, y in dataloader_train:
        x.to(torch.float32).to(device), y.to(torch.float32).to(device)
        optimizer.zero_grad()
        y_hat = model(x)
        l = lossfn(
            y_hat.to(torch.float32),
            y.to(torch.long),
        )
        l.backward()
        optimizer.step()
        print(l)

    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(out.shape)
