"""
Networks used in the assignment.

ArogyaNet - my own small CNN for 1-channel chest X-rays.
  Design decisions that get ablated:
    (A) residual skip connections in every block      -> residual=False removes them
    (B) squeeze-and-excitation channel attention       -> se=False removes it
  Also used (not ablated): depthwise separable convs and global average pooling,
  that's what keeps it far under the 2M parameter limit.

ArogyaNet(meta=True) is the two-branch version for task 8: 3 handcrafted numbers go
through a tiny MLP and get concatenated with the image embedding.
"""
import torch
import torch.nn as nn
import torchvision


class SE(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.fc1 = nn.Conv2d(c, max(c // r, 8), 1)
        self.fc2 = nn.Conv2d(max(c // r, 8), c, 1)

    def forward(self, x):
        s = x.mean((2, 3), keepdim=True)
        s = torch.sigmoid(self.fc2(torch.relu(self.fc1(s))))
        return x * s


class DSConv(nn.Module):
    """depthwise 3x3 + pointwise 1x1, each followed by BN"""
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False)
        self.bn1 = nn.BatchNorm2d(cin)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)

    def forward(self, x):
        x = torch.relu(self.bn1(self.dw(x)))
        return self.bn2(self.pw(x))


class Block(nn.Module):
    def __init__(self, cin, cout, stride, residual=True, se=True):
        super().__init__()
        self.conv1 = DSConv(cin, cout, stride)
        self.conv2 = DSConv(cout, cout, 1)
        self.se = SE(cout) if se else nn.Identity()
        self.residual = residual
        if residual and (stride != 1 or cin != cout):
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
        else:
            self.short = nn.Identity()

    def forward(self, x):
        out = torch.relu(self.conv1(x))
        out = self.se(self.conv2(out))
        if self.residual:
            out = out + self.short(x)
        return torch.relu(out)


class ArogyaNet(nn.Module):
    def __init__(self, residual=True, se=True, meta=False, widths=(32, 64, 128, 256), dropout=0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 24, 3, 2, 1, bias=False), nn.BatchNorm2d(24), nn.ReLU(inplace=True))
        layers = []
        cin = 24
        for w in widths:
            layers.append(Block(cin, w, 2, residual, se))
            layers.append(Block(w, w, 1, residual, se))
            cin = w
        self.features_ = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.emb_dim = cin
        self.use_meta = meta
        head_in = cin
        if meta:
            self.meta_net = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU())
            head_in += 16
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(head_in, 1)

    def embed(self, x, meta=None):
        z = self.pool(self.features_(self.stem(x))).flatten(1)
        if self.use_meta:
            z = torch.cat([z, self.meta_net(meta)], 1)
        return z

    def forward(self, x, meta=None):
        return self.fc(self.drop(self.embed(x, meta))).squeeze(1)


class ResNet18Transfer(nn.Module):
    """ImageNet ResNet18, grey image copied into 3 channels, new 1-logit head"""
    def __init__(self, pretrained=True, dropout=0.2):
        super().__init__()
        w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = torchvision.models.resnet18(weights=w)
        net.fc = nn.Identity()
        self.body = net
        self.emb_dim = 512
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(512, 1)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def embed(self, x, meta=None):
        # x comes in as 1 channel in [0,1]-ish normalised space, undo and use imagenet stats
        x = x * 0.25 + 0.5
        x = (x.repeat(1, 3, 1, 1) - self.mean) / self.std
        return self.body(x)

    def forward(self, x, meta=None):
        return self.fc(self.drop(self.embed(x))).squeeze(1)


def build(cfg):
    if cfg["model"] == "resnet18":
        return ResNet18Transfer(pretrained=cfg.get("pretrained", True), dropout=cfg.get("dropout", 0.2))
    return ArogyaNet(residual=cfg.get("residual", True), se=cfg.get("se", True),
                     meta=cfg.get("meta", False), dropout=cfg.get("dropout", 0.2))


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def arch_table(model, res=224):
    """layer / output shape / params, via forward hooks on the top level blocks"""
    rows = []
    hooks = []

    def hook(name):
        def f(mod, inp, out):
            rows.append((name, tuple(out.shape[1:]), count_params(mod)))
        return f

    hooks.append(model.stem.register_forward_hook(hook("stem conv3x3 s2")))
    for i, b in enumerate(model.features_):
        kind = "Block s2" if b.conv1.dw.stride[0] == 2 else "Block s1"
        hooks.append(b.register_forward_hook(hook(f"stage{i // 2 + 1}.{kind}")))
    hooks.append(model.pool.register_forward_hook(hook("global avg pool")))
    hooks.append(model.fc.register_forward_hook(hook("dropout + linear")))
    model.eval()
    with torch.no_grad():
        meta = torch.zeros(1, 3) if model.use_meta else None
        model(torch.zeros(1, 1, res, res), meta)
    for h in hooks:
        h.remove()
    return rows
