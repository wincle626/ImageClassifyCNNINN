"""Layer-by-layer LNN explainability for a ResNet-18, at the true layer depth.

ResNet-18 is named for its **18 weight layers**:

    1  stem convolution (conv1)
    16 convolutions inside the residual blocks
       (4 stages x 2 BasicBlocks x 2 conv layers)
    1  final fully-connected classifier (fc)
    -----------------------------------------------
    18 layers total

This model makes the network explainable *layer by layer* by attaching a concept
probe + an LNN reasoner after **each of the 17 convolutional layers**. Each conv
layer's feature map is global-average-pooled, mapped to the 49 human concepts,
and reasoned over by an LNN to a class guess -- so we can read what the network
has understood after every convolution. The 18th layer, ``fc``, is the ordinary
classifier; here its role is taken by the deepest LNN (after conv layer 17), so
the LNN literally stands in for the final layer.

    image ─► conv1 ──► ... ──► layer4.1.conv2 ──► (fc replaced by LNN)
               │                     │
             probe+LNN             probe+LNN
             class@1     ...       class@17  = final prediction

Training uses deep supervision: every conv layer is pushed to detect the
concepts and to reason to the class from them.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

from .lnn_layer import LNNReasoningLayer

# The 17 convolutional weight layers of ResNet-18, in forward order, as
# (tap id, module path, output channels, human-readable label). The 18th weight
# layer -- the ``fc`` classifier -- is listed separately below.
CONV_LAYER_SPECS: list[tuple[str, str, int, str]] = [
    ("L1",  "conv1",          64,  "conv1 (7x7 stem)"),
    ("L2",  "layer1.0.conv1", 64,  "stage1 block0 conv1"),
    ("L3",  "layer1.0.conv2", 64,  "stage1 block0 conv2"),
    ("L4",  "layer1.1.conv1", 64,  "stage1 block1 conv1"),
    ("L5",  "layer1.1.conv2", 64,  "stage1 block1 conv2"),
    ("L6",  "layer2.0.conv1", 128, "stage2 block0 conv1"),
    ("L7",  "layer2.0.conv2", 128, "stage2 block0 conv2"),
    ("L8",  "layer2.1.conv1", 128, "stage2 block1 conv1"),
    ("L9",  "layer2.1.conv2", 128, "stage2 block1 conv2"),
    ("L10", "layer3.0.conv1", 256, "stage3 block0 conv1"),
    ("L11", "layer3.0.conv2", 256, "stage3 block0 conv2"),
    ("L12", "layer3.1.conv1", 256, "stage3 block1 conv1"),
    ("L13", "layer3.1.conv2", 256, "stage3 block1 conv2"),
    ("L14", "layer4.0.conv1", 512, "stage4 block0 conv1"),
    ("L15", "layer4.0.conv2", 512, "stage4 block0 conv2"),
    ("L16", "layer4.1.conv1", 512, "stage4 block1 conv1"),
    ("L17", "layer4.1.conv2", 512, "stage4 block1 conv2"),
]
FC_LAYER = ("L18", "fc", "fully-connected classifier (replaced by the LNN)")


def resnet18_weight_layers() -> list[tuple[str, str]]:
    """Return all 18 weight layers of ResNet-18 as (tap id, label)."""
    layers = [(tid, label) for tid, _, _, label in CONV_LAYER_SPECS]
    layers.append((FC_LAYER[0], FC_LAYER[2]))
    return layers


@dataclass
class LayerwiseOutput:
    stages: list[str]                         # the 17 conv-layer tap ids
    concept_logits: dict[str, torch.Tensor]   # tap -> [B, K]
    concept_probs: dict[str, torch.Tensor]    # tap -> [B, K] in [0,1]
    class_truth: dict[str, torch.Tensor]      # tap -> [B, C] in [0,1]
    class_logits: dict[str, torch.Tensor]     # tap -> [B, C]
    final_logits: torch.Tensor                # == class_logits[last conv layer]


def _get_module(root: nn.Module, path: str) -> nn.Module:
    mod = root
    for part in path.split("."):
        mod = mod[int(part)] if part.isdigit() else getattr(mod, part)
    return mod


class LayerwiseCNNLNN(nn.Module):
    def __init__(
        self,
        concept_matrix: torch.Tensor,
        pretrained: bool = True,
        small_input: bool = False,
        temperature: float = 8.0,
    ):
        super().__init__()
        _, num_concepts = concept_matrix.shape
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        net = resnet18(weights=weights)
        if small_input:
            net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                                  padding=1, bias=False)
            net.maxpool = nn.Identity()
        net.fc = nn.Identity()  # the 18th layer is replaced by the LNN
        self.net = net

        # tap ids in depth order (L1..L17); keep the name ``stages`` so the
        # existing evaluation / explanation utilities work unchanged.
        self.stages = [tid for tid, _, _, _ in CONV_LAYER_SPECS]
        self.layer_labels = {tid: label for tid, _, _, label in CONV_LAYER_SPECS}
        self._paths = {tid: path for tid, path, _, _ in CONV_LAYER_SPECS}

        self.probes = nn.ModuleDict({
            tid: nn.Linear(ch, num_concepts)
            for tid, _, ch, _ in CONV_LAYER_SPECS
        })
        self.reasoners = nn.ModuleDict({
            tid: LNNReasoningLayer(concept_matrix) for tid in self.stages
        })
        self.temperature = temperature

        # forward hooks capture each conv layer's output during the pass
        self._feats: dict[str, torch.Tensor] = {}
        for tid, path, _, _ in CONV_LAYER_SPECS:
            _get_module(net, path).register_forward_hook(self._make_hook(tid))

    def _make_hook(self, tid: str):
        def hook(_module, _inp, out):
            self._feats[tid] = out
        return hook

    def _run_backbone(self, x: torch.Tensor) -> None:
        n = self.net
        x = n.maxpool(n.relu(n.bn1(n.conv1(x))))
        x = n.layer1(x)
        x = n.layer2(x)
        x = n.layer3(x)
        n.layer4(x)  # hooks capture everything; final tensor not needed

    def forward(self, x: torch.Tensor) -> LayerwiseOutput:
        self._feats = {}
        self._run_backbone(x)
        c_logits, c_probs, cls_truth, cls_logits = {}, {}, {}, {}
        for tid in self.stages:
            pooled = F.adaptive_avg_pool2d(self._feats[tid], 1).flatten(1)
            logit = self.probes[tid](pooled)
            probs = torch.sigmoid(logit)
            raw = self.reasoners[tid].raw_score(probs)
            c_logits[tid] = logit
            c_probs[tid] = probs
            cls_truth[tid] = raw.clamp(0.0, 1.0)
            cls_logits[tid] = raw * self.temperature
        return LayerwiseOutput(
            stages=self.stages,
            concept_logits=c_logits,
            concept_probs=c_probs,
            class_truth=cls_truth,
            class_logits=cls_logits,
            final_logits=cls_logits[self.stages[-1]],
        )
