"""The value/policy network.

One shared trunk feeding two heads, as in McAleer et al.: a value head
estimating how close a state is to solved, and a policy head scoring each of
the 18 moves. Sharing the trunk is what makes the pair cheaper to train than
two separate networks, since both tasks need the same features.

The default trunk (2048 -> 1024) is smaller than the paper's (4096 -> 2048).
They trained on roughly 8 billion cube visits across three GPUs for 44 hours;
within this project's budget a smaller network that sees more states is the
better trade. Sizes are constructor arguments so this can be revisited.
"""

from __future__ import annotations

import torch
from torch import nn

from rubiks.cube import ALL_MOVES
from rubiks.encoding import ENCODED_SIZE

NUM_MOVES = len(ALL_MOVES)


class CubeNet(nn.Module):
    def __init__(
        self,
        trunk: tuple[int, ...] = (2048, 1024),
        head: int = 512,
        num_moves: int = NUM_MOVES,
    ):
        super().__init__()
        layers = []
        width = ENCODED_SIZE
        for size in trunk:
            layers += [nn.Linear(width, size), nn.ELU()]
            width = size
        self.trunk = nn.Sequential(*layers)
        self.value_head = nn.Sequential(
            nn.Linear(width, head), nn.ELU(), nn.Linear(head, 1)
        )
        self.policy_head = nn.Sequential(
            nn.Linear(width, head), nn.ELU(), nn.Linear(head, num_moves)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(x)
        value = self.value_head(features).squeeze(-1)
        policy_logits = self.policy_head(features)
        return value, policy_logits


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
