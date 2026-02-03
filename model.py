import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class OthelloPlayer(nn.Module):
    def __init__(
            self, 
            in_channels=3, 
            base_channels=64, 
            num_blocks=8, 
            hidden_dim=1024,
        ):
        """
        ResNet-based DQN for Othello.

        Args:
            in_channels (int): number of input channels (3 if you encode board as 3 planes: own pieces, opp pieces, valid moves)
            base_channels (int): number of channels in the residual stream
            num_blocks (int): how many residual blocks to stack
            hidden_dim (int): hidden size of fully connected layer
        """
        super().__init__()

        # Initial stem
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(base_channels)

        # Residual tower
        self.res_blocks = nn.Sequential(*[ResidualBlock(base_channels) for _ in range(num_blocks)])

        # Fully connected head
        self.flatten = nn.Flatten()
        # self.q_score = nn.Sequential(
        #     nn.Linear(base_channels * 8 * 8, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, 64)  # 64 Q-values (8x8 board)
        # )

        self.value = nn.Sequential(
            nn.Linear(base_channels * 8 * 8, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.advantage = nn.Sequential(
            nn.Linear(base_channels * 8 * 8, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64)
        )

    def forward(self, x):
        # x shape: (batch, in_channels, 8, 8)
        x = F.relu(self.bn_in(self.conv_in(x)))
        x = self.res_blocks(x)
        x = self.flatten(x)

        value = self.value(x)
        advantage = self.advantage(x)
        
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


if __name__ == "__main__":
    model = OthelloPlayer()
    total = sum(p.numel() for p in model.parameters())
    print(total)
