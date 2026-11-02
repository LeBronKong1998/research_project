import torch
from torch import nn

from .data import DT, FUTURE


class TrajectoryGAT(nn.Module):
    def __init__(self, hidden=64, modes=3, future=FUTURE):
        super().__init__()
        self.modes = modes
        self.future = future
        self.encoder = nn.Sequential(nn.Linear(6, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)
        self.edge = nn.Linear(2, 1)
        self.temporal = nn.GRU(hidden * 2, hidden, batch_first=True)
        self.mode_embedding = nn.Embedding(modes, hidden)
        self.decoder = nn.GRUCell(hidden + 2, hidden)
        self.motion = nn.Linear(hidden, 3)
        self.mode_head = nn.Linear(hidden, modes)

    def forward(self, history, mask):
        batch, steps, agents, _ = history.shape
        scale = history.new_tensor([40., 40., 15., 15., 1., 1.])
        encoded = self.encoder(history / scale)
        ego = encoded[:, :, 0]
        scores = (self.query(ego).unsqueeze(2) * self.key(encoded)).sum(-1) / encoded.shape[-1] ** 0.5
        relative = history[:, :, :, :2] - history[:, :, 0:1, :2]
        scores = scores + self.edge(relative / 40.).squeeze(-1)
        close = relative.square().sum(-1) < 40.0 ** 2
        scores = scores.masked_fill(~(mask[:, None, :] & close), -1e4)
        attention = torch.softmax(scores, dim=-1)
        context = (attention.unsqueeze(-1) * self.value(encoded)).sum(2)
        _, hidden = self.temporal(torch.cat((ego, context), -1))
        hidden = hidden[0]
        logits = self.mode_head(hidden)
        decoder_state = hidden[:, None, :] + self.mode_embedding.weight[None, :, :]
        position = history[:, -1, 0, :2][:, None, :].expand(-1, self.modes, -1)
        velocity = history[:, -1, 0, 2:4][:, None, :].expand(-1, self.modes, -1)
        outputs = []
        for _ in range(self.future):
            decoder_state = self.decoder(torch.cat((hidden[:, None, :].expand_as(decoder_state), velocity), -1).reshape(batch * self.modes, -1), decoder_state.reshape(batch * self.modes, -1)).reshape(batch, self.modes, -1)
            delta = self.motion(decoder_state)
            velocity = velocity + delta[..., :2] * DT
            position = position + velocity * DT
            heading = torch.atan2(velocity[..., 1], velocity[..., 0]) + 0.05 * torch.tanh(delta[..., 2])
            outputs.append(torch.cat((position, velocity, heading.unsqueeze(-1)), -1))
        return torch.stack(outputs, dim=2), logits


def multimodal_loss(prediction, logits, truth):
    position = torch.linalg.vector_norm(prediction[..., :2] - truth[:, None, :, :2], dim=-1).mean(-1)
    velocity = torch.linalg.vector_norm(prediction[..., 2:4] - truth[:, None, :, 2:4], dim=-1).mean(-1)
    target_heading = torch.atan2(truth[..., 4], truth[..., 5])[:, None, :]
    heading_error = torch.atan2(torch.sin(prediction[..., 4] - target_heading), torch.cos(prediction[..., 4] - target_heading)).abs().mean(-1)
    per_mode = position + 0.3 * velocity + 0.1 * heading_error
    best = per_mode.argmin(-1)
    return per_mode.gather(1, best[:, None]).mean() + 0.1 * nn.functional.cross_entropy(logits, best)
