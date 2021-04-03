import torch

from data import data
from model import GCN


class Runtime:
    def __init__(self, data, hidden_channels: int) -> None:
        self.data = data
        self.model = GCN(hidden_channels=hidden_channels)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=0.01,
                                     weight_decay=5e-4)
        self.criterion = torch.nn.CrossEntropyLoss()

    def train(self):
        self.model.train()
        self.optimizer.zero_grad()                                  # Clear gradients.
        out = self.model(self.data.x, self.data.edge_index)         # Perform a single forward pass.
        loss = self.criterion(out[self.data.train_mask],
                              self.data.y[self.data.train_mask])    # Compute the loss solely based on the training nodes.
        loss.backward()                                             # Derive gradients.
        self.optimizer.step()                                       # Update parameters based on gradients.
        return loss

    def test(self):
      self.model.eval()
      out = self.model(self.data.x, self.data.edge_index)
      pred = out.argmax(dim=1)                                                      # Use the class with highest probability.
      test_correct = pred[self.data.test_mask] == self.data.y[self.data.test_mask]  # Check against ground-truth labels.
      test_acc = int(test_correct.sum()) / int(self.data.test_mask.sum())           # Derive ratio of correct predictions.
      return test_acc