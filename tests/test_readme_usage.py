import unittest

import torch
from torch import nn

from gradient_reversal import GradientReversal, revgrad


class TestReadmeUsage(unittest.TestCase):
    def test_revgrad_matches_readme_behavior(self):
        alpha = torch.tensor([1.0])

        x = torch.tensor([4.0], requires_grad=True)
        x_rev = torch.tensor([4.0], requires_grad=True)

        y = x * 5
        y = y + 6

        y_rev = x_rev * 5
        y_rev = revgrad(y_rev, alpha)
        y_rev = y_rev + 6

        y.backward()
        y_rev.backward()

        self.assertTrue(torch.equal(x.grad, -x_rev.grad))
        self.assertEqual(x.grad.item(), 5.0)
        self.assertEqual(x_rev.grad.item(), -5.0)

    def test_module_usage_produces_reversed_gradient(self):
        net = nn.Sequential(
            nn.Linear(1, 1, bias=False),
            GradientReversal(alpha=1.0),
        )
        with torch.no_grad():
            net[0].weight.fill_(1.0)

        x = torch.tensor([[2.0]], requires_grad=True)
        y = net(x)
        loss = y.sum()
        loss.backward()

        # The linear layer has weight 1, so upstream gradient is 1.
        # GradientReversal(alpha=1) flips it to -1 before reaching input.
        self.assertEqual(x.grad.item(), -1.0)


if __name__ == "__main__":
    unittest.main()
