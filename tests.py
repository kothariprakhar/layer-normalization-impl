import unittest
import torch
import torch.nn as nn

# Assuming the provided code is available in the scope or imported.
# For the purpose of this test suite, we reference the classes directly.

class TestLayerNormImplementation(unittest.TestCase):
    def setUp(self):
        self.batch_size = 8
        self.features = 16
        self.eps = 1e-5
        self.ln = CustomLayerNorm(self.features, self.eps)

    def test_output_shape(self):
        x = torch.randn(self.batch_size, self.features)
        out = self.ln(x)
        self.assertEqual(out.shape, x.shape)

    def test_normalization_statistics(self):
        """Check if output has mean 0 and std 1 (roughly) along feature dim."""
        x = torch.randn(self.batch_size, self.features)
        # Initialize gamma to 1 and beta to 0 to test pure normalization
        nn.init.ones_(self.ln.gamma)
        nn.init.zeros_(self.ln.beta)
        
        out = self.ln(x)
        
        # Mean should be close to 0
        means = out.mean(dim=-1)
        self.assertTrue(torch.allclose(means, torch.zeros_like(means), atol=1e-5))
        
        # Std/Var should be close to 1 (calculated with unbiased=False)
        # Note: var(out) ~= 1. 
        vars_ = out.var(dim=-1, unbiased=False)
        self.assertTrue(torch.allclose(vars_, torch.ones_like(vars_), atol=1e-3))

    def test_compare_with_pytorch_native(self):
        """Compare CustomLayerNorm against nn.LayerNorm."""
        x = torch.randn(self.batch_size, self.features)
        
        # PyTorch LayerNorm
        # elementwise_affine=True by default, matching our custom one
        pytorch_ln = nn.LayerNorm(self.features, eps=self.eps)
        
        # Sync parameters
        with torch.no_grad():
            self.ln.gamma.copy_(pytorch_ln.weight)
            self.ln.beta.copy_(pytorch_ln.bias)

        out_custom = self.ln(x)
        out_pytorch = pytorch_ln(x)

        self.assertTrue(torch.allclose(out_custom, out_pytorch, atol=1e-6))

class TestLayerNormLSTMCell(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.input_size = 10
        self.hidden_size = 20
        self.cell = LayerNormLSTMCell(self.input_size, self.hidden_size)

    def test_forward_step(self):
        x = torch.randn(self.batch_size, self.input_size)
        h_prev = torch.zeros(self.batch_size, self.hidden_size)
        c_prev = torch.zeros(self.batch_size, self.hidden_size)

        h_t, c_t = self.cell(x, (h_prev, c_prev))

        self.assertEqual(h_t.shape, (self.batch_size, self.hidden_size))
        self.assertEqual(c_t.shape, (self.batch_size, self.hidden_size))

    def test_gradient_flow(self):
        """Ensure gradients propagate through the cell and LN layers."""
        x = torch.randn(self.batch_size, self.input_size, requires_grad=True)
        h_prev = torch.randn(self.batch_size, self.hidden_size)
        c_prev = torch.randn(self.batch_size, self.hidden_size)

        h_t, c_t = self.cell(x, (h_prev, c_prev))
        loss = h_t.sum()
        loss.backward()

        # Check if gradients exist for input weights and LN parameters
        self.assertIsNotNone(self.cell.weight_ih.grad)
        self.assertIsNotNone(self.cell.weight_hh.grad)
        self.assertIsNotNone(self.cell.ln_ih.gamma.grad)
        self.assertIsNotNone(self.cell.ln_c.beta.grad)
        self.assertIsNotNone(x.grad)

class TestSequenceModel(unittest.TestCase):
    def test_full_sequence_forward(self):
        batch_size = 2
        input_size = 28
        hidden_size = 32
        num_classes = 10
        seq_len = 28

        model = SequenceModel(input_size, hidden_size, num_classes)
        # Input shape: [Batch, Seq, Features]
        dummy_input = torch.randn(batch_size, seq_len, input_size)
        
        output = model(dummy_input)
        self.assertEqual(output.shape, (batch_size, num_classes))

if __name__ == '__main__':
    unittest.main()