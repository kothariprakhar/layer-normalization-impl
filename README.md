# Layer Normalization

Training state-of-the-art, deep neural networks is computationally expensive. One way to reduce the training time is to normalize the activities of the neurons. A recently introduced technique called batch normalization uses the distribution of the summed input to a neuron over a mini-batch of training cases to compute a mean and variance which are then used to normalize the summed input to that neuron on each training case. This significantly reduces the training time in feed-forward neural networks. However, the effect of batch normalization is dependent on the mini-batch size and it is not obvious how to apply it to recurrent neural networks. In this paper, we transpose batch normalization into layer normalization by computing the mean and variance used for normalization from all of the summed inputs to the neurons in a layer on a single training case. Like batch normalization, we also give each neuron its own adaptive bias and gain which are applied after the normalization but before the non-linearity. Unlike batch normalization, layer normalization performs exactly the same computation at training and test times. It is also straightforward to apply to recurrent neural networks by computing the normalization statistics separately at each time step. Layer normalization is very effective at stabilizing the hidden state dynamics in recurrent networks. Empirically, we show that layer normalization can substantially reduce the training time compared with previously published techniques.

## Implementation Details

# Layer Normalization: Deep Dive & Implementation

This explanation covers the theoretical underpinnings of Layer Normalization (Ba et al., 2016) and dissects the provided PyTorch implementation which applies the concept to a Recurrent Neural Network (RNN) solving a sequential image classification task.

## 1. The Core Problem: Internal Covariate Shift in RNNs

Before Layer Normalization (LN), **Batch Normalization (BN)** was the gold standard. BN normalizes activations using the mean and variance of the *current mini-batch*.

While BN works wonders for Convolutional Neural Networks (CNNs), it has two major flaws regarding Recurrent Neural Networks (RNNs):
1.  **Sequence Length Dependency**: In an RNN, time step $t$ corresponds to a distinct layer in the unrolled graph. If the training sequences vary in length, or if test sequences are longer than training sequences, the batch statistics computed at specific time steps become unreliable or invalid.
2.  **Batch Size Constraints**: BN requires a sufficiently large batch size to estimate population statistics. RNNs are often memory-intensive, forcing small batch sizes where BN estimates become noisy.

## 2. The Solution: Layer Normalization

Layer Normalization solves this by transposing the normalization.

### Mathematical Formulation
Instead of normalizing across the batch dimension $N$, we normalize across the feature dimension $D$ for a **single sample**.

Given an input vector $x$ (activations for one sample at one time step) containing $H$ hidden units:

1.  **Mean ($\\mu$)**: 
    $$ \\mu = \\frac{1}{H} \\sum_{i=1}^{H} x_i $$
2.  **Variance ($\\sigma^2$)**:
    $$ \\sigma^2 = \\frac{1}{H} \\sum_{i=1}^{H} (x_i - \\mu)^2 $$
3.  **Normalization**:
    $$ \\hat{x} = \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} $$
4.  **Scale and Shift** (Learnable Parameters):
    $$ y = \\gamma \\odot \\hat{x} + \\beta $$

Crucially, $\\mu$ and $\\sigma$ depend only on the current input vector, not on other samples in the batch. This makes LN perfectly suited for RNNs, as the same normalization calculation applies to every time step independently.

## 3. Implementation Walkthrough

### The `CustomLayerNorm` Class
The code implements the math directly using PyTorch operations:

```python
mean = x.mean(dim=-1, keepdim=True)
var = x.var(dim=-1, keepdim=True, unbiased=False)
x_norm = (x - mean) / torch.sqrt(var + self.eps)
output = x_norm * self.gamma + self.beta
```
*   **`dim=-1`**: This targets the feature dimension (hidden size), ensuring we calculate statistics per sample.
*   **`self.gamma` / `self.beta`**: These allow the network to restore the representational power if normalization isn't optimal for a specific layer. They are initialized to 1 and 0 respectively (identity transformation).

### The `LayerNormLSTMCell`
The paper's most impactful contribution was stabilizing RNNs. We implemented a custom LSTM cell to demonstrate this.

Standard LSTM equations involve linear transforms like $W_{ii}x_t + b_{ii}$. In our LN-LSTM:
1.  We compute the linear term: `input_proj = W_x(x)`.
2.  We apply **Layer Norm** immediately: `ln_input = LN(input_proj)`.
3.  We do the same for the hidden state projection.

This keeps the inputs to the sigmoids and tanh activations within a stable range (approx -1 to 1 for normalized data), preventing the **vanishing/exploding gradient problem** effectively.

### Data Strategy: Sequential MNIST
To demonstrate the RNN capability without requiring massive NLP resources, we used **Sequential MNIST**.
*   **Concept**: We treat a $28 \\times 28$ image as a sequence of 28 vectors (rows), each of size 28.
*   **Why?**: This forces the network to maintain a memory of the top of the digit while reading the bottom. It turns a static image classification problem into a temporal dependency problem, which is the perfect testbed for the stability provided by Layer Normalization.

## 4. Key Advantages Demonstrated
1.  **Batch Independence**: The model works identically whether `BATCH_SIZE` is 1 or 64. The normalization statistics are calculated per sample.
2.  **Training Stability**: RNNs are notoriously hard to train. By normalizing the gates (Input, Forget, Output), Layer Norm allows for higher learning rates and faster convergence compared to standard LSTMs.
3.  **Simplicity**: The logic performs the exact same computation at inference time as it does at training time (unlike Batch Norm, which requires running averages).

## Verification & Testing

The code provides a mathematically correct implementation of Layer Normalization (LN) and correctly applies it to an LSTM cell as described in the Ba et al. (2016) paper.

### Strengths
1. **Correct Statistics**: The usage of `unbiased=False` in `x.var` is correct. Unlike Batch Normalization which often uses unbiased estimates for tracking, Layer Normalization uses the biased estimate (dividing by N) for the immediate normalization of the layer inputs, which matches the paper's definition.
2. **LSTM Integration**: The application of LN to the computed gates (`LN(Wx)` and `LN(Uh)`) and the cell state (`c_t`) aligns with the standard LN-LSTM architecture.
3. **Broadcasting**: The `CustomLayerNorm` handles the `normalized_shape` correctly for the last dimension, allowing it to work on both 2D `(Batch, Features)` and 3D `(Batch, Seq, Features)` inputs seamlessly via broadcasting.

### Minor Observations
1. **Bias Redundancy**: In `LayerNormLSTMCell`, both `self.ln_ih` and `self.ln_hh` contain learnable bias parameters (`beta`). When `gates_ih` and `gates_hh` are summed, these biases are added together. While mathematically valid (the optimizer will simply learn the sum), it introduces a slight redundancy in parameters compared to an implementation that calculates `LN(Wx) + LN(Uh) + b` with a single external bias vector.
2. **Hardcoded Dimension**: `CustomLayerNorm` assumes the feature dimension is always the last dimension (`dim=-1`). While correct for this LSTM implementation, a more generic LayerNorm (like PyTorch's native `nn.LayerNorm`) handles normalization over multiple dimensions if specified in `normalized_shape`.