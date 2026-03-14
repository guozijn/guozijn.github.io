---
title: Transformer
tags:
  - machine learning
  - transformer
---

## Transformer Core Concepts

### From Tokens to Embeddings

Raw tokens are first mapped to dense vectors through an embedding matrix so that the model can work in a continuous space. The embedding size (`n_embd`) defines the dimensionality of this space and controls both the model capacity and its memory footprint.

### Positional Information

Because self-attention is permutation-invariant, Transformers inject order information with positional encodings. Classical sinusoidal encodings let the model generalise to longer sequences, while learnable embeddings allow the model to adapt positions during training. Modern variants sometimes rely on relative position encodings or rotary embeddings to better capture long context interactions.

### Scaled Dot-Product Self-Attention

For each token, the model projects embeddings into queries (Q), keys (K), and values (V). Attention weights are computed as `softmax(QKᵀ / sqrt(d_k))`, where `d_k` is the head dimension to prevent large dot products from saturating the softmax. The output is a weighted sum of the value vectors, allowing every position to gather information from the entire context window (`block_size`).

### Multi-Head Attention

Multiple attention heads run in parallel on different learned projections of the same sequence. This design allows the model to capture heterogeneous relationships (syntax, long-range dependencies, coreference) in the same layer. The concatenated head outputs are linearly projected back into the model dimension.

### Position-Wise Feed-Forward Network

Each Transformer block follows attention with a two-layer feed-forward network applied independently to every position. A typical configuration is `Linear(n_embd → 4 × n_embd)`, an activation (GELU or ReLU), then `Linear(4 × n_embd → n_embd)`. This component mixes features learned by attention and introduces non-linearity.

### Residual Connections and Normalisation

Skip connections wrap both the attention sublayer and the feed-forward sublayer so that gradients flow directly to earlier blocks. LayerNorm (or RMSNorm in some modern designs) keeps activations well-scaled during training. Variants such as Pre-LN place the normalisation before each sublayer, which improves stability for deeper models.

### Encoder-Decoder vs. Decoder-Only

The original Transformer pairs an encoder that builds contextualised representations with a decoder that performs autoregressive generation, both stacked with attention and feed-forward modules. Many language models today use only the decoder stack with causal masking, which enforces that each token can only attend to previous positions, enabling left-to-right generation.

### Training and Scaling Considerations

- **Optimiser choice**: AdamW remains the default, but large models may benefit from learning rate warm-up, cosine decay, and parameter-specific weight decay.
- **Regularisation**: Dropout complements attention masking, while techniques such as label smoothing or stochastic depth can help deep stacks converge.
- **Precision and compilation**: Training in mixed precision (`bfloat16`/`fp16`) and enabling compiler optimisations (`torch.compile`) significantly reduces memory use and speeds up training.
- **Scaling laws**: Empirically, model performance improves predictably with more data, parameters, and compute, guiding decisions about `n_layer`, `n_head`, and dataset size.

### Inference-Time Generation

During autoregressive generation, the model caches key-value pairs to avoid recomputing attention for past tokens. Sampling strategies such as temperature, top-k, and nucleus sampling trade off creativity against determinism. For instruction-following models, additional techniques like contrastive decoding or aligning with human feedback further shape the output distribution.

### PyTorch Skeleton

```python
import torch
import torch.nn as nn


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int = 192,
        n_embd: int = 192,
        n_layer: int = 3,
        n_head: int = 6,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.token_embed = nn.Embedding(vocab_size, n_embd)
        self.pos_embed = nn.Parameter(torch.zeros(1, block_size, n_embd))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=n_head,
            dim_feedforward=4 * n_embd,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.layers = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        self.norm = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        if idx.size(1) > self.block_size:
            raise ValueError("Sequence length exceeds block size.")
        x = self.token_embed(idx) + self.pos_embed[:, : idx.size(1)]
        x = self.layers(x)
        x = self.norm(x)
        return self.lm_head(x)


def training_step(model, batch, optimizer, scaler=None):
    model.train()
    inputs, targets = batch
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(enabled=scaler is not None):
        logits = model(inputs)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )
    if scaler:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return loss.item()
```

## Hyperparameters
### Minimal Viable Training Config

| **Parameter**       | **Sample Value**      | **Meaning**                          |
| ------------------- | --------------------- | ------------------------------------ |
| `batch_size`        | `48`                  | Samples per optimisation step        |
| `block_size`        | `192`                 | Context window length                |
| `max_iters`         | `300`                 | Maximum number of optimisation steps |
| `learning_rate`     | `3e-4`                | Optimiser step size                  |
| `n_embd`            | `192`                 | Transformer embedding dimension      |
| `n_head`            | `6`                   | Number of attention heads            |
| `n_layer`           | `3`                   | Number of Transformer layers         |
| `dropout`           | `0.2`                 | Regularisation probability           |

### Full Training Configuration

| **Category**                | **Parameter**       | **Sample value**                            | **Meaning**                                         |
| --------------------------- | ------------------- | ------------------------------------------- | --------------------------------------------------- |
| **Data**                    | `block_size`        | `192`                                       | Context window length                               |
|                             | `vocab_size`        | *(auto from tokenizer)*                     | Number of tokens in the vocabulary                  |
| **Model**                   | `n_embd`            | `192`                                       | Embedding dimension                                 |
|                             | `n_head`            | `6`                                         | Number of attention heads                           |
|                             | `n_layer`           | `3`                                         | Transformer depth                                   |
|                             | `dropout`           | `0.2`                                       | Dropout probability                                 |
|                             | `tie_weights`       | `True`                                      | Share token embedding and output projection weights |
| **Training Loop**           | `batch_size`        | `48`                                        | Number of samples per update                        |
|                             | `max_iters`         | `300`                                       | Total training iterations                           |
|                             | `grad_clip`         | `1.0`                                       | Gradient norm clipping                              |
| **Optimiser**               | `learning_rate`     | `3e-4`                                      | Base learning rate                                  |
|                             | `weight_decay`      | `0.1`                                       | AdamW weight decay                                  |
|                             | `betas`             | `(0.9, 0.95)`                               | AdamW momentum coefficients                         |
|                             | `eps`               | `1e-8`                                      | AdamW epsilon                                       |
| **LR Scheduler**            | `lr_decay`          | `True`                                      | Enable learning rate decay                          |
|                             | `warmup_iters`      | `100`                                       | Warm-up steps before decay                          |
|                             | `min_lr`            | `1e-5`                                      | Final learning rate after decay                     |
|                             | `scheduler_type`    | `"cosine"`                                  | Scheduler function                                  |
| **Precision / Hardware**    | `device`            | `"cuda"`                                    | Compute device                                      |
|                             | `dtype`             | `"bfloat16"`                                | Precision mode                                      |
|                             | `compile`           | `True`                                      | Enable Torch 2.x compile optimisation               |
| **Validation / Early Stop** | `eval_interval`     | `100`                                       | Evaluation frequency                                |
|                             | `eval_iters`        | `20`                                        | Mini-batches used for validation loss estimation    |
|                             | `patience`          | `6`                                         | Early stopping patience                             |
|                             | `min_delta`         | `1e-3`                                      | Minimum improvement threshold                       |
| **Checkpoint / Logging**    | `save_interval`     | `100`                                       | Model checkpoint interval                           |
|                             | `log_interval`      | `50`                                        | Logging interval                                    |
|                             | `wandb_project`     | `"gpt-debug"`                               | Optional logging project name                       |
| **Generation**              | `temperature`       | `0.8`                                       | Softmax temperature for sampling                    |
|                             | `top_k`             | `50`                                        | Top-K sampling                                      |
|                             | `top_p`             | `0.95`                                      | Nucleus sampling                                    |
|                             | `max_new_tokens`    | `200`                                       | Maximum number of new tokens to generate            |
