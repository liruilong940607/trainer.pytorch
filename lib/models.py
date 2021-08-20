"""Model Defination."""
import torch
import torch.nn as nn
import tqdm


def truncated_normal(t, mean=0.0, std=0.02):
    torch.nn.init.normal_(t, mean=mean, std=std)
    while True:
      cond = torch.logical_or(t < mean - 2*std, t > mean + 2*std)
      if not torch.sum(cond):
        break
      t = torch.where(cond, torch.nn.init.normal_(torch.ones(t.shape), mean=mean, std=std), t)
    return t


class Dense(nn.Linear):
  
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    torch.nn.init.xavier_uniform_(self.weight)
    if self.bias is not None:
      torch.nn.init.zeros_(self.bias)


class Norm(nn.Module):
  """Layer normalization."""

  def __init__(self, fn, dim):
    super().__init__()
    self.norm = nn.LayerNorm(dim)
    self.fn = fn

  def forward(self, x):
    return self.fn(self.norm(x))


class Residual(nn.Module):
  """Residual layer."""

  def __init__(self, fn):
    super().__init__()
    self.fn = fn

  def forward(self, x):
    return self.fn(x) + x


class MLP(nn.Module):
  """Feedforward layer."""

  def __init__(self, in_dim, out_dim, hidden_dim):
    super().__init__()
    self.net = nn.Sequential(
        Dense(in_dim, hidden_dim),
        nn.GELU(),
        Dense(hidden_dim, out_dim)
    )

  def forward(self, x):
    return self.net(x)


class Attention(nn.Module):
  """Attention layer."""

  def __init__(self, in_dim, out_dim, heads=10):
    super().__init__()
    self.heads = heads
    self.scale = out_dim**-0.5

    self.to_qkv = Dense(in_dim, out_dim * 3, bias=False)
    self.to_out = Dense(out_dim, out_dim)

  def forward(self, x):
    batch_size, seq_len, feature_dim = x.shape
    assert feature_dim % self.heads == 0
    # [batch_size, seq_len, 3, heads, dim]
    qkv = self.to_qkv(x).view(
        batch_size, seq_len, 3, self.heads, feature_dim // self.heads
    )
    # [3, batch_size, heads, seq_len, dim]
    qkv = qkv.permute(2, 0, 3, 1, 4)
    # [batch_size, heads, seq_len, dim]
    q, k, v = qkv[0, ...], qkv[1, ...], qkv[2, ...]

    dots = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
    attn = torch.nn.functional.softmax(dots, dim=-1)

    out = torch.einsum("bhij,bhjd->bhid", attn, v)
    out = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, feature_dim)
    out = self.to_out(out)
    return out


class Transformer(nn.Module):
  """Transformer Encoder."""

  def __init__(self,
               hidden_size=800,
               num_hidden_layers=12,
               num_attention_heads=10,
               intermediate_size=3072):
    super().__init__()
    blocks = []
    for _ in range(num_hidden_layers):
      blocks.extend([
          Residual(Norm(
            Attention(hidden_size, hidden_size, heads=num_attention_heads), 
            dim=hidden_size
          )),
          Residual(Norm(
            MLP(hidden_size, hidden_size, intermediate_size), 
            dim=hidden_size
          ))
      ])
    self.net = nn.Sequential(*blocks)

  def forward(self, x):
    return self.net(x)


class LinearEmbedding(nn.Module):
  """Linear projection."""

  def __init__(self, in_dim, out_dim):
    super().__init__()
    self.net = Dense(in_dim, out_dim)

  def forward(self, x):
    return self.net(x)


class PositionEmbedding(nn.Module):
  """Position Embedding layer."""

  def __init__(self, seq_length, dim):
    super().__init__()

    self.pos_embedding = nn.Parameter(
      truncated_normal(torch.empty(1, seq_length, dim)), requires_grad=True
    )

  def forward(self, x):
    """Call embedding layer."""
    return x + self.pos_embedding


class CrossModalLayer(nn.Module):
  """Cross-modal layer."""

  def __init__(self, hidden_size=800, out_dim=219):
    super().__init__()
    self.transformer_layer = Transformer(
        hidden_size=hidden_size,
        num_hidden_layers=12,
        num_attention_heads=10,
        intermediate_size=3072)
    self.cross_output_layer = Dense(hidden_size, out_dim)
    self.cross_output_layer.weight.data = truncated_normal(self.cross_output_layer.weight)

  def forward(self, modal_a_sequences, modal_b_sequences):
    """Get output for the cross-modal tasks."""
    _, _, modal_a_width = modal_a_sequences.shape
    _, _, modal_b_width = modal_b_sequences.shape
    if modal_a_width != modal_b_width:
      raise ValueError(
          "The modal_a hidden size (%d) should be the same with the modal_b "
          "hidden size (%d)" % (modal_a_width, modal_b_width))
    # [batch_size, modal_a_seq_len + modal_b_seq_len, width]
    merged_sequences = torch.cat([modal_a_sequences, modal_b_sequences], dim=1)
    # [batch_size, modal_a_seq_len + modal_b_seq_len, width]
    merged_sequences = self.transformer_layer(merged_sequences)
    logits = self.cross_output_layer(merged_sequences)
    return logits


class FACTModel(nn.Module):
  """Audio Motion Multi-Modal model."""

  def __init__(self):
    """Initializer for FACTModel."""
    super().__init__()
    self.motion_transformer = Transformer(num_hidden_layers=2)
    self.motion_linear_embedding = LinearEmbedding(in_dim=219, out_dim=800)
    self.motion_pos_embedding = PositionEmbedding(seq_length=120, dim=800)
    self.audio_transformer = Transformer(num_hidden_layers=2)
    self.audio_linear_embedding = LinearEmbedding(in_dim=35, out_dim=800)
    self.audio_pos_embedding = PositionEmbedding(seq_length=240, dim=800)
    self.cross_modal_layer = CrossModalLayer(hidden_size=800, out_dim=219)

  def forward(self, motion_input, audio_input):
    """Predict sequences from inputs.

    Args:
      inputs: Input dict of tensors, the output from the provide_inputs().

    Returns:
      motion_sequence_output: Tensor of shape
        [batch_size, seq_length, motion_feature_dimension]
      motion_last_output: Tensor of shape [batch_size, motion_feature_dimension]
    """
    # Computes motion features.
    motion_features = self.motion_linear_embedding(motion_input)
    # `motion_features` shape = [batch_size, seq_length, hidden_size].
    motion_features = self.motion_pos_embedding(motion_features)
    motion_features = self.motion_transformer(motion_features)
    # Computes audio features.
    audio_features = self.audio_linear_embedding(audio_input)
    audio_features = self.audio_pos_embedding(audio_features)
    audio_features = self.audio_transformer(audio_features)
    # Computes cross modal output.
    output = self.cross_modal_layer(motion_features, audio_features)
    return output

  def loss(self, target, pred):
    target_seq_len = target.shape[1]
    return torch.nn.functional.mse_loss(target, pred[:, :target_seq_len])

  @torch.no_grad()
  def inference(self, motion_seed, audio_full, gen_seq_length=120):
    motion_frames = [motion_seed[:, i:i+1] for i in range(motion_seed.shape[1])]
    audio_frames = [audio_full[:, i:i+1] for i in range(audio_full.shape[1])]
    results = []
    for _ in tqdm.tqdm(range(gen_seq_length)):
      assert len(motion_frames) == 120, len(motion_frames)
      motion_input = torch.cat(motion_frames, dim=1)
      if len(audio_frames) < 240:
        break
      audio_input = torch.cat(audio_frames[:240], dim=1) 
      output = self.forward(motion_input, audio_input)[:, 0:1]  # first frame
      results.append(output)
      motion_frames.append(output)
      motion_frames.pop(0)
      audio_frames.pop(0)
    return torch.cat(results, dim=1)


if __name__ == "__main__":
    model = FACTModel()
    motion = torch.randn(16, 120, 219)
    audio = torch.randn(16, 240, 35)

    out = model(motion, audio)
    print (out.shape)
