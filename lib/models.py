"""Model Defination."""
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size=800, dropout=0.1, max_len=500):
        super().__init__()
        self.pe = nn.Embedding(max_len, hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        indexs = torch.arange(0, x.size(1), dtype=torch.long, device=x.device)
        x = x + self.pe(indexs).unsqueeze(dim=0)
        return self.dropout(self.norm(x))


class FACT(nn.Module):
    def __init__(self,
                 m_feat_dim=219, a_feat_dim=35, out_seq_len=20,
                 hidden_size=800, n_head=10, dim_feedforward=8192):
        super().__init__()
        self.out_seq_len = out_seq_len

        self.audio_transformer = nn.Sequential(
            nn.Linear(a_feat_dim, hidden_size),
            PositionalEncoding(hidden_size),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(hidden_size, n_head, dim_feedforward),
                num_layers=2)
        )
        self.motion_transformer = nn.Sequential(
            nn.Linear(m_feat_dim, hidden_size),
            PositionalEncoding(hidden_size),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(hidden_size, n_head, dim_feedforward),
                num_layers=2)
        )
        self.cross_modal_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, n_head, dim_feedforward),
            num_layers=12)
        self.last_layer = nn.Linear(hidden_size, m_feat_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, motion, audio):
        """
        Args:
            motion: motion features. [batch_size, m_seq_len, m_feat_dim]
            audio: audio features. [batch_size, a_seq_len, a_feat_dim]
        Returns:
            predicted future motion features. [batch_size, out_seq_len, m_feat_dim]
        """
        # audio_embed: [batch_size, a_seq_len, embed_dim]
        audio_embed = self.audio_transformer(audio)
        # motion_embed: [batch_size, m_seq_len, embed_dim]
        motion_embed = self.motion_transformer(motion)
        # embed: [batch_size, m_seq_len + a_seq_len, embed_dim]
        embed = torch.cat([motion_embed, audio_embed], dim=1)
        # out: [batch_size, m_seq_len + a_seq_len, embed_dim]
        out = self.cross_modal_transformer(embed)
        # out: [batch_size, m_seq_len + a_seq_len, m_feat_dim]
        out = self.last_layer(out)
        return out[:, :self.out_seq_len, :]


if __name__ == "__main__":
    model = FACT()
    motion = torch.randn(16, 120, 219)
    audio = torch.randn(16, 240, 35)

    out = model(motion, audio)
    print (out.shape)
