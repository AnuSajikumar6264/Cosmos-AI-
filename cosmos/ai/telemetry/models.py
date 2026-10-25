"""Deep telemetry models used by COSMOS-AI.

Fragment classifiers (OPSSAT-AD: one labelled fragment -> P(anomalous))

``FCNN``                  MLP on the 18 official handcrafted features (re-implementation of
                          the best model of Ruszczak et al. 2025; their exact layer sizes are
                          not published, so ours are stated in the class)
``CNN1DClassifier``       fully convolutional network on the raw fragment (Wang et al. 2017
                          style: three conv blocks + global average pooling)
``PatchTransformerClassifier``  patch embedding + transformer encoder + mean pooling
``HybridClassifier``      CNN branch on the raw fragment + MLP branch on the features, late
                          fused; used to ask whether raw-signal learning adds to the features

Channel-independent forecasters (ESA-ADB: one channel's past -> quantiles of its next H steps)

``TCNForecaster``         causal dilated TCN (Bai et al. 2018); fully convolutional, so one pass
                          over a long record yields a forecast at every position
``PatchTSTForecaster``    PatchTST-style encoder (Nie et al., ICLR 2023) with reversible
                          instance normalisation

Both forecasters share weights across channels and add a learned channel embedding; the
optional telecommand stream enters through :class:`GatedCovariateFusion`
(``h' = h + sigmoid(W_g [h; e]) * W_e e``), so a zero gate recovers the telemetry-only model.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

QUANTILES = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)


# ============================================================================ classifiers
class FCNN(nn.Module):
    """MLP 18 -> 128 -> 64 -> 32 -> 1 with BatchNorm, SiLU and dropout."""

    def __init__(self, n_features: int = 18, hidden=(128, 64, 32), dropout: float = 0.2):
        super().__init__()
        layers, d = [], n_features
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.SiLU(), nn.Dropout(dropout)]
            d = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, 1)
        self.out_dim = d

    def embed(self, feats):
        return self.body(feats)

    def forward(self, x=None, feats=None, mask=None):
        return self.head(self.embed(feats)).squeeze(-1)


def _conv_block(i, o, k):
    return nn.Sequential(nn.Conv1d(i, o, k, padding=k // 2), nn.BatchNorm1d(o), nn.SiLU())


class CNN1DClassifier(nn.Module):
    """Conv(8) -> Conv(5) -> Conv(3) -> masked global average pool -> logit."""

    def __init__(self, in_ch: int = 3, width: int = 64, dropout: float = 0.2):
        super().__init__()
        self.body = nn.Sequential(_conv_block(in_ch, width, 7), _conv_block(width, 2 * width, 5),
                                  _conv_block(2 * width, width, 3))
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(width, 1)
        self.out_dim = width

    def embed(self, x, mask=None):
        h = self.body(x)                                      # (B, W, L)
        if mask is None:
            return h.mean(-1)
        m = mask[:, None, :]
        return (h * m).sum(-1) / m.sum(-1).clamp(min=1.0)

    def forward(self, x=None, feats=None, mask=None):
        return self.head(self.drop(self.embed(x, mask))).squeeze(-1)


class PatchTransformerClassifier(nn.Module):
    """Non-overlapping patches -> linear embedding -> transformer -> mean pool -> logit."""

    def __init__(self, in_ch: int = 3, length: int = 256, patch: int = 16, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        assert length % patch == 0
        self.patch = patch
        self.embed_in = nn.Linear(in_ch * patch, d_model)
        self.pos = nn.Parameter(torch.zeros(1, length // patch, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        enc = nn.TransformerEncoderLayer(d_model, n_heads, 2 * d_model, dropout,
                                         batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)
        self.out_dim = d_model

    def embed(self, x, mask=None):
        b, c, length = x.shape
        p = x.reshape(b, c, length // self.patch, self.patch).permute(0, 2, 1, 3)
        h = self.embed_in(p.reshape(b, length // self.patch, c * self.patch)) + self.pos
        return self.norm(self.encoder(h)).mean(1)

    def forward(self, x=None, feats=None, mask=None):
        return self.head(self.embed(x, mask)).squeeze(-1)


class HybridClassifier(nn.Module):
    """Late fusion of a raw-signal CNN embedding and a feature-MLP embedding."""

    def __init__(self, in_ch: int = 3, n_features: int = 18, width: int = 64,
                 dropout: float = 0.2):
        super().__init__()
        self.cnn = CNN1DClassifier(in_ch, width, dropout)
        self.mlp = FCNN(n_features, (64, 32), dropout)
        self.head = nn.Sequential(nn.Linear(width + 32, 32), nn.SiLU(), nn.Dropout(dropout),
                                  nn.Linear(32, 1))
        self.out_dim = width + 32

    def embed(self, x, feats, mask=None):
        return torch.cat([self.cnn.embed(x, mask), self.mlp.embed(feats)], -1)

    def forward(self, x=None, feats=None, mask=None):
        return self.head(self.embed(x, feats, mask)).squeeze(-1)


# ============================================================================ forecasters
class GatedCovariateFusion(nn.Module):
    """``h + sigmoid(W_g [h; e]) * (W_e e)``: gate 0 recovers the telemetry-only model."""

    def __init__(self, d_model: int, d_cov: int):
        super().__init__()
        self.proj = nn.Linear(d_cov, d_model)
        self.gate = nn.Linear(d_model + d_cov, d_model)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)             # start almost closed

    def forward(self, h, e):                               # h, e: (..., d)
        g = torch.sigmoid(self.gate(torch.cat([h, e], -1)))
        return h + g * self.proj(e), g


class _CausalConv(nn.Module):
    def __init__(self, i, o, k, dilation):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv = nn.Conv1d(i, o, k, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))


class ChannelNorm(nn.Module):
    """LayerNorm over channels at each time step.

    Causal by construction.  GroupNorm(1, d) would normalise over the time axis too, which in
    a causal model leaks future samples into every position (and makes a 512-step training
    window behave differently from a long inference chunk).
    """

    def __init__(self, d):
        super().__init__()
        self.ln = nn.LayerNorm(d)

    def forward(self, x):                                   # (B, C, T)
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


class _TCNBlock(nn.Module):
    def __init__(self, d, k, dilation, dropout):
        super().__init__()
        self.c1 = _CausalConv(d, d, k, dilation)
        self.c2 = _CausalConv(d, d, k, dilation)
        self.n1 = ChannelNorm(d)
        self.n2 = ChannelNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.drop(F.silu(self.n1(self.c1(x))))
        h = self.drop(F.silu(self.n2(self.c2(h))))
        return x + h


class TCNForecaster(nn.Module):
    """Causal TCN: input (B, 1 + K, T) -> quantiles (B, T, H, Q) of steps t+1 .. t+H.

    Inputs are already robust-normalised per channel (train-half median / IQR).  Receptive
    field = 1 + 2 (k - 1)(2^levels - 1).
    """

    def __init__(self, n_channels: int, horizon: int = 16, d_model: int = 64, levels: int = 9,
                 kernel: int = 3, n_cov: int = 0, dropout: float = 0.1,
                 quantiles=QUANTILES):
        super().__init__()
        self.horizon, self.quantiles = horizon, tuple(quantiles)
        self.inp = nn.Conv1d(1, d_model, 1)
        self.chan = nn.Embedding(n_channels, d_model)
        self.blocks = nn.ModuleList([_TCNBlock(d_model, kernel, 2 ** i, dropout)
                                     for i in range(levels)])
        self.receptive_field = 1 + 2 * (kernel - 1) * (2 ** levels - 1)
        self.n_cov = n_cov
        if n_cov:
            self.cov_enc = nn.Sequential(_CausalConv(n_cov, d_model, 9, 1), nn.SiLU(),
                                         _CausalConv(d_model, d_model, 9, 4))
            self.fuse = GatedCovariateFusion(d_model, d_model)
        self.head = nn.Conv1d(d_model, horizon * len(self.quantiles), 1)

    def forward(self, x, chan_id, cov=None, cov_dropout: float = 0.0, last_only: bool = False):
        """``last_only`` evaluates the head at the final position only (training)."""
        h = self.inp(x[:, :1]) + self.chan(chan_id)[:, :, None]
        for blk in self.blocks:
            h = blk(h)
        gate = None
        if self.n_cov and cov is not None:
            e = self.cov_enc(cov)
            if self.training and cov_dropout > 0:            # modality dropout
                keep = (torch.rand(e.shape[0], 1, 1, device=e.device) > cov_dropout).float()
                e = e * keep
            hh, gate = self.fuse(h.transpose(1, 2), e.transpose(1, 2))
            h = hh.transpose(1, 2)
        if last_only:
            h = h[:, :, -1:]
        q = self.head(h)                                      # (B, H*Q, T)
        b, _, t = q.shape
        q = q.view(b, self.horizon, len(self.quantiles), t).permute(0, 3, 1, 2)
        return _sort_quantiles(q), gate


class PatchTSTForecaster(nn.Module):
    """Context window (B, 1 + K, L) -> quantiles (B, H, Q) of the next H steps.

    Reversible instance normalisation (RevIN) removes the window's level and scale before
    the encoder and restores them on the output — the standard PatchTST recipe for
    non-stationary series.
    """

    def __init__(self, n_channels: int, context: int = 512, horizon: int = 16,
                 patch: int = 16, d_model: int = 128, n_heads: int = 8, n_layers: int = 3,
                 n_cov: int = 0, dropout: float = 0.1, quantiles=QUANTILES):
        super().__init__()
        assert context % patch == 0
        self.context, self.horizon, self.patch = context, horizon, patch
        self.quantiles = tuple(quantiles)
        n_tok = context // patch
        self.embed_in = nn.Linear(patch, d_model)
        self.pos = nn.Parameter(torch.zeros(1, n_tok, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.chan = nn.Embedding(n_channels, d_model)
        enc = nn.TransformerEncoderLayer(d_model, n_heads, 2 * d_model, dropout,
                                         batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.n_cov = n_cov
        if n_cov:
            self.cov_embed = nn.Linear(n_cov * patch, d_model)
            self.fuse = GatedCovariateFusion(d_model, d_model)
        self.head = nn.Linear(n_tok * d_model, horizon * len(self.quantiles))

    def forward(self, x, chan_id, cov=None, cov_dropout: float = 0.0):
        v = x[:, 0]                                           # (B, L)
        mu = v.mean(-1, keepdim=True)
        sd = v.std(-1, keepdim=True).clamp(min=1e-3)
        z = (v - mu) / sd
        b, length = z.shape
        tok = self.embed_in(z.view(b, length // self.patch, self.patch))
        h = tok + self.pos + self.chan(chan_id)[:, None, :]
        gate = None
        if self.n_cov and cov is not None:
            c = cov.view(b, self.n_cov, length // self.patch, self.patch).permute(0, 2, 1, 3)
            e = self.cov_embed(c.reshape(b, length // self.patch, -1))
            if self.training and cov_dropout > 0:
                keep = (torch.rand(b, 1, 1, device=e.device) > cov_dropout).float()
                e = e * keep
            h, gate = self.fuse(h, e)
        h = self.norm(self.encoder(h))
        q = self.head(h.reshape(b, -1)).view(b, self.horizon, len(self.quantiles))
        q = q * sd[:, :, None] + mu[:, :, None]
        return _sort_quantiles(q), gate


def _sort_quantiles(q):
    """Monotone quantiles by sorting the last axis (no crossing at inference)."""
    return torch.sort(q, dim=-1).values


def pinball_loss(q_pred, y, quantiles=QUANTILES, weight=None):
    """Mean quantile (pinball) loss; ``q_pred`` (..., Q), ``y`` (...)."""
    tau = torch.as_tensor(quantiles, dtype=q_pred.dtype, device=q_pred.device)
    d = y[..., None] - q_pred
    loss = torch.maximum(tau * d, (tau - 1.0) * d)
    if weight is not None:
        loss = loss * weight[..., None]
        return loss.sum() / (weight.sum() * len(quantiles)).clamp(min=1.0)
    return loss.mean()


def build_classifier(kind: str, **kw) -> nn.Module:
    return {"fcnn": FCNN, "cnn": CNN1DClassifier, "transformer": PatchTransformerClassifier,
            "hybrid": HybridClassifier}[kind](**kw)


def build_forecaster(kind: str, **kw) -> nn.Module:
    return {"tcn": TCNForecaster, "patchtst": PatchTSTForecaster}[kind](**kw)
