# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from marlin_pytorch import Marlin


# ---------------------------
# Audio branch on mel (구조 고정)
# ---------------------------
class MelErrPatchTransformer(nn.Module):
    def __init__(
        self,
        n_mels=128,
        t_pad=768,
        patch_len=64,
        d_model=512,
        nhead=4,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
    ):
        super().__init__()
        assert t_pad % patch_len == 0
        self.patch_len = patch_len
        self.num_patches = t_pad // patch_len
        patch_dim = n_mels * patch_len

        self.patch_proj = nn.Linear(patch_dim, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.num_patches, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        nn.init.trunc_normal_(self.patch_proj.weight, std=0.02)
        nn.init.zeros_(self.patch_proj.bias)
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

    def forward(self, mel: torch.Tensor):
        """
        mel: [B,128,768]
        return
            feat: [B,D]   (clip-level audio feature)
            tok:  [B,S,D] (S = num_patches, time-patch tokens)
        """
        B, F_, T = mel.shape
        x_patches = mel.unfold(2, self.patch_len, self.patch_len)  # [B,128,S,patch_len]
        x_patches = x_patches.permute(0, 2, 1, 3).contiguous()     # [B,S,128,patch_len]
        x_patches = x_patches.view(B, self.num_patches, -1)        # [B,S,128*patch_len]

        tok = self.patch_proj(x_patches) + self.pos_emb
        tok = self.encoder(tok)                                    # [B,S,D]
        feat = tok.mean(dim=1)                                     # [B,D]
        return feat, tok


# ---------------------------
# MARLIN-based Visual Encoder
# ---------------------------
class MarlinVisualEncoder(nn.Module):
    """
    입력:
      v_clip: [B,T,3,224,224]
    출력:
      v_seq: [B,S_vis,D]
      g_vis: [B,D]  (global / anchor feature)
    """
    def __init__(
        self,
        backbone_name: str = "marlin_vit_base_ytf",
        out_dim: int = 512,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.backbone = Marlin.from_online(backbone_name)
        embed_dim = getattr(self.backbone, "embed_dim", 768)
        self.proj = nn.Linear(embed_dim, out_dim)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, v_clip: torch.Tensor):
        if v_clip.dim() != 5:
            raise ValueError(f"v_clip must be [B,T,3,H,W], got {v_clip.shape}")

        x = v_clip.permute(0, 2, 1, 3, 4).contiguous()  # [B,3,T,H,W]
        seq_tokens = self.backbone.extract_features(x, keep_seq=True)    # [B,S_vis,E]
        global_feat = self.backbone.extract_features(x, keep_seq=False)  # [B,E]

        v_seq = self.proj(seq_tokens)    # [B,S_vis,D]
        g_vis = self.proj(global_feat)   # [B,D]
        return v_seq, g_vis


# ---------------------------
# Audio-conditioned visual predictor (A2V mapper)
# ---------------------------
class AudioCondVisualPredictor(nn.Module):
    def __init__(self, d_model=512, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.to_frame_feat = nn.Linear(d_model, d_model)

    def forward(self, audio_bins: torch.Tensor, g_vis: torch.Tensor) -> torch.Tensor:
        g_tok = g_vis.unsqueeze(1)                  # [B,1,D]
        x = torch.cat([g_tok, audio_bins], dim=1)   # [B,1+S_vis,D]
        h = self.encoder(x)                         # [B,1+S_vis,D]
        h_audio = h[:, 1:]                          # [B,S_vis,D]
        return self.to_frame_feat(h_audio)          # [B,S_vis,D]


# ---------------------------
# Pretrain-only audio net (unchanged)
# ---------------------------
class AudioPretrainNet(nn.Module):
    def __init__(self, d_model=512, proj_dim=256, patch_len=64):
        super().__init__()
        self.audio_enc = MelErrPatchTransformer(
            n_mels=128,
            t_pad=768,
            patch_len=patch_len,
            d_model=d_model,
        )
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, proj_dim),
        )
        self.aux_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(d_model, 1),
        )
        self.proto_real = nn.Parameter(torch.zeros(proj_dim))
        nn.init.normal_(self.proto_real, mean=0.0, std=0.02)

    def forward(self, mel: torch.Tensor):
        feat, _ = self.audio_enc(mel)
        z = F.normalize(self.proj(feat), dim=1)
        logit = self.aux_head(feat).squeeze(-1)
        return feat, z, logit


# ---------------------------
# Dual detector (main train)
# ---------------------------
class AVDetector(nn.Module):
    """
    변경점 요약
    1) a2v_diff_feat(maxpool diff)와 g_vis(anchor)를 concat하여 a2v head로 보냄:
         fused_feat = [a2v_diff_feat ; g_vis]  -> dim = 2D
    2) confidence는 "head 이후 확률 결합" 방식 유지하되,
       conf_net 입력을 feature(+각 branch logit)로 강화:
         conf_in = [audio_feat_detach ; fused_feat ; logit_audio_detach ; logit_a2v]
       -> 목적(강한 단서가 있는 branch에 가중치)을 더 잘 학습 가능
    """

    def __init__(
        self,
        marlin_name: str = "marlin_vit_base_ytf",
        freeze_vis_backbone: bool = True,
        a2v_nhead: int = 4,
        a2v_layers: int = 2,
        d_model: int = 512,
    ):
        super().__init__()
        self.d_model = d_model

        # Audio encoder (frozen)
        self.audio_enc = MelErrPatchTransformer(
            n_mels=128,
            t_pad=768,
            patch_len=64,
            d_model=d_model,
        )

        # Visual encoder
        self.vis_enc = MarlinVisualEncoder(
            backbone_name=marlin_name,
            out_dim=d_model,
            freeze_backbone=freeze_vis_backbone,
        )

        # A2V mapper (trainable)
        self.predictor = AudioCondVisualPredictor(
            d_model=d_model,
            nhead=a2v_nhead,
            num_layers=a2v_layers,
        )

        # Audio-only head (frozen)
        self.audio_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(d_model, 1),
        )

        # A2V head: (변경) 입력이 2D (diff_feat + anchor)
        self.a2v_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(d_model, 1),
        )

        # Confidence network: (변경) feature + logit까지 입력으로 사용
        # input dim = audio_feat(D) + fused_feat(2D) + 2 logits = 3D + 2
        conf_in_dim = d_model * 3 + 2
        self.conf_net = nn.Sequential(
            nn.Linear(conf_in_dim, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(d_model, 2),
        )

        # freeze audio-only branch
        for p in self.audio_enc.parameters():
            p.requires_grad = False
        for p in self.audio_head.parameters():
            p.requires_grad = False

    def _audio_to_seq_len(self, audio_tok: torch.Tensor, target_len: int) -> torch.Tensor:
        B, S_a, D = audio_tok.shape
        x = audio_tok.permute(0, 2, 1)  # [B,D,S_a]
        x = F.interpolate(x, size=target_len, mode="linear", align_corners=False)  # [B,D,S_vis]
        x = x.permute(0, 2, 1)  # [B,S_vis,D]
        return x

    def forward(self, mel: torch.Tensor, v_clip: torch.Tensor, first_frame: torch.Tensor, return_aux: bool = False):
        """
        returns:
          - logit_audio: [B]
          - logit_final: [B]
          - (optional) aux dict: mismatch_mse [B], logit_a2v [B]
        """
        # ---- Audio (frozen) ----
        audio_feat, audio_tok = self.audio_enc(mel)  # [B,D], [B,S_a,D]

        # ---- Visual ----
        v_seq, g_vis = self.vis_enc(v_clip)          # [B,S_vis,D], [B,D]
        S_vis = v_seq.size(1)

        # ---- Align audio tokens to S_vis ----
        audio_bins = self._audio_to_seq_len(audio_tok, S_vis)  # [B,S_vis,D]

        # ---- A2V ----
        pred_v_seq = self.predictor(audio_bins, g_vis)         # [B,S_vis,D]

        # ---- Diff feature (per time) ----
        diff_seq = torch.abs(pred_v_seq - v_seq)               # [B,S_vis,D]
        a2v_diff_feat, _ = diff_seq.max(dim=1)                 # [B,D]

        # ---- (변경 1) fused_feat = concat(diff_feat, anchor=g_vis) ----
        fused_feat = torch.cat([a2v_diff_feat, g_vis], dim=-1)  # [B,2D]

        # ---- mismatch for align/repel (if you use it in train.py) ----
        mismatch_mse = (pred_v_seq - v_seq).pow(2).mean(dim=(1, 2))  # [B]

        # ---- Branch logits ----
        logit_audio = self.audio_head(audio_feat).squeeze(-1)        # [B]
        logit_a2v = self.a2v_head(fused_feat).squeeze(-1)            # [B]

        # ---- Branch probabilities ----
        p_audio = torch.sigmoid(logit_audio.detach())                # [B] (frozen)
        p_a2v = torch.sigmoid(logit_a2v)                             # [B]

        # ---- (변경 2) confidence uses features + logits ----
        # 목적: 한쪽이 "강하게 fake" (|logit| 크거나 p가 극단)일 때 그쪽을 더 믿도록 학습 가능
        conf_in = torch.cat(
            [
                audio_feat.detach(),                  # [B,D]
                fused_feat,                           # [B,2D]
                logit_audio.detach().unsqueeze(-1),   # [B,1]
                logit_a2v.unsqueeze(-1),              # [B,1]
            ],
            dim=-1,
        )  # [B, 3D+2]

        conf_logits = self.conf_net(conf_in)          # [B,2]
        conf_weight = torch.softmax(conf_logits, dim=-1)
        w_audio = conf_weight[:, 0]                   # [B]
        w_a2v = conf_weight[:, 1]                     # [B]

        # ---- Weighted probability fusion ----
        p_final = w_audio * p_audio + w_a2v * p_a2v
        p_final = torch.clamp(p_final, 1e-6, 1.0 - 1e-6)
        logit_final = torch.log(p_final / (1.0 - p_final))     # [B]

        if return_aux:
            return logit_audio, logit_final, {
                "mismatch_mse": mismatch_mse,
                "logit_a2v": logit_a2v,
                "w_audio": w_audio,
                "w_a2v": w_a2v,
            }
        return logit_audio, logit_final
