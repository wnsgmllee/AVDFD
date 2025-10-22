# python tools/convert_stage2_to_stage3.py \
#   --stage2 checkpoints/stage2_pretrained.pth \
#   --out    checkpoints/stage3_init_from_stage2.pth \
#   --num_classes 2
import argparse, os, torch
from src.models.video_cav_mae import VideoCAVMAEFT  # ← 분류 모델 (mlp_head 포함)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage2", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num_classes", type=int, required=True)
    args = ap.parse_args()

    # 1) Stage-3 분류 모델 인스턴스 (헤드는 여기서 '랜덤'으로 생성됨)
    model = VideoCAVMAEFT(n_classes=args.num_classes)

    # 2) Stage-2 state_dict 로드
    sd2 = torch.load(args.stage2, map_location="cpu")
    if isinstance(sd2, dict) and "state_dict" in sd2:
        sd2 = sd2["state_dict"]

    # 3) 인코더 키만 이식 (decoder/head/proj 계열은 버림)
    sd3 = model.state_dict()
    new_sd, used, skipped = {}, [], []
    def ok(k, v):
        low = k.lower()
        # 디코더/헤드/프로젝션/학습전용 키 제외
        bad = ["decoder", "mlp", "head", "classifier", "proj", "logit", "fc"]
        return not any(b in low for b in bad) and (k in sd3) and (sd3[k].shape == v.shape)

    for k, v in sd2.items():
        kk = k[7:] if k.startswith("module.") else k  # DDP prefix 제거
        if ok(kk, v):
            new_sd[kk] = v; used.append(kk)
        else:
            # 흔한 프리픽스 교정 (encoder/backbone 표기차)
            for cand in (kk.replace("encoder.", "audio_encoder."),
                         kk.replace("backbone.", "audio_encoder."),
                         kk.replace("encoder.", "visual_encoder."),
                         kk.replace("backbone.", "visual_encoder.")):
                if cand in sd3 and sd3[cand].shape == v.shape and \
                   not any(t in cand.lower() for t in ["decoder","mlp","head","classifier","proj","logit","fc"]):
                    new_sd[cand] = v; used.append(cand); break
            else:
                skipped.append(kk)

    merged = sd3.copy(); merged.update(new_sd)
    model.load_state_dict(merged, strict=False)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, args.out)

    print(f"[OK] saved: {args.out}")
    print(f"  loaded keys: {len(used)} | skipped (decoder/head/etc): {len(skipped)}")

if __name__ == "__main__":
    main()

