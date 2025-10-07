# (선택) 고속 다운로드 준비: hf_transfer

# pip install -U "huggingface_hub[hf_transfer]"
# export HF_HUB_ENABLE_HF_TRANSFER=1   # 빠른 병렬 전송 활성화

# 전체 샤드(82GB+)를 원하는 폴더로 받기
hf download gaunernst/voxceleb2-dev-wds \
  --repo-type dataset \
  --include "voxceleb2-dev-*.tar" \
  --local-dir voxceleb_v2 \
  --local-dir-use-symlinks False
