python3 - <<'PY'
from pathlib import Path
import os, sys

root = Path('.').resolve()
out = Path('/data/jhlee39/workspace/repos/OpenAVFF/data').resolve()
print('[root]', root)
print('[out ]', out)

expect = ['RealVideo-RealAudio','RealVideo-FakeAudio','FakeVideo-RealAudio','FakeVideo-FakeAudio']
for d in expect:
    p = root/d
    print(f'  - {d}:', 'OK dir' if p.is_dir() else 'MISSING')

try:
    out.mkdir(parents=True, exist_ok=True)
    testf = out/'__write_test__.tmp'
    testf.write_text('ok')
    print('[write test] OK ->', testf)
    testf.unlink()
except Exception as e:
    print('[write test] FAIL:', repr(e))
    sys.exit(1)
PY

