#!/usr/bin/env bash
# Standalone bounded SEAL environment/ICL worker. It never calls a provider API.
set -euo pipefail
: "${SLEEP_SEAL_JOB_JSON:?required reviewed baseline job specification}"

# This metadata-only path is safe to run locally, without CUDA or HF credentials.
python3 - <<'PY'
import json, os, re
from pathlib import PurePosixPath
cfg=json.loads(os.environ['SLEEP_SEAL_JOB_JSON'])
required={'source_commit','mount_path','run_id','image','resource_spec','timeout_seconds','stage'}
if not isinstance(cfg,dict) or set(cfg)!=required:
    raise ValueError('baseline job fields must match the reviewed schema; secret values are forbidden')
if cfg['image']!='pytorch/pytorch@sha256:3deb633094acb02ba490e4da21bb37337327860e884c5b2290042a6db7d6b572':
    raise ValueError('baseline image must match the reviewed Torch 2.7 CUDA development digest')
if not re.fullmatch(r'[a-f0-9]{40}',str(cfg['source_commit'])):
    raise ValueError('source_commit must be an immutable full SHA')
for key in ('run_id','resource_spec'):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}',str(cfg[key])):
        raise ValueError('invalid baseline run/resource identifier')
mount=cfg['mount_path']
if (not isinstance(mount,str) or not re.fullmatch(r'/[A-Za-z0-9_/-]+',mount)
    or '..' in PurePosixPath(mount).parts
    or mount.rstrip('/') in ('','/tmp','/var','/etc','/usr','/root','/home')):
    raise ValueError('mount_path must be a dedicated private output volume')
if type(cfg['timeout_seconds']) is not int or not 1<=cfg['timeout_seconds']<=86400:
    raise ValueError('timeout_seconds must be an integer in 1..86400')
if cfg['stage'] not in ('environment','icl'):
    raise ValueError('this separate worker supports environment validation and ICL only')
PY
if [[ "${1:-}" == '--validate-only' && $# -eq 1 ]]; then
    python3 - <<'PY'
import json,os
print(json.dumps({'validated':True,'stage':json.loads(os.environ['SLEEP_SEAL_JOB_JSON'])['stage'],'submitted':False}))
PY
    exit 0
fi
[[ $# -eq 0 ]] || { echo 'Only --validate-only is supported as an argument' >&2; exit 2; }
command -v timeout >/dev/null
seconds=$(python3 -c 'import json,os; print(json.loads(os.environ["SLEEP_SEAL_JOB_JSON"])["timeout_seconds"])')
deadline="${SLEEP_DEADLINE_UNIX:-$(( $(date +%s) + seconds ))}"
[[ "$deadline" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid shared deadline' >&2; exit 2; }
remaining=$(( deadline - $(date +%s) ))
(( remaining > 0 )) || { echo 'Baseline deadline expired before startup' >&2; exit 124; }
(( remaining >= seconds )) || seconds="$remaining"

# Includes source/data downloads, setup/build, import checks, model download and ICL.
# A caller should also wrap checkout and monitor this one Cloud job to terminal.
timeout --signal=TERM --kill-after=30s "${seconds}s" bash -s <<'BOUNDED_SEAL'
set -euo pipefail
readarray -t settings < <(python3 - <<'PY'
import json,os
j=json.loads(os.environ['SLEEP_SEAL_JOB_JSON'])
for key in ('mount_path','run_id','source_commit','stage'):
    print(j[key])
PY
)
mount_path="${settings[0]}"
test -d "$mount_path" && test -w "$mount_path"
run_dir="$mount_path/${settings[1]}"
mkdir "$run_dir" # Unique output; never replace any earlier attempt.
work=$(mktemp -d /tmp/sleep-seal.XXXXXX)
export SLEEP_SEAL_WORK="$work" SLEEP_SEAL_RUN_DIR="$run_dir"
export HF_HOME="$work/hf-cache" XDG_CACHE_HOME="$work/cache"
export TORCH_EXTENSIONS_DIR="$work/torch-extensions"
export HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR"
exec > >(tee -a "$run_dir/console.log") 2>&1
finish() {
    result=$?
    trap - EXIT
    set +e
    SLEEP_SEAL_EXIT_CODE="$result" python3 - <<'PY'
import json,os,shutil
from pathlib import Path
root=Path(os.environ['SLEEP_SEAL_WORK'])/'package'
run=Path(os.environ['SLEEP_SEAL_RUN_DIR'])
# Copy ordinary files only; model snapshots/caches/venvs stay on ephemeral POSIX storage.
for name in ('manifest.json','config.json','commands.json','source.patch','setup.sh','run.sh'):
    if (root/name).is_file():
        shutil.copyfile(root/name,run/name)
for name in ('LICENSE','requirements.txt'):
    path=root/'upstream/original'/name
    if path.is_file():
        shutil.copyfile(path,run/('upstream-'+name))
if (root/'results').is_dir():
    shutil.copytree(root/'results',run/'results',dirs_exist_ok=True,copy_function=shutil.copyfile)
(run/'exit.json').write_text(json.dumps({'exit_code':int(os.environ['SLEEP_SEAL_EXIT_CODE'])})+'\n')
PY
    copy_status=$?
    if (( copy_status != 0 )); then
        echo 'Baseline artifact copy failed; inspect streamed logs and provider state' >&2
        (( result != 0 )) || result=1
    fi
    exit "$result"
}
trap finish EXIT
trap 'exit 124' TERM
trap 'exit 130' INT
for tool in git python3.11 nvcc; do command -v "$tool" >/dev/null; done
test "$(git rev-parse HEAD)" = "${settings[2]}"
python3.11 - <<'PY'
import sys,torch
if sys.version_info[:2]!=(3,11):
    raise RuntimeError('reviewed baseline setup requires Python 3.11')
if torch.__version__.split('+')[0]!='2.7.0':
    raise RuntimeError('reviewed image must supply Torch 2.7.0')
if not torch.cuda.is_available():
    raise RuntimeError('CUDA unavailable; refusing a CPU cloud substitute')
PY
python3.11 scripts/prepare_data.py --output "$work/data"
python3.11 scripts/prepare_seal_baseline.py prepare \
    --config configs/seal_upstream_icl.json --data-dir "$work/data" --output "$work/package"
python3.11 scripts/prepare_seal_baseline.py verify --package "$work/package"
bash "$work/package/setup.sh"
"$work/package/.venv/bin/python" - <<'PY'
import importlib.metadata,json,os,platform,subprocess
from pathlib import Path
import torch,transformers,peft,vllm,flashinfer
root=Path(os.environ['SLEEP_SEAL_WORK'])/'package'
job=json.loads(os.environ['SLEEP_SEAL_JOB_JSON'])
expected={line.split('==')[0]:line.split('==')[1] for line in (root/'upstream/original/requirements.txt').read_text().splitlines() if '==' in line}
observed={name:importlib.metadata.version(name) for name in expected}
if any(observed[name].split('+')[0]!=version for name,version in expected.items()):
    raise RuntimeError('resolved baseline direct pins differ from upstream requirements')
if platform.python_version_tuple()[:2]!=('3','11') or not torch.cuda.is_available():
    raise RuntimeError('baseline environment requires Python 3.11 and CUDA')
if torch.cuda.device_count()!=1:
    raise RuntimeError('reviewed baseline worker expects one GPU')
gpu=torch.cuda.get_device_name(0)
capacity=torch.cuda.get_device_properties(0).total_memory
if not any(name in gpu for name in ('A100','H100')) or capacity<75*1024**3:
    raise RuntimeError('reviewed baseline worker expects an A100/H100 with 80 GB')
report={'passed':True,'stage':job['stage'],'source_commit':job['source_commit'],'image':job['image'],
        'python':platform.python_version(),'cuda':torch.version.cuda,'gpu':gpu,'physical_memory_bytes':capacity,
        'dependencies':observed,'baseline_config':json.loads((root/'config.json').read_text()),
        'nvcc':subprocess.check_output(['nvcc','--version'],text=True),'gpu_benchmark_executed':False}
Path(os.environ['SLEEP_SEAL_RUN_DIR'],'environment.json').write_text(json.dumps(report,indent=2)+'\n')
PY
if [[ "${settings[3]}" == 'icl' ]]; then
    : "${HF_TOKEN:?HF_TOKEN must be injected from the existing Cloud secret reference}"
    bash "$work/package/run.sh"
else
    echo 'Baseline dependency/import/device validation complete; no model or ARC inference executed.'
fi
BOUNDED_SEAL
