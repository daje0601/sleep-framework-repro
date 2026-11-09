#!/usr/bin/env bash
# Called inside a pinned Linux CUDA image after checkout of a pinned source SHA.
set -euo pipefail
: "${SLEEP_TIMEOUT_SECONDS:?required finite timeout}"
: "${SLEEP_JOB_JSON:?required reviewed job specification}"
: "${SLEEP_EXPERIMENT_JSON:?required reviewed experiment configuration}"
: "${HF_TOKEN:?HF_TOKEN must be injected from a VESSL secret reference}"
[[ "$SLEEP_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] && (( SLEEP_TIMEOUT_SECONDS <= 86400 ))
deadline="${SLEEP_DEADLINE_UNIX:-$(( $(date +%s) + SLEEP_TIMEOUT_SECONDS ))}"
[[ "$deadline" =~ ^[1-9][0-9]*$ ]]
remaining=$(( deadline - $(date +%s) ))
if (( remaining <= 0 )); then
    echo 'Cloud job deadline expired before workload startup' >&2
    exit 124
fi
if (( remaining < SLEEP_TIMEOUT_SECONDS )); then
    SLEEP_TIMEOUT_SECONDS="$remaining"
fi
command -v timeout >/dev/null

# The launcher bounds checkout as well; this guard also bounds direct invocation.
timeout --signal=TERM --kill-after=30s "${SLEEP_TIMEOUT_SECONDS}s" bash -s <<'BOUNDED_JOB'
set -euo pipefail
readarray -t settings < <(python3 - <<'PY'
import json, os
j=json.loads(os.environ['SLEEP_JOB_JSON'])
print(j['mount_path'])
print(j['run_id'])
print(j['source_commit'])
print(j.get('stage','preflight'))
PY
)
mount_path="${settings[0]}"
run_dir="$mount_path/${settings[1]}"
# Hugging Face cache uses symlinks/atomic files: keep it on ephemeral POSIX
# storage. The mounted object volume is reserved for explicit run artifacts.
export HF_HOME="/tmp/sleep-hf-cache/${settings[1]}"
mkdir -p "$HF_HOME"
test "$(git rev-parse HEAD)" = "${settings[2]}"
test -d "$mount_path" && test -w "$mount_path"
mkdir "$run_dir" # Refuse to overwrite an earlier attempt, including failures.
export SLEEP_RUN_DIR="$run_dir"
exec > >(tee -a "$run_dir/console.log") 2>&1
finish() {
    result=$?
    SLEEP_EXIT_CODE="$result" python3 - <<'PY' || true
import json, os
from pathlib import Path
Path(os.environ['SLEEP_RUN_DIR'],'exit.json').write_text(json.dumps({'exit_code':int(os.environ['SLEEP_EXIT_CODE'])})+'\n')
PY
    exit "$result"
}
trap finish EXIT
python3 -m venv --system-site-packages /tmp/sleep-job-venv
source /tmp/sleep-job-venv/bin/activate
python - <<'PY'
import json, os, subprocess
job=json.loads(os.environ['SLEEP_JOB_JSON'])
subprocess.run(['python','-m','pip','install',*job['linux_dependencies']],check=True)
subprocess.run(['python','-m','pip','install','--no-deps','--no-build-isolation','-e','.'],check=True)
import torch
if torch.__version__.split('+')[0] != job['torch_version']:
    raise ValueError('image torch version does not match reviewed torch_version')
if not torch.cuda.is_available():
    raise RuntimeError('CUDA GPU unavailable; refusing a CPU cloud substitute')
PY
python -m pip freeze > "$run_dir/requirements-linux.txt"
python - <<'PY'
import json, os, platform, subprocess
from pathlib import Path
import torch
run=Path(os.environ['SLEEP_RUN_DIR'])
job=json.loads(os.environ['SLEEP_JOB_JSON'])
experiment=json.loads(os.environ['SLEEP_EXPERIMENT_JSON'])
(run/'config.json').write_text(json.dumps(experiment,indent=2)+'\n')
(run/'job.json').write_text(json.dumps(job,indent=2)+'\n')
(run/'environment.json').write_text(json.dumps({'python':platform.python_version(),'torch':torch.__version__,
    'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),
    'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()},indent=2)+'\n')
subprocess.run(['python','scripts/prepare_data.py','--output',experiment.get('data_dir','data/arc')],check=True)
PY
run_preflight() {
    sleep-repro preflight --config "$run_dir/config.json" --output "$run_dir/preflight"
    python - <<'PY'
import hashlib, json, os
from pathlib import Path
run=Path(os.environ['SLEEP_RUN_DIR'])
job=json.loads(os.environ['SLEEP_JOB_JSON'])
config=json.loads(os.environ['SLEEP_EXPERIMENT_JSON'])
gate={'passed':True,'source_commit':job['source_commit'],'image':job['image'],
      'resource_spec':job['resource_spec'],
      'config_sha256':hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()}
(run/'preflight_gate.json').write_text(json.dumps(gate,indent=2)+'\n')
PY
}

case "${settings[3]}" in
    preflight)
        run_preflight
        ;;
    train)
        resume_checkpoint=$(python scripts/prepare_resume.py "$run_dir")
        resume_args=()
        if [[ -n "$resume_checkpoint" ]]; then
            resume_args=(--resume "$resume_checkpoint")
        fi
        sleep-repro train --config "$run_dir/config.json" --output "$run_dir/train" "${resume_args[@]}"
        ;;
    evaluate)
        checkpoint=$(python -c 'import json,os; print(json.loads(os.environ["SLEEP_JOB_JSON"]).get("checkpoint") or "")')
        if [[ -n "$checkpoint" ]]; then
            sleep-repro evaluate --config "$run_dir/config.json" --output "$run_dir/evaluate" --checkpoint "$checkpoint"
        else
            sleep-repro evaluate --config "$run_dir/config.json" --output "$run_dir/evaluate"
        fi
        ;;
    evaluate-resume)
        run_preflight
        python scripts/continue_evaluation.py "$run_dir"
        ;;
    *) echo 'Invalid stage' >&2; exit 2 ;;
esac
BOUNDED_JOB
