# Project instructions

## VESSL cost control — explicit user requirement

- Do not leave paid compute running when no work is being performed.
- Use bounded batch jobs and keep monitoring until they reach a terminal state.
- Terminate this project's job on interruption, execution failure, or its total
  deadline, including a stalled scheduling/image-pull phase.
- Verify terminal job state and VESSL's current billing burn rate after each
  experiment. Do not stop unrelated user workloads.
- Keep experiment artifacts in persistent storage; use an ephemeral POSIX
  filesystem for the Hugging Face cache.

## Evidence

This is an independent protocol reconstruction. CPU tests, a successful GPU
preflight, and quantitative paper reproduction are separate claims. Record
failed runs as well as successful ones. Preserve the pinned data split and
keep final evaluation solutions outside all adaptation and selection paths.
