# QuillFuzz

QuillFuzz is a quantum compiler fuzzing tool powered by Large Language Models (LLMs). It automates the generation and refinement of quantum circuits to test and validate quantum compilers such as Guppy, Qiskit, and Pytket.

## Setup

QuillFuzz can be run with Docker Compose, Docker, or Podman. The container is kept alive with `tail -f /dev/null`, so you can enter and leave an interactive shell without stopping the container.

### Docker Compose

Recommended when Compose is available:

```bash
USER_ID=$(id -u) GROUP_ID=$(id -g) docker compose up -d --build
docker compose exec quillfuzz /bin/bash
```

### Docker

Use this if you want a persistent container without Compose:

```bash
docker build -t quillfuzz .

docker run -d \
  --user "$(id -u):$(id -g)" \
  -v "$(pwd):/QuillFuzz" \
  -w /QuillFuzz \
  -e RUST_BACKTRACE=1 \
  --name quillfuzz \
  quillfuzz \
  tail -f /dev/null

docker exec -it quillfuzz /bin/bash
```

### Podman

Use this if you prefer Podman or Docker is unavailable:

```bash
podman build -t quillfuzz .

podman run -d \
  --userns=keep-id \
  -v "$(pwd):/QuillFuzz" \
  -w /QuillFuzz \
  -e RUST_BACKTRACE=1 \
  --name quillfuzz \
  quillfuzz:latest \
  tail -f /dev/null

podman exec -it quillfuzz /bin/bash
```

To leave the shell, use `exit` or Ctrl+D. That only closes the shell session; it does not stop the detached container.

If you want to keep watching the output of a long-running test or fuzzing run after closing the shell, start tmux explicitly after entering the container and detach with Ctrl+B, then D:

```bash
tmux new -A -s quillfuzz
```

Later, reattach with:

```bash
docker exec -it quillfuzz tmux attach -t quillfuzz
podman exec -it quillfuzz tmux attach -t quillfuzz
```

The `docker exec` and `podman exec` examples below apply to the standalone container named `quillfuzz` created by the Docker or Podman commands above. If you start QuillFuzz with Compose, keep using `docker compose exec quillfuzz /bin/bash`.

To stop a standalone container later, use the matching engine:

```bash
docker stop quillfuzz
docker rm quillfuzz

podman stop quillfuzz
podman rm quillfuzz
```

## Running QuillFuzz

### Prerequisites

1.  **API Keys**: You need to provide API keys for LLM access. Create a `.env` file in the root directory and add the supported keys there:
    - OpenAI: `OPENAI_API_KEY`
    - Anthropic: `ANTHROPIC_API_KEY`
    - Google: `GEMINI_API_KEY`
    - Deepseek: `DEEPSEEK_API_KEY`

    Copy and paste this starter template into your `.env` file:

    ```env
    OPENAI_API_KEY=your_openai_api_key_here
    ANTHROPIC_API_KEY=your_anthropic_api_key_here
    GEMINI_API_KEY=your_gemini_api_key_here
    DEEPSEEK_API_KEY=your_deepseek_api_key_here
    ```

### Running Campaigns

To run pre-configured fuzzing campaigns (Guppy, Qiskit, or Pytket), ensure the scripts are executable once:

```bash
chmod +x ./scripts/Complete_run_guppy.sh ./scripts/Complete_run_qiskit.sh ./scripts/Complete_run_pytket.sh
```

Then run the desired script inside the container:

**For Guppy Fuzzing:**

```bash
./scripts/Complete_run_guppy.sh
```

**For Qiskit Fuzzing:**

```bash
./scripts/Complete_run_qiskit.sh
```

**For Pytket Fuzzing:**

```bash
./scripts/Complete_run_pytket.sh
```

## Coverage HTML Reports

Each run saves per-program `.coverage` data files under `<run_dir>/coverage_artifacts/`. At the end of a run these are combined into a single `<run_dir>/coverage_artifacts/.coverage` file and a `coverage_summary.json` is written alongside it.

You can generate a line-level HTML heatmap (file list + per-file hit/miss annotations) from any completed run retroactively with the standard `coverage html` command:

```bash
# Inside the container, from the repo root
COVERAGE_FILE=local_saved_circuits/<run_name>/coverage_artifacts/.coverage \
  python -m coverage html -d local_saved_circuits/<run_name>/coverage_artifacts/htmlcov
```

Then open `local_saved_circuits/<run_name>/coverage_artifacts/htmlcov/index.html` in a browser. Each file shows green (hit) and red (missed) line annotations. The `generate_coverage_report_from_data_file` helper in `src/utils/execution.py` wraps this call and also supports `report_format="xml"` and `report_format="lcov"` for other tooling.

# Bugs found

| Issue | Language | Status | Notes |
|---|---|---|---|
| [Qiskit/qiskit-aer#2404](https://github.com/Qiskit/qiskit-aer/issues/2404) | Qiskit | Open | Not fixed |
| [Qiskit/qiskit#15734](https://github.com/Qiskit/qiskit/issues/15734) | Qiskit | Fixed |  |
| [Qiskit/qiskit#15748](https://github.com/Qiskit/qiskit/issues/15748) | Qiskit | Fixed |  |
| [Qiskit/qiskit#15747](https://github.com/Qiskit/qiskit/issues/15747) | Qiskit | Fixed (duplicate) | Same cause as #15748 |
| [Qiskit/qiskit#15733](https://github.com/Qiskit/qiskit/issues/15733) | Qiskit | In progress (duplicate) | Being fixed |
| [Qiskit/qiskit#16223](https://github.com/Qiskit/qiskit/issues/16223) | Qiskit | In progress | Being fixed |
| [Quantinuum/tket2#1604](https://github.com/Quantinuum/tket2/issues/1604) | Guppy | In progress (duplicate) | Being fixed |
| [Quantinuum/tket2#1577](https://github.com/Quantinuum/tket2/issues/1577) | Guppy | Fixed |  |
| [Quantinuum/tket2#1375](https://github.com/Quantinuum/tket2/issues/1375) | Guppy | In progress | Fixing |
| [Quantinuum/guppylang#1442](https://github.com/Quantinuum/guppylang/issues/1442) | Guppy | Fixed |  |