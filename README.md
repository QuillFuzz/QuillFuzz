# QuillFuzz

QuillFuzz is a quantum compiler fuzzing tool powered by Large Language Models (LLMs). It automates the generation and refinement of quantum circuits to test and validate quantum compilers such as Guppy, Qiskit, and Pytket.

## Setup

QuillFuzz can be run with Docker Compose, Docker, or Podman. The container is kept alive with `tail -f /dev/null`, so you can enter and leave an interactive shell without stopping the container.

### Docker Compose

Recommended when Compose is available:

```bash
UID=$(id -u) GID=$(id -g) docker compose up -d --build
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

To leave the shell, use `exit` or Ctrl+D. That only closes the shell session; it does not stop the detached container. Reconnect with the same `exec` command at any time.

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

### Detached Campaigns

If you want a campaign to keep running after you disconnect from SSH, start it from the host and write logs to the mounted project directory. These commands apply to the standalone `quillfuzz` container; if you are using Compose, replace them with `docker compose exec quillfuzz ...`.

```bash
mkdir -p logs

docker exec -d quillfuzz bash -lc 'cd /QuillFuzz && ./scripts/Complete_run_guppy.sh > /QuillFuzz/logs/guppy_run.log 2>&1'

podman exec -d quillfuzz bash -lc 'cd /QuillFuzz && ./scripts/Complete_run_guppy.sh > /QuillFuzz/logs/guppy_run.log 2>&1'
```

Use the same pattern for Qiskit and Pytket by swapping the script and log filename. To monitor a detached run:

```bash
docker exec quillfuzz pgrep -af Complete_run_guppy.sh
docker exec quillfuzz tail -f /QuillFuzz/logs/guppy_run.log

podman exec quillfuzz pgrep -af Complete_run_guppy.sh
podman exec quillfuzz tail -f /QuillFuzz/logs/guppy_run.log
```

Stop a detached run with the matching engine:

```bash
docker exec quillfuzz pkill -f Complete_run_guppy.sh
podman exec quillfuzz pkill -f Complete_run_guppy.sh
```

### Reports-only Analysis (No Generation)

To run coverage analysis and generate complexity plots for an existing directory of circuit files:

```bash
python src/generate_coverage_and_complexity.py <input_dir> --language guppy
```
