# QuillFuzz

QuillFuzz is a quantum compiler fuzzing tool powered by Large Language Models (LLMs). It automates the generation and refinement of quantum circuits to test and validate quantum compilers such as Qiskit and Guppy.

## Setup

To install the necessary dependencies and set up the environment, simply run:

```bash
UID=$(id -u) GID=$(id -g) docker compose up -d --build
```

and then

```bash
docker compose exec quillfuzz bash
```

### Alternative Setup (Docker)

If `docker compose` is not available, you can build and run using standard docker commands:

```bash
# Build the image
docker build -t quillfuzz .

# Run the container (mounts current directory and runs in background)
docker run -d \
  --user "$(id -u):$(id -g)" \
  -v "$(pwd):/QuillFuzz" \
  -e RUST_BACKTRACE=1 \
  --name quillfuzz \
  quillfuzz \
  tail -f /dev/null

# Enter the container
docker exec -it quillfuzz bash
```

### Alternative Setup (Podman)

If you prefer using Podman or if Docker is not available:

```bash
# Build the image
podman build -t quillfuzz .

# Run the container (interactive mode with volume mount)
podman run -it --rm \
  --userns=keep-id \
  -v "$(pwd):/QuillFuzz" \
  -e RUST_BACKTRACE=1 \
  --name quillfuzz \
  quillfuzz:latest \
  bash
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

To run pre-configured fuzzing campaigns (Guppy or Qiskit), ensure the scripts are executable:

```bash
chmod +x ./scripts/Complete_run_guppy.sh ./scripts/Complete_run_qiskit.sh
```

Then run either script inside the container:

**For Guppy Fuzzing:**

```bash
./scripts/Complete_run_guppy.sh
```

**For Qiskit Fuzzing:**

```bash
./scripts/Complete_run_qiskit.sh
```

### Reports-only Analysis (No Generation)

To run coverage analysis and generate complexity plots for an existing directory of circuit files:

```bash
python src/generate_coverage_and_complexity.py <input_dir> --language guppy
```
