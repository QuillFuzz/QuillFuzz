# QuillFuzz

QuillFuzz is a quantum compiler fuzzing tool powered by Large Language Models (LLMs). It automates the generation and refinement of quantum circuits to test and validate quantum compilers such as Qiskit and Guppy.

## Setup

To install the necessary dependencies and set up the environment, simply run:

```bash
docker compose up -d --build
```

and then

```bash
docker compose exec quillfuzz bash
```

## Running QuillFuzz

Note: You need to provide API keys for LLM access, stored in the `.env` environment file.

To run pre-configured tests, you can run:

```bash
./scripts/Complete_run_guppy.sh
```

or 

```bash
./scripts/Complete_run_qiskit.sh
```

to run guppy or qiskit fuzzing campaigns. Remember to grant them execution permission:

```bash
chmod +x ./scripts/Complete_run_qiskit.sh
chmod +x ./scripts/Complete_run_guppy.sh
```