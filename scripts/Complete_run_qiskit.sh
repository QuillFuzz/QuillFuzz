#!/bin/bash
source .venv/bin/activate || { echo "Failed to activate virtual environment"; exit 1; }
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

# Run the generator first, saving into a named run with date-stamped folder
# Get date and time for unique run naming
RUN_NAME="Complete_run_qiskit_$(date +'%Y%m%d_%H%M%S')"
export QUILLFUZZ_RUN_DIR="$(pwd)/local_saved_circuits/$RUN_NAME"
python src/gen_w_improve.py --config_file run_configs/qiskit_full_run_config.yaml --run_name $RUN_NAME

ASSEMBLED_DIR="local_saved_circuits/$RUN_NAME/assembled"
if [ ! -d "$ASSEMBLED_DIR" ]; then
    echo "Error: Assembled directory '$ASSEMBLED_DIR' not found."
    echo "This likely means no valid circuits were generated."
    exit 1
fi

# After generation is complete and circuits are assembled, run the test_existing_circuits.py script to test assembled circuits
python src/test_existing_circuits.py "$ASSEMBLED_DIR" --workers 4 --language qiskit
