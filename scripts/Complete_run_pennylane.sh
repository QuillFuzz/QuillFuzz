#!/bin/bash
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

# Run the generator first, saving into a named run with date-stamped folder
# Get date and time for unique run naming
RUN_NAME="Complete_run_pennylane_$(date +'%Y%m%d_%H%M%S')"
RUN_DIR="$(pwd)/local_saved_circuits/$RUN_NAME"
python src/gen_w_improve.py --config_file run_configs/pennylane_full_run_config.yaml --run_name "$RUN_NAME" --output_dir "$RUN_DIR"

ASSEMBLED_DIR="$RUN_DIR/assembled"
if [ ! -d "$ASSEMBLED_DIR" ]; then
    echo "Error: Assembled directory '$ASSEMBLED_DIR' not found."
    echo "This likely means no valid circuits were generated."
    exit 1
fi

echo "Generation and assembly complete. Interesting assembled files are in $ASSEMBLED_DIR"