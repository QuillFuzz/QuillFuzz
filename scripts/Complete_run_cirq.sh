#!/bin/bash
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

RUN_NAME="Complete_run_cirq_$(date +'%Y%m%d_%H%M%S')"
RUN_DIR="$(pwd)/local_saved_circuits/$RUN_NAME"
python src/gen_w_improve.py --config_file run_configs/cirq_full_run_config.yaml --run_name "$RUN_NAME" --output_dir "$RUN_DIR"

ASSEMBLED_DIR="$RUN_DIR/assembled"
if [ ! -d "$ASSEMBLED_DIR" ]; then
    echo "Error: Assembled directory '$ASSEMBLED_DIR' not found."
    echo "This likely means no valid circuits were generated."
    exit 1
fi

echo "Generation and assembly complete. Interesting assembled files are in $ASSEMBLED_DIR"
