#!/bin/bash
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

# Run the generator first, saving into a named run with date-stamped folder
RUN_NAME="Test_run_pytket_$(date +'%Y%m%d_%H%M%S')"
export QUILLFUZZ_RUN_DIR="$(pwd)/local_saved_circuits/$RUN_NAME"
python src/gen_w_improve.py --config_file run_configs/pytket_test_run_config.yaml --run_name $RUN_NAME

ASSEMBLED_DIR="local_saved_circuits/$RUN_NAME/assembled"
if [ ! -d "$ASSEMBLED_DIR" ]; then
    echo "Error: Assembled directory '$ASSEMBLED_DIR' not found."
    echo "This likely means no valid circuits were generated."
    exit 1
fi

echo "Test run complete. Assembled files are in $ASSEMBLED_DIR"
