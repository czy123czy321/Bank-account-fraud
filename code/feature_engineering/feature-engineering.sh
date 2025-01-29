#! /usr/bin/bash

BASE_DIR=$(cd ../../ && pwd)
CODE_DIR="$BASE_DIR/code"
OUTPUT_DIR="$BASE_DIR/output"
DATA_DIR="$BASE_DIR/data"
LOG_FILE="$CODE_DIR/feature_engineering/feature_engineering_workflow.log"
PYTHON_EXEC="/opt/homebrew/bin/python3.11"

mkdir -p "$OUTPUT_DIR"
> $LOG_FILE

log() {
  echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

log "Starting workflow for feature engineering."

echo "Converting preprocessing notebook to Python script..."
if jupyter nbconvert --to script $CODE_DIR/feature_engineering/preprocessing_notebook.ipynb --output $CODE_DIR/feature_engineering/preprocessing >>"$LOG_FILE" 2>&1; then
    log "preprocessing notebook converted to script in $CODE_DIR/feature_engineering."
else
	log "Failed to convert preprocessing notebook to script"
	exit 1
fi

echo "Converting feature engineering notebook to script..."
if jupyter nbconvert --to script $CODE_DIR/feature_engineering/final_feature_engineering.ipynb --output $CODE_DIR/feature_engineering/final_feature_engineering >>"$LOG_FILE" 2>&1; then
    log "feature engineering notebook converted to script in $CODE_DIR/feature_engineering."
else
	log "Failed to convert feature engineering notebook to script"
	exit 1
fi

# Execute Python script
if $PYTHON_EXEC "$CODE_DIR/feature_engineering/final_feature_engineering.py" >>"$LOG_FILE" 2>&1; then
  log "Feature engineering Python script executed successfully."
  
  # Validate the existence of saved CSV files
  for file in x_train_data.csv x_test_data.csv y_train_data.csv y_test_data.csv; do
    if [ -f "$DATA_DIR/$file" ]; then
      log "$file successfully saved in $DATA_DIR."
    else
      log "$file was not created in $DATA_DIR."
      exit 1
    fi
  done
else
  log "Failed to execute the feature engineering Python script."
  exit 1
fi

log "Feature engineering completed successfully. Final dataset stored in $DATA_DIR."

echo "Feature Engineering Completed."