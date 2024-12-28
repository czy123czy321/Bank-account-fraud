#! /usr/bin/bash

BASE_DIR=$(pwd)
CODE_DIR="$BASE_DIR/code"
OUTPUT_DIR="$BASE_DIR/output"
LOG_FILE="$OUTPUT_DIR/eda_workflow.log"
PYTHON_EXEC="/opt/homebrew/bin/python3.11"

HTML_FILE="$OUTPUT_DIR/eda_output.html"

mkdir -p "$OUTPUT_DIR"
> $LOG_FILE

log() {
  echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$LOG_FILE"
}

log "Starting workflow for EDA."

# if jupyter nbconvert "$CODE_DIR/eda.ipynb" --to html --output-dir "$OUTPUT_DIR" >>"$LOG_FILE" 2>&1; then
#   log "Notebook exported to HTML: $OUTPUT_DIR/eda.html"
# else
#   log "Failed to export notebook to HTML."
#   exit 1
# fi

if jupyter nbconvert --execute "$CODE_DIR/TEST.ipynb" --to script >>"$LOG_FILE" 2>&1; then
	log "EDA notebook converted to script in $CODE_DIR."
else
	log "Failed to convert notebook to script"
	exit 1
fi

if $PYTHON_EXEC "$CODE_DIR/TEST.py" > "$HTML_FILE" 2>>"$LOG_FILE"; then
  log "Python script executed successfully, saved to $OUTPUT_DIR"
else
  log "Failed to execute the Python script."
  exit 1
fi

log "Opening the output in the browser."
if [[ "$OSTYPE" == "darwin"* ]]; then
  open "$HTML_FILE"  # macOS
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
  xdg-open "$HTML_FILE"  # Linux
else
  log "Unsupported OS for automatic browser launch. Open $HTML_FILE manually."
fi

log "Workflow completed. Output logged to $LOG_FILE"