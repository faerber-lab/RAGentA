#!/bin/bash
# run_rag_parallel.sh

# Parameters
QUESTION_FILE="cleaned_500_questions.jsonl"  # Input JSONL file
NUM_SPLITS=5                            # Number of splits to create
SPLIT_DIR="split_questions"              # Directory to store split files
RESULTS_DIR="results_test"                    # Directory to store results

# Ensure directories exist
mkdir -p $SPLIT_DIR
mkdir -p $RESULTS_DIR
mkdir -p logs

# Split the question file
echo "Splitting question file into $NUM_SPLITS parts..."
python split_questions.py --input_file $QUESTION_FILE --output_dir $SPLIT_DIR --num_splits $NUM_SPLITS

# Array to store job IDs
declare -a JOB_IDS

# Submit jobs for each split
for SPLIT_FILE in $SPLIT_DIR/questions_split_*.jsonl; do
    # Extract split number for naming
    SPLIT_NAME=$(basename $SPLIT_FILE .jsonl)
    
    echo "Submitting job for $SPLIT_FILE..."
    JOB_ID=$(sbatch process_split.sh "$SPLIT_FILE" "$SPLIT_NAME" | awk '{print $NF}')
    
    echo "Submitted job ID: $JOB_ID"
    JOB_IDS+=($JOB_ID)
done

# Print summary
echo "Submitted $NUM_SPLITS jobs with IDs: ${JOB_IDS[@]}"
echo "Results will be stored in the $RESULTS_DIR directory"

# Write job IDs to a file for tracking
echo "${JOB_IDS[@]}" > job_ids.txt

echo "Job submission complete"
