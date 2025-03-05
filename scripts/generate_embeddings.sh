set -e # Exit immediately if a command exits with a non-zero status.
set -u # Treat unset variables as an error when substituting.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORK_DIR=$SCRIPT_DIR/..
DATA_DIR=$WORK_DIR/data
EMBEDDING_MODEL=perceptiveshawty/compositional-bert-large-uncased

echo "Generating embeddings for training data..."

python3 $WORK_DIR/generate_embeddings.py \
    --input_path $DATA_DIR/sts/train.jsonl \
    --output_path $DATA_DIR/embeddings/train_embeddings.txt \
    --model_path $EMBEDDING_MODEL \
    --size 4000

echo "Generating embeddings for validation data..."

python3 $WORK_DIR/generate_embeddings.py \
    --input_path $DATA_DIR/sts/validation.jsonl \
    --output_path $DATA_DIR/embeddings/validation_embeddings.txt \
    --model_path $EMBEDDING_MODEL \
    --size 2000

echo "Generating embeddings for test data..."

python3 $WORK_DIR/generate_embeddings.py \
    --input_path $DATA_DIR/sts/test.jsonl \
    --output_path $DATA_DIR/embeddings/test_embeddings.txt \
    --model_path $EMBEDDING_MODEL \
    --size 1000
