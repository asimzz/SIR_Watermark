set -e
set -u

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORK_DIR=$SCRIPT_DIR/..
DATA_DIR=$WORK_DIR/data
GEN_DIR=$WORK_DIR/gen
MODEL_DIR=$WORK_DIR/model
MODEL_DIR=$WORK_DIR/model

echo "Training surrogate model with watermark logits..."

python3 $WORK_DIR/surrogate_model/train_surrogate_model.py \
    --embedding_data $DATA_DIR/embeddings/train_embeddings.txt \
    --watermark_logits $GEN_DIR/watermark_logits/train_watermark_logits.txt \
    --output_dir $MODEL_DIR/surrogate_model.pth \
    --epochs 100

echo "Evaluating surrogate model on validation data..."

python3 $WORK_DIR/surrogate_model/evaluate_surrogate_model.py \
    --embedding_data $DATA_DIR/embeddings/validation_embeddings.txt \
    --watermark_logits $GEN_DIR/watermark_logits/validation_watermark_logits.txt \
    --model_path $MODEL_DIR/surrogate_model.pth

echo "Evaluating surrogate model on test data..."

python3 $WORK_DIR/surrogate_model/evaluate_surrogate_model.py \
    --embedding_data $DATA_DIR/embeddings/test_embeddings.txt \
    --watermark_logits $GEN_DIR/watermark_logits/test_watermark_logits.txt \
    --model_path $MODEL_DIR/surrogate_model.pth