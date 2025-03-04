set -e
set -u

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORK_DIR=$SCRIPT_DIR/..
DATA_DIR=$WORK_DIR/data
GEN_DIR=$WORK_DIR/gen
MODEL_DIR=$WORK_DIR/model
MODEL_DIR=$WORK_DIR/model

echo "Generating watermark logits for surrogate model from embeddings..."

python3 $WORK_DIR/generate_watermark_logits.py \
    --original_model $MODEL_DIR/transform_model_cbert.pth \
    --embedding_data $DATA_DIR/embeddings/train_embeddings.txt \
    --input_dim 1024 \
    --output_dir $GEN_DIR/watermark_logits.txt

echo "Training surrogate model with watermark logits..."

python3 $WORK_DIR/surrogate_model/train_surrogate_model.py \
    --embedding_data $DATA_DIR/embeddings/train_embeddings.txt \
    --watermark_logits $GEN_DIR/watermark_logits.txt \
    --output_dir $MODEL_DIR/surrogate_model.pth \
    --epochs 100