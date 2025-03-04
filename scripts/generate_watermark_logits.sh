set -e # Exit immediately if a command exits with a non-zero status.
set -u # Treat unset variables as an error when substituting.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORK_DIR=$SCRIPT_DIR/..
DATA_DIR=$WORK_DIR/data
GEN_DIR=$WORK_DIR/gen
TRANSFORM_MODEL_DIR=$WORK_DIR/model

echo "Generating training watermarks logits..."

python3 $WORK_DIR/generate_watermark_logits.py \
    --original_model $TRANSFORM_MODEL_DIR/transform_model_cbert.pth \
    --embedding_data $DATA_DIR/embeddings/train_embeddings.txt \
    --input_dim 1024 \
    --output_dir $GEN_DIR/watermark_logits/train_watermark_logits.txt

echo "Generating validation watermarks logits..."

python3 $WORK_DIR/generate_watermark_logits.py \
    --original_model $TRANSFORM_MODEL_DIR/transform_model_cbert.pth \
    --embedding_data $DATA_DIR/embeddings/validation_embeddings.txt \
    --input_dim 1024 \
    --output_dir $GEN_DIR/watermark_logits/validation_watermark_logits.txt

echo "Generating test watermarks logits..."

python3 $WORK_DIR/generate_watermark_logits.py \
    --original_model $TRANSFORM_MODEL_DIR/transform_model_cbert.pth \
    --embedding_data $DATA_DIR/embeddings/test_embeddings.txt \
    --input_dim 1024 \
    --output_dir $GEN_DIR/watermark_logits/test_watermark_logits.txt