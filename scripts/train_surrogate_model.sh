set -e
set -u

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORK_DIR=$SCRIPT_DIR/..
DATA_DIR=$WORK_DIR/data
GEN_DIR=$WORK_DIR/gen
SURROGATE_MODEL_DIR=$WORK_DIR/model

python $WORK_DIR/train_surrogate_model.py \
    --embedding_data $DATA_DIR/embeddings/train_embeddings.txt \
    --watermark_logits $GEN_DIR/watermark_logits.txt \
    --output_dir $GEN_DIR/surrogate_model.pth \
    --epochs 100