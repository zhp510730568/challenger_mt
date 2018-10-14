t2t-trainer --t2t_usr_dir=./mt --registry_help

PROBLEM=translate_enzh_token32k
MODEL=transformer
HPARAMS=transformer_base_single_gpu

DATA_DIR=/home/zhangpengpeng/t2t_train/challenger_mt
TMP_DIR=/tmp/t2t_datagen
TRAIN_DIR=$HOME/t2t_train/$PROBLEM/$MODEL-$HPARAMS

mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR

# Generate data
t2t-datagen \
  --data_dir=$DATA_DIR \
  --tmp_dir=$TMP_DIR \
  --problem=$PROBLEM
