CURRENT_DIR=`pwd`

python main.py \
  --model_type bert \
  --model_name_or_path bert-base-chinese \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir ../data \
  --train_file DRCD_training.json \
  --predict_file DRCD_dev.json \
  --per_gpu_train_batch_size 12 \
  --learning_rate 1e-5 \
  --num_train_epochs 5.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir ../output \
  --logging_steps 4000 \
  --save_steps 10000  \
  --threads 12 \
  --version_2_with_negative \
  --overwrite_output_dir \
  --evaluate_during_training
