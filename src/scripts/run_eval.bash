CURRENT_DIR=`pwd`

python main.py \
  --model_type bert \
  --model_name_or_path ../output/model_v1 \
  --do_eval \
  --do_lower_case \
  --data_dir ../data \
  --train_file DRCD_training.json \
  --predict_file DRCD_test.json \
  --per_gpu_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 15.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir ../output \
  --save_steps 3000  \
  --threads 12 \
  --version_2_with_negative \
  --overwrite_output_dir 