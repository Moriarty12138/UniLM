CUDA_VISIBLE_DEVICES=4 python decode_seq2seq.py --bert_model torch_unilm_model/ --new_segment_ids --mode s2s --need_score_traces \
  --input_file datasets/LCSTS.src --split LCSTS \
  --model_recover_path output/bert_save/model.8.bin \
  --max_seq_length 256 --max_tgt_length 64 \
  --beam_size 5 --length_penalty 0 \
  --forbid_duplicate_ngrams --forbid_ignore_word "."
