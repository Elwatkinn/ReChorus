python ./src/main.py --model_name DIEN --lr 5e-3 --l2 1e-6 --history_max 20 --alpha_aux 0.5 --aux_hidden_layers "[64,64,64]" --fcn_hidden_layers "[256]" --evolving_gru_type AIGRU --dropout 0.2 --dataset MovieLens_1mimpCTRAll --path 'data/MovieLens_1M/ML_1MCTR/' --num_neg 0 --batch_size 1024 --metric AUC,Log_loss --include_item_features 1 --include_situation_features 1 --model_mode CTR --loss_n BCE
python ./src/main.py --model_name FinalMLP --lr 5e-3 --l2 1e-6 --history_max 20  --dropout 0.2 --dataset MovieLens_1mimpCTRAll --path 'data/MovieLens_1M/ML_1MCTR/' --num_neg 0 --batch_size 1024 --metric AUC,Log_loss --include_item_features 1 --include_situation_features 1 --model_mode CTR --loss_n BCE
python ./src/main.py --model_name final_mlp --lr 1e-3 --l2 1e-6 --history_max 20  --dropout 0.2 --dataset MovieLens_1mimpCTRAll --path 'data/MovieLens_1M/ML_1MCTR/' --num_neg 0 --batch_size 4096 --metric AUC,Log_loss --include_item_features 0 --include_situation_features 0 --model_mode CTR --loss_n BCE
python ./src/main.py --model_name DCN --lr 5e-3 --l2 1e-6 --history_max 20  --dropout 0.2 --dataset MovieLens_1mimpCTRAll --path 'data/MovieLens_1M/ML_1MCTR/' --num_neg 0 --batch_size 1024 --metric AUC,Log_loss --include_item_features 1 --include_situation_features 1 --model_mode CTR --loss_n BCE
python ./src/main.py --model_name DIN --lr 5e-3 --l2 1e-6 --history_max 20  --dropout 0.2 --dataset MovieLens_1mimpCTRAll --path 'data/MovieLens_1M/ML_1MCTR/' --num_neg 0 --batch_size 1024 --metric AUC,Log_loss --include_item_features 1 --include_situation_features 1 --model_mode CTR --loss_n BCE

python ./src/main.py --model_name FinalMLP --mlp1_dropout 0.2 --mlp2_dropout 0.5 --mlp1_batch_norm 1 --mlp2_batch_norm 1 --use_fs 1 --lr 5e-3 --l2 1e-6 --dataset MovieLens_1mimpCTRAll --path 'data/MovieLens_1M/ML_1MCTR/' --num_neg 0 --batch_size 1024 --metric AUC,Log_loss --include_item_features 1 --include_situation_features 1 --model_mode CTR --loss_n BCE --fs1_context c_hour_c,c_weekday_c,c_period_c,c_day_f --fs2_context i_genre_c,i_title_c --mlp1_hidden_units "[64]" --mlp2_hidden_units "[64,64]" --fs_hidden_units "[256,64]"

python ./src/main.py --model_name final_mlp --lr 1e-3 --l2 1e-6 --history_max 20  --dropout 0.2 --dataset MINDCTR --path 'data/MIND_Large/' --num_neg 0 --batch_size 4096 --metric AUC,Log_loss --include_item_features 0 --include_user_features 0 --include_situation_features 0 --model_mode CTR --loss_n BCE
python ./src/main.py --model_name final_mlp --lr 1e-3 --l2 1e-6 --history_max 20  --dropout 0.2 --dataset MovieLens_1mimpCTRAll --path 'data/MovieLens_1M/ML_1MCTR/' --num_neg 0 --batch_size 4096 --metric AUC,Log_loss --include_item_features 0 --include_user_features 0 --include_situation_features 0 --model_mode CTR --loss_n BCE

python ./src/main.py --model_name DCN --lr 1e-3 --l2 1e-6 --history_max 20  --emb_size 10 --dropout 0.2 --dataset MINDCTR --path 'data/MIND_Large/' --num_neg 0 --batch_size 4096 --metric AUC,Log_loss --include_item_features 0 --include_user_features 0 --include_situation_features 0 --model_mode CTR --loss_n BCE
python ./src/main.py --model_name DCN --lr 1e-3 --l2 1e-6 --history_max 20  --emb_size 10 --dropout 0.2 --dataset MovieLens_1mimpCTRAll --path 'data/MovieLens_1M/ML_1MCTR/' --num_neg 0 --batch_size 4096 --metric AUC,Log_loss --include_item_features 0 --include_user_features 0 --include_situation_features 0 --model_mode CTR --loss_n BCE


python ./src/main.py --model_name FM --lr 1e-3 --l2 1e-6 --history_max 20  --emb_size 10 --dropout 0.2 --dataset MINDCTR --path 'data/MIND_Large/' --num_neg 0 --batch_size 4096 --metric AUC,Log_loss --include_item_features 0 --include_user_features 0 --include_situation_features 0 --model_mode CTR --loss_n BCE
python ./src/main.py --model_name FM --lr 1e-3 --l2 1e-6 --history_max 20  --emb_size 10 --dropout 0.2 --dataset MovieLens_1mimpCTRAll --path 'data/MovieLens_1M/ML_1MCTR/' --num_neg 0 --batch_size 4096 --metric AUC,Log_loss --include_item_features 0 --include_user_features 0 --include_situation_features 0 --model_mode CTR --loss_n BCE


FM DCN
mindctr_auc = [0.6307, 0.6167, 0.6107]
movielens_auc = [0.7815, 0.7820, 0.7733]

mindctr_logloss = [0.1661, 0.1635, 0.1605]
movielens_logloss = [0.5671, 0.5568, 0.5711]
