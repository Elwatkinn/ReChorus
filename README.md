# Forked from [ReChorus](https://github.com/THUwangcy/ReChorus.git)

### 介绍

Reproduction of FinalMLP based on the ReChorus framework.

基于ReChorus的FinalMLP复现。

我们复现的脚本位于`src/models/mymodel/final_mlp.py`

### 数据集准备

如果你已有能成功运行ReChorus，可以直接把该仓库的`src/models/mymodel/final_mlp.py`复制到你自己ReChorus的`src/models`中的任意一个文件夹下，例如`src/models/context`。

如果你尚未有能成功运行ReChorus，请跟随 `data/MovieLens_1M/MovieLens-1M.ipynb` and `data/MIND_Large/MIND-large.ipynb`的指引建立数据库文件。

若成功则有如下数据集文件结构：
```
data
  |——MIND_Large
  |      |——train
  |      |——dev
  |      |——test
  |      |——MINDCTR
  |            |——train.csv
  |            |——dev.csv
  |            |——test.csv
  |
  |——Movielens_1M
        |——ML_1MCTR
              |——train.csv
              |——dev.csv
              |——test.csv
```

### 运行

在终端运行如下指令

#### MINDCTR数据集
```
python ./src/main.py --model_name final_mlp --lr 1e-3 --l2 1e-6 --history_max 20  --dropout 0.2 --dataset MINDCTR --path 'data/MIND_Large/' --num_neg 0 --batch_size 4096 --metric AUC,Log_loss --include_item_features 0 --include_situation_features 0 --model_mode CTR --loss_n BCE
```




#### MovieLens数据集
```
python ./src/main.py --model_name final_mlp --lr 1e-3 --l2 1e-6 --history_max 20  --dropout 0.2 --dataset ML_1MCTR --path 'data/MovieLens_1M/' --num_neg 0 --batch_size 4096 --metric AUC,Log_loss --include_item_features 0 --include_user_features 0 --include_situation_features 0 --model_mode CTR --loss_n BCE

```



#### 参数解释
| 参数|类型| 默认值|解释 |
| ----- | ----- | ----- | ----- |
--emb_size|int|10|特征嵌合维度|
--mlp1_hidden_units|str|'[256,256,256]'|MLP1的隐藏层大小列表|
--mlp1_hidden_activations|str|'ReLU'|MLP1的激活函数|
--mlp1_dropout|float|0|MLP1随机忽略的神经元比例|
--mlp1_batch_norm|int|0|MLP1是否使用批归一化|
--mlp2_hidden_units|str|'[256,256,256]'|MLP2的隐藏层大小列表|
--mlp2_hidden_activations|str|'ReLU'|MLP2的激活函数|
--mlp2_dropout|float|0|MLP2随机忽略的神经元比例|
--mlp2_batch_norm|int|0|MLP2是否使用批归一化|
--use_fs|int|1|是否使用特征选择模块|
--use_opti|int|1|是否使用优化后的NewFinalMLP|
--fs_hidden_units|str|'[256]'|特征选择模块隐藏层大小|
--fs1_context|str|'user_id'|特征选择模块门控1的制定特征）|
--fs2_context|str|'item_id'|特征选择模块门控2的制定特征|
--num_heads|int|16|多头融合时指定的头数量|
