# Forked from [ReChorus](https://github.com/THUwangcy/ReChorus.git)

### 介绍

Reproduction of FinalMLP based on the ReChorus framework.

基于ReChorus的FinalMLP复现。

我们复现的脚本位于`src/models/mymodel/final_mlp.py`

### 数据集准备

如果你已有能成功运行ReChorus，可以直接把该仓库的`src/models/mymodel/final_mlp.py`复制到你自己ReChorus的`src/models`中的任意一个文件夹下，例如`src/models/context`。

如果你尚未有能成功运行ReChorus，请跟随 `data/MovieLens_1M/MovieLens-1M.ipynb` and `data/MIND_Large/MIND-large.ipynb`的指引建立数据库文件。若成功则你会在`data/MIND_Large/MINDCTR`等目录下见到`test.csv`,`dev.csv`,`train.csv`等数据集文件。

### 运行

在终端运行如下指令
```
python ./src/main.py --model_name final_mlp --lr 1e-3 --l2 1e-6 --history_max 20  --dropout 0.2 --dataset MINDCTR --path 'data/MIND_Large/' --num_neg 0 --batch_size 4096 --metric AUC,Log_loss --include_item_features 0 --include_situation_features 0 --model_mode CTR --loss_n BCE
```

运行前请确保指令中的`--dataset MINDCTR --path 'data/MIND_Large/'`在`data/MIND_Large/MINDCTR`路径下有`test.csv`,`dev.csv`,`train.csv`等数据集文件。如果你的数据集文件不在`data/MIND_Large/MINDCTR`目录请修改对应的`--dataset A --path 'B' `使路径`B/A`下有所需的数据集文件。