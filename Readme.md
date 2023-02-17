该工程文件为KDD'19《dEFEND: Explainable Fake News Detection》论文代码的Pytorch版本
工程文件目录说明：
./data: 数据集存放路径
	./data/embeddings_data: golve词向量存放路径
./results: 模型存放路径以及注意力权重输出路径

config.py ---模型超参数
data_utils.py ---数据处理
defend.py  ---模型代码
glove.py --- glove词向量加载代码
metrics.py  ---评价指标
train.py ---模型训练、测试以及预测代码


根据目前的模型参数设置，得到可复现的最佳结果为：
	Test Loss:1.6194    Test Accuracy:0.9223
	Test Results:    Precision:0.9306, Recall:0.9571,  F1:0.9437

目前得到的结果已较优于论文中呈现结果，相较于源代码主要做了以下改动：
	1.新增数据清洗流程；
	2.embedding的部分采用glove+随机embedding，随机embedding主要用于glove词表中未出现的词；
	3.GRU的输入采用变长序列打包的形式；
	4.word attention和co-attention的部分根据序列真实长度进行掩码计算；
	5.使用高斯分布(mean=0, std=0.1)对模型参数进行初始化。


```
{author = {Cheng Zheng},
 e-mail = {ch_zheng1997@qq.com},
 time = {Nov 27, 2022}
 }
```