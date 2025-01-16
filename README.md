# HCCF-ReChorus
本项目是中山大学机器学习课程作业，旨在使用ReChorus框架评估推荐算法HCCF的表现。

### HCCF（Hypergraph Contrastive Collaborative Filtering）推荐算法
HCCF是基于协同过滤算法提出的超图协同过滤算法，通过超图增强的跨视图对比学习架构，联合捕获局部和全局协同关系。
+ [HCCF](https://github.com/akaxlh/HCCF)

### ReChorus框架
ReChorus框架是一个基于PyTorch的轻量化、易拓展的推荐系统评价框架。该框架可用于评估不同算法在相同基准上的表现。
ReChorus框架分为Input、Reader、Model和Runner层。其中，本次作业在`src/models/general`中增加了`MyModel.py`，在`src/helper`中增加了`MyRunner.py`，即通过修改Model层和Runner层实现HCCF推荐算法的接入。实验过程详见实验报告。
+ [Rechorus](https://github.com/THUwangcy/ReChorus)

### 运行
下载该项目，根据[Rechorus](https://github.com/THUwangcy/ReChorus)中要求配置环境。
+ 在cmd中运行：
  
  `python main.py --model_name MyModel --reg 1e-7 --ssl_reg 1e-3 --temp 0.3 --path ../data/MovieLens_1M --dataset ML_1MTOPK --lr 0.001 --emb_size 32 --hyper_num 128 --keepRate 0.5`
  
  可复现目前代码在数据集`MovieLens_1M`上的最佳表现。
  
  `python main.py --model_name MyModel --reg 1e-6 --temp 0.3 --dataset Grocery_and_Gourmet_Food --lr 0.0003 --emb_size 64 --hyper_num 8 --keepRate 1 --ssl_reg 0.3`
  
  可复现目前代码在数据集`Grocery_and_Gourmet_Food`上的最佳表现。
