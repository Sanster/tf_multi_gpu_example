原文：http://blog.s-schoener.com/2017-12-15-parallel-tensorflow-intro/

多 GPU 训练可以分为两种：
1. 单机多 GPU（multiple GPUs on your local machine）
2. 多机多 GPU（distributed TensorFlow）

使用 GPU 并行训练的两种模式：
1. horziontal slices：每一块 GPU 只跑网络的一部分，这意味着可以使用更大的 batch_size，但是要针对不同的网络进行设计，通用性较差。
2. vertical slices（replicated training）：每一块 GPU 都跑完整的网络，包括前向计算、loss 计算和梯度计算，
最终将梯度汇总到 parameter server（例如 CPU），parameter server 进行梯度平均和参数更新，最后将更新好的参数发给 GPU。
batch_size 可以增大为 GPU 数量的倍数。

一般来说 replicated training 方法用得比较多，因为可以很方便地添加更多 GPU 来训练。parameter server 更新参数时也有两种模式：
1. synchronously（同步）：parameter server 等待所有 GPU 计算完参数后进行平均，并参数更新，这种方法相当于增加了 mini_batch 的大小，
意味着对于每一次更新**对于梯度的估计更加准确**，理论上应该能够减少训练过程中的震荡。
2. asynchronously（异步）：parameter server 单独对每一个 GPU 更新参数，因此每块 GPU 上参数可能会不一样（训练结果怎么保存参数？取平均？）。
异步方法可以减少训练时间，因为参数更新的次数多了。

`train.py` 中的代码使用了 `in-graph replication with synchronous training` 的方法进行 MNIST 分类模型的训练，即在一台机器上
使用多块 GPU 以同步参数更新的方式进行训练。使用 CPU 来作为 parameter server。
数据的读取使用 tensorflow [dataset api](https://www.tensorflow.org/api_docs/python/tf/data/Dataset)






