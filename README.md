# 情感分析

## 案例简介

情感分析旨在挖掘文本中的主观信息，它是自然语言处理中的经典任务。在本次任务中，我们将在影评文本数据集（Rotten Tomato）上进行情感分析，通过实现课堂讲授的模型方法，深刻体会自然语言处理技术在生活中的应用。

同学们需要实现自己的情感分析器，包括特征提取器（可以选择词袋模型、n-gram模型或词向量模型）、简单的线性分类器以及梯度下降函数。随后在数据集上进行训练和验证。我们提供了代码框架，同学们只需补全`model.py`中的两个函数。

## 数据说明

我们使用来自Rotten Tomato的影评文本数据。其中训练集`data_rt.train`和测试集`data_rt.test`均包含了3554条影评，每条影评包含了文本和情感标签。示例如下：

```
+1 visually , 'santa clause 2' is wondrously creative .
```

其中，`+1` 表示这条影评蕴涵了正面感情，后面是影评的具体内容。

## 数据特征提取

TODO：补全`featureExtractor`函数

在这个步骤中，同学们需要读取给定的训练和测试数据集，并提取出文本中的特征，输出特征向量。

同学们可以选择选择词袋模型、n-gram模型或词向量模型中的一种，也可以对比三者的表现有何差异。

## 训练分类器

TODO：补全`learnPredictor`函数

我们提供的训练数据集中，每句话的标签在文本之前，其中`+1`表示这句话蕴涵了正面感情，`-1`表示这句话蕴涵了负面感情。因此情感分析问题就成为一个分类问题。

我们采用最小化hinge loss的方法训练分类器，假设我们把每条影评文本$x$映射为对应的特征向量$\phi(x)$，hinge loss的定义为
$$
L(x,y; \mathbf{w})=\max(0,1-\mathbf{w}\cdot\phi(x)y)
$$
同学们需要实现一个简单的线性分类器，并推导出相应的梯度下降函数。

## 实验与结果分析

在训练集上完成训练后，同学们需要在测试集上测试分类器性能。本小节要求同学们画出训练集上的损失函数下降曲线和测试集的最终结果，并对结果进行分析。



## 评分要求

同学们需要提交源代码和实验报告。实验报告中应包含两部分内容：

- 对hinge loss反向传播的理论推导，请写出参数的更新公式。
- 对实验结果的分析，请描述采用的模型结构、模型在训练集上的损失函数下降曲线和测试集的最终结果，并对结果进行分析。分析可以从模型的泛化能力、参数对模型性能的影响以及不同特征的影响等方面进行。


## 注意:
ipynb文件为几个py文件的整合，二者基本等价（即使用了ipynb便无需补全py，反之亦然）。ipynb为方便同学们作业而写。