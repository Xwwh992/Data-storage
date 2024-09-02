

## Pytorch 

### 训练流程

1. 下载数据 `Download=True`

2. Dataloader(batch)

3. 神经网络

4. 损失函数 `nn.CrossEntropyLoss()`

5. 优化器 `torch.optim.SGD`

6. 设置训练参数 `epoch`

7. `optimizer.zero_grad() loss.backward() optimizer.step()`

8. 测试的时候设置`with torch.no_grad()`

   ---
   
   

```python
accuracy=(output.argmax(1)==target).sum()
```

- 在分类识别的情况下，输出结果是一个n维的向量，其中是预测为每种种类的概率，用`argmax(1)`来横向阅读，然后判断是否为目标值，根据判断把向量中置为True或者False，然后相加得出正确识别的个数。

  ---
  
  

```
writer=SummaryWriter()
tensorboard --logdir=" "
```

- 使用Tensorboard的方法

  ---

  

### 使用GPU

```python
# 检查是否有GPU可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 将模型移到GPU上
network = Network().to(device)
 loss_fn = nn.CrossEntropyLoss().to(device)  # 将损失函数移动到 GPU
   # 将输入和目标都移到GPU上
        input, target = input.to(device), target.to(device)
```

---



- 用`model.train()`切换到训练模式
- `model.eval()`与`torch.no_grad`结合来评估
- 用不同device训练的模型再不同的device调用时候要转换

---

- `for` 循环可以用在列表推导式（list comprehension）中，用于生成列表

```python
self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
# _ 是一个常见的惯用符号，用于表示循环变量在循环体中不需要被使用。它只是占位符，表示我们并不关心循环变量的值。
```

---

## Python

```python
wi * xi for wi, xi in zip(self.w, x)
```

这一部分是一个生成器表达式，它不是直接生成一个列表，而是创建一个惰性求值的生成器。生成器在需要时才逐个生成元素，而不会一次性把所有元素都计算并存储在内存中。
具体来说，生成器表达式 (wi * xi for wi, xi in zip(self.w, x)) 会在 sum 函数中被逐个消费，并计算加权和

```python
sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
```

`sum(iterable, start)` 接收两个参数：

1. `iterable`: 一个可迭代对象（例如列表或生成器）
2. `start`: 求和的起始值（初始值）

第一次：`累加器 = self.b + w1 * x1`

第二次：`累加器 = (self.b + w1 * x1) + w2 * x2`

这里的最终结果是 `self.b + w1 * x1 + w2 * x2`

---











