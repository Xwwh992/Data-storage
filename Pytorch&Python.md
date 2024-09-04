

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
writer=SummaryWriter(“dir")
tensorboard --logdir="dir"
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
- `torch.multinomial` 函数用于从给定的概率分布中采样,返回根据概率分布得出的元素的位置索引，有如下的参数：
  - **`input`**: 一个表示概率分布的 1D 张量，每个元素表示一个事件发生的概率。该张量不需要归一化（即不必总和为 1），因为 `multinomial` 会自动归一化输入。
  - **`num_samples`**: 要采样的数量，即要从输入的概率分布中抽取多少个样本。
  - **`replacement`**: 采样时是否有放回：
    - `True`: 表示有放回的采样，即同一个事件可以被多次选中。
    - `False`: 表示无放回的采样，即一个事件一旦被选中，就不会再被选中（不能重复）。
  - **`generator`**: 用于控制随机性，保证结果的可复现性。使用 `torch.Generator` 来控制种子。















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

```python
list1 = [10, 20, 30]
list2 = [1, 2, 3]

result = [a - b for a, b in zip(list1, list2)]
print(result)  # 输出: [9, 18, 27]

```

使用for+zip组合来实现对应列表元素的多种计算操作（列表推导式）

---

- 当调用 `zip(a, b)` 时，它不会立即生成所有可能的元组，而是创建一个迭代器。在你遍历这个迭代器时，`zip` 会逐对地从输入的可迭代对象中取出元素并生成一个元组。这个过程是“懒”生成的，只有在你实际使用这些元组时才会生成它们。
- 元组是不可变的有序数据结构，可以存储多个元素。
- 元组可以用于多种操作，如解包、连接和遍历。
- 元组可以作为字典的键，因为它们是不可变的。这使得元组在需要多维索引或复合键时非常有用。
- `sorted` 是 Python 内置的一个函数，用于对可迭代对象进行排序并返回一个新的排序后的列表。与 `sort()` 方法不同，`sort()` 只能对列表进行原地排序（即修改原列表），而 `sorted` 则不会修改原对象，而是返回一个新的排序后的列表。

```python
sorted(iterable, key=None, reverse=False)
```

- 在排序列表或字典时，`lambda`函数经常用于指定排序的依据（即`key`参数）。例如，根据列表元素的某个属性进行排序：

```python
# 对一组元组进行排序，按第二个元素排序
data = [(1, 'c'), (2, 'a'), (3, 'b')]
sorted_data = sorted(data, key=lambda x: x[1])
print(sorted_data)  # 输出: [(2, 'a'), (3, 'b'), (1, 'c')]
```

---

```python
stoi = {s: i for i, s in enumerate(chars)}
```

- `enumerate` 是 Python 的一个内置函数，用于将一个可迭代对象（如列表、字符串等）组合为一个索引序列，即每个元素会与一个索引值组合成一个元组。具体来说，`enumerate(chars)` 会返回一个迭代器，这个迭代器生成的每个元素都是一个元组，包含一个索引（`i`）和字符（`s`）。

---



