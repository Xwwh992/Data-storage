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

```py
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.model1 = Sequential(
            Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            MaxPool2d(2),
            Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(in_features=1024, out_features=64),
            Linear(in_features=64, out_features=10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x
```

```py
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    train_data = torchvision.datasets.CIFAR10("../datasetCIF", train=True, transform=torchvision.transforms.ToTensor(),
                                              download=True)
    test_data = torchvision.datasets.CIFAR10("../datasetCIF", train=False, transform=torchvision.transforms.ToTensor(),
                                             download=True)
    test_datasize = len(test_data)

    # print(test_data.class_to_idx)

    # 尝试降低 num_workers 的值
    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True, drop_last=True)

    network = Network().to(device)  # 将模型移动到 GPU
     loss_fn = nn.CrossEntropyLoss().to(device)  # 将损失函数移动到 GPU
    learning_rate = 1e-2
    optimizer = torch.optim.SGD(network.parameters(), learning_rate)

    total_train_step = 0
    total_test_step = 0
    epoch = 20
writer = SummaryWriter("../0831")

for i in range(epoch):
    print("--------------第{}轮训练----------------".format(i + 1))
    for input, target in train_dataloader:
        input, target = input.to(device), target.to(device)  # 将数据移动到 GPU
        output = network(input)
        loss = loss_fn(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数:{},Loss:{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for input, target in test_dataloader:
            input, target = input.to(device), target.to(device)  # 将数据移动到 GPU
            output = network(input)
            loss = loss_fn(output, target)
            total_test_loss += loss
            accuracy = (output.argmax(1) == target).sum()
            total_accuracy += accuracy
    print("整体测试集上的误差：{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy / test_datasize))
    writer.add_scalar("test_accuracy", total_accuracy / test_datasize, total_test_step)
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    total_test_step += 1
    torch.save(network, "network_{}".format(i))
```

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
### Pytorch函数
- `torch.multinomial` 函数用于从给定的概率分布中采样,返回根据概率分布得出的元素的位置索引，有如下的参数：
  - **`input`**: 一个表示概率分布的 1D 张量，每个元素表示一个事件发生的概率。该张量不需要归一化（即不必总和为 1），因为 `multinomial` 会自动归一化输入。
  - **`num_samples`**: 要采样的数量，即要从输入的概率分布中抽取多少个样本。
  - **`replacement`**: 采样时是否有放回：
    - `True`: 表示有放回的采样，即同一个事件可以被多次选中。
    - `False`: 表示无放回的采样，即一个事件一旦被选中，就不会再被选中（不能重复）。
  - **`generator`**: 用于控制随机性，保证结果的可复现性。使用 `torch.Generator` 来控制种子。

---

- torch.sum(*input*, *dim*, *keepdim=False*, ***, *dtype=None*) → [Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)
  - 维度0的方向是行方向，对每一列求和。维度1的方向是列方向，对每一行求和


```python
>>> a = torch.randn(4, 4)     # 想要压缩哪一个维度 对于列求和压缩了行维度 对于行求和压缩了列维度
>>> a
tensor([[ 0.0569, -0.2475,  0.0737, -0.3429],
        [-0.2993,  0.9138,  0.9337, -1.6864],
        [ 0.1132,  0.7892, -0.1003,  0.5688],
        [ 0.3637, -0.9906, -0.4752, -1.5197]])
>>> torch.sum(a, 1)
tensor([-0.4598, -0.1381,  1.3708, -2.6217])
```
---
- **Broadcasting** semantics

```python
>>> x=torch.empty(5,7,3)
>>> y=torch.empty(5,7,3)
# same shapes are always broadcastable (i.e. the above rules always hold)

>>> x=torch.empty((0,))
>>> y=torch.empty(2,2)
# x and y are not broadcastable, because x does not have at least 1 dimension

# can line up trailing dimensions
>>> x=torch.empty(5,3,4,1)
>>> y=torch.empty(  3,1,1)
# x and y are broadcastable.
# 1st trailing dimension: both have size 1
# 2nd trailing dimension: y has size 1
# 3rd trailing dimension: x size == y size
# 4th trailing dimension: y dimension doesn't exist
# but:
>>> x=torch.empty(5,2,4,1)
>>> y=torch.empty(  3,1,1)
# x and y are not broadcastable, because in the 3rd trailing dimension 2 != 3
```

---

- **torch.nn.functional.one_hot(***tensor*, *num_classes=-1*) → LongTensor

```py
>>> F.one_hot(torch.arange(0, 5) % 3)
tensor([[1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0]])
>>> F.one_hot(torch.arange(0, 5) % 3, num_classes=5)
tensor([[1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0]])
>>> F.one_hot(torch.arange(0, 6).view(3,2) % 3)
tensor([[[1, 0, 0],
         [0, 1, 0]],
        [[0, 0, 1],
         [1, 0, 0]],
        [[0, 1, 0],
         [0, 0, 1]]])
# 在分类问题中，每个类别（如字母）之间没有实际的大小或顺序关系。例如，字母 'a' 和 'b' 分别编码为 1 和 2，它们之间的数值差异没有任何意义。直接使用整数会导致模型可能误解这些数字之间的大小关系，产生错误的学习方向。直接使用整数编码的输入，意味着模型的权重将直接与这些整数相乘。这会导致输入类别之间的参数共享问题，难以学到类别特定的特征。
# one-hot 编码要求输入的类别数 (num_classes) 大于等于输入数字的范围。如果 num_classes 设置得过小，输入中超过这个范围的数字将无法正确编码，甚至导致错误。
```
- 如果你不确定类别数，可以根据数据动态设置 `num_classes`，例如使用 `max(xs) + 1` 来确保 `num_classes` 足够大

```py
num_classes = xs.max().item() + 1
xenc = F.one_hot(xs, num_classes=num_classes)
```

---

torch.randn(**size*, ***, *generator=None*, *out=None*, *dtype=None*, *layout=torch.strided*, *device=None*, *requires_grad=False*, *pin_memory=False*) → [Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)

```py
>>> torch.randn(4)
tensor([-2.1436,  0.9966,  2.3426, -0.6366])
>>> torch.randn(2, 3)
tensor([[ 1.5954,  2.8929, -1.0923],
        [ 1.1719, -0.4709, -0.1996]])
```

---

torch.max(*input*, *dim*, *keepdim=False*, ***, *out=None*)

```py
>>> a = torch.randn(4, 4)
>>> a
tensor([[-1.2360, -0.2942, -0.1222,  0.8475],
        [ 1.1949, -1.1127, -2.2379, -0.6702],
        [ 1.5717, -0.9207,  0.1297, -1.8768],
        [-0.6172,  1.0036, -0.6060, -0.2432]])
>>> torch.max(a, 1)
torch.return_types.max(values=tensor([0.8475, 1.1949, 1.5717, 1.0036]), indices=tensor([3, 0, 0, 1]))
```

---

torch.no_grad()

```py
>>> x = torch.tensor([1.], requires_grad=True)
>>> with torch.no_grad():
...     y = x * 2
>>> y.requires_grad
False
>>> @torch.no_grad()
... def doubler(x):
...     return x * 2
>>> z = doubler(x)
>>> z.requires_grad
False
>>> @torch.no_grad
... def tripler(x):
...     return x * 3
>>> z = tripler(x)
>>> z.requires_grad
False
>>> # factory function exception
>>> with torch.no_grad():
...     a = torch.nn.Parameter(torch.rand(10))
>>> a.requires_grad
True
```


































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

```py
x,y = {
    'train': (Xtr, Ytr),
    'val': (Xdev, Ydev),
    'test': (Xte, Yte),
  }[split]     # 根据split来索引字典，然后x,y是对应的拆包
```

---
```py
pair = min(stats, key=lambda p: merges.get(p, float("inf")))
```

- 如果你直接对字典使用 `for` 循环或传递给像 `min()` 这样的函数，它默认会迭代 **键**
- **如果你想迭代字典的值**，可以显式地使用 `my_dict.values()`
- **如果你想迭代键值对**，可以使用 `my_dict.items()`

---

- **`0x80`** 在二进制形式是 `10000000`。根据 UTF-8 规则，`0x80` 是一个 **后续字节** 的格式，表示它应该跟随一个起始字节（比如 `110xxxxx` 或 `1110xxxx`），不能单独作为一个字符。**`0x80`** 符合 `10xxxxxx` 的模式，这意味着它只能作为多字节字符的一部分，而不能单独解码。因此，单独解码 `0x80` 时会失败，因为它没有前导字节来指示它属于多字节字符的一部分。

---

