# LLM

---

## Autograd

- gradient可以看成指向loss增加的向量
- **`isinstance()`**用来判断实例对象是否属于某个类。对象不属于类时候可以用于判定然后做类的转换再使用
- 卷积核在卷积操作中的作用确实可以理解为衡量它与对应图像区块之间的**相似程度**
- 只定义了**`__repr__`**方法而没有定义**`__str__`**方法，那当调用`str()`或`print()`时，Python会自动使用**`__repr__`**方法的返回值
- **`__repr__`**应该用于返回更详细的开发者视角的信息，而**`__str__`**则用于简洁的用户视角的信息
- 反向传播：用local gradient乘上传递过来的gradient，单纯的加法只会让梯度 **flow over**不改变数值

```python
class Value:
  
  def __init__(self, data, _children=(), _op='', label=''):
    self.data = data
    self.grad = 0.0
    self._backward = lambda: None
    self._prev = set(_children)  # 记录前驱节点
    self._op = _op    # 记录运算符号
    self.label = label   #记录变量的名字

  def __repr__(self):
    return f"Value(data={self.data})"
  
  def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), '+')   #加法不改变梯度
    
    def _backward():
      self.grad += 1.0 * out.grad
      other.grad += 1.0 * out.grad
    out._backward = _backward
    
    return out

  def __mul__(self, other): # 乘法用local gradient * 传递过来的梯度
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), '*')
    
    def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
    out._backward = _backward
      
    return out
  
  def __pow__(self, other):
    assert isinstance(other, (int, float)), "only supporting int/float powers for now"
    out = Value(self.data**other, (self,), f'**{other}')

    def _backward():
        self.grad += other * (self.data ** (other - 1)) * out.grad
    out._backward = _backward

    return out
  
  def __rmul__(self, other): # other * self  解决int和value相乘的问题 python会自动调用这个函数
    return self * other

  def __truediv__(self, other): # self / other
    return self * other**-1

  def __neg__(self): # -self
    return self * -1

  def __sub__(self, other): # self - other
    return self + (-other)

  def __radd__(self, other): # other + self
    return self + other

  def tanh(self):
    x = self.data
    t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
    out = Value(t, (self, ), 'tanh')
    
    def _backward():
      self.grad += (1 - t**2) * out.grad     #tanh的导数
    out._backward = _backward
    
    return out
  
  def exp(self):
    x = self.data
    out = Value(math.exp(x), (self, ), 'exp')
    
    def _backward():		#使用+=可以防止一个变量不止使用了一次，另一个变量对前驱节点反向传播时覆盖前一个结果							
      self.grad += out.data * out.grad 
    out._backward = _backward
    
    return out
  
  
  def backward(self):           # 神经网络的遍历问题可以看成图的遍历的问题（有向无环图）
                                # 使用DFS来生成拓扑排序              
    topo = []
    visited = set()
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)
    build_topo(self)
    
    self.grad = 1.0      #自身对自身的梯度是1
    for node in reversed(topo):
      node._backward()
```

### MLP(Perceptron)

```python
class Neuron:
  
  def __init__(self, nin):
    self.w = [Value(random.uniform(-1,1)) for _ in range(nin)] # 生成对应输入个数维数的权重 转化成了Value型
    self.b = Value(random.uniform(-1,1))
  
  def __call__(self, x):
    # w * x + b
    act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)   # 计算w*x+b
    out = act.tanh()    # 激活函数
    return out
  
  def parameters(self):
    return self.w + [self.b]

class Layer:
  
  def __init__(self, nin, nout):
    self.neurons = [Neuron(nin) for _ in range(nout)]  # 生成的神经元的个数取决于有几个输出 输入用来计算输出值
  
  def __call__(self, x):
    outs = [n(x) for n in self.neurons] # n是neurons中的元素，而neurous是一个neuron列表。每个neuron对象在创建的时候 													         		 	 # 就已经被实例化，按照简单的循环理解即可。
    return outs[0] if len(outs) == 1 else outs
  
  def parameters(self):
    return [p for neuron in self.neurons for p in neuron.parameters()]
			# 最里面的for循环的变量和最外面的变量名字应该对应，使用了嵌套循环
class MLP:
  
  def __init__(self, nin, nouts):
    sz = [nin] + nouts # 把输入层、隐含层和输出层合并成一个列表
    self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]  # 调用layer层生成每层神经元，同时生成多个层
  																																# 每层的神经元个数取决于这一层要向下一层输出多少
  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x
  
  def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]
```

```python
xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0] # desired targets
for k in range(20):
  
  # forward pass
  ypred = [n(x) for x in xs]      # 可以选择不同的损失函数
  loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
  
  # backward pass
  for p in n.parameters():
    p.grad = 0.0       # 在新一次的反向传播前一定要梯度归零
  loss.backward()
  
  # update
  for p in n.parameters():     # 手动的梯度下降  可以采取 learning_rate = 1.0 - 0.9*k/100
    p.data += -0.1 * p.grad
  
  print(k, loss.data)   
```

## Bigram

-  **单个神经元一次只对一个输入进行运算**，但通过**矩阵运算和批处理**，神经网络能够一次并行处理多个输入样本，把几个输入组合成一个大的矩阵，认为这个矩阵就是当前的一个输入。通过**批处理（batch processing）**实现并行计算
- 如果你只生成 `5x1` 的输出，那么你只能为每个输入样本输出一个**具体的结果**，而无法得到预测的概率分布。这种情况下，你的模型只能做出一个**硬预测**（即每次只能预测一个特定的字符），但在大多数机器学习问题中，我们希望模型能够**输出概率分布**，以便：
  - **进行进一步分析**（比如衡量模型的不确定性）。
  - **用于训练和优化**，特别是在反向传播中，概率分布可以帮助计算更合适的损失函数（如交叉熵损失）。
- **L2正则化**
  - **防止权重过大**：L2正则化通过限制权重的大小，避免模型过度拟合训练数据。
  - **简化模型**：减小权重的数值，等同于减少模型的复杂性，使其更平滑，从而提升泛化能力。
  - **抑制噪声影响**：正则化帮助模型忽略数据中的噪声，避免模型对训练数据的过度拟合。

```py
words = open('names.txt', 'r').read().splitlines()

N = torch.zeros((27, 27), dtype=torch.int32) #用来记录每个字母对出现的频数

chars = sorted(list(set(''.join(words)))) # 构建一个26个字母的字母集合而且从小到大排列
stoi = {s:i+1 for i,s in enumerate(chars)} # 构建了一个包含'.'的从字母到数字的字典 a-1,b-2,etc
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}  #构建从数字到字母的映射
for w in words:
  chs = ['.'] + list(w) + ['.']   # 利用添加.来记录名字开头的字母和名字结尾的字母
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    N[ix1, ix2] += 1
    
P = (N+1).float()  #加1是为了避免有些组合出现的频数是0，这样会导致计算负对数似然的时候出现无穷，加的越多，越uniform
# module smoothing
P /= P.sum(1, keepdims=True)# 这里求的是行向量的和，如果不使用keepdims，会导致本该产生的列向量变行向量   
g = torch.Generator().manual_seed(2147483647) # 创建generator

for i in range(5):  # 测试输出的结果
  out = []
  ix = 0
  while True:
    p = P[ix]
    ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()#给出按照概率来预测的采样
    out.append(itos[ix])
    if ix == 0:
      break
  print(''.join(out))
# GOAL: maximize likelihood of the data w.r.t. model parameters (statistical modeling)
# equivalent to maximizing the log likelihood (because log is monotonic)
# equivalent to minimizing the negative log likelihood
# equivalent to minimizing the average negative log likelihood
log_likelihood = 0.0
n = 0

 #-----------------------------通过简单的计算出现的频率来计算loss------------------------
for w in words:
#for w in ["andrejq"]:       #可以衡量给出的一个具体的名字用当前模型预测的quality
  chs = ['.'] + list(w) + ['.'] 
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    prob = P[ix1, ix2]
    logprob = torch.log(prob)
    log_likelihood += logprob
    n += 1
    #print(f'{ch1}{ch2}: {prob:.4f} {logprob:.4f}')

print(f'{log_likelihood=}')
nll = -log_likelihood    #负对数似然
print(f'{nll=}')
print(f'{nll/n}')    # 负对数似然的平均值

# ----------------- !!! OPTIMIZATION !!! yay, but this time actually --------------
# create the dataset
xs, ys = [], []
for w in words:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    xs.append(ix1)
    ys.append(ix2)
xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()
print('number of examples: ', num)

# initialize the 'network'
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g, requires_grad=True)
# gradient descent
for k in range(1):
  
  # forward pass
  xenc = F.one_hot(xs, num_classes=27).float() # input to the network: one-hot encoding
  logits = xenc @ W # predict log-counts
  counts = logits.exp() # counts, equivalent to N
  probs = counts / counts.sum(1, keepdims=True) # probabilities for next character
  loss = -probs[torch.arange(num), ys].log().mean() + 0.01*(W**2).mean() # regularization term
  print(loss.item())
  
  # backward pass
  W.grad = None # set to zero the gradient
  loss.backward()
  
  # update
  W.data += -50 * W.grad
# finally, sample from the 'neural net' model
g = torch.Generator().manual_seed(2147483647)

for i in range(5):
  
  out = []
  ix = 0
  while True:
    
    # ----------
    # BEFORE:
    #p = P[ix]
    # ----------
    # NOW:
    xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
    logits = xenc @ W # predict log-counts
    counts = logits.exp() # counts, equivalent to N
    p = counts / counts.sum(1, keepdims=True) # probabilities for next character
    # ----------
    
    ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
    out.append(itos[ix])
    if ix == 0:
      break
  print(''.join(out))
```

## Trigram+MLP

- **对数空间的均匀分布**
  - 如果你在 `-3` 到 `0` 之间均匀生成数字，那么这些数字的指数值（即 `10^lre`）将覆盖从 `10^(-3)` 到 `10^0`（即 0.001 到 1）之间的范围。
  - 这样做是为了避免学习率在小值范围内变化过快，而在大值范围内变化过慢。
  
- 将**隐藏层变换到更高的维度**的主要原因包括：

  - 提升模型的表达能力，帮助模型学习到更多的复杂特征。
  - 增强模型的非线性变换能力，使得它能更好地捕捉上下文关系。
  - 通过更高维度的特征空间，处理更多的信息和模式。
  - 帮助在欠拟合和过拟合之间找到一个平衡点。

- **展开嵌入向量**

  - 在自然语言处理任务中，比如字符或词的预测，我们通常会输入多个上下文字符或词，通过这些上下文预测下一个字符。为了让模型能够同时看到这些上下文字符的嵌入信息，我们需要将它们拼接成一个完整的向量，这样神经网络可以处理整个上下文，而不仅仅是单个字符。在这种情况下，每个字符的嵌入向量是 10 维的，3 个字符的上下文就会形成一个 30 维的向量。通过展开，神经网络就能同时看到这 3 个字符的信息，并通过训练找到它们和下一个字符之间的关系。
  
- **Mini-batch&Batch&SGD**
  - **大数据集**：小批量梯度下降（Mini-batch）是常用选择。可以利用并行计算，平衡了计算效率与更新稳定性。
  - **数据量较小**：批量梯度下降可以直接使用。
  - **高频率快速更新需求**：随机梯度下降适合快速迭代和跳出局部最优，但需要配合其他技术（如学习率调整）稳定训练。

| 方法       | 每次更新样本量 | 优点                                   | 缺点                       |
| ---------- | -------------- | -------------------------------------- | -------------------------- |
| SGD        | 1              | 更新频繁，适合大数据集，能跳出局部最优 | 噪声大，收敛不稳定         |
| Mini-batch | 小批量 (如32)  | 利用硬件并行，稳定性好，收敛速度快     | 仍有噪声，批量大小需要调节 |
| Batch      | 全集           | 稳定，准确                             | 计算代价大，收敛速度慢     |

```py
import torch
import torch.nn.functional as F
words = open('names.txt', 'r').read().splitlines()
len(words)
# build the vocabulary of characters and mappings to/from integers
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
print(itos)
# build the dataset

block_size = 3 # context length: how many characters do we take to predict the next one?
X, Y = [], []
for w in words:
  
  #print(w)
  context = [0] * block_size
  for ch in w + '.':
    ix = stoi[ch]
    X.append(context)
    Y.append(ix)
    #print(''.join(itos[i] for i in context), '--->', itos[ix])
    context = context[1:] + [ix] # crop and append
  
X = torch.tensor(X)
Y = torch.tensor(Y)

def build_dataset(words):  #分割数据集
  X, Y = [], []
  for w in words:

    #print(w)
    context = [0] * block_size
    for ch in w + '.':
      ix = stoi[ch]
      X.append(context)
      Y.append(ix)
      #print(''.join(itos[i] for i in context), '--->', itos[ix])
      context = context[1:] + [ix] # crop and append  去除第一个然后新加入一个

  X = torch.tensor(X)
  Y = torch.tensor(Y)
  print(X.shape, Y.shape)
  return X, Y

import random
random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))  #划分训练集、验证集和测试集
n2 = int(0.9*len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])
# ------------ now made respectable :) ---------------
g = torch.Generator().manual_seed(2147483647) # for reproducibility
C = torch.randn((27, 10), generator=g)
W1 = torch.randn((30, 200), generator=g)  #上文的数目*映射到的维度数
b1 = torch.randn(200, generator=g)   # MLP隐藏层
W2 = torch.randn((200, 27), generator=g)
b2 = torch.randn(27, generator=g)
parameters = [C, W1, b1, W2, b2]
sum(p.nelement() for p in parameters) # number of parameters in total
for p in parameters:
  p.requires_grad = True
  
lre = torch.linspace(-3, 0, 1000)
lrs = 10**lre
lri = []  # 存储每次迭代的学习率
lossi = []  # 存储每次迭代的损失值
stepi = []  # 存储每次迭代的步数

for i in range(200000):  # 训练 200,000 步
    # minibatch construct
    ix = torch.randint(0, Xtr.shape[0], (32,))  # 随机选择 32 个样本进行训练
 
  # 前向传播
    emb = C[Xtr[ix]]  # (32, 3, 10) 从嵌入矩阵中查找对应的嵌入向量
    h = torch.tanh(emb.view(-1, 30) @ W1 + b1)  # (32, 200) 使用 tanh 激活函数，计算隐藏层输出
    logits = h @ W2 + b2  # (32, 27) 计算输出层 logits
    loss = F.cross_entropy(logits, Ytr[ix])  # 计算交叉熵损失

  #print(loss.item())
  
  # backward pass
  for p in parameters:
    p.grad = None
  loss.backward()
  
  # update
  #lr = lrs[i]
  lr = 0.1 if i < 100000 else 0.01
  for p in parameters:
    p.data += -lr * p.grad

  # track stats
  #lri.append(lre[i])
  stepi.append(i)
  lossi.append(loss.log10().item())

#print(loss.item())
emb = C[Xtr] # (32, 3, 2)
h = torch.tanh(emb.view(-1, 30) @ W1 + b1) # (32, 100)
logits = h @ W2 + b2 # (32, 27)
loss = F.cross_entropy(logits, Ytr)

emb = C[Xdev] # (32, 3, 2)
h = torch.tanh(emb.view(-1, 30) @ W1 + b1) # (32, 100)
logits = h @ W2 + b2 # (32, 27)
loss = F.cross_entropy(logits, Ydev)

# sample from the model
g = torch.Generator().manual_seed(2147483647 + 10)

for _ in range(20):
    
    out = []
    context = [0] * block_size # initialize with all ...
    while True:
      emb = C[torch.tensor([context])] # (1,block_size,d)
      h = torch.tanh(emb.view(1, -1) @ W1 + b1)
      logits = h @ W2 + b2
      probs = F.softmax(logits, dim=1)
      ix = torch.multinomial(probs, num_samples=1, generator=g).item()
      context = context[1:] + [ix]
      out.append(ix)
      if ix == 0:
        break
    
    print(''.join(itos[i] for i in out))
```

## Activation&Gradient&Batchnorm
- **tanh引发的梯度消失**

  - 大输入值的区域：当输入值非常大时（正数或负数），tanh 的输出接近 1 或 -1。这时，tanh 的导数会非常接近 0，反向传播时梯度将会极小，这就导致梯度消失。
- 深层网络：在深度网络中，如果使用 tanh 作为每一层的激活函数，随着层数的增加，传递到早期层的梯度会因为 tanh 导数非常小而不断衰减，最终导致网络无法有效更新权重。
  - 1. **ReLU 替代**：ReLU（Rectified Linear Unit）常常被用作 tanh 的替代激活函数，因为 ReLU 在正区间上没有饱和区域，导数为 1，不容易出现梯度消失问题。但 ReLU 也有其他问题，如死 ReLU（Dead ReLU）现象。
  2. 权重初始化技巧：使用 **Xavier（Glorot）**初始化或 He 初始化。这些初始化方法能够确保在网络的每一层，激活函数输出的方差保持稳定，帮助缓解梯度消失或爆炸问题。
    3. **Batch Normalization**：Batch Normalization 可以在每一层之后对输出进行归一化，保证激活值在训练过程中保持一个稳定的分布，从而减少梯度消失问题。
  4. 其他激活函数：除了 ReLU，也可以尝试使用 **Leaky ReLU、ELU** 或 **SELU** 等激活函数，这些函数在深度网络中表现得更好，尤其是在避免梯度消失问题方面。
- Python 是一种动态语言，允许在运行时对对象进行修改，可以**动态地为任何对象添加属性**
```py
# MLP revisited
n_embd = 10 # the dimensionality of the character embedding vectors
n_hidden = 200 # the number of neurons in the hidden layer of the MLP

#初始化的时候应该尽可能假设27个字母中下一个出现的概率都是相等的，这样可以降低初始loss，减少squeeze weights的过程
#初始化的时候减少神经元的饱和状态，增加productive的训练轮数
#b1的值在计算batchnorm的时候不起任何作用
g = torch.Generator().manual_seed(2147483647) # for reproducibility
C  = torch.randn((vocab_size, n_embd),            generator=g)
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * (5/3)/((n_embd * block_size)**0.5) #* 0.2
#b1 = torch.randn(n_hidden,                        generator=g) * 0.01  
W2 = torch.randn((n_hidden, vocab_size),          generator=g) * 0.01 
b2 = torch.randn(vocab_size,                      generator=g) * 0

#BatchNorm 可能会导致本来没有交集的数据在同一个 batch 中产生一定的联系
# BatchNorm parameters
bngain = torch.ones((1, n_hidden))
bnbias = torch.zeros((1, n_hidden))
bnmean_running = torch.zeros((1, n_hidden))
bnstd_running = torch.ones((1, n_hidden))

parameters = [C, W1, W2, b2, bngain, bnbias]
print(sum(p.nelement() for p in parameters)) # number of parameters in total
for p in parameters:
  p.requires_grad = True
  
# same optimization as last time
max_steps = 200000
batch_size = 32
lossi = []

for i in range(max_steps):
  
  # minibatch construct
  ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
  Xb, Yb = Xtr[ix], Ytr[ix] # batch X,Y
  
  # forward pass
  emb = C[Xb] # embed the characters into vectors
  embcat = emb.view(emb.shape[0], -1) # concatenate the vectors
  # Linear layer
  hpreact = embcat @ W1 #+ b1 # hidden layer pre-activation
  # BatchNorm layer
  # -------------------------------------------------------------
  #在训练期间持续更新运行均值和标准差，以便在推断时使用
  bnmeani = hpreact.mean(0, keepdim=True)
  bnstdi = hpreact.std(0, keepdim=True)
  hpreact = bngain * (hpreact - bnmeani) / bnstdi + bnbias
  with torch.no_grad():
    bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
    bnstd_running = 0.999 * bnstd_running + 0.001 * bnstdi
  # -------------------------------------------------------------
  # Non-linearity
  h = torch.tanh(hpreact) # hidden layer
  logits = h @ W2 + b2 # output layer
  loss = F.cross_entropy(logits, Yb) # loss function
  
  # backward pass
  for p in parameters:
    p.grad = None
  loss.backward()
  
  # update
  lr = 0.1 if i < 100000 else 0.01 # step learning rate decay
  for p in parameters:
    p.data += -lr * p.grad

  # track stats
  if i % 10000 == 0: # print every once in a while
    print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
  lossi.append(loss.log10().item())
  
  # calibrate the batch norm at the end of training

with torch.no_grad():
  # pass the training set through
  emb = C[Xtr]
  embcat = emb.view(emb.shape[0], -1)
  hpreact = embcat @ W1 # + b1
  # measure the mean/std over the entire training set
  bnmean = hpreact.mean(0, keepdim=True)
  bnstd = hpreact.std(0, keepdim=True)
@torch.no_grad() # this decorator disables gradient tracking
def split_loss(split):
  x,y = {
    'train': (Xtr, Ytr),
    'val': (Xdev, Ydev),
    'test': (Xte, Yte),
  }[split]
  emb = C[x] # (N, block_size, n_embd)
  embcat = emb.view(emb.shape[0], -1) # concat into (N, block_size * n_embd)
  hpreact = embcat @ W1 # + b1
  #hpreact = bngain * (hpreact - hpreact.mean(0, keepdim=True)) / hpreact.std(0, keepdim=True) + bnbias
  hpreact = bngain * (hpreact - bnmean_running) / bnstd_running + bnbias
  h = torch.tanh(hpreact) # (N, n_hidden)
  logits = h @ W2 + b2 # (N, vocab_size)
  loss = F.cross_entropy(logits, y)
  print(split, loss.item())

split_loss('train')
split_loss('val')

# SUMMARY + PYTORCHIFYING -----------
# Let's train a deeper network
# The classes we create here are the same API as nn.Module in PyTorch

class Linear:
  
  def __init__(self, fan_in, fan_out, bias=True):
    self.weight = torch.randn((fan_in, fan_out), generator=g) / fan_in**0.5
    self.bias = torch.zeros(fan_out) if bias else None
  
  def __call__(self, x):
    self.out = x @ self.weight
    if self.bias is not None:
      self.out += self.bias
    return self.out
  
  def parameters(self):
    return [self.weight] + ([] if self.bias is None else [self.bias])


class BatchNorm1d:
  
  def __init__(self, dim, eps=1e-5, momentum=0.1):
    self.eps = eps
    self.momentum = momentum
    self.training = True
    # parameters (trained with backprop)
    self.gamma = torch.ones(dim)
    self.beta = torch.zeros(dim)
    # buffers (trained with a running 'momentum update')
    self.running_mean = torch.zeros(dim)
    self.running_var = torch.ones(dim)
  
  def __call__(self, x):
    # calculate the forward pass
    if self.training:
      xmean = x.mean(0, keepdim=True) # batch mean
      xvar = x.var(0, keepdim=True) # batch variance
    else:
      xmean = self.running_mean
      xvar = self.running_var
    xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
    self.out = self.gamma * xhat + self.beta
    # update the buffers
    if self.training:
      with torch.no_grad():
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
    return self.out
   def parameters(self):
    return [self.gamma, self.beta]

class Tanh:
  def __call__(self, x):
    self.out = torch.tanh(x)
    return self.out
  def parameters(self):
    return []

n_embd = 10 # the dimensionality of the character embedding vectors
n_hidden = 100 # the number of neurons in the hidden layer of the MLP
g = torch.Generator().manual_seed(2147483647) # for reproducibility

C = torch.randn((vocab_size, n_embd),            generator=g)
layers = [
  Linear(n_embd * block_size, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
  Linear(           n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
  Linear(           n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
  Linear(           n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
  Linear(           n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
  Linear(           n_hidden, vocab_size, bias=False), BatchNorm1d(vocab_size),
]
# layers = [
#   Linear(n_embd * block_size, n_hidden), Tanh(),
#   Linear(           n_hidden, n_hidden), Tanh(),
#   Linear(           n_hidden, n_hidden), Tanh(),
#   Linear(           n_hidden, n_hidden), Tanh(),
#   Linear(           n_hidden, n_hidden), Tanh(),
#   Linear(           n_hidden, vocab_size),
# ]
with torch.no_grad():
  # last layer: make less confident
  layers[-1].gamma *= 0.1
  #layers[-1].weight *= 0.1
  # all other layers: apply gain
  for layer in layers[:-1]:
    if isinstance(layer, Linear):
      layer.weight *= 1.0 #5/3

parameters = [C] + [p for layer in layers for p in layer.parameters()]
print(sum(p.nelement() for p in parameters)) # number of parameters in total
for p in parameters:
  p.requires_grad = True
  
# same optimization as last time
max_steps = 200000
batch_size = 32
lossi = []
ud = []

for i in range(max_steps):
  
  # minibatch construct
  ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
  Xb, Yb = Xtr[ix], Ytr[ix] # batch X,Y
  
  # forward pass
  emb = C[Xb] # embed the characters into vectors
  x = emb.view(emb.shape[0], -1) # concatenate the vectors
  for layer in layers:
    x = layer(x)
  loss = F.cross_entropy(x, Yb) # loss function
  
  # backward pass
  for layer in layers:
    layer.out.retain_grad() # AFTER_DEBUG: would take out retain_graph
  for p in parameters:
    p.grad = None
  loss.backward()
  
  # update
  lr = 0.1 if i < 150000 else 0.01 # step learning rate decay
  for p in parameters:
    p.data += -lr * p.grad

  # track stats
  if i % 10000 == 0: # print every once in a while
    print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
  lossi.append(loss.log10().item())
  with torch.no_grad():
    ud.append([((lr*p.grad).std() / p.data.std()).log10().item() for p in parameters])  
  if i >= 1000:
     break # AFTER_DEBUG: would take out obviously to run full optimization
      
@torch.no_grad() # this decorator disables gradient tracking
def split_loss(split):
  x,y = {
    'train': (Xtr, Ytr),
    'val': (Xdev, Ydev),
    'test': (Xte, Yte),
  }[split]
  emb = C[x] # (N, block_size, n_embd)
  x = emb.view(emb.shape[0], -1) # concat into (N, block_size * n_embd)
  for layer in layers:
    x = layer(x)
  loss = F.cross_entropy(x, y)
  print(split, loss.item())

# put layers into eval mode
for layer in layers:      
  layer.training = False    #评估的时候设置不处于训练模式
split_loss('train')
split_loss('val')     

# sample from the model
g = torch.Generator().manual_seed(2147483647 + 10)

for _ in range(20):
    
    out = []
    context = [0] * block_size # initialize with all ...
    while True:
      # forward pass the neural net
      emb = C[torch.tensor([context])] # (1,block_size,n_embd)
      x = emb.view(emb.shape[0], -1) # concatenate the vectors
      for layer in layers:
        x = layer(x)
      logits = x
      probs = F.softmax(logits, dim=1)
      # sample from the distribution
      ix = torch.multinomial(probs, num_samples=1, generator=g).item()
      # shift the context window and track the samples
      context = context[1:] + [ix]
      out.append(ix)
      # if we sample the special '.' token, break
      if ix == 0:
        break
    
    print(''.join(itos[i] for i in out)) # decode and print the generated word
```

## WaveNet

- `e=torch.randn(4,8,10)`,`e[:, ::2, :]` 和 `e[:, 1::2, :]` 从偶数和奇数索引产生了形状为 `(4, 4, 10)` 的张量。
- `torch.cat` 沿着维度 `2` 进行拼接，将两个 10 个元素的维度拼接成 20 个元素，因此最终的形状为 `(4, 4, 20)`。

```py
import torch
import torch.nn.functional as F

# 读取所有的名字数据，并将每行作为一个单独的词存储在words列表中
words = open('names.txt', 'r').read().splitlines()

# 构建字符词汇表，将每个字符映射为整数（字符到索引的映射）和索引到字符的反向映射
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)} # stoi将字符映射到整数（1开始），并为特殊符号'.'设置0
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()} # itos是从整数映射回字符
vocab_size = len(itos) # 字符的词汇表大小

# 随机打乱words列表的顺序，以便后续训练
import random
random.seed(42)
random.shuffle(words)

# 定义块大小，即上下文长度：我们使用多少个字符来预测下一个字符
block_size = 8

# 构建数据集，X是输入字符的上下文，Y是对应的下一个字符
def build_dataset(words):  
  X, Y = [], []
  
  for w in words:
    context = [0] * block_size # 初始化为全0的上下文
    for ch in w + '.': # 遍历每个名字的字符，并在末尾添加句点'.'，表示词的结束
      ix = stoi[ch] # 将字符转换为索引
      X.append(context) # 将当前上下文添加到输入X中
      Y.append(ix) # 将下一个字符的索引作为标签Y
      context = context[1:] + [ix] # 滑动窗口，更新上下文，移除最早的字符并添加当前字符

  # 将X和Y转换为PyTorch张量
  X = torch.tensor(X)
  Y = torch.tensor(Y)
  print(X.shape, Y.shape)
  return X, Y

# 将数据集划分为训练集、验证集和测试集（80%训练，10%验证，10%测试）
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))
Xtr,  Ytr  = build_dataset(words[:n1])     # 80% 训练集
Xdev, Ydev = build_dataset(words[n1:n2])   # 10% 验证集
Xte,  Yte  = build_dataset(words[n2:])     # 10% 测试集

# -----------------------------------------------------------------------------------------------
# 自定义的线性层，进行线性变换
class Linear:
  
  def __init__(self, fan_in, fan_out, bias=True):
    self.weight = torch.randn((fan_in, fan_out)) / fan_in**0.5 # 使用Kaiming初始化方法
    self.bias = torch.zeros(fan_out) if bias else None # 如果bias=True，初始化偏置为0
  
  def __call__(self, x):
    # 前向传播：进行矩阵乘法并加上偏置
    self.out = x @ self.weight
    if self.bias is not None:
      self.out += self.bias
    return self.out
  
  def parameters(self):
    # 返回层的参数：权重和（如果有的话）偏置
    return [self.weight] + ([] if self.bias is None else [self.bias])

# -----------------------------------------------------------------------------------------------
# 自定义的一维批归一化层，用于加速训练并稳定模型
class BatchNorm1d:
  
  def __init__(self, dim, eps=1e-5, momentum=0.1):
    self.eps = eps
    self.momentum = momentum
    self.training = True
    # 可训练的参数：gamma和beta
    self.gamma = torch.ones(dim)
    self.beta = torch.zeros(dim)
    # 存储训练过程中的均值和方差
    self.running_mean = torch.zeros(dim)
    self.running_var = torch.ones(dim)
  
  def __call__(self, x):
    # 计算前向传播
    if self.training:
      # 计算当前批次的均值和方差
      if x.ndim == 2:
        dim = 0
      elif x.ndim == 3:
        dim = (0,1)
      xmean = x.mean(dim, keepdim=True) # 批均值
      xvar = x.var(dim, keepdim=True) # 批方差
    else:
      # 使用移动平均和方差
      xmean = self.running_mean
      xvar = self.running_var
    
    # 标准化输入
    xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
    self.out = self.gamma * xhat + self.beta
    
    # 如果在训练模式，更新移动平均和方差
    if self.training:
      with torch.no_grad():
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
    return self.out
  
  def parameters(self):
    # 返回gamma和beta
    return [self.gamma, self.beta]

# -----------------------------------------------------------------------------------------------
# 自定义的Tanh激活函数层
class Tanh:
  def __call__(self, x):
    self.out = torch.tanh(x) # 使用PyTorch内置的tanh函数
    return self.out
  def parameters(self):
    return [] # Tanh层没有可训练的参数

# -----------------------------------------------------------------------------------------------
# 嵌入层，将索引映射到高维向量空间
class Embedding:
  
  def __init__(self, num_embeddings, embedding_dim):
    self.weight = torch.randn((num_embeddings, embedding_dim)) # 随机初始化嵌入矩阵
    
  def __call__(self, IX):
    self.out = self.weight[IX] # 根据索引返回对应的嵌入向量
    return self.out
  
  def parameters(self):
    return [self.weight] # 返回嵌入矩阵

# -----------------------------------------------------------------------------------------------
# FlattenConsecutive层，指定想输入连续数据多少部分
class FlattenConsecutive:
  
  def __init__(self, n):
    self.n = n
    
  def __call__(self, x):
    B, T, C = x.shape
    x = x.view(B, T//self.n, C*self.n) # 展平操作
    if x.shape[1] == 1:
      x = x.squeeze(1)     #中间的维度是1的时候，将中间的维度压缩掉
    self.out = x
    return self.out
  
  def parameters(self):
    return [] # 展平层没有可训练的参数

# -----------------------------------------------------------------------------------------------
# Sequential类，串联多个层，按顺序执行
class Sequential:
  
  def __init__(self, layers):
    self.layers = layers
  
  def __call__(self, x):
    # 顺序执行每一层
    for layer in self.layers:
      x = layer(x)
    self.out = x
    return self.out
  
  def parameters(self):
    # 获取所有层的参数并展平为一个列表
    return [p for layer in self.layers for p in layer.parameters()]

# 设置随机种子，保证结果可复现
torch.manual_seed(42)

# 层次化网络定义
n_embd = 24 # 字符嵌入向量的维度
n_hidden = 128 # 隐藏层的神经元数量
model = Sequential([
  Embedding(vocab_size, n_embd), # 嵌入层，将字符映射为向量
  FlattenConsecutive(2), Linear(n_embd * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
  FlattenConsecutive(2), Linear(n_hidden*2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
  FlattenConsecutive(2), Linear(n_hidden*2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
  Linear(n_hidden, vocab_size),
])

# 参数初始化，将最后一层的权重初始化为较小值，降低预测置信度
with torch.no_grad():
  model.layers[-1].weight *= 0.1

parameters = model.parameters()
print(sum(p.nelement() for p in parameters)) # 打印总参数数量
for p in parameters:
  p.requires_grad = True # 所有参数开启反向传播

# 优化器设置，简单的随机梯度下降
max_steps = 200000
batch_size = 32
lossi = []

for i in range(max_steps):
  
  # 构造小批量数据
  ix = torch.randint(0, Xtr.shape[0], (batch_size,))
  Xb, Yb = Xtr[ix], Ytr[ix]
  
  # 前向传播
  logits = model(Xb)
  loss = F.cross_entropy(logits, Yb) # 计算交叉熵损失
  
  # 反向传播
  for p in parameters:
    p.grad = None # 清空之前的梯度
  loss.backward()
  
  # 参数更新
  lr = 0.1 if i < 100000 else 0.01 # 学习率计划，前100k步使用较大学习率
  for p in parameters:
    p.data -= lr * p.grad
  
  # 日志记录和展示
  if i % 10000 == 0:
    print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
  lossi.append(loss.log10().item())

# -----------------------------------------------------------------------------------------------
# 切换为评估模式，即不再更新批归一化的统计量（均值和方差）
for layer in model.layers:
    if isinstance(layer, BatchNorm1d):
        layer.training = False

# 在训练集上采样，生成一些新名字
for _ in range(20):
    out = []
    context = [0] * block_size  # 初始化上下文为句点开始的全0向量
    while True:
        logits = model(torch.tensor([context]))  # 通过模型进行前向传播
        probs = F.softmax(logits, dim=1)  # 将模型的输出logits转为概率分布
        ix = torch.multinomial(probs, num_samples=1).item()  # 根据概率分布采样下一个字符的索引
        context = context[1:] + [ix]  # 更新上下文窗口
        out.append(ix)  # 将采样到的索引加入结果中
        if ix == 0:  # 如果遇到句点，表示名字结束
            break
    print(''.join(itos[i] for i in out))  # 输出生成的名字，将索引转为字符

# -----------------------------------------------------------------------------------------------
# 在验证集上计算损失，用于评估模型的泛化性能
@torch.no_grad()  # 禁用梯度计算，以节省内存
def eval_loss(split):
    X, Y = (Xtr, Ytr) if split == 'train' else (Xdev, Ydev)  # 根据split选择训练集或验证集
    logits = model(X)  # 前向传播
    loss = F.cross_entropy(logits, Y)  # 计算交叉熵损失
    print(f'{split} loss: {loss.item()}')  # 打印损失
    return loss.item()

# 训练结束后评估训练集和验证集的损失
eval_loss('train')
eval_loss('dev')

```

## GPT（Pre-train)

### Attention

- Attention is a **communication mechanism**. Can be seen as nodes in a directed graph looking at each other and aggregating information with a weighted sum from all nodes that point to them, with data-dependent weights.
- There is no notion of space. Attention simply acts over a set of vectors. This is why we need to positionally encode tokens.
- Each example across batch dimension is of course processed completely independently and never "talk" to each other
- In an "encoder" attention block just delete the single line that does masking with `tril`, allowing all tokens to communicate. This block here is called a "decoder" attention block because it has triangular masking, and is usually used in autoregressive settings, like language modeling.
- **"self-attention"** just means that the keys and values are produced from the same source as queries. In "cross-attention", the queries still get produced from x, but the keys and values come from some other, external source (e.g. an encoder module)
- "Scaled" attention additional **divides `wei` by 1/sqrt(head_size)**. This makes it so when input Q,K are unit variance, wei will be unit variance too and Softmax will stay diffuse and not saturate too much.

```py
# toy example illustrating how matrix multiplication can be used for a "weighted aggregation"
torch.manual_seed(42)
a = torch.tril(torch.ones(3, 3))
a = a / torch.sum(a, 1, keepdim=True)
b = torch.randint(0,10,(3,2)).float()
c = a @ b    #矩阵乘法可以用来进行加权聚合
print('a=')
print(a)
print('--')
print('b=')
print(b)
print('--')
print('c=')
print(c)

a=
tensor([[1.0000, 0.0000, 0.0000],
        [0.5000, 0.5000, 0.0000],
        [0.3333, 0.3333, 0.3333]])
--
b=
tensor([[2., 7.],
        [6., 4.],
        [6., 5.]])
--
c=
tensor([[2.0000, 7.0000],
        [4.0000, 5.5000],
        [4.6667, 5.3333]])  
```

```py
# consider the following toy example:

torch.manual_seed(1337)
B,T,C = 4,8,2 # batch, time, channels
x = torch.randn(B,T,C)

# We want x[b,t] = mean_{i<=t} x[b,i]
xbow = torch.zeros((B,T,C))
for b in range(B):
    for t in range(T):
        xprev = x[b,:t+1] # (t,C)
        xbow[b,t] = torch.mean(xprev, 0)
# version 2: using matrix multiply for a weighted aggregation
wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(1, keepdim=True)
xbow2 = wei @ x # (B, T, T) @ (B, T, C) ----> (B, T, C)

# version 3: use Softmax
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T,T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)
xbow3 = wei @ x

# version 4: self-attention!
torch.manual_seed(1337)
B,T,C = 4,8,32 # batch, time, channels
x = torch.randn(B,T,C)

# let's see a single Head perform self-attention
head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
k = key(x)   # (B, T, 16)
q = query(x) # (B, T, 16)
wei =  q @ k.transpose(-2, -1) # (B, T, 16) @ (B, 16, T) ---> (B, T, T)

tril = torch.tril(torch.ones(T, T))
#wei = torch.zeros((T,T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)

v = value(x)
out = wei @ v
#out = wei @ x
```

### GPT

```py
import torch
import torch.nn as nn
from torch.nn import functional as F

# 超参数设置
batch_size = 64  # 并行处理的独立序列数
block_size = 256  # 预测的最大上下文长度
max_iters = 5000  # 最大训练迭代次数
eval_interval = 500  # 评估间隔
learning_rate = 3e-4  # 学习率
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 使用GPU或CPU
eval_iters = 200  # 评估迭代次数
n_embd = 384  # 嵌入维度
n_head = 6  # 注意力头的数量
n_layer = 6  # Transformer层的数量
dropout = 0.2  # dropout概率

torch.manual_seed(1337)  # 设置随机种子以保证结果可复现

# 读取文本数据
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 获取文本中的所有唯一字符
chars = sorted(list(set(text)))
vocab_size = len(chars)  # 词汇表大小
# 创建字符到整数的映射
stoi = {ch: i for i, ch in enumerate(chars)}  # 字符到索引的映射
itos = {i: ch for i, ch in enumerate(chars)}  # 索引到字符的映射
encode = lambda s: [stoi[c] for c in s]  # 编码函数：字符串转整数列表
decode = lambda l: ''.join([itos[i] for i in l])  # 解码函数：整数列表转字符串

# 训练和测试数据划分
data = torch.tensor(encode(text), dtype=torch.long)  # 将编码后的数据转换为Tensor
n = int(0.9 * len(data))  # 前90%作为训练集，后10%作为验证集
train_data = data[:n]
val_data = data[n:]

# 数据加载函数
def get_batch(split):
    # 生成一小批次的输入x和目标y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))  # 随机选择起始索引  减去blocksize防止越界
    x = torch.stack([data[i:i + block_size] for i in ix])  # 生成输入序列
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])  # 生成目标序列
    x, y = x.to(device), y.to(device)  # 将数据移动到指定设备
    return x, y

@torch.no_grad()  # 禁用梯度计算
def estimate_loss():
    out = {}
    model.eval()  # 切换到评估模式
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)  # 初始化损失数组
        for k in range(eval_iters):
            X, Y = get_batch(split)  # 获取批次数据
            logits, loss = model(X, Y)  # 前向传播
            losses[k] = loss.item()  # 记录损失
        out[split] = losses.mean()  # 计算平均损失
    model.train()  # 切换回训练模式
    return out

class Head(nn.Module):
    """ 单头自注意力 """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)  # 线性变换生成键
        self.query = nn.Linear(n_embd, head_size, bias=False)  # 线性变换生成查询
        self.value = nn.Linear(n_embd, head_size, bias=False)  # 线性变换生成值
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))  # 下三角矩阵用于遮蔽

        self.dropout = nn.Dropout(dropout)  # dropout层

    def forward(self, x):
        # 输入形状为 (batch, time-step, channels)
        B, T, C = x.shape
        k = self.key(x)   # (B, T, hs)
        q = self.query(x) # (B, T, hs)
        # 计算注意力分数（"亲和度"）
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # 应用遮蔽
        wei = F.softmax(wei, dim=-1)  # 归一化
        wei = self.dropout(wei)  # 应用dropout
        # 进行加权聚合
        v = self.value(x)  # (B, T, hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ 多头自注意力 """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])  # 创建多个头
        self.proj = nn.Linear(head_size * num_heads, n_embd)  # 学习到不同权重组合而不是简单拼接
        self.dropout = nn.Dropout(dropout)  # dropout层

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # 合并所有头的输出
        out = self.dropout(self.proj(out))  # 线性变换和dropout
        return out

class FeedFoward(nn.Module):
    """ 简单的线性层后跟非线性激活 """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(         
            nn.Linear(n_embd, 4 * n_embd),  # 扩展维度 在小的维度上，激活函数可能无法发挥其潜力。扩展维度后，再进行非线性激活，能够有效地使得更多的神经元激活，产生丰富的特征表示 
            nn.ReLU(),  # 非线性激活
            nn.Linear(4 * n_embd, n_embd),  # 收缩回原维度
            nn.Dropout(dropout),  # dropout层
        )

    def forward(self, x):
        return self.net(x)  # 前向传播

class Block(nn.Module):
    """ Transformer块：通信后计算 """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head  # 每个头的维度
        self.sa = MultiHeadAttention(n_head, head_size)  # 自注意力层
        self.ffwd = FeedFoward(n_embd)  # 前馈层
        self.ln1 = nn.LayerNorm(n_embd)  # 第一个层归一化
        self.ln2 = nn.LayerNorm(n_embd)  # 第二个层归一化

    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # 残差连接和自注意力
        x = x + self.ffwd(self.ln2(x))  # 残差连接和前馈层
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # 每个token直接从查找表中读取下一个token的logits
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)  # 词嵌入层
        self.position_embedding_table = nn.Embedding(block_size, n_embd)  # 位置嵌入层
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])  # 多个Transformer块 *用于解包block 将每个block以独立的参数传入sequential，按顺序组合形成整体网络结构
        self.ln_f = nn.LayerNorm(n_embd)  # 最后的层归一化
        self.lm_head = nn.Linear(n_embd, vocab_size)  # 输出层

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)  # 使用正态分布初始化权重
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)  # 偏置初始化为零
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)  # 嵌入权重初始化

    def forward(self, idx, targets=None):
      B, T = idx.shape  # 获取批量大小B和时间步长T

      # idx和targets都是形状为(B,T)的整数张量
      tok_emb = self.token_embedding_table(idx)  # 将输入索引转换为token嵌入，形状为(B,T,C)
      pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # 获取位置嵌入，形状为(T,C)
      x = tok_emb + pos_emb  # 将token嵌入和位置嵌入相加，形状为(B,T,C)
      x = self.blocks(x)  # 通过多个Transformer块处理嵌入，形状保持为(B,T,C)
      x = self.ln_f(x)  # 应用最终的LayerNorm，形状依旧为(B,T,C)
      logits = self.lm_head(x)  # 通过线性层得到每个时间步的logits，形状为(B,T,vocab_size)

      if targets is None:
          loss = None  # 如果没有提供targets，则损失为None
      else:
          B, T, C = logits.shape  # 获取logits的形状
          logits = logits.view(B*T, C)  # 将logits重塑为(B*T, C)以便计算损失
          targets = targets.view(B*T)  # 将targets重塑为(B*T)
          loss = F.cross_entropy(logits, targets)  # 计算交叉熵损失

      return logits, loss  # 返回logits和损失

    def generate(self, idx, max_new_tokens):
        # idx是当前上下文的(B, T)索引数组
        for _ in range(max_new_tokens):
            # 裁剪idx到最后的block_size个token
            idx_cond = idx[:, -block_size:]
            # 获取预测结果
            logits, loss = self(idx_cond)
            # 只关注最后一个时间步的logits
            logits = logits[:, -1, :]  # 变为(B, C)
            # 应用softmax获取概率分布
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # 从概率分布中采样下一个token的索引
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # 将采样的索引添加到当前序列中
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx  # 返回生成的新索引序列

model = GPTLanguageModel()  # 实例化模型
m = model.to(device)  # 将模型转移到计算设备上
# 打印模型参数数量
print(sum(p.numel() for p in m.parameters()) / 1e6, 'M parameters')

# 创建PyTorch优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # 每隔一段时间评估训练集和验证集的损失
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()  # 评估损失
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # 采样一批数据
    xb, yb = get_batch('train')

    # 评估损失
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)  # 清除梯度
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数

# 从模型生成文本
context = torch.zeros((1, 1), dtype=torch.long, device=device)  # 初始化上下文为零
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))  # 生成500个新token并解码为字符串
# open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))  # 生成10000个token并保存到文件

```

