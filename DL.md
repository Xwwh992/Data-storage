# DL

- **K折交叉验证**
- **`isinstance()`**用来判断实例对象是否属于某个类。对象不属于类时候可以用于判定然后做类的转换再使用

- 卷积核在卷积操作中的作用确实可以理解为衡量它与对应图像区块之间的**相似程度**

- 重点关注**减少泛化误差**，过度训练会导致与训练集很相似，但是没办法应对测试集，最好是在最优处减少泛化与训练误差的差距

- **LayerNorm** 的归一化是针对单个样本进行的，不依赖于批次大小

- **BatchNorm**：主要用于 CNN 等模型，适合较大批次的训练

- **LayerNorm**：主要用于 RNN、Transformer 等模型，适合小批次或变长序列的训练

- **matplotlib.pyplot**

- 只定义了**`__repr__`**方法而没有定义**`__str__`**方法，那么当调用`str()`或`print()`时，Python会自动使用**`__repr__`**方法的返回值

- **`__repr__`**应该用于返回更详细的开发者视角的信息，而**`__str__`**则用于简洁的用户视角的信息

- **Doing the back propagation maunally is obviously ridiculous** 

- 反向传播：用local gradient乘上传递过来的gradient，单纯的加法只会让梯度 **flow over**不改变数值

- **删除偶数下表**标元素(C++20)

```c++
  #include <iostream>
  #include <vector>
  #include <algorithm>  // for std::erase_if
  int main() {
      std::vector<int> v = {1, 2, 3, 4, 5};
      // 使用 std::erase_if 进行删除操作
      std::erase_if(v, [&v](int num) {
          return (&num - &v[0] + 1) % 2 == 0;
      });
      // 输出剩下的元素
      for (auto i : v) {
          std::cout << i << ",";
      }
      return 0;
  }
```

---

## 反向传播

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
      self.grad += out.data * out.grad # NOTE: in the video I incorrectly used = instead of +=. Fixed here.
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

## 生成MLP(Perceptron)

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

- **gradient可以看成指向loss增加的向量**
## 梯度下降

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

## 语言模型(bigram)

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

