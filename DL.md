- **K折交叉验证**
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

  

- **`isinstance()`**用来判断实例对象是否属于某个类。对象不属于类时候可以用于判定然后做类的转换再使用
- 建立一个**神经网络的数据类型并实现反向传播**

```python
class Value:
  
  def __init__(self, data, _children=(), _op='', label=''):
    self.data = data
    self.grad = 0.0
    self._backward = lambda: None
    self._prev = set(_children)
    self._op = _op
    self.label = label

  def __repr__(self):
    return f"Value(data={self.data})"
  
  def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), '+')
    
    def _backward():
      self.grad += 1.0 * out.grad
      other.grad += 1.0 * out.grad
    out._backward = _backward
    
    return out

  def __mul__(self, other):
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
      self.grad += (1 - t**2) * out.grad
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
    
    self.grad = 1.0
    for node in reversed(topo):
      node._backward()
```
