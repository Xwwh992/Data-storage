## Pytorch 

### 训练流程

1. 下载数据 `Download=True`

2. Dataloader

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
  
  
  
  



