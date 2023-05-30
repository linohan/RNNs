import numpy as np

# Sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# RNN层
class RNNLayer:
    def __init__(self, input_shape, hidden_units, output_units):
        self.input_shape = input_shape
        self.hidden_units = hidden_units
        self.output_units = output_units
        
        # 初始化权重和偏置
        self.Whh = np.random.randn(hidden_units, hidden_units) * 0.01
        self.Wxh = np.random.randn(input_shape, hidden_units) * 0.01
        self.Why = np.random.randn(hidden_units, output_units) * 0.01
        self.bh = np.zeros((1, hidden_units))
        self.by = np.zeros((1, output_units))
    
    def forward(self, input_data, h_prev):
        self.h_prev = h_prev
        self.x = input_data
        
        # 计算隐藏状态
        self.a = np.dot(self.x, self.Wxh) + np.dot(self.h_prev, self.Whh) + self.bh
        self.h = np.tanh(self.a)
        
        # 计算输出
        self.o = np.dot(self.h, self.Why) + self.by
        self.y = sigmoid(self.o)
        
        return self.y, self.h
    
    def backward(self, output_error, h_next, learning_rate):
        # 计算损失关于输出的梯度
        dL_do = output_error * (self.y * (1 - self.y))
        
        # 计算损失关于隐藏状态的梯度
        dL_dh = np.dot(dL_do, self.Why.T) + np.dot(h_next, self.Whh.T)
        dL_da = dL_dh * (1 - self.h ** 2)
        
        # 计算损失关于权重的梯度
        dL_dWxh = np.dot(self.x.T, dL_da)
        dL_dWhh = np.dot(self.h_prev.T, dL_da)
        dL_dWhy = np.dot(self.h.T, dL_do)
        
        # 计算损失关于偏置的梯度
        dL_dbh = np.sum(dL_da, axis=0, keepdims=True)
        dL_dby = np.sum(dL_do, axis=0, keepdims=True)
        
        # 更新权重和偏置
        self.Whh -= learning_rate * dL_dWhh
        self.Wxh -= learning_rate * dL_dWxh
        self.Why -= learning_rate * dL_dWhy
        self.bh -= learning_rate * dL_dbh
        self.by -= learning_rate * dL_dby
        
        return dL_dh
    
# 测试代码
if __name__ == '__main__':
    # 定义输入数据和标签
    x = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
    y = np.array([[1], [0], [1], [0]])
    
    # 定义超参数
    hidden_units = 6
    output_units = 1
    learning_rate = 0.1
    
    # 初始化RNN
    rnn = RNNLayer(input_shape=x.shape[1], hidden_units=hidden_units, output_units=output_units)
    
    # 迭代训练
    for epoch in range(10000):
        h_prev = np.zeros((1, hidden_units))
        loss = 0
        
        for i in range(x.shape[0]):
            # 前向传播
            y_pred, h_prev = rnn.forward(x[i:i+1], h_prev)
            
            # 计算损失
            loss += np.mean(-(y[i:i+1] * np.log(y_pred) + (1 - y[i:i+1]) * np.log(1 - y_pred)))
            
            # 反向传播
            output_error = y_pred - y[i:i+1]
            h_prev = rnn.backward(output_error, h_prev, learning_rate)
        
        if epoch % 1000 == 0:
            print('Epoch:', epoch, 'Loss:', loss)