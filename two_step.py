import pickle
import pandas as pd
import numpy as np

# step 1
# test_data=np.array([[100,20,10000,100000]])

# 加载模型
with open('./rf_model.pkl', 'rb') as f:
    loaded_rf_model = pickle.load(f)
# predicted_value = loaded_rf_model.predict(test_data)
# print("预测值：", predicted_value)

#step2
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# Transformer 模型
class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim*4,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # Add sequence dimension
        x = x.unsqueeze(1)  
        
        x = self.embedding(x)
        
        # Transformer layers
        x = self.transformer_encoder(x)
        x = x.squeeze(1)
        x = self.fc(x)
        return x

# 创建预测函数
def predict_co2(input_data, model, scaler_X, scaler_y):

    model.eval()
    
    # 转换输入数据格式
    if isinstance(input_data, pd.DataFrame):
        input_data = input_data.values
    
    input_scaled = scaler_X.transform(input_data)
    input_tensor = torch.FloatTensor(input_scaled)
    
    # 预测
    with torch.no_grad():
        predictions = model(input_tensor)
        
    # 转换回原始尺度
    predictions = scaler_y.inverse_transform(predictions.numpy().reshape(-1, 1))
    
    return predictions

input_dim = 7  # 特征数量
# input_dim=X_train.shape[1]
model = TransformerModel(
input_dim=input_dim,
hidden_dim=64,
num_layers=2,
num_heads=4
)
model.load_state_dict(torch.load('best_model.pth'))
    # 示例数据
#     features = df[['Time spent at sea [hours]',
#                'avg_speed',
#                'max_speed',
#                'length',
#                'breadth',
#                'gross_tonnage',
#                'deadweight',
#               'engine_power']]
test_data = pd.DataFrame({
    'Time spent at sea [hours]': [2031.43],
    'avg_speed': [18.2],
    # 'max_speed':[16.0],
    'length': [183],
    'breadth': [26.0],
    'gross_tonnage': [22528.0],
    'deadweight': [3335.0],
    'engine_power': [32484.72],
})
import joblib
scaler_X = joblib.load('best_scaler_X.pkl')
scaler_y = joblib.load('best_scaler_y.pkl')
predictions = predict_co2(test_data, model, scaler_X, scaler_y)
print(f"预测的CO2排放量: {predictions[0][0]:.2f} 吨")

if __name__=="__main__":
    # test_data = pd.DataFrame({
    
    # 'Time spent at sea [hours]': [2031.43],
    # 'avg_speed': [18.2],
    # # 'max_speed':[16.0],
    # 'length': [183],
    # 'breadth': [26.0],
    # 'gross_tonnage': [22528.0],
    # 'deadweight': [3335.0],
    # 'engine_power': [32484.72],
    # })  
    input_data={
        'ship type':0,
        # 'Time spent at sea [hours]':0,
        'length':0,
        'breadth':0,
        'deadweight':0,
        'gross tonnage': 0,
        'speed':0,
        'time': 0
        }
    for key in input_data:
        while True:  # 使用循环确保用户输入有效的数字
            try:
                value = float(input(f"请输入 {key}: "))  # 输入转换为浮点数
                input_data[key] = value  # 更新字典中的值
                break  # 输入正确时退出循环
            except ValueError:
                print("输入无效，请输入一个数字！")

    print("输入的数据为：")
    for key in input_data:
        print(f'{key}: {input_data[key]}')

    step1_data=[[
        input_data['length'],
        input_data['breadth'],
        input_data['gross tonnage'],
        input_data['deadweight'],
        input_data['ship type']
        ]]
    print(step1_data)
    predicted_value = loaded_rf_model.predict(step1_data)
    print("engine power 预测值：", predicted_value)

    step2_data=[[
        input_data['time'],
        input_data['speed'],
        input_data['length'],
        input_data['breadth'],
        input_data['gross tonnage'],
        input_data['deadweight'],
        predicted_value
    ]]
    scaler_X = joblib.load('best_scaler_X.pkl')
    scaler_y = joblib.load('best_scaler_y.pkl')
    predictions = predict_co2(step2_data, model, scaler_X, scaler_y)
    print(f"预测的CO2排放量: {predictions[0][0]:.2f} 吨")