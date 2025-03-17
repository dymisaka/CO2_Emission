import torch
import joblib
import pandas as pd
import numpy as np
from torch import nn

class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, d_model=32, nhead=4, num_layers=2, dim_feedforward=128):
        super().__init__()
        
        self.embedding = nn.Linear(input_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, 1)
        )
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]
        x = self.regression_head(x)
        return x.squeeze()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型
model = TransformerRegressor(input_dim=7)  # 必须与原始特征数量一致
model.load_state_dict(torch.load('./TransformerModel.pth', map_location=device))
model.to(device)
model.eval()  # 切换到评估模式

# 加载标准化器
feature_scaler = joblib.load('transformer_feature_scaler.pkl')  # 特征标准化器
target_scaler = joblib.load('transformer_target_scaler.pkl')      # 目标标准化器

# 3. 准备新数据示例
new_data = pd.DataFrame([{
    'Time spent at sea [hours]': 1500,
    'avg_speed': 14.2,
    'length': 180,
    'breadth': 28,
    'gross_tonnage': 12000,
    'deadweight': 20000,
    'type no': 2  # 注意保留所有训练时使用的特征
}])

# 4. 数据预处理
# 确保特征顺序与训练时一致
expected_features = [
    'Time spent at sea [hours]',
    'avg_speed',
    'length',
    'breadth',
    'gross_tonnage',
    'deadweight',
    'type no'
]
X_new = new_data[expected_features] 

# 应用特征标准化
X_new_scaled = feature_scaler.transform(X_new) 

# 转换为Tensor
inputs = torch.FloatTensor(X_new_scaled).unsqueeze(1).to(device)  # 添加序列维度

# 5. 进行预测
with torch.no_grad():
    prediction_scaled = model(inputs).cpu().numpy()

# 6. 反标准化预测结果
prediction = target_scaler.inverse_transform(prediction_scaled.reshape(-1, 1))

print(f"预测的CO₂排放量为: {prediction[0][0]:.2f} 公吨")