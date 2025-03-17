import torch
import joblib
from torch import nn
import numpy as np

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

Transformer_device=None
Transformer_model=None
Transformer_feature_scaler=None
Transformer_target_scaler=None


def load_Transformer_model():
    global Transformer_device, Transformer_model, Transformer_feature_scaler, Transformer_target_scaler
    Transformer_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Transformer_model = TransformerRegressor(input_dim=7)
    Transformer_model.load_state_dict(torch.load('./TransformerModel.pth', map_location=Transformer_device))
    Transformer_model.to(Transformer_device)
    Transformer_model.eval()
    
    Transformer_feature_scaler = joblib.load('transformer_feature_scaler.pkl')
    Transformer_target_scaler = joblib.load('transformer_target_scaler.pkl')

def get_transformer_predict_result(input_data):
    global Transformer_feature_scaler, Transformer_target_scaler
    
    # 数据预处理
    data_np = np.array(input_data, dtype=np.float32)
    data_scaled = Transformer_feature_scaler.transform(data_np)
    
    # 转换为Tensor
    data_tensor = torch.FloatTensor(data_scaled).unsqueeze(1).to(Transformer_device)
    
    with torch.no_grad():
        prediction = Transformer_model(data_tensor).cpu().numpy()
    
    return Transformer_target_scaler.inverse_transform(prediction.reshape(-1, 1))

# # 3. 准备新数据示例
# new_data = pd.DataFrame([{
#     'Time spent at sea [hours]': 1500,
#     'avg_speed': 14.2,
#     'length': 180,
#     'breadth': 28,
#     'gross_tonnage': 12000,
#     'deadweight': 20000,
#     'type no': 2  # 注意保留所有训练时使用的特征
# }])

# # 4. 数据预处理
# # 确保特征顺序与训练时一致
# expected_features = [
#     'Time spent at sea [hours]',
#     'avg_speed',
#     'length',
#     'breadth',
#     'gross_tonnage',
#     'deadweight',
#     'type no'
# ]
# if __name__=='__main__':
#     data=[[1500,14.2,180,28,12000,20000,2]]
#     load_Transformer_model()
#     pre=get_transformer_predict_result(data)
#     print(f'Transformer result: {pre}')