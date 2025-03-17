import pickle
import pandas as pd
import torch
import torch.nn as nn
import joblib

# Transformer 模型
class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, dropout=0.1):
        super(TransformerModel, self).__init__()

        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = x.squeeze(1)
        x = self.fc(x)
        return x


# 全局变量
rf_model = None
my_Transformer_model = None
scaler_X = None
scaler_y = None


def load_two_model():
    global rf_model, my_Transformer_model, scaler_X, scaler_y
    if rf_model is None:
        with open('./rf_model.pkl', 'rb') as f:
            rf_model = pickle.load(f)
    
    if scaler_X is None or scaler_y is None:
        scaler_X = joblib.load('best_scaler_X.pkl')
        scaler_y = joblib.load('best_scaler_y.pkl')
    
    if my_Transformer_model is None:
        my_Transformer_model = TransformerModel(input_dim=7, hidden_dim=64, num_layers=2, num_heads=4)
        my_Transformer_model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))
        my_Transformer_model.eval()

def get_two_model_result(input_data):
    # 确保输入是DataFrame格式
    if not isinstance(input_data, pd.DataFrame):
        input_data = pd.DataFrame(input_data)
    # if isinstance(input_data, np.ndarray):
    #     input_data = pd.DataFrame(input_data, columns=scaler_X.feature_names_in_)
    # elif isinstance(input_data, pd.DataFrame):
    #     input_data = input_data[scaler_X.feature_names_in_]  # 只保留需要的特征列


    
    # 特征工程
    step1_features = input_data[['length', 'breadth', 'gross_tonnage', 'deadweight', 'ship type']].to_numpy()
    input_data['engine_power'] = rf_model.predict(step1_features)
    
    # 准备最终特征
    step2_features = input_data[['Time spent at sea [hours]', 'avg_speed', 'length', 'breadth', 'gross_tonnage', 'deadweight', 'engine_power']].to_numpy()
    
    # 标准化并预测
    input_scaled = scaler_X.transform(step2_features)
    input_tensor = torch.FloatTensor(input_scaled)
    
    with torch.no_grad():
        predictions = my_Transformer_model(input_tensor)
    
    return float(scaler_y.inverse_transform(predictions.numpy().reshape(-1, 1))[0][0])

# def get_user_input():
#     """ 获取用户输入数据 """
#     input_keys = ['ship type', 'length', 'breadth', 'deadweight', 'gross tonnage', 'speed', 'time']
#     user_data = {}

#     for key in input_keys:
#         while True:
#             try:
#                 user_data[key] = float(input(f"请输入 {key}: "))
#                 break
#             except ValueError:
#                 print("输入无效，请输入一个数字！")

#     return user_data


# if __name__ == "__main__":
#     # user_input = get_user_input()
#     user_input = pd.DataFrame({
#         'time': [2031.43],
#         'speed': [18.2],
#         # 'max_speed':[16.0],
#         'length': [183],
#         'breadth': [26.0],
#         'gross tonnage': [22528.0],
#         'deadweight': [3335.0],
#         # 'engine power': [32484.72],
#         'ship type': [1]
#     })
#     co2_result = get_two_model_result(user_input)
#     print(f"预测的 CO2 排放量: {co2_result:.2f} 吨")
