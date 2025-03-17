# import pandas as pd
import numpy as np
import joblib

# 全局变量，用于存储加载的 MLP 模型
mlp_model = None
mlp_scaler = None

def load_mlp_model():
    global mlp_model
    global mlp_scaler
    mlp_model = joblib.load('mlp_model.pkl')
    mlp_scaler = joblib.load('mlp_scaler.pkl')

def get_mlp_predict_result(data):
    global mlp_model
    global mlp_scaler
    if(mlp_model==None):
        print("mlp_model is None. Loading MLP model...")
        mlp_model = joblib.load('mlp_model.pkl')
        print("MLP model loaded successfully.")
    if(mlp_scaler==None):
        print("mlp_scaler is None. Loading MLP scaler...")
        mlp_scaler = joblib.load('mlp_scaler.pkl')
        print("MLP scaler loaded successfully.")

    # 确保输入是numpy数组并标准化
    X_new = np.array(data, dtype=np.float32)
    X_new_scaled = mlp_scaler.transform(X_new)
    return mlp_model.predict(X_new_scaled)

# if __name__=='__main__':
#     data=[[120,15.5,200,30,15000,25000]]
# #     new_data = pd.DataFrame([{
# #     'Time spent at sea [hours]': 120,
# #     'avg_speed': 15.5,
# #     'length': 200,
# #     'breadth': 30,
# #     'gross_tonnage': 15000,
# #     'deadweight': 25000
# # }])
#     load_mlp_model()
#     pre=get_mlp_predict_result(data)
#     print(f"MLP result: {pre}")
