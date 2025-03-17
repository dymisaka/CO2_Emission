import pandas as pd
import joblib

# 加载保存的模型和标准化器
scaler = joblib.load('mlp_scaler.pkl')
model = joblib.load('mlp_model.pkl')

# 加载新数据（示例数据，需替换为实际数据）
new_data = pd.DataFrame([{
    'Time spent at sea [hours]': 120,
    'avg_speed': 15.5,
    'length': 200,
    'breadth': 30,
    'gross_tonnage': 15000,
    'deadweight': 25000
}])

# 确保特征顺序与训练时一致
features = ['Time spent at sea [hours]', 'avg_speed', 'length', 
           'breadth', 'gross_tonnage', 'deadweight']
new_features = new_data[features]

# 数据预处理
X_new_scaled = scaler.transform(new_features)

# 进行预测
prediction = model.predict(X_new_scaled)

print(f"预测的CO2排放量为: {prediction[0]:.2f} 公吨")