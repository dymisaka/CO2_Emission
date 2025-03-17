from flask import Flask, request, jsonify, render_template
import pandas as pd
import mlp_model
import Transformer_model
import two_model
from finddata import find_nearest

app = Flask(__name__)

# # global mlp_model
mlp_model.load_mlp_model()
Transformer_model.load_Transformer_model()
two_model.load_two_model()

# 应用启动时加载所有模型
# @app.before_first_request
# def load_models():
#     print("Loading MLP model...")
#     mlp_model.load_mlp_model()
#     print("Loading Transformer model...")
#     Transformer_model.load_Transformer_model()
#     print("Loading Two-stage model...")
#     two_model.load_two_model()
#     print("All models loaded successfully")

@app.route('/')
def index():
    return render_template('index.html')  # 确保 HTML 文件在 "templates" 目录下


def format_nearest_data(row):
    return {
        "type": row.get('type no', 'N/A'),
        "length": round(row.get('length', 0), 1),
        "breadth": round(row.get('breadth', 0), 1),
        "deadweight": round(row.get('deadweight', 0), 1),
        "gross_tonnage": round(row.get('gross_tonnage', 0), 1),
        "speed": round(row.get('avg_speed', 0), 1),  # 将 avg_speed 格式化后返回为 speed
        "time": round(row.get('Time spent at sea [hours]',0), 2),  # 格式化时间
        "co2": round(row.get('Total CO₂ emissions [m tonnes]', 0), 2)  # 实际 CO₂
    }



@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 获取前端输入的数据
        # 获取并转换数据
        data = {
            'ship type': request.form.get('type', '0'),
            'length': float(request.form.get('length', 0)),
            'breadth': float(request.form.get('breadth', 0)),
            'deadweight': float(request.form.get('deadweight', 0)),
            'gross tonnage': float(request.form.get('gross_tonnage', 0)),
            'speed': float(request.form.get('speed', 0)),
            'time': float(request.form.get('time', 0))
        }
        print(f'input data: {data}')
        mlp_input = [[
            data['time'],
            data['speed'],
            data['length'],
            data['breadth'],
            data['gross tonnage'],
            data['deadweight'],
            int(data['ship type'])
        ]]
        transformer_input = [[
            data['time'],
            data['speed'],
            data['length'],
            data['breadth'],
            data['gross tonnage'],
            data['deadweight'],
            int(data['ship type'])
        ]]
        two_model_input = pd.DataFrame([{
            'Time spent at sea [hours]': data['time'],
            'avg_speed': data['speed'],
            'length': data['length'],
            'breadth': data['breadth'],
            'gross_tonnage': data['gross tonnage'],
            'deadweight': data['deadweight'],
            'ship type': int(data['ship type'])
        }])
        mlp_result = mlp_model.get_mlp_predict_result(mlp_input)[0]
        print(f'mlp result: {mlp_result}')
        transformer_result = Transformer_model.get_transformer_predict_result(transformer_input)[0][0]
        print(f'transformer result: {transformer_result}')
        two_steps_result = two_model.get_two_model_result(two_model_input)
        print(f'two step result: {two_steps_result}')

        # 调用 find_nearest 获取最近数据（返回一个列表，每个元素为一个数据点的字典）
        nearest_data = find_nearest({
            'length': data['length'],
            'breadth': data['breadth'],
            'gross_tonnage': data['gross tonnage'],
            'deadweight': data['deadweight'],
            'avg_speed': data['speed'],
            'Time spent at sea [hours]': data['time'],
            'ship type': data['ship type']
        })

        # 对每个返回的数据点调用 format_nearest_data 进行格式化
        formatted_nearest = [format_nearest_data(row) for row in nearest_data]

        return jsonify({
            "predictions": {
                "MLP": float(mlp_result),
                "Transformer": float(transformer_result),
                "2steps": float(two_steps_result)
            },
            "nearest": formatted_nearest
        })

    except Exception as e:
        return jsonify({"error": str(e)})

    #     # 避免 Transformer 计算结果溢出
    #     if transformer_result == float('inf') or transformer_result != transformer_result:
    #         transformer_result = "Overflow"

    #     # # 根据 Category 调整计算结果
    #     # category_factor = {
    #     #     "0": 0,  # Bulk carrier
    #     #     "1": 1, # Chemical tanker
    #     #     "2": 2, # Combination carrier
    #     #     "3": 3, # Container/ro-ro cargo ship
    #     #     "4": 4,  # Container ship
    #     #     "5": 5, # Gas carrier
    #     #     "6": 6,  # General cargo ship
    #     #     "7": 7, # LNG carrier
    #     #     "8": 8,  # Oil tanker
    #     #     "9": 9, # Passenger ship
    #     #     "10":10, # Refrigerated cargo carrier
    #     #     "11":11, # Ro-pax ship
    #     #     "12":12  # Ro-ro ship
    #     # }
        
    #     factor = category_factor.get(ship_type, 1.0)  # 默认 1.0
    #     mlp_result *= factor
    #     transformer_result = transformer_result if isinstance(transformer_result, str) else transformer_result * factor
    #     two_steps_result = mlp_result + (transformer_result if isinstance(transformer_result, (int, float)) else 0)

    #     # 返回 JSON 数据
    #     return jsonify({
    #         "Category": ship_type,
    #         "MLP": mlp_result,
    #         "Transformer": transformer_result,
    #         "2steps": two_steps_result
    #     })
    # except ValueError:
    #     return jsonify({"error": "Invalid input. Please enter numerical values."})

if __name__ == '__main__':
    app.run(debug=True)
