import pandas as pd 
import numpy as np

def load_data():
    df = pd.read_csv("co2_0122.csv")
    # 注意这里 features 列名需与 CSV 文件保持一致
    features = ['length', 'breadth', 'gross_tonnage', 'deadweight', 
                'avg_speed', 'Time spent at sea [hours]']
    
    # 计算归一化参数，并对数据归一化
    normalization = {}
    for col in features:
        min_val = df[col].min()
        max_val = df[col].max()
        normalization[col] = {'min': min_val, 'range': max_val - min_val}
        df[f'{col}_norm'] = (df[col] - min_val) / (max_val - min_val)
    
    return df, normalization

def calculate_distances(input_vector, row, features):
    """
    计算欧氏距离和曼哈顿距离：
    - 欧氏距离：直线距离，即各维度差值平方和再开根号。
    - 曼哈顿距离：各维度差值绝对值的和。
    """
    euclidean = np.sqrt(sum(
        (input_vector[i] - row[f'{col}_norm'])**2 for i, col in enumerate(features)
    ))
    manhattan = sum(
        abs(input_vector[i] - row[f'{col}_norm']) for i, col in enumerate(features)
    )
    return euclidean, manhattan

def find_nearest(input_data, metric="euclidean"):
    """
    根据选择的距离公式查找最近的三个数据点，并返回指定的属性。
    
    参数:
      input_data: 包含待查询数据的字典，需包含 'ship type' 以及各特征值（键名须与 features 一致）
      metric: 距离公式，"euclidean" 或 "manhattan"，默认为 "euclidean"
      
    返回:
      最近的三个数据点（列表），每个数据点为一个字典，仅包含以下字段：
         'length', 'breadth', 'deadweight', 'gross_tonnage',
         'avg_speed', 'Time spent at sea [hours]', 'CO2'
    """
    df, normalization = load_data()
    # 注意：此处 CSV 文件中船舶类型的列假设为 'type no'
    ship_type = int(input_data['ship type'])
    filtered_df = df[df['type no'] == ship_type].copy()
    
    if filtered_df.empty:
        return None

    features = ['length', 'breadth', 'gross_tonnage', 'deadweight', 
                'avg_speed', 'Time spent at sea [hours]']
    
    # 构建输入向量（归一化后）
    input_vector = [
        (input_data[col] - normalization[col]['min']) / normalization[col]['range']
        for col in features
    ]
    
    # 计算每一行的距离
    distances = []
    for _, row in filtered_df.iterrows():
        euclidean, manhattan = calculate_distances(input_vector, row, features)
        distances.append((euclidean, manhattan))
    
    filtered_df['euclidean'] = [d[0] for d in distances]
    filtered_df['manhattan'] = [d[1] for d in distances]
    
    # 根据所选距离公式排序，并选取最近的三个数据点
    if metric.lower() == "manhattan":
        sorted_df = filtered_df.sort_values(by='manhattan')
    else:
        sorted_df = filtered_df.sort_values(by='euclidean')
    
    nearest_rows = sorted_df.head(3).copy()
    # 删除距离字段
    nearest_rows = nearest_rows.drop(columns=["euclidean", "manhattan"])
    # 只保留所需列（请确保这些列名与 CSV 中一致）
    columns_to_return = ['length', 'breadth', 'deadweight', 'gross_tonnage', 
                         'avg_speed', 'Time spent at sea [hours]', 'Total CO₂ emissions [m tonnes]']
    nearest_rows = nearest_rows[columns_to_return]
    print("nearest rows:")
    print(nearest_rows)
    return nearest_rows.to_dict(orient="records")
