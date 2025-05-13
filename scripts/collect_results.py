import os
import json

base_dir = '/home/NeuralCausalModels/out'
results = []

for pipeline in os.listdir(base_dir):
    pipeline_dir = os.path.join(base_dir, pipeline)
    if not os.path.isdir(pipeline_dir):
        continue
    for exp in os.listdir(pipeline_dir):
        exp_dir = os.path.join(pipeline_dir, exp)
        results_path = os.path.join(exp_dir, 'results.json')
        if os.path.isfile(results_path):
            try:
                with open(results_path, 'r') as f:
                    data = json.load(f)
                # 解析 graph 和 n_samples
                info = dict(item.split('=') for item in exp.split('-') if '=' in item)
                row = {
                    'pipeline': pipeline,
                    'graph': info.get('graph'),
                    'n_samples': info.get('n_samples'),
                }
                row.update(data)
                results.append(row)
            except Exception as e:
                print(f"Failed to read {results_path}: {e}")

# 输出为表格
import pandas as pd
df = pd.DataFrame(results)
print(df)
# 如需保存
df.to_csv('/home/NeuralCausalModels/out/all_results.csv', index=False)