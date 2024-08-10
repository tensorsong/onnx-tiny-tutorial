import onnx
import time
import onnxruntime as ot
import numpy as np
from onnxsim import simplify

model_path = 'models/resnet50-v1-12.onnx'
model = onnx.load(model_path)
model_sim, check = simplify(model)
onnx.save(model, 'models/sim_resnet50-v1-12.onnx')

input = {'data': np.random.randn(1, 3, 224, 224).astype(np.float32)}
ori_sess = ot.InferenceSession(
    'models/resnet50-v1-12.onnx', providers=['CPUExecutionProvider', 'CUDAExecutionProvider']
)
sim_sess = ot.InferenceSession(
    'models/sim_resnet50-v1-12.onnx', providers=['CPUExecutionProvider', 'CUDAExecutionProvider']
)
infer_num = 100

start = time.time()
for _ in range(infer_num):
    output = ori_sess.run([], input)
end = time.time()
print(f'origin resnet50 spend time: {(end - start) * 1000 / infer_num} ms.')

start = time.time()
for _ in range(infer_num):
    output = sim_sess.run([], input)
end = time.time()
print(f'simplify resnet50 spend time: {(end - start) * 1000 / infer_num} ms.')
