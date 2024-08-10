import time
import onnx
import onnxruntime as ot
import numpy as np
from onnxconverter_common import float16, auto_convert_mixed_precision

def compute_mse(predict, target):
    return np.mean((np.array(predict) - np.array(target)) ** 2) < 1e-4

input = {'data': np.random.randn(1, 3, 224, 224).astype(np.float32)}
model = onnx.load('models/resnet50-v1-12.onnx')
# 生成fp16精度模型
model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
onnx.save(model_fp16, 'models/resnet50-v1-12_fp16.onnx')
# 生成混合精度模型，即部分算子使用fp32精度的模型
model_mixed = auto_convert_mixed_precision(model, input, validate_fn=compute_mse, keep_io_types=True)
onnx.save(model_mixed, 'models/resnet50-v1-12_mixed.onnx')

ori_sess = ot.InferenceSession(
    'models/resnet50-v1-12.onnx', providers=['CPUExecutionProvider', 'CUDAExecutionProvider']
)
fp16_sess = ot.InferenceSession(
    'models/resnet50-v1-12_fp16.onnx', providers=['CPUExecutionProvider', 'CUDAExecutionProvider']
)
infer_num = 100

start = time.time()
for _ in range(infer_num):
    output = ori_sess.run([], input)
end = time.time()
print(f'origin resnet50 spend time: {(end - start) * 1000 / infer_num} ms.')

start = time.time()
for _ in range(infer_num):
    output_fp16 = fp16_sess.run([], input)
end = time.time()
print(f'fp16 resnet50 spend time: {(end - start) * 1000 / infer_num} ms.')

mse = np.mean((output[0] - output_fp16[0]) ** 2)
print(f"MSE: {mse}")
