import onnx
import numpy as np
import onnxruntime as ot
from onnx import helper, numpy_helper

add_node_0 = helper.make_node('Add', ['add_node_0_input_0', 'add_0_constant_tensor'], 
                              outputs=['add_node_0_output'], name='Add_0')
mul_0 = helper.make_node('Mul', ['add_node_0_output', 'mul_0_constant_tensor'],
                         outputs=['mul_node_0_output'], name='Mul_0')
add_node_1 = helper.make_node('Add', ['mul_node_0_output', 'add_1_constant_tensor'],
                              outputs=['add_node_1_output'], name='Add_1')
shape = [3, 224, 224]
input_value_info = helper.make_tensor_value_info(
    'add_node_0_input_0', onnx.TensorProto.FLOAT, shape=shape
)
output_value_info = helper.make_tensor_value_info(
    'add_node_1_output', onnx.TensorProto.FLOAT, shape=shape
)

add_0_constant_tensor = numpy_helper.from_array(
    np.random.randint(1, 10, size=shape).astype(np.float32), name='add_0_constant_tensor'
)
mul_0_constant_tensor = numpy_helper.from_array(
    np.random.randint(1, 10, size=shape).astype(np.float32), name='mul_0_constant_tensor'
)
add_1_constant_tensor = numpy_helper.from_array(
    np.random.randint(1, 10, size=shape).astype(np.float32), name='add_1_constant_tensor'
)
graph = helper.make_graph(
    nodes=[add_node_0, mul_0, add_node_1], name='test_model',
    inputs=[input_value_info], outputs=[output_value_info],
    initializer=[add_0_constant_tensor, mul_0_constant_tensor, add_1_constant_tensor]
)
model = helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid('ai.onnx', 13)])
onnx.save(model, 'test.onnx')

sess = ot.InferenceSession(
    'test.onnx', providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
input = {'add_node_0_input_0': np.random.randint(1, 10, size=shape).astype(np.float32)}
output = sess.run(['add_node_1_output'], input)
print(output[0].shape)