import onnx
import numpy as np
from onnx import helper, numpy_helper, onnx_ml_pb2

##################################################################################################

model: onnx_ml_pb2.ModelProto = onnx.load('test.onnx')
shape = [3, 224, 224]

# 创建新的add节点
add_node_inert = helper.make_node(
    'Add', ['add_node_0_output', 'add_insert_constant_tensor'],
    outputs=['add_node_insert_output'], name='Add_insert'
)
add_insert_constant_tensor = numpy_helper.from_array(
    np.random.randint(1, 10, size=shape).astype(np.float32), name='add_insert_constant_tensor'
)

# 插入新节点
model.graph.node.append(add_node_inert)
model.graph.initializer.append(add_insert_constant_tensor)

# 修改原始节点的输入name
for idx in range(len(model.graph.node)):
    if model.graph.node[idx].name == 'Mul_0':
        model.graph.node[idx].input[0] = 'add_node_insert_output'
onnx.save(model, 'insert_add.onnx')

##################################################################################################

model: onnx_ml_pb2.ModelProto = onnx.load('test.onnx')

# 删除节点
for idx in range(len(model.graph.node)):
    if model.graph.node[idx].name == 'Mul_0':
        del model.graph.node[idx]
        break

# 修改节点输入名称
for idx in range(len(model.graph.node)):
    if model.graph.node[idx].name == 'Add_1':
        model.graph.node[idx].input[0] = 'add_node_0_output'
        break
onnx.save(model, 'del_mul.onnx')

##################################################################################################

model: onnx_ml_pb2.ModelProto = onnx.load('test.onnx')
add_node_insert = helper.make_node('Add', ['add_node_0_output', 'add_insert_constant_tensor'],
                                   outputs=['add_node_insert_output'], name='Add_insert')
add_insert_constant_tensor = numpy_helper.from_array(
    np.random.randint(1, 10, size=shape).astype(np.float32), name='add_insert_constant_tensor'
)
# 删除节点
for idx in range(len(model.graph.node)):
    if model.graph.node[idx].name == 'Mul_0':
        del model.graph.node[idx]
        break

# 插入新节点
model.graph.node.append(add_node_inert)
model.graph.initializer.append(add_insert_constant_tensor)

# 修改节点输入名称
for idx in range(len(model.graph.node)):
    if model.graph.node[idx].name == 'Add_1':
        model.graph.node[idx].input[0] = 'add_node_insert_output'
        break
onnx.save(model, 'replace_mul.onnx')