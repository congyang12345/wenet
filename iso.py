import onnx_graphsurgeon as gs
import numpy as np
import onnx

model = onnx.load("ModifiedDecoder.onnx")
graph = gs.import_onnx(model)

tensors = graph.tensors()

graph.inputs = [tensors["377"].to_variable(dtype=np.float32, shape=(10, 63, 256))]
graph.outputs = [tensors["388"].to_variable(dtype=np.float32)]

graph.cleanup()

onnx.save(gs.export_onnx(graph), "subgraph.onnx")
