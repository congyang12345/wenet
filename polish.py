import onnx
import onnx_graphsurgeon as gs
import numpy as np

######################### layernorm ######################################

graph = gs.import_onnx(onnx.load('./encoder_sim.onnx'))
graph.cleanup().toposort()

list_ = {}
for idx, node in enumerate(graph.nodes):
    if 1:
        list_.update({node.name: idx})
        # print(node.name)
        if 'Add' in node.name and 'norm' in node.inputs[1].name:
            # print(list_)
            index = node.name.split('_')[1]
            sub_name = "ReduceMean_" + str(int(index)-10)
            reduce_name = "Sub_" + str(int(index)-9)
            mul_name = "Mul_" + str(int(index)-1)
            sub_index = list_[sub_name]
            reduce_index = list_[reduce_name]
            mul_index = list_[mul_name]

            attrs = {}
            attrs["num_groups"] = 1
            attrs["eps"] = 1e-5
            attrs["plugin_version"] = "1"

            attrs["plugin_namespace"] = ""

            layer_norm_out = gs.Variable("layernorm_out_"+str(idx), dtype=np.float32)
            layer_norm = gs.Node(op="LayerNormPlugin", name= "LayerNorm_" + str(idx), inputs=[graph.nodes[reduce_index].inputs[0], graph.nodes[mul_index].inputs[1], node.inputs[1]], outputs=node.outputs, attrs=attrs)
            graph.nodes[sub_index].inputs.clear()
            graph.nodes[reduce_index].inputs.clear()
            node.outputs.clear()
            graph.nodes.append(layer_norm)
    '''
    if 'Sigmoid' in node.name:
        if 'Split' in graph.nodes[idx-1].name:
            split_index = idx - 1
            mul_index = idx + 1
            attrs = {}
            attrs["plugin_version"] = "1"
            attrs["plugin_namespace"] = ""

            glu = gs.Node(op="GluPlugin", name= "Glu_" + str(idx), inputs=graph.nodes[split_index].inputs, outputs=graph.nodes[mul_index].outputs, attrs=attrs)

            graph.nodes[split_index].inputs.clear()
            graph.nodes[mul_index].outputs.clear()
            graph.nodes.append(glu)

        else:
            mul_index = idx + 1
            attrs = {}
            attrs["plugin_version"] = "1"
            attrs["plugin_namespace"] = ""
    
            swish = gs.Node(op="SwishPlugin", name= "Swish_" + str(idx), inputs=node.inputs, outputs=graph.nodes[mul_index].outputs, attrs=attrs)
    
            graph.nodes[mul_index].inputs.clear()
            graph.nodes[mul_index].outputs.clear()
            node.inputs.clear()
            graph.nodes.append(swish)
    '''

Unsqueeze_29 = graph.nodes[list_["Unsqueeze_29"]]
Not_30 = graph.nodes[list_["Not_30"]]
Slice_79 = graph.nodes[list_["Slice_79"]]
Slice_84 = graph.nodes[list_["Slice_84"]]

start_node = Unsqueeze_29.outputs[0]
Unsqueeze_29_Cast_output = gs.Variable(name="Unsqueeze_29_Cast_output", dtype=None, shape=None)
attrs_dict = {}
attrs_dict['to'] = 6
newNode = gs.Node(name="Slice_84_Cast", op="Cast", inputs=[start_node],
                  outputs=[Unsqueeze_29_Cast_output], attrs=attrs_dict)
graph.nodes.append(newNode)
Slice_79.inputs[0] = Unsqueeze_29_Cast_output
Slice_84_outputs = Not_30.outputs[0]
end_node = Slice_84.outputs[0]
Not_30.outputs[0] = end_node
Slice_84.outputs[0] = Slice_84_outputs
Not_30.inputs[0] = Slice_84.outputs[0]

Slice_84_Cast_output = gs.Variable(name="Slice_84_Cast_output", dtype=None, shape=None)
attrs_dict = {}
attrs_dict['to'] = 9
newNode = gs.Node(name="Slice_84_Cast", op="Cast", inputs=[Slice_84_outputs ],
                  outputs=[Slice_84_Cast_output], attrs=attrs_dict)
graph.nodes.append(newNode)
Not_30.inputs[0] = Slice_84_Cast_output

graph.cleanup().toposort()

onnx.save(gs.export_onnx(graph), "./ModifyEncoder.onnx")

########################################## Not ########################################################
