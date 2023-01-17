################################################################################
# Copyright [yyyy] [name of copyright owner]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################
from tqdm import tqdm
import torch
import cutlass
import pycutlass
from pycutlass import *
from passes.epilogue_parser import EpilogueVisitTreeDAG
import operator


def get_alignment(dim):
    if dim % 8 == 0: return 8
    elif dim % 4 == 0: return 4
    elif dim % 2 == 0: return 2
    else: return 1

def erase_node_recursive(graph, node):
    inputs = node.all_input_nodes
    if node.op == "call_function":
        graph.erase_node(node)
    for input in inputs:
        if len(input.users) == 0:
            erase_node_recursive(graph, input)

def update_topological_index(graph):
    """
    Update the topological index of each node
    """
    for idx, node in enumerate(graph.nodes):
        node.meta["topo_idx"] = idx

def get_topological_index(graph, node):
    """
    Get the node index in topological order
    """
    for idx, node_ in enumerate(graph.nodes):
        if node == node_:
            return idx

class FusedConv2d:
    __name__ = "cutlass_conv2d_with_visitor"
    def __init__(self, node) -> None:
        assert node.target == torch.ops.aten.convolution
        self.node = node
        input = node.args[0]
        weight = node.args[1]
        bias = node.args[2]

        assert bias is None

        stride = node.args[3]
        padding = node.args[4]
        dilation = node.args[5]
        transposed = node.args[6]
        output_paddig = node.args[7]
        groups = node.args[8]

        N, C, H, W = list(input.meta["tensor_meta"].shape)
        K, C_, R, S = list(weight.meta["tensor_meta"].shape)

        assert C_ == C


        element_a = cutlass.float16
        element_b = cutlass.float16
        element_c = cutlass.float16
        element_acc = cutlass.float32

        math_operation = MathOperation.multiply_add
        opclass = cutlass.OpClass.TensorOp
        
        math_inst = MathInstruction(
            [16, 8, 16], element_a, element_b,
            element_acc, opclass, math_operation
        )

        # TODO: tune this problem size
        tile_description = TileDescription(
            [128, 128, 32], 2, [2, 2, 1], math_inst
        )
        split_k_slices = 1

        layout_a = cutlass.TensorNHWC
        layout_b = cutlass.TensorNHWC
        layout_c = cutlass.TensorNHWC

        align_a = get_alignment(C)
        align_b = get_alignment(C)
        align_c = get_alignment(K)

        print(align_a)
        print(align_b)
        print(align_c)

        A = TensorDescription(
            element_a, layout_a, align_a
        )

        B = TensorDescription(
            element_b, layout_b, align_b
        )

        D = TensorDescription(
            element_c, layout_c, align_c
        )

        element_epilogue = cutlass.float32

        epilogue_functor = pycutlass.LinearCombination(
            D.element, D.alignment, math_inst.element_accumulator, element_epilogue
        )

        # Creating the epilogue visitor tree
        self.epilogue_functor = EpilogueVisitTreeDAG(
            elementwise_functor=epilogue_functor,
            element_accumulator=math_inst.element_accumulator,
            elements_per_access=D.alignment,
            element_compute=cutlass.float32, element_output=D.element
        )

        self.epilogue_functor.initialize(node)
        self.epilogue_functor.optimize(tile_description)

        iterator_algorithm = cutlass.conv.IteratorAlgorithm.optimized
        swizzling_functor = cutlass.IdentitySwizzle8
        stride_support = StrideSupport.Strided
        conv_kind = cutlass.conv.Operator.fprop

        self.operation = Conv2dOperation(
            conv_kind, iterator_algorithm,
            80, tile_description, A, B, D, stride_support,
            self.epilogue_functor, swizzling_functor, visitor=True
        )

        pycutlass.compiler.add_module([self.operation,])

        self.problem_size = cutlass.conv.Conv2dProblemSize(
            cutlass.Tensor4DCoord(N, H, W, C),
            cutlass.Tensor4DCoord(K, R, S, C),
            cutlass.Tensor4DCoord(padding[0], padding[0], padding[1], padding[1]),
            cutlass.MatrixCoord(stride[0], stride[1]),
            cutlass.MatrixCoord(dilation[0], dilation[1]),
            cutlass.conv.Mode.cross_correlation,
            split_k_slices, 1
        )

        N_, K_, P, Q = list(node.meta["tensor_meta"].shape)
        assert N_ == N
        assert K_ == K
        self.output_shape = (N, P, Q, K)
        self.implicit_mn = [N * P * Q, K]
        self.output_numel = N * P * Q * K

        self.args = self.epilogue_functor.args + list(node.args)
        self.outputs = self.epilogue_functor.outputs
        self.kernel_outputs = self.epilogue_functor.kernel_outputs
        self.output_2_kernel_output = self.epilogue_functor.output_2_kernel_output
    
    def __call__(self, *args):
        input = args[-9]  # N, C, H, W
        weight = args[-8] # K, C, R, S

        # permute the input as they are under the HCHW layout
        # cl stands for channel last
        input_cl = input.permute(0, 2, 3, 1).contiguous()
        weight_cl = weight.permute(0, 2, 3, 1).contiguous()

        kwargs = {"problem_size": self.implicit_mn}
        for output_node in self.kernel_outputs:
            if output_node.target in [torch.ops.aten.sum]:
                kwargs["output_" + output_node.name] = torch.zeros(
                    size=output_node.meta['tensor_meta'].shape,
                    dtype=output_node.meta['tensor_meta'].dtype,
                    device="cuda"
                )
            else:
                kwargs["output_" + output_node.name] = torch.empty(
                    size=output_node.meta['tensor_meta'].shape,
                    dtype=output_node.meta['tensor_meta'].dtype,
                    device="cuda"
                )
        
        for idx, epilogue_arg in enumerate(self.epilogue_functor.args):
            kwargs[epilogue_arg.name] = args[idx]
        
        output_op = self.operation.epilogue_type(**kwargs)

        # tensor_D = torch.empty(size=self.output_shape, dtype=torch.float16, device="cuda")

        arguments = Conv2dArguments(
            self.operation, self.problem_size,
            A=input_cl, B=weight_cl, C=input_cl, D=input_cl,
            output_op=output_op,
            split_k_mode=cutlass.conv.SplitKMode.Serial,
            split_k_slices=1
        )
        stream = torch.cuda.current_stream()

        self.operation.run(arguments, stream=stream.cuda_stream)

        results = []
        for output_node in self.outputs:
            output_tensor = kwargs["output_" + self.output_2_kernel_output[output_node].name].view(output_node.meta["tensor_meta"].shape)
            if output_tensor.numel() == self.output_numel:
                results.append(output_tensor.view(self.output_shape).permute(0, 3, 1, 2).contiguous())
            else:
                results.append(output_tensor)

        return results


def pass_conv_fusion(module, graph, verbose=True):
    """
    Fuse Conv kernels with its epilogue
    """

    conv_idx = 0
    for node in tqdm(graph.nodes):
        if node.op == "call_function":
            if node.target == torch.ops.aten.convolution:
                if conv_idx >= 1: continue

                update_topological_index(graph)
                fused_conv = FusedConv2d(node)
                inserting_point = fused_conv.epilogue_functor.root
                inserting_idx = get_topological_index(graph, inserting_point)

                for output in fused_conv.outputs:
                    idx = get_topological_index(graph, output)
                    if idx < inserting_idx:
                        inserting_idx = idx
                        inserting_point = output

                graph.inserting_after(inserting_point)
                fused_node = graph.call_function(fused_conv, args=tuple(fused_conv.args))
                fused_node.meta = {}
                fused_node.meta['tensor_meta'] = fused_conv.epilogue_functor.root.meta['tensor_meta']._replace()
                graph.inserting_after(fused_node)

                for idx, output_node in enumerate(fused_conv.outputs):
                    get_item_node = graph.call_function(operator.getitem, args=(fused_node, idx))
                    get_item_node.meta["tensor_meta"] = output_node.meta["tensor_meta"]._replace()
                    get_item_node.meta["unfusible"] = True
                    graph.inserting_after(get_item_node)
                    output_node.replace_all_uses_with(get_item_node)
                    erase_node_recursive(graph, output_node)

                # node.replace_all_uses_with(fused_node)
                # erase_node_recursive(graph, node)
                
                conv_idx += 1
    
    graph.lint()
    graph.eliminate_dead_code()