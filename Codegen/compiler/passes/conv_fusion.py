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
from passes import autotuner
from autotuner.design_space_descriptor import ConvConfigDescriptor
from passes.nvfuser_parser import NvfuserParser
import nvtx


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
        # tile_description = TileDescription(
        #     [128, 64, 32], 2, [2, 2, 1], math_inst
        # )
        # split_k_slices = 1

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

        self.problem_size = cutlass.conv.Conv2dProblemSize(
            cutlass.Tensor4DCoord(N, H, W, C),
            cutlass.Tensor4DCoord(K, R, S, C),
            cutlass.Tensor4DCoord(padding[0], padding[0], padding[1], padding[1]),
            cutlass.MatrixCoord(stride[0], stride[1]),
            cutlass.MatrixCoord(dilation[0], dilation[1]),
            cutlass.conv.Mode.cross_correlation,
            1, 1
        )

        self.conv_descriptor = ConvConfigDescriptor(
            problem_size=self.problem_size, element_a=element_a,
            element_b=element_b, element_c=element_c, 
            element_accumulator=element_acc, alignment_a=align_a,
            alignment_b=align_b, alignment_c=align_c
        )

        best_config = autotuner(self.conv_descriptor)

        print(best_config)

        threadblock_shape = [
            best_config['block_x'], best_config['block_y'], best_config['block_z']]
        warp_count = [
            best_config['block_x'] // best_config['warp_x'],
            best_config['block_y'] // best_config['warp_y'],
            best_config['block_z'] // best_config['warp_z']
        ]

        stages = best_config['stage']
        log_swizzle = best_config['log_swizzle']
        self.split_k_slices = best_config['split_k_slices']
        swizzling_functor = getattr(cutlass, "IdentitySwizzle%d"%int(pow(2, log_swizzle)))

        tile_description = TileDescription(
            threadblock_shape, stages, warp_count, math_inst
        )
        
        self.epilogue_functor.optimize(tile_description)

        iterator_algorithm = self.conv_descriptor.heuristic.available_iterator_algorithms[best_config["iterator_algorithm"]]
        stride_support = StrideSupport.Strided
        conv_kind = cutlass.conv.Operator.fprop

        self.operation = Conv2dOperation(
            conv_kind, iterator_algorithm,
            80, tile_description, A, B, D, stride_support,
            self.epilogue_functor, swizzling_functor, visitor=True
        )

        pycutlass.compiler.add_module([self.operation,])


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

        # we permute it before training
        # weight_cl = weight.permute(0, 2, 3, 1).contiguous()
        weight_cl = weight

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
                results.append(output_tensor.view(self.output_shape).permute(0, 3, 1, 2))#.contiguous())
            else:
                results.append(output_tensor)

        return results


class FusedConv2dDgrad:
    __name__ = "cutlass_conv2d_dgrad_with_visitor"
    def __init__(self, node) -> None:
        assert node.target == torch.nn.grad.conv2d_input
        self.node = node

        input_size = node.args[0]
        weight = node.args[1]
        grad_output = node.args[2]

        stride = node.args[3]
        padding = node.args[4]
        dilation = node.args[5]
        groups = node.args[6]

        N, C, H, W = input_size
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

        layout_a = cutlass.TensorNHWC
        layout_b = cutlass.TensorNHWC
        layout_c = cutlass.TensorNHWC

        align_a = get_alignment(K)
        align_b = get_alignment(C)
        align_c = get_alignment(C)

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

        self.problem_size = cutlass.conv.Conv2dProblemSize(
            cutlass.Tensor4DCoord(N, H, W, C),
            cutlass.Tensor4DCoord(K, R, S, C),
            cutlass.Tensor4DCoord(padding[0], padding[0], padding[1], padding[1]),
            cutlass.MatrixCoord(stride[0], stride[1]),
            cutlass.MatrixCoord(dilation[0], dilation[1]),
            cutlass.conv.Mode.cross_correlation,
            1, 1
        )

        self.conv_descriptor = ConvConfigDescriptor(
            problem_size=self.problem_size, element_a=element_a,
            element_b=element_b, element_c=element_c, 
            element_accumulator=element_acc, alignment_a=align_a,
            alignment_b=align_b, alignment_c=align_c, 
            conv_kind=cutlass.conv.Operator.dgrad
        )

        best_config = autotuner(self.conv_descriptor)

        print(best_config)

        threadblock_shape = [
            best_config['block_x'], best_config['block_y'], best_config['block_z']]
        warp_count = [
            best_config['block_x'] // best_config['warp_x'],
            best_config['block_y'] // best_config['warp_y'],
            best_config['block_z'] // best_config['warp_z']
        ]

        stages = best_config['stage']
        log_swizzle = best_config['log_swizzle']
        self.split_k_slices = best_config['split_k_slices']
        # handle dgrad
        if self.problem_size.stride_h == 1 and self.problem_size.stride_w == 1:
            swizzling_functor = getattr(cutlass, "IdentitySwizzle%d"%int(pow(2, log_swizzle)))
            stride_support = StrideSupport.Unity
        else:
            swizzling_functor = getattr(cutlass, "StridedDgradIdentitySwizzle%d"%int(pow(2, log_swizzle)))
            stride_support = StrideSupport.Strided

        tile_description = TileDescription(
            threadblock_shape, stages, warp_count, math_inst
        )

        iterator_algorithm = self.conv_descriptor.heuristic.available_iterator_algorithms[best_config["iterator_algorithm"]]
        
        conv_kind = cutlass.conv.Operator.dgrad

        self.epilogue_functor.optimize(tile_description)

        self.operation = Conv2dOperation(
            conv_kind, iterator_algorithm,
            80, tile_description, A, B, D, stride_support,
            self.epilogue_functor, swizzling_functor, visitor=True
        )

        pycutlass.compiler.add_module([self.operation,])

        self.output_shape = (N, H, W, C)
        self.implicit_mn = [N * H * W, C]
        self.output_numel = N * H * W * C

        self.args = self.epilogue_functor.args + list(node.args)
        self.outputs = self.epilogue_functor.outputs
        self.kernel_outputs = self.epilogue_functor.kernel_outputs
        self.output_2_kernel_output = self.epilogue_functor.output_2_kernel_output
    
    def __call__(self, *args):
        grad_output = args[-5]  # N, K, P, Q
        weight = args[-6]  # K, C, S, R

        # permute the input as they are under the HCHW layout
        # cl stands for channel last
        grad_output_cl = grad_output.permute(0, 2, 3, 1).contiguous()
        
        # we permute it before training
        # weight_cl = weight.permute(0, 2, 3, 1).contiguous()
        weight_cl = weight

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
            A=grad_output_cl, B=weight_cl, C=grad_output_cl, D=grad_output_cl,
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
                results.append(output_tensor.view(self.output_shape).permute(0, 3, 1, 2))#.contiguous())
            else:
                results.append(output_tensor)

        return results


class FusedConv2dWgrad:
    # we use the standalone conv2d wgrad kernels
    __name__ = "cutlass_conv2d_wgrad"
    def __init__(self, node) -> None:
        raise DeprecationWarning("This module has low performance")
        assert node.target == torch.nn.grad.conv2d_weight
        self.node = node

        input = node.args[0]
        K, C, R, S = node.args[1]
        grad_output = node.args[2]
        stride = node.args[3]
        padding = node.args[4]
        dilation = node.args[5]

        N, C_, H, W = list(input.meta["tensor_meta"].shape)

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

        layout_a = cutlass.TensorNHWC
        layout_b = cutlass.TensorNHWC
        layout_c = cutlass.TensorNHWC

        align_a = get_alignment(K)
        align_b = get_alignment(C)
        align_c = get_alignment(C)

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

        self.problem_size = cutlass.conv.Conv2dProblemSize(
            cutlass.Tensor4DCoord(N, H, W, C),
            cutlass.Tensor4DCoord(K, R, S, C),
            cutlass.Tensor4DCoord(padding[0], padding[0], padding[1], padding[1]),
            cutlass.MatrixCoord(stride[0], stride[1]),
            cutlass.MatrixCoord(dilation[0], dilation[1]),
            cutlass.conv.Mode.cross_correlation,
            1, 1
        )

        self.conv_descriptor = ConvConfigDescriptor(
            problem_size=self.problem_size, element_a=element_a,
            element_b=element_b, element_c=element_c, 
            element_accumulator=element_acc, alignment_a=align_a,
            alignment_b=align_b, alignment_c=align_c, 
            conv_kind=cutlass.conv.Operator.wgrad, split_k_mode="Serial",
            autotuning_rounds=30
        )

        best_config = autotuner(self.conv_descriptor)

        print(best_config)

        threadblock_shape = [
            best_config['block_x'], best_config['block_y'], best_config['block_z']]
        warp_count = [
            best_config['block_x'] // best_config['warp_x'],
            best_config['block_y'] // best_config['warp_y'],
            best_config['block_z'] // best_config['warp_z']
        ]

        stages = best_config['stage']
        log_swizzle = best_config['log_swizzle']
        self.split_k_slices = best_config['split_k_slices']
        # handle dgrad

        swizzling_functor = getattr(cutlass, "IdentitySwizzle%d"%int(pow(2, log_swizzle)))
        stride_support = StrideSupport.Strided

        tile_description = TileDescription(
            threadblock_shape, stages, warp_count, math_inst
        )

        iterator_algorithm = self.conv_descriptor.heuristic.available_iterator_algorithms[best_config["iterator_algorithm"]]
        
        conv_kind = cutlass.conv.Operator.wgrad

        self.operation = Conv2dOperation(
            conv_kind, iterator_algorithm,
            80, tile_description, A, B, D, stride_support,
            epilogue_functor, swizzling_functor, visitor=False
        )

        pycutlass.compiler.add_module([self.operation,])

        self.output_shape = (K, C, R, S)
        self.implicit_mn = [K * R * S, C]
        self.output_numel = K * R * S * C

        self.args = [grad_output, input]
    
    def __call__(self, *args):
        raise DeprecationWarning("This module has low performance")
        grad_output = args[-2]  # N, K, P, Q
        input = args[-1]  # N, H, W, C

        # permute the input as they are under the HCHW layout
        # cl stands for channel last
        grad_output_cl = grad_output.permute(0, 2, 3, 1).contiguous()
        input_cl = input.permute(0, 2, 3, 1).contiguous()

        grad_weight_cl = torch.empty(
            size=self.output_shape, dtype=torch.float16,
            device="cuda"
        )

        # tensor_D = torch.empty(size=self.output_shape, dtype=torch.float16, device="cuda")

        arguments = Conv2dArguments(
            self.operation, self.problem_size,
            A=grad_output_cl, B=input_cl, C=grad_weight_cl, D=grad_weight_cl,
            output_op=self.operation.epilogue_type(1.0, 0.0),
            split_k_mode=cutlass.conv.SplitKMode.Serial,
            split_k_slices=self.split_k_slices
        )
        stream = torch.cuda.current_stream()

        self.operation.run(arguments, stream=stream.cuda_stream)

        return grad_weight_cl#.permute(0, 3, 1, 2).contiguous()


class FusedBNForward:
    __name__ = "nvfuser_bn_fp"
    def __init__(self, node, module) -> None:
        assert node.target == torch.ops.aten.native_batch_norm.default
        self.node = node

        self.nvfuser_parser = NvfuserParser(self.node)

        self.scripted_splitted_batch_norm = self.nvfuser_parser.extract_sub_module(module)

        self.args = self.nvfuser_parser.input_nodes
        self.outputs = self.nvfuser_parser.outputs
    
    def __call__(self, *args):
        permuted_args = []
        for arg in args:
            if len(arg.size()) == 4:
                permuted_args.append(arg.permute(0, 2, 3, 1).contiguous())
            else:
                permuted_args.append(arg)

        with nvtx.annotate("fused BN ReLU"):
            outputs = self.scripted_splitted_batch_norm(*permuted_args)

        permuted_outputs = []
        for output in outputs:
            if len(list(output.size())) == 4:
                permuted_outputs.append(output.permute(0, 3, 1, 2))
            else:
                permuted_outputs.append(output)
        return permuted_outputs


class FusedBNBackward:
    __name__ = "nvfuser_bn_bp"
    def __init__(self, node, module) -> None:
        assert node.target == torch.ops.aten.native_batch_norm_backward
        self.node = node

        self.nvfuser_parser = NvfuserParser(self.node)

        self.scripted_splitted_batch_norm_backward = self.nvfuser_parser.extract_sub_module(module)

        self.args = self.nvfuser_parser.input_nodes
        self.outputs = self.nvfuser_parser.outputs
    
    def __call__(self, *args):
        permuted_args = []
        for arg in args:
            if len(arg.size()) == 4:
                permuted_args.append(arg.permute(0, 2, 3, 1).contiguous())
            else:
                permuted_args.append(arg)

        with nvtx.annotate("fused BN ReLU"):
            outputs = self.scripted_splitted_batch_norm_backward(*permuted_args)

        permuted_outputs = []
        for output in outputs:
            if len(list(output.size())) == 4:
                permuted_outputs.append(output.permute(0, 3, 1, 2))
            else:
                permuted_outputs.append(output)
        # print(permuted_outputs[0].contiguous().view(-1))
        # return [torch.ops.aten.native_batch_norm_backward(args[0], args[1], args[2], args[3], args[4], args[5], args[6], True, 1e-5, [True, True, True])[0].contiguous(), permuted_outputs[1], permuted_outputs[2]]
        return permuted_outputs

class FusedElementwise:
    __name__ = "nvfuser_fused_elementwise"
    def __init__(self, node, module) -> None:

        self.node = node

        self.nvfuser_parser = NvfuserParser(self.node)

        self.fused_elementwise = self.nvfuser_parser.extract_sub_module(module)

        self.args = self.nvfuser_parser.input_nodes
        self.outputs = self.nvfuser_parser.outputs
    
    def __call__(self, *args):
        permuted_args = []
        for arg in args:
            if len(arg.size()) == 4:
                permuted_args.append(arg.permute(0, 2, 3, 1).contiguous())
            else:
                permuted_args.append(arg)

        with nvtx.annotate("fused element wise"):
            outputs = self.fused_elementwise(*permuted_args)

        permuted_outputs = []
        for output in outputs:
            if len(list(output.size())) == 4:
                permuted_outputs.append(output.permute(0, 3, 1, 2))
            else:
                permuted_outputs.append(output)
        return permuted_outputs


def pass_conv_fusion(module, graph, verbose=True):
    """
    Fuse Conv kernels with its epilogue
    """

    conv_idx = 0
    conv_dgrad_idx = 0
    conv_wgrad_idx = 0
    bn_fp_idx = 0
    bn_bp_idx = 0
    el_idx = 0
    for node in tqdm(graph.nodes):
        if node.op == "call_function":
            if node.target == torch.ops.aten.convolution:
                if conv_idx >= 20: continue

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
            
            elif node.target == torch.nn.grad.conv2d_input:
                if conv_dgrad_idx >= 19: continue  # conv_dgrad_2 is wrong
                update_topological_index(graph)
                fused_conv_dgrad = FusedConv2dDgrad(node)
                inserting_point = node

                graph.inserting_after(inserting_point)
                fused_node_dgrad = graph.call_function(fused_conv_dgrad, args=tuple(fused_conv_dgrad.args))
                fused_node_dgrad.meta = {}
                fused_node_dgrad.meta['tensor_meta'] = fused_conv_dgrad.epilogue_functor.root.meta['tensor_meta']._replace()
                # node.replace_all_uses_with(fused_node_dgrad)
                # erase_node_recursive(graph, node.users[0])

                graph.inserting_after(fused_node_dgrad)
                for idx, output_node in enumerate(fused_conv_dgrad.outputs):
                    get_item_node = graph.call_function(operator.getitem, args=(fused_node_dgrad, idx))
                    get_item_node.meta["tensor_meta"] = output_node.meta["tensor_meta"]._replace()
                    get_item_node.meta["unfusible"] = True
                    graph.inserting_after(get_item_node)
                    output_node.replace_all_uses_with(get_item_node)
                    erase_node_recursive(graph, output_node)
                
                conv_dgrad_idx += 1
            
            # elif node.target == torch.nn.grad.conv2d_weight:
            #     if conv_wgrad_idx >= 20: continue
            #     update_topological_index(graph)
            #     fused_conv_wgrad = FusedConv2dWgrad(node)
            #     inserting_point = node

            #     graph.inserting_after(inserting_point)
            #     fused_node_wgrad = graph.call_function(fused_conv_wgrad, args=tuple(fused_conv_wgrad.args))
            #     fused_node_wgrad.meta = {}
            #     fused_node_wgrad.meta['tensor_meta'] = node.meta['tensor_meta']._replace()
            #     node.replace_all_uses_with(fused_node_wgrad)
            #     erase_node_recursive(graph, node)

            #     # graph.inserting_after(fused_node_wgrad)
            #     # for idx, output_node in enumerate(fused_conv_wgrad.outputs):
            #     #     get_item_node = graph.call_function(operator.getitem, args=(fused_node_wgrad, idx))
            #     #     get_item_node.meta["tensor_meta"] = output_node.meta["tensor_meta"]._replace()
            #     #     get_item_node.meta["unfusible"] = True
            #     #     graph.inserting_after(get_item_node)
            #     #     output_node.replace_all_uses_with(get_item_node)
            #     #     erase_node_recursive(graph, output_node)
            #     conv_wgrad_idx += 1

            
            elif node.target == torch.ops.aten.native_batch_norm.default:
                if bn_fp_idx >= 20: continue
                print("=========================================================")
                print(node)

                update_topological_index(graph)
                fused_bn_fp = FusedBNForward(node, module)

                # inserting_idx = 1e+16
                # for output in fused_bn_fp.outputs:
                #     idx = get_topological_index(graph, output)
                #     if idx < inserting_idx:
                #         inserting_idx = idx
                #         inserting_point = output
                inserting_idx = 0
                for input in fused_bn_fp.args:
                    idx = get_topological_index(graph, input)
                    if idx > inserting_idx:
                        inserting_idx = idx
                        inserting_point = input

                
                graph.inserting_after(inserting_point)
                fused_node = graph.call_function(fused_bn_fp, args=tuple(fused_bn_fp.args))
                graph.inserting_after(fused_node)

                for idx, output_node in enumerate(fused_bn_fp.outputs):
                    get_item_node = graph.call_function(operator.getitem, args=(fused_node, idx))
                    get_item_node.meta["tensor_meta"] = output_node.meta["tensor_meta"]._replace()
                    get_item_node.meta["unfusible"] = True
                    graph.inserting_after(get_item_node)
                    output_node.replace_all_uses_with(get_item_node)
                    erase_node_recursive(graph, output_node)

                # node.replace_all_uses_with(fused_node)
                bn_fp_idx += 1
            
            elif node.target == torch.ops.aten.native_batch_norm_backward:
                if bn_bp_idx >= 16: continue
                print("=========================================================")
                print(node)

                update_topological_index(graph)
                fused_bn_bp = FusedBNBackward(node, module)

                inserting_idx = 0
                for input in fused_bn_bp.args:
                    idx = get_topological_index(graph, input)
                    if idx > inserting_idx:
                        inserting_idx = idx
                        inserting_point = input
                
                graph.inserting_after(inserting_point)
                fused_node = graph.call_function(fused_bn_bp, args=tuple(fused_bn_bp.args))
                graph.inserting_after(fused_node)

                for idx, output_node in enumerate(fused_bn_bp.outputs):
                    get_item_node = graph.call_function(operator.getitem, args=(fused_node, idx))
                    get_item_node.meta["tensor_meta"] = output_node.meta["tensor_meta"]._replace()
                    get_item_node.meta["unfusible"] = True
                    graph.inserting_after(get_item_node)
                    output_node.replace_all_uses_with(get_item_node)
                    erase_node_recursive(graph, output_node)

                # node.replace_all_uses_with(fused_node)
                bn_bp_idx += 1
            
            elif node.target in [torch.ops.aten.expand]:
                if el_idx >= 1: continue
                print("=================================================")
                print("Elementwise")
                update_topological_index(graph)
                fused_elementwise = FusedElementwise(node, module)

                inserting_idx = 0
                for input in fused_elementwise.args:
                    idx = get_topological_index(graph, input)
                    if idx > inserting_idx:
                        inserting_idx = idx
                        inserting_point = input
                
                graph.inserting_after(inserting_point)
                fused_node = graph.call_function(fused_elementwise, args=tuple(fused_elementwise.args))
                graph.inserting_after(fused_node)

                for idx, output_node in enumerate(fused_elementwise.outputs):
                    get_item_node = graph.call_function(operator.getitem, args=(fused_node, idx))
                    get_item_node.meta["tensor_meta"] = output_node.meta["tensor_meta"]._replace()
                    get_item_node.meta["unfusible"] = True
                    graph.inserting_after(get_item_node)
                    output_node.replace_all_uses_with(get_item_node)
                    erase_node_recursive(graph, output_node)
                
                el_idx += 1

                


    
    graph.lint()
    graph.eliminate_dead_code()