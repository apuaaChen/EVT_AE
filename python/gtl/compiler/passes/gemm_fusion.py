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
import pycutlass
from pycutlass import *
import cutlass
import torch
import torch.nn.functional as F
from gtl.compiler.passes.epilogue_parser import EpilogueVisitTreeDAG, TensorOutputNodeDAG, AccumulatorNodeDAG
from gtl.compiler.passes import autotuner
import nvtx
import operator
from .softmax_operation import SoftmaxArguments, SoftmaxOperation
from .layernorm_operation import LayerNormArguments, LayerNormOperation
from .softmax_backward_operation import SoftmaxBackwardArguments, SoftmaxBackwardOperation
from .layernorm_backward_operation import LayerNormBackwardArguments, LayerNormBackwardOperation
from .spmm_operator import SpmmOperation, SpmmArguments
from gtl.compiler.autotuner.design_space_descriptor import GemmConfigDescriptor, BatchedGemmConfigDescriptor
from gtl.compiler.nodes import *
from tqdm import tqdm
import logging

def get_priority(node):
    if len(node.users) > 1:
        return -1
    user = list(node.users.keys())[0]
    if user.op == "output":
        return 0
    elif user.op == "call_function":
        if user.target in [torch.ops.aten.view, torch.ops.aten.unsqueeze, torch.ops.aten.squeeze, torch.ops.aten._unsafe_view, operator.getitem]:
            return get_priority(user)
        else:
            return -1
    else:
        raise NotImplementedError()

################################################################################
# Graph-level pass to fuse GEMM kernels
################################################################################

def pass_dead_code_elimination_non_topo(module, graph):
    changed = True
    while(changed):
        changed = False
        for node in graph.nodes:
            if node.op == "call_function":
                if len(node.users) == 0:
                    graph.erase_node(node)
                    changed = True
                    break

def get_alignment(dim):
    if dim % 8 == 0: return 8
    elif dim % 4 == 0: return 4
    elif dim % 2 == 0: return 2
    else: return 1

class FusedSpmm:
    __name__ = "cutlass_spmm_with_visitor"
    def __init__(self, node, module) -> None:
        assert node.target == torch.ops.aten.mm
        self.node = node
        lhs_node = node.args[0]
        rhs_node = node.args[1]

        # only support row-major * row-major now

        # get shape
        lhs_shape = lhs_node.meta["tensor_meta"].shape
        rhs_shape = rhs_node.meta["tensor_meta"].shape

        M, K = lhs_shape[-2:]
        N = rhs_shape[-1]

        align_emb = get_alignment(N)

        self.Output = TensorDescription(cutlass.float16, cutlass.RowMajor, align_emb)

        tile_num_row = 32
        num_warp = tile_num_row * N // align_emb // 32

        self.tile_description = TileDescription(
            threadblock_shape=[tile_num_row, N, 1], stages=1,
            warp_count=[num_warp, 1, 1], math_instruction=None
        )

        epilogue_functor = LinearCombination(
            element_output=self.Output.element,
            epilogue_vector_length=self.Output.alignment,
            element_accumulator=cutlass.float32,
            element_epilogue=cutlass.float32
        )

        self.epilogue_functor = EpilogueVisitTreeDAG(
            elementwise_functor=epilogue_functor,
            element_accumulator=cutlass.float32,
            elements_per_access=self.Output.alignment,
            element_compute=cutlass.float32, element_output=self.Output.element
        )

        self.epilogue_functor.initialize(node)
        self.epilogue_functor.optimize(self.tile_description)

        self.operation = SpmmOperation(
            tile_description=self.tile_description,
            element_input=cutlass.float16, element_accumulator=cutlass.float32,
            alignment_emb=align_emb, alignment_nnz=1, 
            epilogue_functor=self.epilogue_functor
        )

        cutlass_path = os.getenv('CUTLASS_PATH')
        assert cutlass_path is not None, "Environment variable 'CUTLASS_PATH' is not defined."
        cuda_install_path = os.getenv('CUDA_INSTALL_PATH')
        assert cuda_install_path is not None, "Environment variable 'CUDA_INSTALL_PATH' is not defined."
        mlcompiler_path = os.getenv('MLCOMPILER_PATH')
        assert mlcompiler_path is not None, "Environment variable 'CUDA_INSTALL_PATH' is not defined."
        include_paths = [
            cuda_install_path + '/include',
            cutlass_path + '/include',
            cutlass_path + '/tools/util/include',
            cutlass_path + '/tools/library/scripts/pycutlass/src/cpp/include',
            '/opt/conda/lib/python3.8/site-packages/torch/include/',
            mlcompiler_path + '/src/cuda/'
        ]
        compile_options = CompilationOptions(
            ['-std=c++14'], [80, ], include_paths=include_paths
        )

        pycutlass.compiler.add_module([self.operation,], compile_options)

        self.args = self.epilogue_functor.args + list(node.args)
        self.outputs = self.epilogue_functor.outputs
        self.kernel_outputs = self.epilogue_functor.kernel_outputs
        self.output_2_kernel_output = self.epilogue_functor.output_2_kernel_output

        self.stream = None
    
    def __call__(self, *args):
        default_stream = torch.cuda.current_stream()
        if self.stream is not None:
            self.stream.wait_stream(default_stream)
            stream = self.stream
        else:
            stream = torch.cuda.current_stream()
        with torch.cuda.stream(stream):
            lhs = args[-2]
            rhs = args[-1]
            with nvtx.annotate("cutlass_spmm"):
                M, K = lhs.size()
                K, N = rhs.size()
                rhs = rhs.contiguous()
                problem_size = cutlass.MatrixCoord(M, N)
                kwargs = {"problem_size": [M, N]}
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
                arguments = SpmmArguments(
                    operation=self.operation, problem_size=problem_size,
                    csr_matrix=lhs, B=rhs, output_op=output_op
                )

                if self.stream is None: 
                    self.operation.run(arguments, stream=torch.cuda.current_stream().cuda_stream)
                    stream = torch.cuda.current_stream().cuda_stream
                else:
                    self.operation.run(arguments, stream=self.stream.cuda_stream)
                    stream = self.stream.cuda_stream

                results = []
                for output_node in self.outputs:
                    results.append(kwargs["output_" + self.output_2_kernel_output[output_node].name].view(output_node.meta["tensor_meta"].shape))
                
                if self.stream is not None:
                    # without the code below, the caching allocator might reuse
                    # the memory unexpectedly
                    # https://pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html?highlight=stream#torch.Tensor.record_stream
                    for arg in args:
                        if isinstance(arg, torch.Tensor):
                            arg.record_stream(self.stream)
        return results



class FusedGEMM:
    __name__ = "cutlass_gemm_with_visitor"
    def __init__(self, node) -> None:
        assert node.target == torch.ops.aten.mm
        self.node = node
        lhs_node = node.args[0]
        rhs_node = node.args[1]

        # get layout
        # We can only fuse immediate transpose here.
        # For non-immediate transpose / permutations, 
        # case 1: clone + view is between the transpose & mm
        #     the view would break the original continuous and stride dimension
        # case 2: detach, or other transparent ops
        #     these ops will be removed by other compiler passes
        # case 3: other arithmatic ops
        #     mainloop fusion is not yet supported
        if lhs_node.target == torch.ops.aten.t:
            lhs_layout = cutlass.ColumnMajor
            node.replace_input_with(lhs_node, lhs_node.args[0])
        else:
            lhs_layout = cutlass.RowMajor
        
        if rhs_node.target == torch.ops.aten.t:
            rhs_layout = cutlass.ColumnMajor
            node.replace_input_with(rhs_node, rhs_node.args[0])
        else:
            rhs_layout = cutlass.RowMajor
        
        # get shape
        lhs_shape = lhs_node.meta["tensor_meta"].shape
        rhs_shape = rhs_node.meta["tensor_meta"].shape

        M, K = lhs_shape[-2:]
        N = rhs_shape[-1]
        
        self.lhs_layout = lhs_layout
        self.rhs_layout = rhs_layout

        math_inst = MathInstruction(
            instruction_shape=[16, 8, 16],
            element_a=cutlass.float16, element_b=cutlass.float16,
            element_accumulator=cutlass.float32, 
            opcode_class=cutlass.OpClass.TensorOp
        )

        # get max alignment
        if lhs_layout == cutlass.RowMajor:
            align_a = get_alignment(K)
        elif lhs_layout == cutlass.ColumnMajor:
            align_a = get_alignment(M)
        else:
            raise NotImplementedError
        
        if rhs_layout == cutlass.RowMajor:
            align_b = get_alignment(N)
        elif rhs_layout == cutlass.ColumnMajor:
            align_b = get_alignment(K)
        else:
            raise NotImplementedError
        
        align_c = get_alignment(N)

        A = TensorDescription(cutlass.float16, lhs_layout, align_a)
        B = TensorDescription(cutlass.float16, rhs_layout, align_b)
        C = TensorDescription(cutlass.float16, cutlass.RowMajor, align_c)

        # epilogue functor
        epilogue_functor = LinearCombination(
            element_output=C.element, epilogue_vector_length=C.alignment,
            element_accumulator=math_inst.element_accumulator,
            element_epilogue=cutlass.float32)
        
        # Creating the epilogue visitor tree
        self.epilogue_functor = EpilogueVisitTreeDAG(
            elementwise_functor=epilogue_functor, 
            element_accumulator=math_inst.element_accumulator,
            elements_per_access=C.alignment,
            element_compute=cutlass.float32, element_output=C.element
        )

        self.is_direct_output = self.epilogue_functor.initialize(node)
        if not self.is_direct_output:
            self.split_k_mode = "None"
        else:
            self.split_k_mode = "Serial"
        # launch autotuner
        self.gemm_descriptor = GemmConfigDescriptor(
            problem_size=[M, N, K], element_a=cutlass.float16, layout_a=lhs_layout,
            element_b=cutlass.float16, layout_b=rhs_layout, 
            element_c=cutlass.float16, layout_c=cutlass.RowMajor, 
            element_accumulator=cutlass.float32, alignment_a=align_a,
            alignment_b=align_b, alignment_c=align_c,
            split_k_mode=self.split_k_mode
        )
        best_config = autotuner(self.gemm_descriptor)
        
        logging.debug(f"{best_config}")

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

        

        if self.is_direct_output:
            self.operation = GemmOperationUniversal(
                arch=80, tile_description=tile_description,
                A=A, B=B, C=C, epilogue_functor=epilogue_functor,
                swizzling_functor=swizzling_functor, visitor=False
            )
            if self.split_k_mode == "Parallel":
                self.reduction_operation = ReductionOperation(
                    shape=cutlass.MatrixCoord(4, 32 * C.alignment),
                    C=C, element_accumulator=cutlass.float32,
                    element_compute=cutlass.float32,
                    epilogue_functor=epilogue_functor,
                    count=C.alignment
                )
        else:
            self.epilogue_functor.optimize(tile_description)
            self.operation = GemmOperationUniversal(
                arch=80, tile_description=tile_description,
                A=A, B=B, C=C, epilogue_functor=self.epilogue_functor,
                swizzling_functor=swizzling_functor, visitor=True
            )

        if self.split_k_mode == "Parallel":
            pycutlass.compiler.add_module([self.operation, self.reduction_operation])
        else:
            pycutlass.compiler.add_module([self.operation,])

        self.args = self.epilogue_functor.args + list(node.args)
        self.outputs = self.epilogue_functor.outputs
        self.kernel_outputs = self.epilogue_functor.kernel_outputs
        self.output_2_kernel_output = self.epilogue_functor.output_2_kernel_output

        self.stream = None

    def __call__(self, *args):
        default_stream = torch.cuda.current_stream()
        if self.stream is not None:
            self.stream.wait_stream(default_stream)
            stream = self.stream
        else:
            stream = torch.cuda.current_stream()
        with torch.cuda.stream(stream):
            lhs = args[-2]
            rhs = args[-1]
            with nvtx.annotate("cutlass_gemm"):
                if self.is_direct_output:
                    if self.lhs_layout == cutlass.ColumnMajor:
                        lhs = torch.transpose(lhs, -1, -2)
                    if self.rhs_layout == cutlass.ColumnMajor:
                        rhs = torch.transpose(rhs, -1, -2)
                    # it may still contain some reshaping
                    results = [torch.matmul(lhs, rhs).view(self.kernel_outputs[0].meta['tensor_meta'].shape),]
                else:
                    if self.lhs_layout == cutlass.RowMajor:
                        M, K = lhs.size()
                    else:
                        K, M = lhs.size()
                    if self.rhs_layout == cutlass.RowMajor:
                        N = rhs.size(-1)
                    else:
                        N = rhs.size(-2)
                    lhs = lhs.contiguous()
                    rhs = rhs.contiguous()
                    problem_size = cutlass.gemm.GemmCoord(M, N, K)
                    kwargs = {"problem_size": [M, N]}
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

                    if self.is_direct_output:
                        output_tensor = None
                        for key in kwargs.keys():
                            if "output" in key:
                                output_tensor = kwargs[key]
                        output_op = self.operation.epilogue_type(1.0, 0.0)
                        if self.split_k_mode in ["None", "Serial"]:
                            gemm_mode=cutlass.gemm.Mode.Gemm
                        elif self.split_k_mode == "Parallel":
                            gemm_mode= cutlass.gemm.Mode.GemmSplitKParallel
                        else:
                            raise NotImplementedError
                        arguments = GemmArguments(
                            operation=self.operation, problem_size=problem_size,
                            A=lhs, B=rhs, C=output_tensor, D=output_tensor,
                            output_op=output_op, gemm_mode=gemm_mode,
                            split_k_slices=self.split_k_slices
                        )
                        # the code below is no longer required.
                        # as we allocate the workspace under the same stream
                        # of the kernel
                        # if hasattr(arguments, "workspace_buffer"):
                        #     self.workspace_buffer = arguments.workspace_buffer
                    else:
                        output_op = self.operation.epilogue_type(**kwargs)
                        arguments = GemmArguments(
                            operation=self.operation, problem_size=problem_size,
                            A=lhs, B=rhs, C=lhs, D=lhs,
                            output_op=output_op, gemm_mode=cutlass.gemm.Mode.Gemm
                        )
                    if self.stream is None: 
                        self.operation.run(arguments, stream=torch.cuda.current_stream().cuda_stream)
                        stream = torch.cuda.current_stream().cuda_stream
                    else:
                        self.operation.run(arguments, stream=self.stream.cuda_stream)
                        stream = self.stream.cuda_stream
                    
                    if self.split_k_mode == "Parallel":
                        reduction_arguments = ReductionArguments(
                            operation=self.reduction_operation,
                            problem_size=[M, N], partitions=self.split_k_slices,
                            workspace=arguments.ptr_D, destination=output_tensor,
                            source=output_tensor, output_op=output_op)
                        self.reduction_operation.run(reduction_arguments, stream=stream)

                    results = []
                    for output_node in self.outputs:
                        results.append(kwargs["output_" + self.output_2_kernel_output[output_node].name].view(output_node.meta["tensor_meta"].shape))
                    
                if self.stream is not None:
                    # without the code below, the caching allocator might reuse
                    # the memory unexpectedly
                    # https://pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html?highlight=stream#torch.Tensor.record_stream
                    for arg in args:
                        if isinstance(arg, torch.Tensor):
                            arg.record_stream(self.stream)
        return results


permute_2_layout = {
    (0, 1, 2) : cutlass.RowMajor,
    (0, 2, 1) : cutlass.ColumnMajor,
    (1, 0, 2) : cutlass.RowMajor,
    (1, 2, 0) : cutlass.ColumnMajor
}

class FusedBMM:
    __name__ = "cutlass_bmm_with_visitor"
    def __init__(self, node) -> None:
        self.node = node
        assert node.target == torch.ops.aten.bmm
        lhs_node = node.args[0]
        rhs_node = node.args[1]

        lhs_shape = list(lhs_node.meta["tensor_meta"].shape)
        rhs_shape = list(rhs_node.meta["tensor_meta"].shape)

        batch, M, K = lhs_shape
        N = rhs_shape[-1]

        # get layout
        # We can only fuse immediate transpose here.
        # For non-immediate transpose / permutations, 
        # case 1: clone + view is between the transpose & mm
        #     the view would break the original continuous and stride dimension
        # case 2: detach, or other transparent ops
        #     these ops will be removed by other compiler passes
        # case 3: other arithmatic ops
        #     mainloop fusion is not yet supported

        # step 1: get permutation of input multiplicand and multiplier
        lhs_permute = self._trace_permutation(lhs_node, lhs_node)
        rhs_permute = self._trace_permutation(rhs_node, rhs_node)
        self.lhs_permute = lhs_permute
        self.rhs_permute = rhs_permute

        # step 2: get the layout from permutation
        try:
            self.lhs_layout = permute_2_layout[tuple(lhs_permute)]
        except:
            raise NotImplementedError
        
        try:
            self.rhs_layout = permute_2_layout[tuple(rhs_permute)]
        except:
            raise NotImplementedError
        
        math_inst = MathInstruction(
            instruction_shape=[16, 8, 16],
            element_a=cutlass.float16, element_b=cutlass.float16,
            element_accumulator=cutlass.float32,
            opcode_class=cutlass.OpClass.TensorOp
        )

        # get max alignment
        if self.lhs_layout == cutlass.RowMajor:
            align_a = get_alignment(K)
        elif self.lhs_layout == cutlass.ColumnMajor:
            align_a = get_alignment(M)
        else:
            raise NotImplementedError
        
        if self.rhs_layout == cutlass.RowMajor:
            align_b = get_alignment(N)
        elif self.rhs_layout == cutlass.ColumnMajor:
            align_b = get_alignment(K)
        else:
            raise NotImplementedError
        
        align_c = get_alignment(N)

        A = TensorDescription(cutlass.float16, self.lhs_layout, align_a)
        B = TensorDescription(cutlass.float16, self.rhs_layout, align_b)
        C = TensorDescription(cutlass.float16, cutlass.RowMajor, align_c)

        epilogue_functor = LinearCombination(
            element_output=C.element, epilogue_vector_length=C.alignment,
            element_accumulator=math_inst.element_accumulator,
            element_epilogue=cutlass.float32)

        # Creating the epilogue visitor tree
        self.epilogue_functor = EpilogueVisitTreeDAG(
            elementwise_functor=epilogue_functor, 
            element_accumulator=math_inst.element_accumulator,
            elements_per_access=C.alignment,
            element_compute=cutlass.float32, element_output=C.element
        )

        self.epilogue_functor.initialize(node)

        bmm_descriptor = BatchedGemmConfigDescriptor(
            problem_size=[batch, M, N, K], element_a=cutlass.float16, layout_a=self.lhs_layout,
            element_b=cutlass.float16, layout_b=self.rhs_layout,
            element_c=cutlass.float16, layout_c=cutlass.RowMajor,
            element_accumulator=cutlass.float32, alignment_a=align_a,
            alignment_b=align_b, alignment_c=align_c, permute_a=lhs_permute,
            permute_b=rhs_permute
        )
        best_config = autotuner(bmm_descriptor)

        threadblock_shape = [
            best_config['block_x'], best_config['block_y'], best_config['block_z']]
        warp_count = [
            best_config['block_x'] // best_config['warp_x'],
            best_config['block_y'] // best_config['warp_y'],
            best_config['block_z'] // best_config['warp_z']
        ]
        
        stages = best_config['stage']

        swizzling_functor = cutlass.BatchedIdentitySwizzle

        tile_description = TileDescription(
            threadblock_shape, stages, warp_count, math_inst
        )
    
        self.epilogue_functor.optimize(tile_description)

        C = TensorDescription(cutlass.float16, self.epilogue_functor.output_layout, 8)

        self.operation = GemmOperationUniversal(
            arch=80, tile_description=tile_description,
            A=A, B=B, C=C, epilogue_functor=self.epilogue_functor,
            swizzling_functor=swizzling_functor, visitor=True
        )

        self.problem_size = cutlass.gemm.GemmCoord(M, N, K)
        self.batch = batch

        pycutlass.compiler.add_module([self.operation,])

        self.args = self.epilogue_functor.args + list(node.args)
        self.outputs = self.epilogue_functor.outputs
        self.kernel_outputs = self.epilogue_functor.kernel_outputs
        self.output_2_kernel_output = self.epilogue_functor.output_2_kernel_output

        self.stream = None
    
    def _trace_permutation(self, node, root_node):
        """
        helper function to combine several continuous permutations
        it returns the permutation and update the GEMM inputs to the
        unpermuted tensors
        """
        if node.target != torch.ops.aten.permute:
            self.node.replace_input_with(root_node, node)
            return [0, 1, 2]
        else:
            layout = self._trace_permutation(node.args[0], root_node)
            new_layout = [layout[i] for i in node.args[1]]
            return new_layout

    def __call__(self, *args):
        lhs = args[-2]
        rhs = args[-1]
        with nvtx.annotate("cutlass_bmm"):
            lhs = lhs.contiguous()
            rhs = rhs.contiguous()
            if self.epilogue_functor.output_layout == cutlass.RowMajor:
                kwargs = {"problem_size": [self.problem_size.m(), self.problem_size.n()], "batch_size": self.batch}
            else:
                kwargs = {"problem_size": [self.problem_size.n(), self.problem_size.m()], "batch_size": self.batch}
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
            arguments = BatchedGemmPermutedArguments(
                operation=self.operation, problem_size=self.problem_size,
                A=lhs, B=rhs, C=lhs, D=lhs,
                output_op=output_op, gemm_mode=cutlass.gemm.Mode.Batched, batch=self.batch,
                permute_A=self.lhs_permute, permute_B=self.rhs_permute
            )
            if self.stream is None: 
                self.operation.run(arguments, stream=torch.cuda.current_stream().cuda_stream)
            else:
                self.stream.wait_stream(torch.cuda.current_stream())
                self.operation.run(arguments, stream=self.stream.cuda_stream)

            results = []
            for output_node in self.outputs:
                results.append(kwargs["output_" + self.output_2_kernel_output[output_node].name].view(output_node.meta["tensor_meta"].shape))
        return results

################################################################################
# Graph-level pass to fuse ReduceApply Ops
################################################################################
class FusedReduceApply:
    def __init__(self, node) -> None:
        assert node.target == self.target

        # update the softmax shape
        shape = node.meta["tensor_meta"].shape
        if self.reduction_dim_arg_idx is not None:
            reduction_dim = node.args[self.reduction_dim_arg_idx]
            if reduction_dim < 0:
                reduction_dim = len(shape) + reduction_dim
            independent_size = 1
            for idx, dim in enumerate(shape):
                if idx == reduction_dim: continue
                independent_size *= dim
            self.shape = (independent_size, shape[reduction_dim])
            reduction_dim = 1
        else:
            reduction_dim = len(shape) - 1
            independent_size = 1
            for idx, dim in enumerate(shape):
                if idx == reduction_dim: continue
                independent_size *= dim
            self.shape = (independent_size, shape[reduction_dim])
            reduction_dim = 1

        # get alignment
        if self.shape[1] % 8 == 0:
            alignment = 8
        elif self.shape[1] % 4 == 0:
            alignment = 4
        elif self.shape[1] % 2 == 0:
            alignment = 2
        else:
            alignment = 1
        
        self.Input = TensorDescription(cutlass.float16, cutlass.RowMajor, alignment)
        self.Output = TensorDescription(cutlass.float16, cutlass.RowMajor, alignment)

        warp_count = min(max(1, (self.shape[1] + (32 * alignment * 2) - 1) // (32 * alignment * 4)), 4)

        self.tile_description = TileDescription(
            threadblock_shape=[1, self.shape[reduction_dim], 1], stages=1,
            warp_count=[1, warp_count, 1], math_instruction=None
        )

        epilogue_functor = LinearCombination(
            element_output=self.Output.element, 
            epilogue_vector_length=self.Output.alignment,
            element_accumulator=cutlass.float32,
            element_epilogue=cutlass.float32)
        
        self.epilogue_functor = EpilogueVisitTreeDAG(
            elementwise_functor=epilogue_functor, 
            element_accumulator=cutlass.float32,
            elements_per_access=self.Output.alignment,
            element_compute=cutlass.float32, element_output=self.Output.element
        )

        self.epilogue_functor.initialize(node)
        self.epilogue_functor.optimize(self.tile_description)


################################################################################
# Graph-level pass to fuse Softmax kernels
################################################################################

class FusedSoftmax(FusedReduceApply):
    __name__ = "cutlass_softmax_with_visitor"
    target = torch.ops.aten._softmax
    reduction_dim_arg_idx = 1

    def __init__(self, node) -> None:
        super().__init__(node)
        self.operation = SoftmaxOperation(
            input=self.Input, output=self.Output, threadblock_tile=self.tile_description.threadblock_shape,
            warp_count=self.tile_description.warp_count, element_accumulator=cutlass.float32, epilogue_functor=self.epilogue_functor
        )
        cutlass_path = os.getenv('CUTLASS_PATH')
        assert cutlass_path is not None, "Environment variable 'CUTLASS_PATH' is not defined."
        cuda_install_path = os.getenv('CUDA_INSTALL_PATH')
        assert cuda_install_path is not None, "Environment variable 'CUDA_INSTALL_PATH' is not defined."
        mlcompiler_path = os.getenv('MLCOMPILER_PATH')
        assert mlcompiler_path is not None, "Environment variable 'CUDA_INSTALL_PATH' is not defined."
        include_paths = [
            cuda_install_path + '/include',
            cutlass_path + '/include',
            cutlass_path + '/tools/util/include',
            cutlass_path + '/tools/library/scripts/pycutlass/src/cpp/include',
            '/opt/conda/lib/python3.8/site-packages/torch/include/',
            mlcompiler_path + '/src/cuda/'
        ]
        compile_options = CompilationOptions(
            ['-std=c++14'], [80, ], include_paths=include_paths
        )

        pycutlass.compiler.add_module([self.operation,], compile_options)

        self.args = self.epilogue_functor.args + list(node.args)
        self.outputs = self.epilogue_functor.outputs
        self.kernel_outputs = self.epilogue_functor.kernel_outputs
        self.output_2_kernel_output = self.epilogue_functor.output_2_kernel_output

        self.stream = None
    
    def __call__(self, *args):
        kwargs = {"problem_size": [self.shape[0], self.shape[1]]}
        with nvtx.annotate("cutlass_softmax"):
            for output_node in self.kernel_outputs:
                kwargs["output_" + output_node.name] = torch.empty(
                    size=output_node.meta['tensor_meta'].shape,
                    dtype=output_node.meta['tensor_meta'].dtype,
                    device="cuda"
                )
            for idx, epilogue_arg in enumerate(self.epilogue_functor.args):
                kwargs[epilogue_arg.name] = args[idx]

            output_op = self.operation.epilogue_type(**kwargs)            

            arguments = SoftmaxArguments(
                operation=self.operation, problem_size=kwargs["problem_size"],
                input=args[-3], output_op=output_op
            )
            if self.stream is None: 
                self.operation.run(arguments, stream=torch.cuda.current_stream().cuda_stream)
            else:
                self.stream.wait_stream(torch.cuda.current_stream())
                self.operation.run(arguments, stream=self.stream.cuda_stream)

            results = []
            for output_node in self.outputs:
                results.append(kwargs["output_" + self.output_2_kernel_output[output_node].name].view(output_node.meta["tensor_meta"].shape))
        return results


################################################################################
# Graph-level pass to fuse Softmax Backward kernels
################################################################################

class FusedSoftmaxBackward(FusedReduceApply):
    __name__ = "cutlass_softmax_with_visitor"
    target = torch.ops.aten._softmax_backward_data
    reduction_dim_arg_idx = 2
    def __init__(self, node) -> None:
        super().__init__(node)
        self.operation = SoftmaxBackwardOperation(
            input=self.Input, output=self.Output, threadblock_tile=self.tile_description.threadblock_shape,
            warp_count=self.tile_description.warp_count, element_accumulator=cutlass.float32, epilogue_functor=self.epilogue_functor
        )
        cutlass_path = os.getenv('CUTLASS_PATH')
        assert cutlass_path is not None, "Environment variable 'CUTLASS_PATH' is not defined."
        cuda_install_path = os.getenv('CUDA_INSTALL_PATH')
        assert cuda_install_path is not None, "Environment variable 'CUDA_INSTALL_PATH' is not defined."
        mlcompiler_path = os.getenv('MLCOMPILER_PATH')
        assert mlcompiler_path is not None, "Environment variable 'CUDA_INSTALL_PATH' is not defined."
        include_paths = [
            cuda_install_path + '/include',
            cutlass_path + '/include',
            cutlass_path + '/tools/util/include',
            cutlass_path + '/tools/library/scripts/pycutlass/src/cpp/include',
            '/opt/conda/lib/python3.8/site-packages/torch/include/',
            mlcompiler_path + '/src/cuda/'
        ]
        compile_options = CompilationOptions(
            ['-std=c++14'], [80, ], include_paths=include_paths
        )

        pycutlass.compiler.add_module([self.operation,], compile_options)

        self.args = self.epilogue_functor.args + list(node.args)
        self.outputs = self.epilogue_functor.outputs
        self.kernel_outputs = self.epilogue_functor.kernel_outputs
        self.output_2_kernel_output = self.epilogue_functor.output_2_kernel_output
        self.stream = None
    
    def __call__(self, *args):
        kwargs = {"problem_size": [self.shape[0], self.shape[1]]}
        with nvtx.annotate("cutlass_softmax_backward"):
            for output_node in self.kernel_outputs:
                kwargs["output_" + output_node.name] = torch.empty(
                    size=output_node.meta['tensor_meta'].shape,
                    dtype=output_node.meta['tensor_meta'].dtype,
                    device="cuda"
                )
            for idx, epilogue_arg in enumerate(self.epilogue_functor.args):
                kwargs[epilogue_arg.name] = args[idx]

            output_op = self.operation.epilogue_type(**kwargs)            

            arguments = SoftmaxBackwardArguments(
                operation=self.operation, problem_size=kwargs["problem_size"],
                o_softmax=args[-3], grad_softmax=args[-4], output_op=output_op
            )
            if self.stream is None: 
                self.operation.run(arguments, stream=torch.cuda.current_stream().cuda_stream)
            else:
                self.stream.wait_stream(torch.cuda.current_stream())
                self.operation.run(arguments, stream=self.stream.cuda_stream)

            results = []
            for output_node in self.outputs:
                results.append(kwargs["output_" + self.output_2_kernel_output[output_node].name].view(output_node.meta["tensor_meta"].shape))
        return results


################################################################################
# Graph-level pass to fuse LayerNorm kernels
################################################################################

class FusedLayerNormForward(FusedReduceApply):
    __name__ = "cutlass_layernorm_with_visitor"
    target = torch.ops.aten.native_layer_norm
    reduction_dim_arg_idx = None
    def __init__(self, node) -> None:
        super().__init__(node)

        self.operation = LayerNormOperation(
            input=self.Input, output=self.Output, threadblock_tile=self.tile_description.threadblock_shape,
            warp_count=self.tile_description.warp_count, element_accumulator=cutlass.float32, epilogue_functor=self.epilogue_functor
        )
        cutlass_path = os.getenv('CUTLASS_PATH')
        assert cutlass_path is not None, "Environment variable 'CUTLASS_PATH' is not defined."
        cuda_install_path = os.getenv('CUDA_INSTALL_PATH')
        assert cuda_install_path is not None, "Environment variable 'CUDA_INSTALL_PATH' is not defined."
        mlcompiler_path = os.getenv('MLCOMPILER_PATH')
        assert mlcompiler_path is not None, "Environment variable 'CUDA_INSTALL_PATH' is not defined."
        include_paths = [
            cuda_install_path + '/include',
            cutlass_path + '/include',
            cutlass_path + '/tools/util/include',
            cutlass_path + '/tools/library/scripts/pycutlass/src/cpp/include',
            '/opt/conda/lib/python3.8/site-packages/torch/include/',
            mlcompiler_path + '/src/cuda/'
        ]
        compile_options = CompilationOptions(
            ['-std=c++14'], [80, ], include_paths=include_paths
        )

        pycutlass.compiler.add_module([self.operation,], compile_options)

        self.args = self.epilogue_functor.args + list(node.args)
        _, mean, std = list(node.users.keys())

        self.outputs = self.epilogue_functor.outputs
        self.outputs.remove(mean)
        self.outputs.remove(std)
        self.outputs += [mean, std]

        self.kernel_outputs = self.epilogue_functor.kernel_outputs
        self.kernel_outputs.remove(mean)
        self.kernel_outputs.remove(std)
        self.kernel_outputs += [mean, std]

        self.output_2_kernel_output = self.epilogue_functor.output_2_kernel_output
        self.output_2_kernel_output[mean] = mean
        self.output_2_kernel_output[std] = std

        self.stream = None
    
    def __call__(self, *args):
        kwargs = {"problem_size": [self.shape[0], self.shape[1]]}
        with nvtx.annotate("cutlass_layernorm"):
            for output_node in self.kernel_outputs:
                kwargs["output_" + output_node.name] = torch.empty(
                    size=output_node.meta['tensor_meta'].shape,
                    dtype=output_node.meta['tensor_meta'].dtype,
                    device="cuda"
                )
            for idx, epilogue_arg in enumerate(self.epilogue_functor.args):
                kwargs[epilogue_arg.name] = args[idx]
            
            # output_op = self.operation.epilogue_type(**kwargs)
            # TODO: to deprecate
            # softmax_output = torch.empty_like(args[-3])

            output_op = self.operation.epilogue_type(**kwargs)    

            arguments = LayerNormArguments(
                operation=self.operation, problem_size=kwargs["problem_size"],
                input=args[-5], mean=kwargs["output_" + self.kernel_outputs[-2].name],
                std=kwargs["output_" + self.kernel_outputs[-1].name],
                output_op=output_op
            )
            if self.stream is None: 
                self.operation.run(arguments, stream=torch.cuda.current_stream().cuda_stream)
            else:
                self.stream.wait_stream(torch.cuda.current_stream())
                self.operation.run(arguments, stream=self.stream.cuda_stream)

            results = []
            for output_node in self.outputs:
                results.append(kwargs["output_" + self.output_2_kernel_output[output_node].name].view(output_node.meta["tensor_meta"].shape))

            # results_origin = [view_4, view_2]
        return results



################################################################################
# Graph-level pass to fuse LayerNorm kernels
################################################################################

class FusedLayerNormBackward(FusedReduceApply):
    __name__ = "cutlass_layernorm_backward_with_visitor"
    target = torch.ops.aten.native_layer_norm_backward
    reduction_dim_arg_idx = None
    def __init__(self, node) -> None:
        super().__init__(node)

        self.operation = LayerNormBackwardOperation(
            input=self.Input, output=self.Output, threadblock_tile=self.tile_description.threadblock_shape,
            warp_count=self.tile_description.warp_count, element_accumulator=cutlass.float32, epilogue_functor=self.epilogue_functor
        )
        cutlass_path = os.getenv('CUTLASS_PATH')
        assert cutlass_path is not None, "Environment variable 'CUTLASS_PATH' is not defined."
        cuda_install_path = os.getenv('CUDA_INSTALL_PATH')
        assert cuda_install_path is not None, "Environment variable 'CUDA_INSTALL_PATH' is not defined."
        mlcompiler_path = os.getenv('MLCOMPILER_PATH')
        assert mlcompiler_path is not None, "Environment variable 'CUDA_INSTALL_PATH' is not defined."
        include_paths = [
            cuda_install_path + '/include',
            cutlass_path + '/include',
            cutlass_path + '/tools/util/include',
            cutlass_path + '/tools/library/scripts/pycutlass/src/cpp/include',
            '/opt/conda/lib/python3.8/site-packages/torch/include/',
            mlcompiler_path + '/src/cuda/'
        ]
        compile_options = CompilationOptions(
            ['-std=c++14'], [80, ], include_paths=include_paths
        )

        pycutlass.compiler.add_module([self.operation,], compile_options)

        self.args = self.epilogue_functor.args + list(node.args)

        self.outputs = self.epilogue_functor.outputs
        self.kernel_outputs = self.epilogue_functor.kernel_outputs
        self.output_2_kernel_output = self.epilogue_functor.output_2_kernel_output

        self.stream = None
    
    def __call__(self, *args):
        kwargs = {"problem_size": [self.shape[0], self.shape[1]]}
        with nvtx.annotate("cutlass_layernorm_backward"):
            for output_node in self.kernel_outputs:
                kwargs["output_" + output_node.name] = torch.empty(
                    size=output_node.meta['tensor_meta'].shape,
                    dtype=output_node.meta['tensor_meta'].dtype,
                    device="cuda"
                )
            for idx, epilogue_arg in enumerate(self.epilogue_functor.args):
                kwargs[epilogue_arg.name] = args[idx]
            
            # output_op = self.operation.epilogue_type(**kwargs)
            # TODO: to deprecate
            # softmax_output = torch.empty_like(args[-3])

            output_op = self.operation.epilogue_type(**kwargs)    

            arguments = LayerNormBackwardArguments(
                operation=self.operation, problem_size=kwargs["problem_size"],
                gamma=args[-3], grad_layernorm=args[-8], x=args[-7], mean=args[-5],
                std=args[-4], output_op=output_op
            )
            if self.stream is None: 
                self.operation.run(arguments, stream=torch.cuda.current_stream().cuda_stream)
            else:
                self.stream.wait_stream(torch.cuda.current_stream())
                self.operation.run(arguments, stream=self.stream.cuda_stream)

            results = []
            for output_node in self.outputs:
                results.append(kwargs["output_" + self.output_2_kernel_output[output_node].name].view(output_node.meta["tensor_meta"].shape))

            # results_origin = [view_4, view_2]
        return results

def get_topological_index(graph, node):
    """
    Get the node index in topological order
    """
    for idx, node_ in enumerate(graph.nodes):
        if node == node_:
            return idx

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


def get_valid_inserting_range(graph, args, outputs):
    # TODO: this can potentially be optimized
    min_idx = 0
    min_node = None
    max_idx = 1e+16
    for arg in args:
        if isinstance(arg, fx.Node):
            idx = get_topological_index(graph, arg)
            if idx > min_idx:
                min_idx = idx
                min_node = arg
    
    for output in outputs:
        if isinstance(output, fx.Node):
            idx = get_topological_index(graph, output)
            if idx < max_idx:
                max_idx = idx
    return min_idx, max_idx
    

def pass_gemm_fusion(module, graph, verbose=True, disabled_ops=[]):
    """
    Fuse GEMM kernel with its epilogues
    """

    for node in tqdm(graph.nodes):
        if node.op == "call_function":
            if node.target in disabled_ops: continue
            if node.target == torch.ops.aten.mm:
                logging.debug(f"====================={node}======================")
                update_topological_index(graph)
                # check for spmm case
                # currently, we use a simple heuristic that constant tensor
                # is used in spmm case
                if node.args[0].op == "get_attr" or node.args[1].op == "get_attr" or node.args[0].meta["tensor_meta"].stride==(0, 0) or node.args[1].meta["tensor_meta"].stride==(0, 0):
                    node.meta["sparse"] = True
                    fused_gemm = FusedSpmm(node, module)
                else:
                    node.meta["sparse"] = False
                    fused_gemm = FusedGEMM(node)
                inserting_point = fused_gemm.epilogue_functor.root
                inserting_idx = get_topological_index(graph, inserting_point)

                for output in fused_gemm.outputs:
                    idx = get_topological_index(graph, output)
                    if idx < inserting_idx:
                        inserting_idx = idx
                        inserting_point = output

                graph.inserting_after(inserting_point)
                fused_node = graph.call_function(fused_gemm, args=tuple(fused_gemm.args))
                fused_node.meta = {}
                fused_node.meta['tensor_meta'] = fused_gemm.epilogue_functor.root.meta['tensor_meta']._replace()
                graph.inserting_after(fused_node)

                for idx, output_node in enumerate(fused_gemm.outputs):
                    get_item_node = graph.call_function(operator.getitem, args=(fused_node, idx))
                    get_item_node.meta["tensor_meta"] = output_node.meta["tensor_meta"]._replace()
                    get_item_node.meta["unfusible"] = True
                    graph.inserting_after(get_item_node)
                    output_node.replace_all_uses_with(get_item_node)
                    erase_node_recursive(graph, output_node)

            elif node.target == torch.ops.aten.bmm:
                logging.debug(f"====================={node}======================")
                update_topological_index(graph)
                node.meta["sparse"] = False
                fused_bmm = FusedBMM(node)

                inserting_point = fused_bmm.epilogue_functor.root
                inserting_idx = get_topological_index(graph, inserting_point)

                for output in fused_bmm.outputs:
                    idx = get_topological_index(graph, output)
                    if idx < inserting_idx:
                        inserting_idx = idx
                        inserting_point = output

                graph.inserting_after(inserting_point)
                fused_node = graph.call_function(fused_bmm, args=tuple(fused_bmm.args))
                fused_node.meta = {}
                fused_node.meta['tensor_meta'] = fused_bmm.epilogue_functor.root.meta['tensor_meta']._replace()
                graph.inserting_after(fused_node)

                for idx, output_node in enumerate(fused_bmm.outputs):
                    get_item_node = graph.call_function(operator.getitem, args=(fused_node, idx))
                    get_item_node.meta["tensor_meta"] = output_node.meta["tensor_meta"]._replace()
                    get_item_node.meta["unfusible"] = True
                    graph.inserting_after(get_item_node)
                    output_node.replace_all_uses_with(get_item_node)
                    erase_node_recursive(graph, output_node)

            elif node.target == torch.ops.aten._softmax:
                logging.debug(f"====================={node}======================")
                update_topological_index(graph)
                fused_softmax = FusedSoftmax(node)
                inserting_point = fused_softmax.epilogue_functor.root
                inserting_idx = get_topological_index(graph, inserting_point)

                for output in fused_softmax.outputs:
                    idx = get_topological_index(graph, output)
                    if idx < inserting_idx:
                        inserting_idx = idx
                        inserting_point = output
                
                graph.inserting_after(inserting_point)
                fused_node = graph.call_function(fused_softmax, args=tuple(fused_softmax.args))
                fused_node.meta = {}
                fused_node.meta['tensor_meta'] = fused_softmax.epilogue_functor.root.meta['tensor_meta']._replace()
                graph.inserting_after(fused_node)

                for idx, output_node in enumerate(fused_softmax.outputs):
                    get_item_node = graph.call_function(operator.getitem, args=(fused_node, idx))
                    get_item_node.meta["tensor_meta"] = output_node.meta["tensor_meta"]._replace()
                    get_item_node.meta["unfusible"] = True
                    graph.inserting_after(get_item_node)
                    output_node.replace_all_uses_with(get_item_node)
                    erase_node_recursive(graph, output_node)
            
            elif node.target == torch.ops.aten._softmax_backward_data:
                logging.debug(f"====================={node}======================")
                update_topological_index(graph)
                fused_softmax = FusedSoftmaxBackward(node)
                inserting_point = fused_softmax.epilogue_functor.root
                inserting_idx = get_topological_index(graph, inserting_point)

                for output in fused_softmax.outputs:
                    idx = get_topological_index(graph, output)
                    if idx < inserting_idx:
                        inserting_idx = idx
                        inserting_point = output
                
                graph.inserting_after(inserting_point)
                fused_node = graph.call_function(fused_softmax, args=tuple(fused_softmax.args))
                fused_node.meta = {}
                fused_node.meta['tensor_meta'] = fused_softmax.epilogue_functor.root.meta['tensor_meta']._replace()
                graph.inserting_after(fused_node)

                for idx, output_node in enumerate(fused_softmax.outputs):
                    get_item_node = graph.call_function(operator.getitem, args=(fused_node, idx))
                    get_item_node.meta["tensor_meta"] = output_node.meta["tensor_meta"]._replace()
                    get_item_node.meta["unfusible"] = True
                    graph.inserting_after(get_item_node)
                    output_node.replace_all_uses_with(get_item_node)
                    erase_node_recursive(graph, output_node)
            
            elif node.target == torch.ops.aten.native_layer_norm:
                logging.debug(f"====================={node}======================")
                update_topological_index(graph)
                fused_layernorm = FusedLayerNormForward(node)
                inserting_point = fused_layernorm.epilogue_functor.root
                inserting_idx = get_topological_index(graph, inserting_point)

                for output in fused_layernorm.outputs:
                    idx = get_topological_index(graph, output)
                    if idx < inserting_idx:
                        inserting_idx = idx
                        inserting_point = output
                
                graph.inserting_after(inserting_point)
                fused_node = graph.call_function(fused_layernorm, args=tuple(fused_layernorm.args))
                fused_node.meta = {}
                fused_node.meta['tensor_meta'] = fused_layernorm.epilogue_functor.root.meta['tensor_meta']._replace()
                graph.inserting_after(fused_node)
                for idx, output_node in enumerate(fused_layernorm.outputs):
                    get_item_node = graph.call_function(operator.getitem, args=(fused_node, idx))
                    get_item_node.meta["tensor_meta"] = output_node.meta["tensor_meta"]._replace()
                    get_item_node.meta["unfusible"] = True
                    graph.inserting_after(get_item_node)
                    output_node.replace_all_uses_with(get_item_node)
                    erase_node_recursive(graph, output_node)
            
            elif node.target == torch.ops.aten.native_layer_norm_backward:
                if "unfusible" in node.meta.keys():
                    continue
                logging.debug(f"====================={node}======================")
                update_topological_index(graph)

                fused_layernorm = FusedLayerNormBackward(node)
                inserting_point = fused_layernorm.epilogue_functor.root
                inserting_idx = get_topological_index(graph, inserting_point)

                for output in fused_layernorm.outputs:
                    idx = get_topological_index(graph, output)
                    if idx < inserting_idx:
                        inserting_idx = idx
                        inserting_point = output
                
                graph.inserting_after(inserting_point)
                fused_node = graph.call_function(fused_layernorm, args=tuple(fused_layernorm.args))
                fused_node.meta = {}
                fused_node.meta['tensor_meta'] = fused_layernorm.epilogue_functor.root.meta['tensor_meta']._replace()
                graph.inserting_after(fused_node)
                for idx, output_node in enumerate(fused_layernorm.outputs):
                    get_item_node = graph.call_function(operator.getitem, args=(fused_node, idx))
                    get_item_node.meta["tensor_meta"] = output_node.meta["tensor_meta"]._replace()
                    get_item_node.meta["unfusible"] = True
                    graph.inserting_after(get_item_node)
                    output_node.replace_all_uses_with(get_item_node)
                    erase_node_recursive(graph, output_node)



    try:
        graph.lint()
        graph.eliminate_dead_code()
    except:
        pass_dead_code_elimination_non_topo(module, graph)      
        graph.eliminate_dead_code()
