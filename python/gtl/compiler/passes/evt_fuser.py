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
from cutlass import DataType
import cutlass
from cutlass.shape import GemmCoord, MatrixCoord
from torch.fx import Graph, Node
import torch
from cutlass.backend.evt.frontend.frontend_base import EVTFrontendBase
from cutlass.backend.evt.ir.tensor import Tensor as fakeTensor
from cutlass import LayoutType, TensorDescription
from cutlass.backend.library import FunctionalOp, ActivationOp
from typing import Any, Union
import pdb
from torch.fx.graph_module import GraphModule
from torch._functorch.partitioners import _extract_graph_with_inputs_outputs
from cutlass.backend.gemm_operation import GemmArguments
import operator
from cutlass.backend import compiler
from cutlass.backend.utils.datatypes import torch_to_cutlass
from gtl.ops.softmax import SoftmaxOperation, SoftmaxArguments
from gtl.ops.softmax_backward import SoftmaxBackwardOperation, SoftmaxBackwardArguments
from gtl.ops.layernorm import LayerNormOperation, LayerNormArguments
from gtl.ops.layernorm_backward import LayerNormBackwardOperation, LayerNormBackwardArguments
from gtl.ops.spmm import SpmmOperation, SpmmArguments
from cutlass.backend.compiler import CompilationOptions
from torch.fx.passes.tools_common import legalize_graph
import os
from torch.profiler import profile, ProfilerActivity, record_function
from copy import deepcopy
from gtl.ops.bmm import BmmArguments2x

from torch.utils._python_dispatch import _pop_mode_temporarily, _len_torch_dispatch_stack
from contextlib import nullcontext
from gtl.compiler.autotuner.gemm_tuner import MMTuner
from gtl.compiler.autotuner.bmm_tuner import BMMTuner
from gtl.compiler.autotuner.reduce_apply_tuner import SoftmaxTuner, SoftmaxBackwardTuner, LayerNormTuner, LayerNormBackwardTuner
from gtl.compiler.passes.pass_decomposition import spmm


class FusedOpBase:
    """
    The base implementation of all the fused operators
    """
    def __init__(self, node: Node, epilogue_visitor) -> None:
        # assert node.target == self.target
        self.node = node

        # Outputs and inputs in the original graph
        self.outputs = epilogue_visitor.outputs
        self.output2store = epilogue_visitor.output2store
        self.inputs = [
            input for input in epilogue_visitor.inputs 
            if input.target != self.node.target]
        self.post_reshape_permute = epilogue_visitor.post_reshape_permute

        # Stream
        self.stream = None

        #
        # Debug & Profile helpers
        #
        self.mode = os.environ.get('MODE', 'RUN')
        # Flag to avoid running this partition multiple times
        self.launched = False
        self.partition_names = epilogue_visitor.partition_names
        if self.mode == 'DEBUG':
            self.reference_module: GraphModule = epilogue_visitor.reference_gm
            self.reference_module.recompile()
        self.rtol = 1e-2
        self.atol = 1e-3

        self.warmup_iters = 20
        self.profile_iters = 20

    def reference(self, args, visitor_args):
        raise NotImplementedError()

    def pass_rate(self, out, ref, criteria):
        return torch.sum(
            torch.isclose(out, ref, **criteria)
        ) / out.numel()

    def reference_base(self, mainloop_args: list, mainloop_target, visitor_args_, args, inductor_profile=True):
        visitor_args = deepcopy(visitor_args_)
        # Step 1: get inputs
        
        inputs = {}
        for node in self.reference_module.graph.nodes:
            if node.op == "placeholder" and node.name != "accum":
                inputs[node.name] = visitor_args[node.name]
        
        def tmp_call_reference(mainloop_args, inputs):
            accum = mainloop_target(*mainloop_args)
            return self.reference_module(accum=accum, **inputs)
        
        output_tensors = tmp_call_reference(mainloop_args, inputs)

        # Step 2: get output orders
        output_node = [node for node in self.reference_module.graph.nodes if node.op == "output"][0]
        output_node_names = [self.output2store[self.get_output_name(arg)] for arg in output_node.args[0]]

        for name, ref in zip(output_node_names, output_tensors):
            out = visitor_args[name].contiguous().view(-1)
            ref = ref.contiguous().view(-1)
            try:
                assert torch.allclose(out, ref, rtol=self.rtol)
            except:
                # Print the two tensors
                print(f"!!!!!!!!!!!!!!result mismatch: {name}!!!!!!!!!!!!!!")
                print(out)
                print(ref)
                if out.dtype == torch.bool:
                    continue
                print("maximum abs error: ", torch.max(torch.abs(out-ref)))
                print("maximum relative error: ", torch.max(torch.abs(out-ref)/torch.abs(ref)))
                print("atol passed rate: ", self.pass_rate(out, ref, {"atol": self.atol}))
                print("rtol passed rate: ", self.pass_rate(out, ref, {"rtol": self.rtol}))
                # breakpoint()
        
        # Step 3: profiling
        
        ## Profile the baseline
        print("############### Torch Compile ###############")
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
            with record_function("torch"):
                tmp_call_reference(mainloop_args, inputs)
            
        print(prof.key_averages().table(sort_by="cuda_time_total"))

        if inductor_profile:
            ## Profile the basic torch compile
            print("############### Torch ###############")
            inductor_fn = torch.compile(tmp_call_reference)
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
                with record_function("torch"):
                    inductor_fn(mainloop_args, inputs)
            
            print(prof.key_averages().table(sort_by="cuda_time_total"))
        ## Profile the optimized kernel
        print("############### EVT ###############")
        torch.cuda.synchronize()
        stream = torch.cuda.current_stream()
        with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
            with record_function("evt"):
                self.call(visitor_args, stream, *args)
        print(prof.key_averages().table(sort_by="cuda_time_total"))
        breakpoint()

    def get_output_name(self, output):
        if output.target == operator.getitem and output.args[0] == self.node and output.args[1] == 0:
            return "accum"
        elif output.target == operator.getitem and output.args[0].target == "accum" and output.args[1] == 0:
            return "accum"
        if output.target == torch.ops.aten.clone:
            name = output.args[0].name
        elif output.target == self.target:
            name = "accum"
        else:
            name = output.name
        return name

    def call(self, visitor_args, stream, *args) -> Any:
        raise NotImplementedError("Should be Overwritten by child class")
    
    def __call__(self, *args_):
        args = []
        for arg in args_:
            if isinstance(arg, torch.Tensor):
                try:
                    args.append(arg.contiguous())
                except:
                    args.append(arg)
            else:
                args.append(arg)

        default_stream = torch.cuda.current_stream()
        if self.stream is not None:
            self.stream.wait_stream(default_stream)
            stream = self.stream
        else:
            stream = default_stream
        
        with torch.cuda.stream(stream):
            # Create the output nodes
            visitor_args = {}
            for output_node in self.outputs:
                output_name = self.get_output_name(output_node)
                if output_name not in self.output2store:
                    self.output2store[output_name] = output_name
                if output_name in self.output2store and output_name != self.output2store[output_name]:
                    continue
                if output_node.target in [torch.ops.aten.sum]:
                    visitor_args[output_name] = torch.zeros(
                        size=output_node.meta['tensor_meta'].shape,
                        dtype=output_node.meta['tensor_meta'].dtype,
                        device="cuda"
                    )
                else:
                    visitor_args[output_name] = torch.empty(
                        size=output_node.meta['tensor_meta'].shape,
                        dtype=output_node.meta['tensor_meta'].dtype,
                        device="cuda"
                    )
            # Register the inputs
            for idx, input in enumerate(self.inputs):
                visitor_args[input.name] = args[idx]

            self.call(visitor_args, stream, *args)

            if self.stream is not None:
                # without the code below, the caching allocator might reuse
                # the memory unexpectedly
                # https://pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html?highlight=stream#torch.Tensor.record_stream
                for arg in args:
                    if isinstance(arg, torch.Tensor):
                        arg.record_stream(self.stream)
            
            # get the results
            results = []
            for output in self.outputs:
                name = self.get_output_name(output)
                output_tensor = visitor_args[self.output2store[name]]
                if name in self.post_reshape_permute:
                    shape, indices = self.post_reshape_permute[name]
                    output_tensor = output_tensor.view(shape)
                    output_tensor = torch.ops.aten.permute(output_tensor, indices)
                results.append(output_tensor)
            
            if self.mode == "DEBUG" and not self.launched:
                print(f"======================================================")
                print(f"######################################################")
                print("Debugging: ", self.partition_names)
                print(f"######################################################")
                print(f"======================================================")
                
                self.launched = True
                # Step 1: Verify the result
                self.reference(args, visitor_args)
            return results
    
    def get_src(self, node):
        if node.target in self.mainloop_fusion_targets:
            return node.args[0]
        else:
            return node
    
    def get_alignment(self, dim):
        if dim % 8 == 0: return 8
        elif dim % 4 == 0: return 4
        elif dim % 2 == 0: return 2
        else: return 1


class FusedGEMM(FusedOpBase):
    """
    The fused operator of GEMM epilogue_visitor(A*B, args)
    """
    __name__ = "cutlass_gemm_with_visitor"
    target = torch.ops.aten.mm
    mainloop_fusion_targets = [torch.ops.aten.permute]
    def __init__(self, node: Node, epilogue_visitor) -> None:
        super().__init__(node, epilogue_visitor)
        # Get A and B
        lhs_node = node.args[0]
        rhs_node = node.args[1]
        self.lhs_layout = self.get_src_layout(lhs_node)
        self.rhs_layout = self.get_src_layout(rhs_node)

        self.A = self.get_src(lhs_node)
        self.B = self.get_src(rhs_node)
        self.args = [self.A, self.B]

        # Get problem size
        lhs_shape = lhs_node.meta["tensor_meta"].shape
        rhs_shape = rhs_node.meta["tensor_meta"].shape

        M, K = lhs_shape[-2:]
        N = rhs_shape[-1]

        self.problem_size = GemmCoord(M, N, K)
        # Plan
        plan = cutlass.op.Gemm(
            element=lhs_node.meta["tensor_meta"].dtype,
            layout_A=self.lhs_layout, layout_B=self.rhs_layout,
            layout_C=cutlass.LayoutType.RowMajor,
            element_accumulator=torch.float32
        )

        # Get candidate tds
        with (_pop_mode_temporarily() 
            if _len_torch_dispatch_stack() > 0 else nullcontext()):
            autotuner = MMTuner(plan, self.problem_size, epilogue_visitor)
            td, swizzle, stages = autotuner.get_best_config()
        plan.tile_description = td
        plan.swizzling_functor = swizzle

        epilogue_visitor.epilogue_stages = stages
        # Register epilogue
        plan.epilogue_visitor = epilogue_visitor

        self.plan = plan

        # Get Alignment
        if self.lhs_layout == cutlass.LayoutType.RowMajor:
            self.align_A = self.get_alignment(K)
        else:
            self.align_A = self.get_alignment(M)
        
        if self.rhs_layout == cutlass.LayoutType.RowMajor:
            self.align_B = self.get_alignment(N)
        else:
            self.align_B = self.get_alignment(K)
        self.align_C = self.get_alignment(N)

        # Get the underlying kernel
        self.operation = self.plan.compile(
            # tile_description=best_td,
            alignment_A=self.align_A, alignment_B=self.align_B, 
            alignment_C=self.align_C)
    
    def reference(self, args, visitor_args):
        A = args[-2]
        B = args[-1]
        def target_fn(A, B):
            if self.lhs_layout == cutlass.LayoutType.ColumnMajor:
                A = torch.transpose(A, -1, -2)
            if self.rhs_layout == cutlass.LayoutType.ColumnMajor:
                B = torch.transpose(B, -1, -2)
            return self.target(A, B)
        super().reference_base([A,B], target_fn, visitor_args, args)
    
    def call(self, visitor_args, stream, *args) -> Any:
        A = args[-2].contiguous()
        B = args[-1].contiguous()
        if self.lhs_layout == cutlass.LayoutType.ColumnMajor:
            A = torch.transpose(A, -1, -2)
        if self.rhs_layout == cutlass.LayoutType.ColumnMajor:
            B = torch.transpose(B, -1, -2)

        arguments = GemmArguments(
            operation=self.operation, problem_size=self.problem_size,
            A=A, B=B, C=None, D=None,
            output_op = self.operation.epilogue_type(visitor_args),
            gemm_mode=cutlass.GemmUniversalMode.Gemm
        )

        self.operation.run(arguments, stream=stream.cuda_stream)
    
    def get_src_layout(self, node: Node):
        if node.target == torch.ops.aten.permute:
            indices = node.args[1]
            if indices == [0, 1]:
                return cutlass.LayoutType.RowMajor
            elif indices == [1, 0]:
                return cutlass.LayoutType.ColumnMajor
            else:
                raise ValueError(f"Invalid permutation: {indices}")
        else:
            return cutlass.LayoutType.RowMajor


class FusedBMM(FusedOpBase):
    """
    The fused operator of BMM epilogue_visitor(A*B, args)
    """
    __name__ = "cutlass_bmm_with_visitor"
    target = torch.ops.aten.bmm
    mainloop_fusion_targets = [torch.ops.aten.permute]

    def __init__(self, node: Node, epilogue_visitor) -> None:
        super().__init__(node, epilogue_visitor)

        # Whether the problem is tranposed
        self.tranposed = epilogue_visitor.transposed

        # Get A and B
        lhs_node = node.args[0]
        rhs_node = node.args[1]

        # Switch the lhs and rhs nodes
        if self.tranposed:
            lhs_node, rhs_node = rhs_node, lhs_node

        self.lhs_layout, self.permute_A = self.get_src_layout(lhs_node)
        self.rhs_layout, self.permute_B = self.get_src_layout(rhs_node)

        self.A = self.get_src(lhs_node)
        self.B = self.get_src(rhs_node)
        self.args = [self.A, self.B]

        # Get problem size
        lhs_shape = lhs_node.meta["tensor_meta"].shape
        rhs_shape = rhs_node.meta["tensor_meta"].shape

        if self.tranposed:
            lhs_shape = [lhs_shape[i] for i in [0, 2, 1]]
            rhs_shape = [rhs_shape[i] for i in [0, 2, 1]]

        M, K = lhs_shape[-2:]
        N = rhs_shape[-1]
        batch_count = lhs_shape[0]

        self.problem_size = GemmCoord(M, N, K)

        # Plan
        plan = cutlass.op.Gemm(
            element=lhs_node.meta["tensor_meta"].dtype,
            layout_A=self.lhs_layout, layout_B=self.rhs_layout,
            layout_C=cutlass.LayoutType.RowMajor,
            element_accumulator=torch.float32
        )

        with (_pop_mode_temporarily() 
            if _len_torch_dispatch_stack() > 0 else nullcontext()):
            autotuner = BMMTuner(
                plan, batch_count, self.problem_size,
                self.permute_A, self.permute_B, epilogue_visitor)
            td, swizzle, stages = autotuner.get_best_config()
        plan.tile_description = td
        plan.swizzling_functor = swizzle
        epilogue_visitor.epilogue_stages = stages
        # Register epilogue
        plan.epilogue_visitor = epilogue_visitor

        self.plan = plan

        # Get Alignment
        if self.lhs_layout == cutlass.LayoutType.RowMajor:
            self.align_A = self.get_alignment(K)
        else:
            self.align_A = self.get_alignment(M)
        
        if self.rhs_layout == cutlass.LayoutType.RowMajor:
            self.align_B = self.get_alignment(N)
        else:
            self.align_B = self.get_alignment(K)
        self.align_C = self.get_alignment(N)

        # Get the underlying kernel
        self.operation = self.plan.compile(
            # tile_description=best_td,
            alignment_A=self.align_A, alignment_B=self.align_B, 
            alignment_C=self.align_C)
    
    def reference(self, args, visitor_args):
        A = args[-2]
        B = args[-1]
        if self.tranposed:
            A, B = B, A
            permute_A = [self.permute_B[i] for i in [0, 2, 1]]
            permute_B = [self.permute_A[i] for i in [0, 2, 1]]
        else:
            permute_A = self.permute_A
            permute_B = self.permute_B
        def target_fn(A, B):
            A = torch.ops.aten.permute(A, permute_A)
            B = torch.ops.aten.permute(B, permute_B)
            return self.target(A, B)
        super().reference_base([A,B], target_fn, visitor_args, args)
    
    def call(self, visitor_args, stream, *args) -> Any:
        A = args[-2]
        B = args[-1]
        if self.permute_A in [[1, 2, 0], [2, 1, 0]]:
            A = torch.ops.aten.permute(A, self.permute_A).contiguous()
            permute_A = [0, 1, 2]
        else:
            permute_A = self.permute_A
        if self.permute_B in [[1, 2, 0], [2, 1, 0]]:
            B = torch.ops.aten.permute(B, self.permute_B).contiguous()
            permute_B = [0, 1, 2]
        else:
            permute_B = self.permute_B
        
        arguments = BmmArguments2x(
            operation=self.operation, problem_size=self.problem_size,
            A=A, B=B, C=None, D=None, permute_A=permute_A, permute_B=permute_B,
            output_op = self.operation.epilogue_type(visitor_args)
        )

        self.operation.run(arguments, stream=stream.cuda_stream)

    def get_src_layout(self, node: Node):
        if node.target == torch.ops.aten.permute:
            indices = node.args[1]
        else:
            indices = [0, 1, 2]

        if self.tranposed:
            indices = [indices[i] for i in [0, 2, 1]]
        
        if indices in [[0, 1, 2], [1, 0, 2]]:
            return cutlass.LayoutType.RowMajor, indices
        elif indices in [[0, 2, 1], [2, 0, 1]]:
            return cutlass.LayoutType.ColumnMajor, indices
        else:
            # Force RowMajor and set them contiguous
            return cutlass.LayoutType.RowMajor, indices


class FusedSoftmax(FusedOpBase):
    """
    The fused operator of Softmax epilogue_visitor(_softmax, args)
    """
    __name__ = "cutlass_softmax_with_visitor"
    target = torch.ops.aten._softmax
    mainloop_fusion_targets = []

    def __init__(self, node: Node, epilogue_visitor) -> None:
        super().__init__(node, epilogue_visitor)

        # Get input
        input = node.args[0]
        self.args = [input]
        shape = node.meta["tensor_meta"].shape
        reduction_dim = node.args[1]
        if reduction_dim < 0:
            reduction_dim = len(shape) + reduction_dim
        self.problem_size = MatrixCoord(shape[0], shape[2])

        # Get alignment
        alignment = self.get_alignment(shape[-1])
        element = torch_to_cutlass(node.meta["tensor_meta"].dtype)

        with (_pop_mode_temporarily() 
            if _len_torch_dispatch_stack() > 0 else nullcontext()):
            autotuner = SoftmaxTuner(
                epilogue_visitor, self.problem_size, dtype=node.meta["tensor_meta"].dtype
            )
            rows_per_cta, warp_count, cache_input = autotuner.get_best_config()

        # Get softmax operation
        self.operation = SoftmaxOperation(
            input=TensorDescription(element, LayoutType.RowMajor, alignment),
            rows_per_cta=rows_per_cta, num_columns=self.problem_size.column, num_rows=self.problem_size.row, 
            warp_count=warp_count, epilogue_visitor=epilogue_visitor, cache_input=cache_input
        )
        # TODO: hardcode
        MLCOMPILER_SRC_DIR = '/workspace/SEAL-PICASSO-ML-Compiler/src/cuda'
        include_paths = [
            MLCOMPILER_SRC_DIR
        ] + compiler.default_include_paths

        compile_options = CompilationOptions(
            compiler.default_compile_options, 80, include_paths=include_paths
        )
        compiler.add_module([self.operation], compile_options)
    
    def reference(self, args, visitor_args):
        input = args[-1]
        def target_fn(input):
            return self.target(input, -1, False)
        super().reference_base([input], target_fn, visitor_args, args)
    
    def call(self, visitor_args, stream, *args) -> Any:
        input = args[-1]

        arguments = SoftmaxArguments(
            self.operation, self.problem_size,
            input, self.operation.epilogue_type(visitor_args)
        )

        self.operation.run(arguments, stream=stream.cuda_stream)


class FusedSoftmaxBackward(FusedOpBase):
    """
    The fused operator of Softmax epilogue_visitor(_softmax_backward_data, args)
    """
    __name__ = "cutlass_softmax_backward_with_visitor"
    target = torch.ops.aten._softmax_backward_data
    mainloop_fusion_targets = []

    def __init__(self, node: Node, epilogue_visitor) -> None:
        super().__init__(node, epilogue_visitor)

        # Get input
        grad, softmax, reduction_dim = node.args[:3]
        self.args = [grad, softmax]
        shape = node.meta["tensor_meta"].shape
        if reduction_dim < 0:
            reduction_dim = len(shape) + reduction_dim
        self.problem_size = MatrixCoord(shape[0], shape[2])

        # Get alignment
        alignment = self.get_alignment(shape[-1])
        element = torch_to_cutlass(node.meta["tensor_meta"].dtype)

        with (_pop_mode_temporarily() 
            if _len_torch_dispatch_stack() > 0 else nullcontext()):
            autotuner = SoftmaxBackwardTuner(
                epilogue_visitor, self.problem_size, dtype=node.meta["tensor_meta"].dtype
            )
            rows_per_cta, warp_count, cache_input = autotuner.get_best_config()

        # Get softmax operation
        self.operation = SoftmaxBackwardOperation(
            input=TensorDescription(element, LayoutType.RowMajor, alignment),
            rows_per_cta=rows_per_cta, num_columns=self.problem_size.column, num_rows=self.problem_size.row, 
            warp_count=warp_count, epilogue_visitor=epilogue_visitor, cache_input=cache_input
        )
        # TODO: hardcode
        MLCOMPILER_SRC_DIR = '/workspace/SEAL-PICASSO-ML-Compiler/src/cuda'
        include_paths = [
            MLCOMPILER_SRC_DIR
        ] + compiler.default_include_paths

        compile_options = CompilationOptions(
            compiler.default_compile_options, 80, include_paths=include_paths
        )
        compiler.add_module([self.operation], compile_options)
    
    def reference(self, args, visitor_args):
        grad, softmax = args[-2:]
        def target_fn(grad, softmax):
            return torch.ops.aten._softmax_backward_data(grad, softmax, -1, torch.float16)
        super().reference_base([grad, softmax], target_fn, visitor_args, args)
    
    def call(self, visitor_args, stream, *args) -> Any:
        grad, softmax = args[-2:]

        arguments = SoftmaxBackwardArguments(
            self.operation, self.problem_size,
            grad, softmax, self.operation.epilogue_type(visitor_args)
        )

        self.operation.run(arguments, stream=stream.cuda_stream)


class FusedLayerNorm(FusedOpBase):
    """
    The fused operator of LayerNorm epilogue_visitor(native_layer_norm, args)
    """
    __name__ = "cutlass_layernorm_with_visitor"
    target = torch.ops.aten.native_layer_norm
    mainloop_fusion_targets = []

    def __init__(self, node: Node, epilogue_visitor) -> None:
        super().__init__(node, epilogue_visitor)

        # Get input
        input = node.args[0]
        self.reduction_length = node.args[1]
        self.args = [input]
        shape = node.meta["tensor_meta"].shape
        self.problem_size = MatrixCoord(shape[0], shape[2])

        # get mean & var tensor
        for user in node.users:
            assert user.target == operator.getitem
            if user.args[1] == 1:
                self.mean_name = user.name
            elif user.args[1] == 2:
                self.invstd_name = user.name

        # Get alignment
        alignment = self.get_alignment(shape[-1])
        element = torch_to_cutlass(node.meta["tensor_meta"].dtype)

        with (_pop_mode_temporarily() 
            if _len_torch_dispatch_stack() > 0 else nullcontext()):
            autotuner = LayerNormTuner(
                epilogue_visitor, self.problem_size, dtype=node.meta["tensor_meta"].dtype
            )
            rows_per_cta, warp_count, cache_input = autotuner.get_best_config()

        # Get softmax operation
        self.operation = LayerNormOperation(
            input=TensorDescription(element, LayoutType.RowMajor, alignment),
            rows_per_cta=rows_per_cta, num_columns=self.problem_size.column, num_rows=self.problem_size.row, 
            warp_count=warp_count, epilogue_visitor=epilogue_visitor, cache_input=cache_input
        )

        # TODO: hardcode
        MLCOMPILER_SRC_DIR = '/workspace/SEAL-PICASSO-ML-Compiler/src/cuda'
        include_paths = [
            MLCOMPILER_SRC_DIR
        ] + compiler.default_include_paths

        compile_options = CompilationOptions(
            compiler.default_compile_options, 80, include_paths=include_paths
        )
        compiler.add_module([self.operation], compile_options)
    
    def reference(self, args, visitor_args):
        input = args[-1]
        def target_fn(input):
            return torch.ops.aten.native_layer_norm(
                input, self.reduction_length, None, None, 1e-12
            )
        super().reference_base([input], target_fn, visitor_args, args)
    
    def call(self, visitor_args, stream, *args) -> Any:
        # create the mean & std vectors
        input = args[-1]

        arguments = LayerNormArguments(
            self.operation, self.problem_size,
            input, visitor_args[self.mean_name], visitor_args[self.invstd_name],
            self.operation.epilogue_type(visitor_args)
        )

        self.operation.run(arguments, stream=stream.cuda_stream)


class FusedLayerNormBackward(FusedOpBase):
    """
    The fused operator of LayerNorm epilogue_visitor(native_layer_norm, args)
    """
    __name__ = "cutlass_layernorm_backward_with_visitor"
    target = torch.ops.aten.native_layer_norm_backward
    mainloop_fusion_targets = []

    def __init__(self, node: Node, epilogue_visitor) -> None:
        super().__init__(node, epilogue_visitor)

        # Get input
        grad, x, self.reduction_length, mean, std, gamma = node.args[:6]

        self.args = [grad, x, mean, std, gamma]
        shape = node.meta["tensor_meta"].shape
        self.problem_size = MatrixCoord(shape[0], shape[2])

        # Get alignment
        alignment = self.get_alignment(shape[-1])
        element = torch_to_cutlass(node.meta["tensor_meta"].dtype)

        with (_pop_mode_temporarily() 
            if _len_torch_dispatch_stack() > 0 else nullcontext()):
            autotuner = LayerNormBackwardTuner(
                epilogue_visitor, self.problem_size, dtype=node.meta["tensor_meta"].dtype
            )
            rows_per_cta, warp_count, cache_input = autotuner.get_best_config()

        # Get softmax operation
        self.operation = LayerNormBackwardOperation(
            input=TensorDescription(element, LayoutType.RowMajor, alignment),
            rows_per_cta=rows_per_cta, num_columns=self.problem_size.column, num_rows=self.problem_size.row, 
            warp_count=warp_count, epilogue_visitor=epilogue_visitor, cache_input=cache_input
        )

        # TODO: hardcode
        MLCOMPILER_SRC_DIR = '/workspace/SEAL-PICASSO-ML-Compiler/src/cuda'
        include_paths = [
            MLCOMPILER_SRC_DIR
        ] + compiler.default_include_paths

        compile_options = CompilationOptions(
            compiler.default_compile_options, 80, include_paths=include_paths
        )
        compiler.add_module([self.operation], compile_options)
    
    def reference(self, args, visitor_args):
        grad, x, mean, invstd, gamma = args[-5:]
        beta = torch.zeros_like(gamma)
        def target_fn(grad, x, mean, invstd, gamma, beta):
            return torch.ops.aten.native_layer_norm_backward(
                grad, x, self.reduction_length, mean, invstd, gamma, beta, [True, False, False])
        super().reference_base([grad, x, mean, invstd, gamma, beta], target_fn, visitor_args, args)
    
    def call(self, visitor_args, stream, *args) -> Any:
        # create the mean & std vectors
        grad, x, mean, invstd, gamma = args[-5:]

        arguments = LayerNormBackwardArguments(
            self.operation, self.problem_size,
            gamma, grad, x, mean, invstd,
            self.operation.epilogue_type(visitor_args)
        )

        self.operation.run(arguments, stream=stream.cuda_stream)


class FusedSpmm(FusedOpBase):
    """
    The fused operator of LayerNorm epilogue_visitor(spmm, args)
    """
    __name__ = "cutlass_spmm_with_visitor"
    target = spmm
    mainloop_fusion_targets = []

    def __init__(self, node: Node, epilogue_visitor) -> None:
        super().__init__(node, epilogue_visitor)

        # Get input
        adj_matrix, embedding = node.args
        self.args = [adj_matrix, embedding]
        shape = node.meta["tensor_meta"].shape
        num_nodes = embedding.meta["tensor_meta"].shape[0]
        self.problem_size = GemmCoord(shape[0], shape[2], num_nodes)

        # Get alignment
        alignment = self.get_alignment(shape[-1])
        element = torch_to_cutlass(node.meta["tensor_meta"].dtype)
        element_index = torch_to_cutlass(torch.int64)

        # Get softmax operation
        self.operation = SpmmOperation(
            input=TensorDescription(element, LayoutType.RowMajor, alignment),
            element_index=element_index,
            num_columns=self.problem_size.n, num_rows=self.problem_size.m, 
            epilogue_visitor=epilogue_visitor
        )

        # TODO: hardcode
        MLCOMPILER_SRC_DIR = '/workspace/SEAL-PICASSO-ML-Compiler/src/cuda'
        include_paths = [
            MLCOMPILER_SRC_DIR
        ] + compiler.default_include_paths

        compile_options = CompilationOptions(
            compiler.default_compile_options, 80, include_paths=include_paths
        )
        compiler.add_module([self.operation], compile_options)
    
    def reference(self, args, visitor_args):
        adj_matrix = args[-2]
        embedding = args[-1]
        def target_fn(adj_matrix, embedding):
            return torch.ops.aten.mm(
                adj_matrix, embedding
            )
        super().reference_base([adj_matrix, embedding], target_fn, visitor_args, args, inductor_profile=False)
    
    def call(self, visitor_args, stream, *args) -> Any:
        # create the mean & std vectors
        adj_matrix = args[-2]
        embedding = args[-1]

        arguments = SpmmArguments(
            self.operation, self.problem_size,
            adj_matrix, embedding,
            self.operation.epilogue_type(visitor_args)
        )

        self.operation.run(arguments, stream=stream.cuda_stream)


class EVTFuser(EVTFrontendBase):
    """
    The entry point of CUTLASS EVT Fuser 
    """
    supported_targets = [
        torch.ops.aten.mm, torch.ops.aten.bmm, torch.ops.aten._softmax, 
        torch.ops.aten._softmax_backward_data,
        torch.ops.aten.native_layer_norm, torch.ops.aten.native_layer_norm_backward,
        spmm]
    
    def __init__(self, element_compute=DataType.f32, **kwargs):
        super().__init__(element_compute, **kwargs)
        self.mode = os.environ.get('MODE', 'RUN')

    def get_node_with_name(name, graph_module):
        for node in graph_module.graph.nodes:
            if node.name == name:
                return node
        raise ValueError(f"{name} is not in the graph.")

    def fusible(partition, graph_module):
        for node_name in partition:
            node = EVTFuser.get_node_with_name(node_name, graph_module)
            if node.target in EVTFuser.supported_targets:
                return True
        return False
            
    def trace(self, graph_module: GraphModule, partition_: 'list[Node]'):
        self.partition_names = partition_
        try:
            graph_module.graph.lint()
        except:
            legalize_graph(graph_module)
        partition = [EVTFuser.get_node_with_name(n, graph_module) for n in partition_]

        # Step 1: get the epilogue partition
        # The epilogue partition excludes the mainloop and mainloop-fused ops
        epilogue_partition = []
        # We use the heavy operation (mm, bmm, softmax, etc) as the anchor
        anchor = None
        for node in partition:
            if node.target in self.supported_targets:
                anchor = node
        
        # Heuristic: sum node is excluded from reduce_apply op
        if anchor.target in [torch.ops.aten._softmax, torch.ops.aten._softmax_backward_data,
                             torch.ops.aten.native_layer_norm, torch.ops.aten.native_layer_norm_backward,
                             spmm]:
            partition = [node for node in partition if node.target != torch.ops.aten.sum]

        for node in partition:
            if anchor in node.users:
                continue
            if node == anchor:
                continue
            epilogue_partition.append(node)

        # Step 2: extract the subgraph
        inputs = set()  # In original graph
        outputs = set() # In original graph
        for node in epilogue_partition:
            for input in node.all_input_nodes:
                if input in epilogue_partition:
                    continue
                inputs.add(input)
        
        for node in epilogue_partition:
            if node.target == torch.ops.aten.view and node.args[0] in inputs and node.args[0] != anchor:
                continue
            for user in node.users:
                if user in epilogue_partition:
                    continue
                outputs.add(node)
        if len(epilogue_partition) == 0:
            inputs = set([anchor,])
            outputs = set([anchor,])

        subgraph: Graph = _extract_graph_with_inputs_outputs(graph_module.graph,
                                                            inputs, outputs)
        
        if self.mode == "DEBUG":
            # Deep copy otherwise it will be modified
            self.reference_gm = deepcopy(GraphModule(graph_module, subgraph))
            # Update the name of anchor
            for node in self.reference_gm.graph.nodes:
                if node.name == anchor.name:
                    node.target = "accum"
                    node.name = "accum"

        # Step 2.1 Remove the clone nodes if there are
        for node in subgraph.nodes:
            if node.target == torch.ops.aten.clone:
                input = node.all_input_nodes[0]
                node.replace_all_uses_with(input)
                subgraph.erase_node(node)
            elif node.target == operator.getitem:
                if anchor.target == torch.ops.aten.native_layer_norm:
                    if node.args[1] == 0:
                        input = node.all_input_nodes[0]
                        node.replace_all_uses_with(input)
                        subgraph.erase_node(node)
                    else:
                        list(node.users)[0].replace_input_with(node, None)
                        subgraph.erase_node(node)
                elif anchor.target == torch.ops.aten.native_layer_norm_backward:
                    if node.args[1] == 0:
                        input = node.all_input_nodes[0]
                        node.replace_all_uses_with(input)
                        subgraph.erase_node(node)


        
        # Step 3: register the input and output nodes in the original graph & subgraph
        # 3.1 Output nodes in subgraph
        # This is used to figure out whether a store node should be inserted
        for node in subgraph.nodes:
            if node.op == "output":
                self.outputs_subgraph = set(node.all_input_nodes)
        # In some cases, the anchor node is also expected to be returned
        store_anchor = False
        for user in anchor.users:
            if user not in epilogue_partition:
                store_anchor = True
                break

        for node in subgraph.nodes:
            if node.name == anchor.name:
                node.target = "accum"
                node.name = "accum"
                if store_anchor:
                    self.outputs_subgraph.add(node)
                    outputs.add(anchor)
        # 3.2 Input nodes in the original graph
        self.inputs = list(inputs)

        # 3.3 Output nodes in the original graph
        self.outputs = list(outputs)

        # Visit the nodes
        for node in subgraph.nodes:
            self.visit(node)
        self.pass_manager()
        self.visualize()
        self.epilogue_thread_type = self.dag_ir.epilogue_thread_type
        self.reduction_names = self.dag_ir.reduction_names
        self.return_names = [output.name for output in self.outputs]
        self.transposed = self.dag_ir.transposed
        self.post_reshape_permute = self.dag_ir.post_reshape_permute
        self.output2store = self.dag_ir.output2store


        #
        # Step 4: compose the fused kernel
        #
        
        # Get mainloop args
        fused_kernel = self.get_fused_node(anchor)
        args = [input for input in self.inputs if input!= anchor] + fused_kernel.args
        graph_module.graph.inserting_after(anchor)
        fused_node = graph_module.graph.call_function(fused_kernel, args=tuple(args))
        fused_node.meta = {}
        fused_node.meta["evt"] = True
        graph_module.graph.inserting_after(fused_node)

        for idx, output_node in enumerate(self.outputs):
            # output_node = name_to_node[_output_node.name]
            get_item_node = graph_module.graph.call_function(operator.getitem, args=(fused_node, idx))
            graph_module.graph.inserting_after(get_item_node)
            view_node = graph_module.graph.call_function(torch.ops.aten.view, args=(get_item_node, output_node.meta["tensor_meta"].shape))
            view_node.meta = {}
            view_node.meta["tensor_meta"] = output_node.meta["tensor_meta"]._replace()
            output_node.replace_all_uses_with(view_node)
            graph_module.graph.inserting_after(view_node)

    
    #
    # Epilogue constructor
    #

    def visit(self, node: Node):
        if node.op in ["placeholder", "get_attr"]:
            self._visit_source(node)
        elif node.op == "call_function":
            if hasattr(self, f"visit_{node.target.__name__}"):
                getattr(self, f"visit_{node.target.__name__}")(node)
            else:
                raise NotImplementedError(f"Doesn't support target {node.target}")
    
    def create_fake_tensor(self, node: Node):
        meta = node.meta["tensor_meta"]
        element = meta.dtype
        shape = tuple(meta.shape)
        return fakeTensor(element=element, shape=shape, layout_tag=LayoutType.RowMajor)
    
    def insert_store_node(self, node):
        example = self.create_fake_tensor(node)
        self.add_store_node(node.name)
        self.set_store_tensor(node.name, example)
        name = node.name
        self.mark_output(name)

    def _get_name(self, node: Node):
        if node in self.outputs_subgraph:
            return node.name + "_t"
        else:
            return node.name  
    
    def _visit_source(self, node: Union[Node, float, int]):
        # This function is only to be called internally
        # No matter what is the node's target, it is added as a load node
        if isinstance(node, Node):
            name = self._get_name(node)
            example = self.create_fake_tensor(node)
            self.add_load_node(name, example)
            if node in self.outputs_subgraph:
                self.insert_store_node(node)
                self.add_edge(name, node.name, weight=0)
            return node.name
        elif isinstance(node, float) or isinstance(node, int):
            return self.add_imm(node)
        else:
            raise ValueError(f"Invalid source: {node}")
    
    def _visit_compute(self, node: Node, op: FunctionalOp, **kwargs):
        name = self._get_name(node)
        self.add_compute_node(op=op, name=name, **kwargs)
        # Add the source operands if they are not in graph
        arg_names = []
        for arg in node.args:
            if not isinstance(arg, Node):
                arg_names.append(self._visit_source(arg))
            else:
                arg_names.append(arg.name)
        
        # Add the edges
        for idx, arg in enumerate(arg_names):
            self.add_edge(arg, name, weight=idx)
        if node in self.outputs_subgraph:
            self.insert_store_node(node)
            self.add_edge(name, node.name, weight=0)
        return node.name
    
    def visit_add(self, node: Node):
        return self._visit_compute(node, FunctionalOp.Plus)

    def visit_mul(self, node: Node):
        return self._visit_compute(node, FunctionalOp.Multiplies)
    
    def visit_gelu(self, node: Node):
        return self._visit_compute(node, ActivationOp.Gelu)
    
    def visit_tanh(self, node: Node):
        return self._visit_compute(node, ActivationOp.Tanh)

    def visit_rand_like(self, node: Node):
        return self._visit_compute(node, FunctionalOp.Rand)
    
    def visit_one_hot(self, node: Node):
        one_hot_name: str = self._visit_compute(
            node, FunctionalOp.OneHot, num_classes=node.kwargs["num_classes"])
        # Insert an unsqueeze node before onehot to ensure correct shape infer
        name = f"unsqueeze_{one_hot_name}"
        op = self.layout_fns["reshape"]
        input = node.args[0]
        new_shape = input.meta["tensor_meta"].shape + (1,)
        kwargs = {"new_shape": new_shape}
        self.add_layout_node(op, kwargs, name)
        # Add edge
        self.add_edge(input.name, name, weight=0)
        self.remove_edge(input.name, one_hot_name)
        self.add_edge(name, one_hot_name)
        return one_hot_name
    
    def visit_ge(self, node: Node):
        return self._visit_compute(node, FunctionalOp.GreaterEqual)
    
    def visit_ne(self, node: Node):
        return self._visit_compute(node, FunctionalOp.Ne)
    
    def visit_sum(self, node: Node):
        name = self._get_name(node)
        op = (FunctionalOp.Plus, FunctionalOp.AtomicAdd)
        self.add_compute_node(op=op, name=name)
        arg_name = node.args[0].name
        # Add the edges
        self.add_edge(arg_name, name, weight=0)
        if node in self.outputs_subgraph:
            self.insert_store_node(node)
            self.add_edge(name, node.name, weight=0)
        return node.name
    
    def visit_tanh_backward(self, node: Node):
        return self._visit_compute(node, ActivationOp.TanhBackward)
    
    def visit_gelu_backward(self, node: Node):
        return self._visit_compute(node, ActivationOp.DGelu)
    
    def visit_relu(self, node: Node):
        return self._visit_compute(node, ActivationOp.ReLU)

    def visit_sigmoid(self, node: Node):
        return self._visit_compute(node, ActivationOp.Sigmoid)
    
    def visit_view(self, node: Node):
        name = self._get_name(node)
        op = self.layout_fns["reshape"]
        input, new_shape = node.args
        kwargs = {"new_shape": new_shape}
        self.add_layout_node(op, kwargs, name)
        # Add edge
        self.add_edge(input.name, name, weight=0)
        if node in self.outputs_subgraph:
            self.insert_store_node(node)
            self.add_edge(name, node.name, weight=0)
        return node.name

    def visit_unsqueeze(self, node: Node):
        name = self._get_name(node)
        op = self.layout_fns["reshape"]
        input = node.args[0]
        new_shape = node.meta["tensor_meta"].shape
        kwargs = {"new_shape": new_shape}
        self.add_layout_node(op, kwargs, name)
        # Add edge
        self.add_edge(input.name, name, weight=0)
        if node in self.outputs_subgraph:
            self.insert_store_node(node)
            self.add_edge(name, node.name, weight=0)
        return node.name

    
    def visit_permute(self, node: Node):
        name = self._get_name(node)
        op = self.layout_fns["permute"]
        input, indices = node.args
        kwargs = {"indices": tuple(indices)}
        self.add_layout_node(op, kwargs, name)
        # Add edge
        self.add_edge(input.name, name, weight=0)
        if node in self.outputs_subgraph:
            self.insert_store_node(node)
            self.add_edge(name, node.name, weight=0)
        return node.name


    #
    # Mainloop Op
    #

    def get_fused_node(self, node: Node):
        if node.target == torch.ops.aten.mm:
            op = FusedGEMM(node, self)
        elif node.target == torch.ops.aten.bmm:
            op = FusedBMM(node, self)
        elif node.target == torch.ops.aten._softmax:
            op = FusedSoftmax(node, self)
        elif node.target == torch.ops.aten._softmax_backward_data:
            op = FusedSoftmaxBackward(node, self)
        elif node.target == torch.ops.aten.native_layer_norm:
            op = FusedLayerNorm(node, self)
        elif node.target == torch.ops.aten.native_layer_norm_backward:
            op = FusedLayerNormBackward(node, self)
        elif node.target == spmm:
            op = FusedSpmm(node, self)
        else:
            raise NotImplementedError()
        return op
