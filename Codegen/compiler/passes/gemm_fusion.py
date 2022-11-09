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
from passes.epilogue_parser import EpilogueVisitTreeDAG
from passes import autotuner
import nvtx
import operator

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

class FusedGEMM:
    __name__ = "cutlass_gemm_with_visitor"
    def __init__(self, node) -> None:
        assert node.target == torch.ops.aten.mm
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

        # launch autotuner
        best_config = autotuner(
            problem_size=[M, N, K], element_a=cutlass.float16, layout_a=lhs_layout,
            element_b=cutlass.float16, layout_b=rhs_layout, 
            element_c=cutlass.float16, layout_c=cutlass.RowMajor, 
            element_accumulator=cutlass.float32, alignment_a=align_a,
            alignment_b=align_b, alignment_c=align_c)

        threadblock_shape = best_config[0:3]
        threadblock_shape = [int(t) for t in threadblock_shape]
        warp_shape = best_config[3:6]
        warp_count = [t // w for t, w in zip(threadblock_shape, warp_shape)]
        warp_count = [int(t) for t in warp_count]
        stages = int(best_config[6])

        log_swizzle = best_config[7]
        swizzling_functor = getattr(cutlass, "IdentitySwizzle%d"%int(pow(2, log_swizzle)))


        tile_description = TileDescription(
            threadblock_shape, stages, warp_count, math_inst
        )


        A = TensorDescription(cutlass.float16, lhs_layout, align_a)
        B = TensorDescription(cutlass.float16, rhs_layout, align_b)
        C = TensorDescription(cutlass.float16, cutlass.RowMajor, align_c)

        epilogue_functor = LinearCombination(
            element_output=C.element, epilogue_vector_length=C.alignment,
            element_accumulator=math_inst.element_accumulator,
            element_epilogue=cutlass.float32)

        # Creating the epilogue visitor tree
        self.epilogue_functor = EpilogueVisitTreeDAG(
            elementwise_functor=epilogue_functor, 
            tile_description=tile_description, 
            element_accumulator=math_inst.element_accumulator,
            elements_per_access=C.alignment,
            element_compute=cutlass.float32, element_output=C.element
        )

        self.epilogue_functor.initialize(node)

        # Initialize the epilogue visitor tree

        self.operation = GemmOperationUniversal(
            arch=80, tile_description=tile_description,
            A=A, B=B, C=C, epilogue_functor=self.epilogue_functor,
            swizzling_functor=swizzling_functor, visitor=True
        )

        pycutlass.compiler.add_module([self.operation,])

        self.args = self.epilogue_functor.args + list(node.args)
        self.outputs = self.epilogue_functor.outputs

    def __call__(self, *args):
        lhs = args[-2]
        rhs = args[-1]
        with nvtx.annotate("cutlass_gemm"):
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
            for output_node in self.outputs:
                kwargs["output_" + output_node.name] = torch.empty(
                    size=output_node.meta['tensor_meta'].shape,
                    dtype=output_node.meta['tensor_meta'].dtype,
                    device="cuda"
                )

            for idx, epilogue_arg in enumerate(self.epilogue_functor.args):
                kwargs[epilogue_arg.name] = args[idx]

            output_op = self.operation.epilogue_type(**kwargs)
            arguments = GemmArguments(
                operation=self.operation, problem_size=problem_size,
                A=lhs, B=rhs, C=lhs, D=lhs,
                output_op=output_op, gemm_mode=cutlass.gemm.Mode.Gemm
            )
            s1 = torch.cuda.current_stream()
            self.operation.run(arguments, stream=s1.cuda_stream)

            results = []
            for output_node in self.outputs:
                results.append(kwargs["output_" + output_node.name])
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

        # launch autotuner
        # TODO: implement the autotuner for batched case
        tile_description = TileDescription(
            threadblock_shape=[128, 128, 64], stages=3, warp_count=[2, 2, 1], math_instruction=math_inst
        )

        A = TensorDescription(cutlass.float16, self.lhs_layout, 8)
        B = TensorDescription(cutlass.float16, self.rhs_layout, 8)
        C = TensorDescription(cutlass.float16, cutlass.RowMajor, 8)

        swizzling_functor = cutlass.BatchedIdentitySwizzle

        epilogue_functor = LinearCombination(
            element_output=C.element, epilogue_vector_length=C.alignment,
            element_accumulator=math_inst.element_accumulator,
            element_epilogue=cutlass.float32)
        
        # Creating the epilogue visitor tree
        self.epilogue_functor = EpilogueVisitTreeDAG(
            elementwise_functor=epilogue_functor, 
            tile_description=tile_description, 
            element_accumulator=math_inst.element_accumulator,
            elements_per_access=C.alignment,
            element_compute=cutlass.float32, element_output=C.element
        )

        self.epilogue_functor.initialize(node)

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
            kwargs = {"problem_size": [self.problem_size.m(), self.problem_size.n()], "batch_size": self.batch}
            for output_node in self.outputs:
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
            s1 = torch.cuda.current_stream()
            self.operation.run(arguments, stream=s1.cuda_stream)

            results = []
            for output_node in self.outputs:
                results.append(kwargs["output_" + output_node.name])
        return results

def pass_gemm_fusion(module, graph):
    """
    Fuse GEMM kernel with its epilogues
    """
    gemm_idx = 0
    bmm_idx = 0
    for node in graph.nodes:
        if node.op == "call_function":
            if node.target == torch.ops.aten.mm:
                if gemm_idx > 29:
                    break

                fused_gemm = FusedGEMM(node)
                graph.inserting_after(fused_gemm.epilogue_functor.root)
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

                
                gemm_idx += 1
                pass_dead_code_elimination_non_topo(module, graph)      
                graph.eliminate_dead_code()
                # graph.eliminate_dead_code()

    for node in graph.nodes:
        if node.op == "call_function":    
            if node.target == torch.ops.aten.bmm:
                if bmm_idx > 4: break
                fused_bmm = FusedBMM(node)
                graph.inserting_after(fused_bmm.epilogue_functor.root)
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
                bmm_idx += 1

                pass_dead_code_elimination_non_topo(module, graph)      
                graph.eliminate_dead_code()
                # graph.eliminate_dead_code()

    pass_dead_code_elimination_non_topo(module, graph)      
    graph.eliminate_dead_code()
    graph.lint()
