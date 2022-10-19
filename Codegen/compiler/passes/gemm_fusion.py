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


class FusedGEMM:
    __name__ = "cutlass_gemm_with_visitor"
    def __init__(self, node) -> None:
        assert node.target == torch.ops.aten.mm
        lhs_node = node.args[0]
        rhs_node = node.args[1]

        # get layout
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

        # launch autotuner
        best_config = autotuner(
            problem_size=[M, N, K], element_a=cutlass.float16, layout_a=lhs_layout,
            element_b=cutlass.float16, layout_b=rhs_layout, 
            element_c=cutlass.float16, layout_c=cutlass.RowMajor, 
            element_accumulator=cutlass.float32)

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

        A = TensorDescription(cutlass.float16, lhs_layout, 8)
        B = TensorDescription(cutlass.float16, rhs_layout, 8)
        C = TensorDescription(cutlass.float16, cutlass.RowMajor, 8)

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


def pass_gemm_fusion(module, graph):
    """
    Fuse GEMM kernel with its epilogues
    """
    for node in graph.nodes:
        if node.op == "call_function":
            if node.target == torch.ops.aten.mm:
                fused_gemm = FusedGEMM(node)
                graph.inserting_after(fused_gemm.epilogue_functor.root)
                fused_node = graph.call_function(fused_gemm, args=tuple(fused_gemm.args))
                fused_node.meta = {}
                fused_node.meta['tensor_meta'] = fused_gemm.epilogue_functor.root.meta['tensor_meta']._replace()
                graph.inserting_after(fused_node)

                for idx, output_node in enumerate(fused_gemm.outputs):
                    get_item_node = graph.call_function(operator.getitem, args=(fused_node, idx))
                    graph.inserting_after(get_item_node)
                    output_node.replace_all_uses_with(get_item_node)

            # if node.target == torch.ops.aten._softmax:
            #     print("============Softmax===============")
            #     epilogue_functor = LinearCombination(
            #         element_output=cutlass.float16, epilogue_vector_length=8,
            #         element_accumulator=cutlass.float32,
            #         element_epilogue=cutlass.float32)
            #     epilogue_tree = EpilogueVisitTreeDAG(
            #         elementwise_functor=epilogue_functor, 
            #         tile_description=None, 
            #         element_accumulator=cutlass.float32,
            #         elements_per_access=8,
            #         element_compute=cutlass.float32, element_output=cutlass.float16
            #     )

            #     epilogue_tree.initialize(node)
                
    graph.eliminate_dead_code()
    graph.lint()
