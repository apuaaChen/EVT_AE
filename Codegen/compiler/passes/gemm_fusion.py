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
import nvtx

################################################################################
# Graph-level pass to fuse GEMM kernels
################################################################################


def pass_gemm_fusion(module, graph):
    """
    Fuse GEMM kernel with its epilogues
    """
    for node in graph.nodes:
        if node.op == "call_function":
            if node.target == torch.ops.aten.mm:

                # Step 1: fuse the source operand transpose nodes
                lhs_node = node.args[0]
                rhs_node = node.args[1]

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

                math_inst = MathInstruction(
                    instruction_shape=[16, 8, 16],
                    element_a=cutlass.float16, element_b=cutlass.float16,
                    element_accumulator=cutlass.float32, 
                    opcode_class=cutlass.OpClass.TensorOp
                )

                tile_description = TileDescription(
                    [128, 128, 64], 3, [2, 2, 1], math_inst
                )

                A = TensorDescription(cutlass.float16, lhs_layout, 8)
                B = TensorDescription(cutlass.float16, rhs_layout, 8)
                C = TensorDescription(cutlass.float16, cutlass.RowMajor, 8)

                epilogue_functor = LinearCombination(
                    element_output=C.element, epilogue_vector_length=C.alignment,
                    element_accumulator=math_inst.element_accumulator,
                    element_epilogue=cutlass.float32)
                
                swizzling_functor = cutlass.IdentitySwizzle1

                # Creating the epilogue visitor tree
                epilogue_functor = EpilogueVisitTreeDAG(
                    elementwise_functor=epilogue_functor, 
                    tile_description=tile_description, 
                    element_accumulator=math_inst.element_accumulator,
                    elements_per_access=C.alignment,
                    element_compute=cutlass.float32, element_output=C.element
                )

                epilogue_functor.initialize(node)

                # Initialize the epilogue visitor tree

                operation = GemmOperationUniversal(
                    arch=80, tile_description=tile_description,
                    A=A, B=B, C=C, epilogue_functor=epilogue_functor,
                    swizzling_functor=swizzling_functor, visitor=True
                )

                # print(operation.rt_module.emit())

                pycutlass.compiler.add_module([operation,])

                args = epilogue_functor.args + list(node.args)
                outputs = epilogue_functor.outputs
                print(args)

                node.meta['op'] = operation

                def cutlass_gemm(*args):
                    lhs = args[-2]
                    rhs = args[-1]
                    with nvtx.annotate("cutlass_gemm"):
                        if lhs_layout == cutlass.RowMajor:
                            M, K = lhs.size()
                        else:
                            K, M = lhs.size()
                        if rhs_layout == cutlass.RowMajor:
                            N = rhs.size(-1)
                        else:
                            N = rhs.size(-2)
                        lhs = lhs.contiguous()
                        rhs = rhs.contiguous()
                        bias = args[0]
                        problem_size = cutlass.gemm.GemmCoord(M, N, K)
                        kwargs = {"problem_size": [M, N]}
                        for output_node in outputs:
                            kwargs[output_node.name] = torch.empty(
                                size=output_node.meta['tensor_meta'].shape,
                                dtype=output_node.meta['tensor_meta'].dtype,
                                device="cuda"
                            )
                        kwargs["primals_2"] = bias

                        for idx, epilogue_arg in enumerate(epilogue_functor.args):
                            kwargs[epilogue_arg.name] = args[idx]

                        output_op = operation.epilogue_type(**kwargs)
                        arguments = GemmArguments(
                            operation=node.meta['op'], problem_size=problem_size,
                            A=lhs, B=rhs, C=lhs, D=lhs,
                            output_op=output_op, gemm_mode=cutlass.gemm.Mode.Gemm
                        )
                        node.meta['op'].run(arguments)
                        # arguments.sync()
                        output_op.sync()
                    return kwargs["view_1"]
                
                graph.inserting_after(epilogue_functor.root)
                fused_node = graph.call_function(cutlass_gemm, args=tuple(args))
                fused_node.meta = {}
                fused_node.meta['tensor_meta'] = epilogue_functor.root.meta['tensor_meta']._replace()
                epilogue_functor.root.replace_all_uses_with(fused_node)

                
                # node.target = cutlass_gemm

                break
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
