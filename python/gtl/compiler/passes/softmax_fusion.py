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
from unittest import result
import cutlass.backend
from cutlass.backend import *
import cutlass
import torch
from gtl.compiler.passes.epilogue_parser import EpilogueVisitTreeDAG
from gtl.compiler.passes import autotuner
import nvtx
import operator

from .softmax_operation import SoftmaxArguments, SoftmaxOperation


################################################################################
# Graph-level pass to fuse Softmax kernels
################################################################################

class FusedSoftmax:
    __name__ = "cutlass_softmax_with_visitor"
    def __init__(self, node) -> None:
        assert node.target == torch.ops.aten._softmax

        # update the softmax shape
        shape = node.meta["tensor_meta"].shape
        reduction_dim = node.args[1]
        if reduction_dim < 0:
            reduction_dim = len(shape) + reduction_dim
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

        Input = TensorDescription(cutlass.float16, cutlass.RowMajor, alignment)
        Output = TensorDescription(cutlass.float16, cutlass.RowMajor, alignment)

        warp_count = min(max(1, (self.shape[1] + (32 * alignment) - 1) // (32 * alignment)), 4)

        # To tune: 
        batch_size = 32

        self.shape = (self.shape[0] // 32, self.shape[1])

        tile_description = TileDescription(
            threadblock_shape=[1, self.shape[reduction_dim], batch_size], stages=1,
            warp_count=[1, warp_count, 1], math_instruction=None
        )

        epilogue_functor = LinearCombination(
            element_output=Output.element, 
            epilogue_vector_length=Output.alignment,
            element_accumulator=cutlass.float32,
            element_epilogue=cutlass.float32)
        
        self.epilogue_functor = EpilogueVisitTreeDAG(
            elementwise_functor=epilogue_functor, 
            tile_description=tile_description, 
            element_accumulator=cutlass.float32,
            elements_per_access=Output.alignment,
            element_compute=cutlass.float32, element_output=Output.element
        )

        self.epilogue_functor.initialize(node)

        self.operation = SoftmaxOperation(
            input=Input, output=Output, threadblock_tile=tile_description.threadblock_shape,
            warp_count=tile_description.warp_count, element_accumulator=cutlass.float32, epilogue_functor=self.epilogue_functor
        )
        cutlass_path = os.getenv('CUTLASS_PATH')
        assert cutlass_path is not None, "Environment variable 'CUTLASS_PATH' is not defined."
        cuda_install_path = os.getenv('CUDA_INSTALL_PATH')
        assert cuda_install_path is not None, "Environment variable 'CUDA_INSTALL_PATH' is not defined."
        include_paths = [
            cuda_install_path + '/include',
            cutlass_path + '/include',
            cutlass_path + '/tools/util/include',
            cutlass_path + '/python/cutlass/cpp/include',
            '/opt/conda/lib/python3.8/site-packages/torch/include/',
            '/workspace/sparseTraining/src/cuda/'
        ]
        compile_options = CompilationOptions(
            ['-std=c++14'], [80, ], include_paths=include_paths
        )

        cutlass.backend.compiler.add_module([self.operation,], compile_options)

        self.args = self.epilogue_functor.args + list(node.args)
        self.outputs = self.epilogue_functor.outputs
    
    def __call__(self, *args):
        kwargs = {"problem_size": [self.shape[0], self.shape[1]]}
        with nvtx.annotate("cutlass_softmax"):
            for output_node in self.outputs:
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

            arguments = SoftmaxArguments(
                operation=self.operation, problem_size=kwargs["problem_size"],
                input=args[-3], output_op=output_op
            )
            s1 = torch.cuda.current_stream()
            self.operation.run(arguments, stream=s1.cuda_stream)

            results = []
            for output_node in self.outputs:
                results.append(kwargs["output_" + output_node.name])

            # results_origin = [view_4, view_2]
        return results

def get_topological_index(graph, node):
    """
    Get the node index in topological order
    """
    for idx, node_ in enumerate(graph.nodes):
        if node == node_:
            return idx

def pass_softmax_fusion(module, graph):
    """
    Fuse Softmax kernel with its epilogues
    """
    softmax_idx = 0
    for node in graph.nodes:
        if node.op == "call_function":
            if node.target == torch.ops.aten._softmax:
                if softmax_idx > 3:
                    break
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
                
                softmax_idx += 1
                # graph.eliminate_dead_code()
    
    graph.eliminate_dead_code()
    graph.lint()
