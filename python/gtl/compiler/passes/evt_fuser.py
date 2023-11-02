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
from cutlass import DataType, SwizzlingFunctor
import cutlass
from cutlass.shape import GemmCoord
from tqdm import tqdm
from torch.fx import Graph, Node
import torch
from torch.fx.passes.utils.fuser_utils import topo_sort, validate_partition
from cutlass.backend.evt.frontend.frontend_base import EVTFrontendBase
from cutlass.backend.evt.ir.tensor import Tensor as fakeTensor
from cutlass import LayoutType
from cutlass.backend.library import FunctionalOp, ActivationOp
from typing import Union
import pdb
from torch.fx.graph_module import GraphModule
from torch._functorch.partitioners import _extract_graph_with_inputs_outputs
from cutlass.backend.gemm_operation import GemmArguments
import operator
from cutlass.swizzle import ThreadblockSwizzleStreamK
from cutlass.profiler import CUDAEventProfiler
from concurrent.futures import ThreadPoolExecutor


################################################################################
# Autotuner
################################################################################
# TODO: probably move the autotuner to other files
class Autotuner:
    def __init__(self, plan, A: Node, B: Node, anchor: Node, visitor_args=None) -> None:
        self.plan = plan
        self.meta_A = A.meta["tensor_meta"]
        self.meta_B = B.meta["tensor_meta"]
        self.op = anchor.target
        self.warmup_iterations=100
        self.profile_iterations=100

        self.swizzling_functors = [
            SwizzlingFunctor.Identity1, SwizzlingFunctor.Identity2,
            SwizzlingFunctor.Identity4, SwizzlingFunctor.Identity8,
            SwizzlingFunctor.StreamK
        ]

        # self.compile(self.plan.tile_descriptions(), self.plan)
        self.num_best_tds = 5
        self.best_tds = self.profile()
    
    def compile(self, tds, plan):
        """
        Perform parallel compilation of the kernels
        """
        for td in tqdm(tds):
            plan.compile(td, alignment_A=8, alignment_B=8, alignment_C=8)

    def profile(self):
        tensor_A = torch.empty(size=self.meta_A.shape, dtype=self.meta_A.dtype, device="cuda")
        tensor_B = torch.empty(size=self.meta_B.shape, dtype=self.meta_B.dtype, device="cuda")
        if self.plan._layout_a == LayoutType.ColumnMajor:
            tensor_A = torch.transpose(tensor_A, -1, -2)
        if self.plan._layout_b == LayoutType.ColumnMajor:
            tensor_B = torch.transpose(tensor_B, -1, -2)
        
        tensor_C = self.op(tensor_A, tensor_B)
        tensor_D = torch.empty_like(tensor_C)

        # Get durations
        durations = []
        tds = []
        for td in tqdm(self.plan.tile_descriptions()):
            self.plan.tile_description = td
            duration = CUDAEventProfiler(
                self.plan, self.warmup_iterations, self.profile_iterations,
                tensor_A, tensor_B, tensor_C, tensor_D
            )()
            durations.append(duration)
            tds.append(td)
        # Sort
        combined = list(zip(durations, tds))
        sorted_combined = sorted(combined, key=lambda x: x[0])
        _, sorted_td = zip(*sorted_combined)

        return sorted_td[:self.num_best_tds]

class FusedGEMM:
    """
    The fused operator of GEMM epilogue_visitor(A*B, args)
    """
    __name__ = "cutlass_gemm_with_visitor"
    def __init__(self, node, epilogue_visitor) -> None:
        assert node.target == torch.ops.aten.mm
        self.node = node

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

        # Outputs and inputs in the original graph
        self.outputs = epilogue_visitor.outputs
        self.inputs = [
            input for input in epilogue_visitor.inputs 
            if input.target != self.node.target]

        # Plan
        plan = cutlass.op.Gemm(
            element=lhs_node.meta["tensor_meta"].dtype,
            layout_A=self.lhs_layout, layout_B=self.rhs_layout,
            layout_C=cutlass.LayoutType.RowMajor,
            element_accumulator=torch.float32
        )

        # Use streamk
        # plan.swizzling_functor = ThreadblockSwizzleStreamK

        # Get candidate tds
        # autotuner = Autotuner(plan, self.A, self.B, self.node)
        # self.best_tds = autotuner.best_tds

        epilogue_visitor.epilogue_stages = 2
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

        # Stream
        self.stream = None
    
    def __call__(self, *args):
        default_stream = torch.cuda.current_stream()

        if self.stream is not None:
            self.stream.wait_stream(default_stream)
            stream = self.stream
        else:
            stream = default_stream
        
        with torch.cuda.stream(stream):
            A = args[-2]
            B = args[-1]
            if self.lhs_layout == cutlass.LayoutType.ColumnMajor:
                A = torch.transpose(A, -1, -2)
            if self.rhs_layout == cutlass.LayoutType.ColumnMajor:
                B = torch.transpose(B, -1, -2)


            # Create the output nodes
            visitor_args = {}
            for output_node in self.outputs:
                visitor_args[output_node.name] = torch.empty(
                    size=output_node.meta['tensor_meta'].shape,
                    dtype=output_node.meta['tensor_meta'].dtype,
                    device="cuda"
                )
            
            # Register the inputs
            for idx, input in enumerate(self.inputs):
                visitor_args[input.name] = args[idx]
            
            # if self.operation is None:
            #     best_td = None
            #     min_duration = 1e+6
            #     C = A @ B
            #     for td in tqdm(self.best_tds):
            #         self.plan.tile_description = td
            #         duration = CUDAEventProfiler(
            #             self.plan, 100, 100, A, B, C, C, 
            #             visitor_args=visitor_args
            #         )()
            #         if duration < min_duration:
            #             best_td = td
            #     self.operation = self.plan.compile(
            #         tile_description=best_td,
            #         alignment_A=self.align_A, alignment_B=self.align_B, 
            #         alignment_C=self.align_C)
            
            arguments = GemmArguments(
                operation=self.operation, problem_size=self.problem_size,
                A=A, B=B, C=None, D=None,
                output_op = self.operation.epilogue_type(visitor_args),
                gemm_mode=cutlass.GemmUniversalMode.Gemm
            )

            self.operation.run(arguments, stream=stream.cuda_stream)

            if self.stream is not None:
                # without the code below, the caching allocator might reuse
                # the memory unexpectedly
                # https://pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html?highlight=stream#torch.Tensor.record_stream
                for arg in args:
                    if isinstance(arg, torch.Tensor):
                        arg.record_stream(self.stream)

        # get the results
        return [visitor_args[output.name] for output in self.outputs]

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
    
    def get_src(self, node):
        if node.target == torch.ops.aten.permute:
            return node.args[0]
        else:
            return node
    
    def get_alignment(self, dim):
        if dim % 8 == 0: return 8
        elif dim % 4 == 0: return 4
        elif dim % 2 == 0: return 2
        else: return 1


class EVTFuser(EVTFrontendBase):
    """
    The entry point of CUTLASS EVT Fuser 
    """
    def trace(self, graph_module: GraphModule, partition: 'list[Node]'):
        # Step 1: get the epilogue partition
        # The epilogue partition excludes the mainloop and mainloop-fused ops
        epilogue_partition = []
        # We use the heavy operation (mm, bmm, softmax, etc) as the anchor
        anchor = None
        for node in partition:
            if node.target in [torch.ops.aten.mm, torch.ops.aten.bmm]:
                anchor = node
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
            for user in node.users:
                if user in epilogue_partition:
                    continue
                outputs.add(node)
        subgraph: Graph = _extract_graph_with_inputs_outputs(graph_module.graph,
                                                            inputs, outputs)
        
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
        # self.visualize()
        self.pass_manager()
        # self.visualize()
        self.epilogue_thread_type = self.dag_ir.epilogue_thread_type
        self.reduction_names = self.dag_ir.reduction_names
        self.return_names = [output.name for output in self.outputs]

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
            self.add_load_node(node.name, example)
            if node in self.outputs_subgraph:
                self.insert_store_node(node)
                self.add_edge(name, node.name, weight=0)
            return node.name
        elif isinstance(node, float) or isinstance(node, int):
            return self.add_imm(node)
        else:
            raise ValueError(f"Invalid source: {node}")
    
    def _visit_compute(self, node: Node, op: FunctionalOp):
        name = self._get_name(node)
        self.add_compute_node(op=op, name=name)
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
    
    def visit_ge(self, node: Node):
        return self._visit_compute(node, FunctionalOp.GreaterEqual)
    
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

    #
    # Mainloop Op
    #

    def get_fused_node(self, node: Node):
        if node.target == torch.ops.aten.mm:
            op = FusedGEMM(node, self)
        else:
            raise NotImplementedError()
        return op
