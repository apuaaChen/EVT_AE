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
from torch.fx.graph_module import GraphModule
from torch.fx.passes.tools_common import legalize_graph
from torch.fx import Graph, Node
import torch
from torch._functorch.partitioners import _extract_graph_with_inputs_outputs
import operator
from gtl.compiler.passes.pass_clean_up import CleanUp
import os
from torch.profiler import profile, ProfilerActivity, record_function
from copy import deepcopy
from torch.fx.subgraph_rewriter import replace_pattern_with_filters
from gtl.compiler.passes.pass_decomposition import batch_norm_elemt, batch_norm_backward_elemt

torch.fx.wrap('batch_norm_elemt')
torch.fx.wrap('batch_norm_backward_elemt')

class FusedInductorOp:
    """
    Implementation of fused inductor op
    """
    __name__ = "inductor_fused_kernel"
    def __init__(self, gm: GraphModule, partition_names) -> None:
        self.inputs = [node for node in gm.graph.nodes if node.op == "placeholder"]
        self.input_names = [node.name for node in self.inputs]
        output_node = [node for node in gm.graph.nodes if node.op == "output"][0]
        self.outputs = list(output_node.args[0])
        self.output_names = [node.name for node in self.outputs]
        CleanUp()(gm)
        gm.recompile()
        # self.inductor_module = torch.compile(gm)
        self.inductor_module = torch.jit.script(gm)
        self.partition_names = partition_names
        #
        # Debug & Profile helpers
        #
        self.mode = os.environ.get('MODE', 'RUN')
        # Flag to avoid running this partition multiple times
        self.launched = False
        if self.mode == 'DEBUG':
            self.reference_module: GraphModule = deepcopy(gm)
            self.reference_module.recompile()
        self.rtol = 1e-2
        self.atol = 1e-3
    
    def __call__(self, *args):
        inputs = {}
        for name, tensor in zip(self.input_names, args):
            inputs[name] = tensor

        outputs = self.inductor_module(**inputs)
        if not isinstance(outputs, list):
            outputs = [outputs,]
        
        if self.mode == "DEBUG" and not self.launched:
            print(f"======================================================")
            print(f"######################################################")
            print(f"Debugging {self.partition_names}")
            print(f"######################################################")
            print(f"======================================================")

            self.launched = True
            self.reference(inputs, outputs)
        
        return outputs
    
    def reference(self, inputs, outputs):
        outputs_ref = self.reference_module(**inputs)
        if not isinstance(outputs_ref, list):
            outputs_ref = [outputs,]
        
        for out, ref in zip(outputs, outputs_ref):
            try:
                assert torch.allclose(out, ref, rtol=self.rtol)
            except:
                # Print the two tensors
                print(f"!!!!!!!!!!!!!!result mismatch!!!!!!!!!!!!!!")
                print(out)
                print(ref)
                if out.dtype == torch.bool:
                    continue
                print("maximum abs error: ", torch.max(torch.abs(out-ref)))
                print("maximum relative error: ", torch.max(torch.abs(out-ref)/torch.abs(ref)))
                print("atol passed rate: ", self.pass_rate(out, ref, {"atol": self.atol}))
                print("rtol passed rate: ", self.pass_rate(out, ref, {"rtol": self.rtol}))
            
            # Step 3: profiling
            ## Profile the baseline
            print("############### Torch Compile ###############")
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
                with record_function("torch"):
                    self.reference_module(**inputs)
                
            print(prof.key_averages().table(sort_by="cuda_time_total"))

            ## Profile the optimized kernel
            print("############### INDUCTOR ###############")
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
                with record_function("evt"):
                    self.inductor_module(**inputs)
            print(prof.key_averages().table(sort_by="cuda_time_total"))
            breakpoint()


class NVFuser:
    """
    The entry point of the default fuser from torch inductor
    """
    def __init__(self) -> None:
        self.mode = os.environ.get('MODE', 'RUN')
    
    def get_node_with_name(name, graph_module):
        for node in graph_module.graph.nodes:
            if node.name == name:
                return node
        raise ValueError(f"{name} is not in the graph.")
    
    def trace(self, graph_module: GraphModule, partition_: 'list[Node]'):
        self.partition_names = partition_
        try:
            graph_module.graph.lint()
        except:
            legalize_graph(graph_module)
        partition = [NVFuser.get_node_with_name(n, graph_module) for n in partition_]

        # Step 2: extract the subgraph
        inputs = set()
        outputs = set()
        for node in partition:
            for input in node.all_input_nodes:
                if input not in partition:
                    inputs.add(input)
            has_external_user = False
            for user in node.users:    
                if user not in partition:
                    has_external_user = True
                    break
            if has_external_user:
                outputs.add(node)
        
        subgraph: Graph = _extract_graph_with_inputs_outputs(graph_module.graph,
                                                            inputs, outputs)
        
        sub_gm = GraphModule(graph_module, subgraph)

        self.decomposition(sub_gm)

        fused_op = FusedInductorOp(sub_gm, self.partition_names)
        input_nodes = [NVFuser.get_node_with_name(n, graph_module) for n in fused_op.input_names]
        output_nodes = [NVFuser.get_node_with_name(n, graph_module) for n in fused_op.output_names]

        fused_node = graph_module.graph.call_function(fused_op, args=tuple(input_nodes))
        fused_node.meta = {}
        graph_module.graph.inserting_after(fused_node)

        for idx, output in enumerate(output_nodes):
            get_item_node = graph_module.graph.call_function(operator.getitem, args=(fused_node, idx))
            get_item_node.meta = {}
            get_item_node.meta["tensor_meta"] = output.meta["tensor_meta"]._replace()
            output.replace_all_uses_with(get_item_node)
            graph_module.graph.inserting_after(get_item_node)
    
    def decomposition(self, gm: GraphModule):
        # Step 1: get the output node names
        output_nodes = [node for node in gm.graph.nodes if node.op == "output"]
        assert len(output_nodes) == 1
        original_output_node_names = [node.name for node in output_nodes[0].args[0]]

        # Step 2: Get the decomposition patterns
        # modified = False
        graph = gm.graph
        for node in graph.nodes:
            if hasattr(node.target, "__qualname__"):
                visitor_name = "requires_" + node.target.__qualname__
                if hasattr(self, visitor_name):
                    pattern, replacement, filter = getattr(self, visitor_name)(node)
                    matches = replace_pattern_with_filters(
                        gm, pattern, replacement, filter
                    )
                    # if len(matches) >= 1:
                    #     modified = True

        # Step 3: update the output node names
        output_nodes = [node for node in gm.graph.nodes if node.op == "output"]
        assert len(output_nodes) == 1
        for name, node in zip(original_output_node_names, output_nodes[0].args[0]):
            node.name = name
        
        # # Filling node meta
        # if modified:
        #     FakeTensorInfer(gm).infer()
    
    def requires_batch_norm_elemt(self, node: torch.fx.Node):
        def pattern(input, weight, bias, mean_vec, mean_x2_vec, eps, K):
            output, saved_mean, saved_rstd = batch_norm_elemt(
                input, weight, bias, mean_vec, mean_x2_vec, eps, K
            )
            return output, saved_mean, saved_rstd
        
        def replacement(input, weight, bias, mean_vec, mean_x2_vec, eps, K):
            var = mean_x2_vec - mean_vec * mean_vec
            rstd = torch.ops.aten.rsqrt(var + eps)
            saved_mean = torch.ops.aten.squeeze(mean_vec)
            saved_rstd = torch.ops.aten.squeeze(rstd)

            sub_mean = torch.ops.aten.sub(input, mean_vec)
            mul_gamma = torch.ops.aten.mul(weight, sub_mean)
            mul_rstd = torch.ops.aten.mul(mul_gamma, rstd)
            add_bias = torch.ops.aten.add(mul_rstd, bias)
            output = torch.ops.aten.to(add_bias, torch.float16)
            return output, saved_mean, saved_rstd
        
        return (pattern, replacement, None)
    
    def requires_batch_norm_backward_elemt(self, node: torch.fx.Node):
        def pattern(y_grad, input, saved_mean, saved_rstd, K, sum_grad_y,
            sum_grad_y_xhat, reduction_factor, gamma):
            x_grad, gamma_grad, beta_grad = batch_norm_backward_elemt(
                y_grad, input, saved_mean, saved_rstd, K, sum_grad_y,
                sum_grad_y_xhat, reduction_factor, gamma
            )
            return x_grad, gamma_grad, beta_grad
        
        def replacement(y_grad, input, saved_mean, saved_rstd, K, sum_grad_y,
            sum_grad_y_xhat, reduction_factor, gamma):
            sub_saved_mean = torch.ops.aten.sub(input, saved_mean)
            x_hat = torch.ops.aten.mul(sub_saved_mean, saved_rstd)

            mean_grad_y_xhat = torch.ops.aten.mul(sum_grad_y_xhat, reduction_factor)
            mean_grad_y = torch.ops.aten.mul(sum_grad_y, reduction_factor)

            var_grad_mean = -mean_grad_y_xhat * x_hat * saved_rstd * gamma
            x_grad = (y_grad - mean_grad_y) * gamma * saved_rstd + var_grad_mean
            x_grad = torch.ops.aten.to(x_grad, torch.float16)
            gamma_grad = torch.ops.aten.to(torch.ops.aten.squeeze(sum_grad_y_xhat), torch.float16)
            beta_grad = torch.ops.aten.to(torch.ops.aten.squeeze(sum_grad_y), torch.float16)
            return x_grad, gamma_grad, beta_grad
        
        return (pattern, replacement, None)

