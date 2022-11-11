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
import torch
import operator

################################################################################
# Graph-level pass to assign streams to different threads to allow streaming
################################################################################
def pass_assign_stream(module, graph):
    default_stream = torch.cuda.default_stream()

    remaining_nodes = []
    executed_nodes = []
    wavefronts = []

    # step 0: get all nodes
    for node in graph.nodes:
        remaining_nodes.append(node)

    while len(remaining_nodes) > 0:

        wavefront = []

        for node in remaining_nodes:
            executable = True
            for arg in node.all_input_nodes:
                if arg in wavefront:
                    executable = False
                if arg not in executed_nodes:
                    executable = False
            
            if executable:
                wavefront.append(node)
                executed_nodes.append(node)
        
        remaining_nodes = [n for n in remaining_nodes if n not in executed_nodes]
        wavefronts.append(wavefront)
    
    for wavefront in wavefronts:
        for node in wavefront:
            if node.op in ["placeholder", "get_attr"]:
                node.meta["stream"] = default_stream
            elif node.op == "call_function":
                if node.target in [torch.ops.aten.view, torch.ops.aten.unsqueeze, operator.getitem, torch.ops.aten.permute, torch.ops.aten.squeeze, torch.ops.aten._unsafe_view]:
                    node.meta["stream"] = node.all_input_nodes[0].meta["stream"]
                else:
                    stream = None
                    if len(node.all_input_nodes) == 1:
                        if len(node.all_input_nodes[0].users) == 1:
                            stream = node.all_input_nodes[0].meta["stream"]
                    if stream is None:
                        stream = torch.cuda.Stream()

                    node.meta["stream"] = stream
                    class CallFunctionWithStream:
                        __name__ = node.target.__name__
                        def __init__(self, node) -> None:
                            # get all input streams
                            self.input_streams = []
                            for input_node in node.all_input_nodes:
                                self.input_streams.append(input_node.meta["stream"])
                            # get self stream
                            self.stream = node.meta["stream"]
                            self.target = node.target
                        
                        def __call__(self, *args, **kwargs):
                            for input_stream in self.input_streams:
                                self.stream.wait_stream(input_stream)
                            with torch.cuda.stream(self.stream):  
                                # print(node.target)          
                                return self.target(*args, **kwargs)

                    node.target = CallFunctionWithStream(node)

            elif node.op == "output":
                # inject the final synchronization node
                node.meta["stream"] = default_stream

                class OutputSyncOp:
                    __name__ = "output_stream_sync"
                    def __init__(self, node) -> None:
                        self.input_streams = []

                        # get self stream
                        self.stream = node.meta["stream"]
                        self.input_nodes = []
                        self.input_indices = []

                        for idx, input_node in enumerate(node.all_input_nodes):
                            input_stream = input_node.meta["stream"]
                            if input_stream != self.stream:
                                self.input_streams.append(input_stream)
                                self.input_nodes.append(input_node)
                                self.input_indices.append(idx)

                        
                        self.target = node.target
                    def __call__(self, *args):
                        for input_stream in self.input_streams:
                            self.stream.wait_stream(input_stream)
                        return args
                
                graph.inserting_before(node)
                output_sync = OutputSyncOp(node)
                input_args = []
                for idx in output_sync.input_indices:
                    input_args.append(node.args[0][idx])
                sync_node = graph.call_function(output_sync, args=tuple(input_args))

                graph.inserting_after(sync_node)
                for idx, output_node in enumerate(output_sync.input_nodes):
                    get_item_node = graph.call_function(operator.getitem, args=(sync_node, idx))
                    graph.inserting_after(get_item_node)

                    node.replace_input_with(output_node, get_item_node)
            else:
                print(node)
                raise NotImplementedError()
    graph.lint()

