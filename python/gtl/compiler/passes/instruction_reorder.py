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
from gtl.compiler.passes.gemm_fusion import FusedGEMM, FusedBMM


################################################################################
# Graph-level pass to assign streams to rearrange operators
################################################################################
def pass_instruction_reorder(module, graph):    
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
    
    for idx, wavefront in enumerate(wavefronts):
        # if idx > 51: continue
        # look for low priority node
        # Our new heuristic is that each low priority GEMM should follow a 
        # high priority GEMM
        non_critical_nodes = []
        for node in wavefront:
            if hasattr(node.target, "stream"):
                if node.target.stream is not None:
                    non_critical_nodes.append(node)
        
        idx = 0
        if len(non_critical_nodes) > 0:
            print("=======================================")
            print(wavefront)
            print(non_critical_nodes)
            insert_point = None
            for node in wavefront:
                if isinstance(node.target, FusedGEMM) or isinstance(node.target, FusedBMM):
                    if node not in non_critical_nodes:
                        insert_point = node
                        break
                    else:
                        break
            
            if insert_point is not None:
                print(insert_point)
                for node in non_critical_nodes:
                    graph.inserting_before(insert_point)
                    new_non_critical_node = graph.node_copy(non_critical_nodes[idx])
                    non_critical_nodes[idx].replace_all_uses_with(new_non_critical_node)
                    graph.erase_node(non_critical_nodes[idx])

            # for node in wavefront:
            #     if isinstance(node.target, FusedGEMM) or isinstance(node.target, FusedBMM):
            #         if node not in non_critical_nodes:
            #             print(node)
            #             graph.inserting_after(list(node.users.keys())[-1])
            #             # graph.inserting_before(list(node)
            #             new_non_critical_node = graph.node_copy(non_critical_nodes[idx])
            #             non_critical_nodes[idx].replace_all_uses_with(new_non_critical_node)
            #             graph.erase_node(non_critical_nodes[idx])
            #             users = new_non_critical_node.users.keys()
            #             # insert get item node
            #             for user in list(users):
            #                 graph.inserting_after(new_non_critical_node)
            #                 new_get_item_node = graph.node_copy(user)
            #                 user.replace_all_uses_with(new_get_item_node)
            #                 graph.erase_node(user)
                        # # inser the stream synchronization node
                        # # node.target.pre_stream_sync = new_non_critical_node.target.stream

                        # class StreamSyncOp:
                        #     __name__ = "uncritical_stream_sync"
                        #     def __init__(self, uncritical_stream) -> None:
                        #         self.uncritical_stream = uncritical_stream
                        #         pass
                            
                        #     def __call__(self, *args):
                        #         print(self.uncritical_stream)
                        #         print(torch.cuda.current_stream())
                        #         self.uncritical_stream.wait_stream(torch.cuda.current_stream())
                        #         return args
                        
                        # graph.inserting_before(node)
                        # stream_sync = StreamSyncOp(new_non_critical_node.target.stream)
                        # sync_node = graph.call_function(stream_sync, args=new_non_critical_node.args)
                        # graph.inserting_after(sync_node)
                        # for idx, arg in enumerate(list(new_non_critical_node.args)):
                        #     get_item_node = graph.call_function(operator.getitem, args=(sync_node, idx))
                        #     graph.inserting_after(get_item_node)
                        #     new_non_critical_node.replace_input_with(arg, get_item_node)

                        # idx += 1

    graph.lint()
    graph.eliminate_dead_code()