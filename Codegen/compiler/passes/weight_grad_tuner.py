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

from passes.gemm_fusion import FusedGEMM
import torch
from pycutlass.test.profiler import GpuTimer
import nvtx
from autotuner.auto_tuner import sample_parameters_with_ML, sample_parameters_without_ML
import numpy as np
import xgboost as xgb


################################################################################
# Graph-level pass to update the weight gradient kernels
################################################################################


def module_profiler(module, graph, profile_iter=10):
    input_nodes = []
    tangents = []
    for node in graph.nodes:
        if node.op == "placeholder":
            if "tangents" in str(node.target):
                tangents.append(
                    torch.ones(
                        size=tuple(node.meta["tensor_meta"].shape), 
                        dtype=node.meta["tensor_meta"].dtype, device="cuda")
                )
            else:
                input_nodes.append(
                    torch.ones(
                        size=tuple(node.meta["tensor_meta"].shape), 
                        dtype=node.meta["tensor_meta"].dtype, device="cuda")
                )
    module.recompile()
    for _ in range(10):
        module(input_nodes, tangents)
    
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(profile_iter):
        output = module(input_nodes, tangents)
    end.record()
    torch.cuda.synchronize()

    print(start.elapsed_time(end) / profile_iter)

    return start.elapsed_time(end) / profile_iter

def pass_weight_gradient_tuner(module, graph):
    weight_gradient_config = {}
    for node in graph.nodes:
        if node.op == "call_function":
            # get the GEMM nodes
            if isinstance(node.target, FusedGEMM):
                if node.target.is_direct_output:
                    gemm_descriptor = node.target.gemm_descriptor
                    key = gemm_descriptor.get_key()
                    if key not in weight_gradient_config:
                        weight_gradient_config[key] = [node]
                    else:
                        weight_gradient_config[key].append(node)
    

    # need helper function to profile the latency of the whole model w/ cuda graph
    # step 1: get static inputs
    input_nodes = []
    tangents = []
    for node in graph.nodes:
        if node.op == "placeholder":
            if "tangents" in str(node.target):
                tangents.append(
                    torch.ones(
                        size=tuple(node.meta["tensor_meta"].shape), 
                        dtype=node.meta["tensor_meta"].dtype, device="cuda")
                )
            else:
                input_nodes.append(
                    torch.ones(
                        size=tuple(node.meta["tensor_meta"].shape), 
                        dtype=node.meta["tensor_meta"].dtype, device="cuda")
                )
    for key in weight_gradient_config.keys():
        config_descriptor = weight_gradient_config[key][0].target.gemm_descriptor
        problem_size = config_descriptor.problem_description["problem_size"]
        model = xgb.XGBRegressor()
        cold = True
        heuristics = config_descriptor.heuristic

        features = np.zeros((0, len(config_descriptor.CONFIG)))
        labels = np.array([])

        sampled_latency = {
            "mean": [],
            "std": []
        }

        best_latency = 1e+16
        global_key = "global____" + config_descriptor.get_key()
        best_parameter = config_descriptor.cache_load(global_key)
        if best_parameter is None:
            checked_configs = set({})
            autotuning_rounds = 15
            samples_per_round = 10
            verbose = True

            for round_idx in range(autotuning_rounds):
                if cold:
                    sampled_parameters = sample_parameters_without_ML(
                        heuristics, 
                        num_samples=samples_per_round,
                        checked_configs=checked_configs
                    )
                else:
                    sampled_parameters = sample_parameters_with_ML(
                        heuristics,
                        num_samples=samples_per_round,
                        model=model,
                        problem_size=problem_size,
                        checked_configs=checked_configs
                    )
                profiled_latency = []
                for i in range(samples_per_round):
                    parameter = sampled_parameters[i]
                    if verbose:
                        print(parameter)
                    assert isinstance(parameter, dict)
                    # update all the same node in the module
                    # create threadblock shape

                    operation = config_descriptor.generate_code(parameter)
                    for node in weight_gradient_config[key]:
                        node.target.operation = operation
                        node.target.split_k_slices = parameter['split_k_slices']
                    
                    module.recompile()
                    for _ in range(10):
                        module(input_nodes, tangents)
                    
                    torch.cuda.synchronize()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    with nvtx.annotate("weight grad tune"):
                        for i in range(10):
                            output = module(input_nodes, tangents)
                    end.record()
                    torch.cuda.synchronize()

                    latency = start.elapsed_time(end) / 10
                    profiled_latency.append(latency)
                
                features = np.vstack([
                    heuristics.parameter_to_feature(
                        parameter, problem_size) for parameter in sampled_parameters
                ])
                labels = profiled_latency
                if cold:
                    model.fit(features, labels)
                    cold = False
                else:
                    model.fit(features, labels, xgb_model=model)
                array_labels = np.array(labels)
                top_idx = np.argsort(array_labels)[0]
                if verbose:
                    print("round: ", round_idx, ", best parameter:", sampled_parameters[top_idx], ", measured latency: ", array_labels[top_idx], ", mean sampled latency: ", np.mean(labels))
                if array_labels[top_idx] < best_latency:
                    best_parameter = sampled_parameters[top_idx]
                    best_latency = array_labels[top_idx]

                sampled_latency["mean"].append(np.mean(labels))
                sampled_latency["std"].append(np.std(labels))
            
            print("best parameter:", best_parameter, ", best latency: ", best_latency)
            config_descriptor.cache_insert_with_key(global_key, list(best_parameter.values()))

        # assign best parameter
        for node in weight_gradient_config[key]:
            operation = config_descriptor.generate_code(best_parameter)
            node.target.operation = operation
            node.target.split_k_slices = best_parameter['split_k_slices']




    # print(start.elapsed_time(end) / 10)
        # module.recompile()
        # latency = module_profiler(module, input_nodes, tangents)
        # print(latency)
    # autotuner.tune_module(
    #     weight_gradient_config["GEMMf16f16f16f32colrowrow_8_8_8_Serial_1024_1024_16384_"][0].target.gemm_descriptor,
    #     module, input_nodes, tangents, weight_gradient_config)

    
    # print(weight_gradient_config)
