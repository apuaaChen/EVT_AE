import numpy as np
import xgboost as xgb
from pycutlass import *
import cutlass
from autotuner.design_space_descriptor import GemmConfigDescriptor

# Number of Features
NUM_FEATURES = 15

def module_profiler(module, input_nodes, tangents, profile_iter=40):
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

    return start.elapsed_time(end) / profile_iter

def sample_implementation_parameters(heuristics):
    # Sample a valid parameter from heuristic
    parameters = heuristics.propose_parameters()
    while not heuristics.is_valid(parameters):
        parameters = heuristics.propose_parameters()
    return parameters

def sample_parameters_without_ML(heuristics, num_samples, checked_configs):
    # Uniformly sample valid parameters given heuristics
    sampled_parameters = []
    while len(sampled_parameters) < num_samples:   
        parameter = heuristics.propose_valid_parameters()     
        parameter_str = str(parameter)
        if parameter_str in checked_configs:
            continue
        else:
            checked_configs.add(parameter_str)
            sampled_parameters.append(parameter)
    return sampled_parameters

def sample_parameters_with_ML(heuristics, num_samples, model, problem_size, checked_configs):
    
    CONFIG_THIS_ROUND = set({})

    sampled_parameters = []
    timeout = 0
    # sample 10 times more samples
    while len(sampled_parameters) < num_samples*10:
        if timeout >= 300:
            break
        parameter = heuristics.propose_valid_parameters()
        parameter_str = str(parameter)
        # print("parameter_str: ", parameter_str)
        if parameter_str in checked_configs or parameter_str in CONFIG_THIS_ROUND:
            timeout += 1
            if timeout % 100 == 0:
                print("timeout: %d" % timeout)
            continue
        else:
            CONFIG_THIS_ROUND.add(parameter_str)
            sampled_parameters.append(parameter)
    # predict performance
    predicted_performance = []
    for i in range(len(sampled_parameters)):
        cur_parameter = sampled_parameters[i]
        feature = heuristics.parameter_to_feature(cur_parameter, problem_size)
        perf = model.predict(feature)[0]
        predicted_performance.append(perf)
    
    predicted_performance = np.array(predicted_performance)
    # select the top $num_samples candidates
    top_idx = np.argsort(predicted_performance)[0:num_samples]
    selected_parameters = [sampled_parameters[idx] for idx in top_idx]

    for param in selected_parameters:
        parameter_str = str(param)
        checked_configs.add(parameter_str)
    return selected_parameters


class Autotuner:
    def __init__(self, 
        autotuning_rounds=10, samples_per_round=10, save_model=False, 
        load_model=True, verbose=False) -> None:
        #
        self.autotuning_rounds = autotuning_rounds
        self.samples_per_round = samples_per_round

        self.verbose = verbose

        

        self.model = xgb.XGBRegressor()
        self.save_model = save_model
        #: indicates whether the model is code
        self.cold = True
        if load_model:
            try:
                self.model.load_model("./xgboost.model")
                self.cold = False
            except:
                if self.verbose:
                    print("Pre-trained model is not found. Train from scratch.")
    
    def tune_module(self, config_descriptor, module, input_nodes, tangents, node_dict):
        problem_size = config_descriptor.problem_description["problem_size"]
        
        self.model = xgb.XGBRegressor()
        self.cold = True
        heuristics = config_descriptor.heuristic

        features = np.zeros((0, len(config_descriptor.CONFIG)))
        labels = np.array([])

        sampled_latency = {
            "mean": [],
            "std": []
        }

        best_latency = 1e+16
        checked_configs = set({})

        for round_idx in range(config_descriptor.autotuning_rounds):
            if self.cold:
                sampled_parameters = sample_parameters_without_ML(
                    heuristics, 
                    num_samples=self.samples_per_round,
                    checked_configs=checked_configs
                )
            else:
                sampled_parameters = sample_parameters_with_ML(
                    heuristics,
                    num_samples=self.samples_per_round,
                    model=self.model,
                    problem_size=problem_size,
                    checked_configs=checked_configs
                )
            profiled_latency = []
            for i in range(self.samples_per_round):
                parameter = sampled_parameters[i]
                if self.verbose:
                    print(parameter)
                assert isinstance(parameter, dict)
                # update all the same node in the module
                # create threadblock shape

                operation = config_descriptor.generate_code(parameter)
                key = config_descriptor.get_key()
                for node in node_dict[key]:
                    pass
                    # node.target.operation = operation
                    # node.target.split_k_slices = parameter['split_k_slices']
                
                latency = module_profiler(module, input_nodes, tangents)
                profiled_latency.append(latency)
            
            features = np.vstack([
                heuristics.parameter_to_feature(
                    parameter, problem_size) for parameter in sampled_parameters
            ])
            labels = profiled_latency
            if self.cold:
                self.model.fit(features, labels)
                self.cold = False
            else:
                self.model.fit(features, labels, xgb_model=self.model)
            array_labels = np.array(labels)
            top_idx = np.argsort(array_labels)[0]
            if self.verbose:
                print("round: ", round_idx, ", best parameter:", sampled_parameters[top_idx], ", measured latency: ", array_labels[top_idx], ", mean sampled latency: ", np.mean(labels))
            if array_labels[top_idx] < best_latency:
                best_parameter = sampled_parameters[top_idx]
                best_latency = array_labels[top_idx]

            sampled_latency["mean"].append(np.mean(labels))
            sampled_latency["std"].append(np.std(labels))
        
    
    def __call__(self, config_descriptor):
        problem_size = config_descriptor.problem_description["problem_size"]
        key = config_descriptor.get_key()
        config_descriptor.cache_init()

        # try load from cache
        best_parameter = config_descriptor.cache_load(key)
        if best_parameter is not None:
            return best_parameter
        
        self.model = xgb.XGBRegressor()
        self.cold = True

        heuristics = config_descriptor.heuristic

        features = np.zeros((0, len(config_descriptor.CONFIG)))
        labels = np.array([])

        sampled_latency = {
            "mean": [],
            "std": []
        }
        best_latency = 1e+16
        checked_configs = set({})

        for round_idx in range(self.autotuning_rounds):
            if self.cold:
                sampled_parameters = sample_parameters_without_ML(
                    heuristics, 
                    num_samples=self.samples_per_round,
                    checked_configs=checked_configs
                )
            else:
                sampled_parameters = sample_parameters_with_ML(
                    heuristics,
                    num_samples=self.samples_per_round,
                    model=self.model,
                    problem_size=problem_size,
                    checked_configs=checked_configs
                )
            profiled_latency = []
            for i in range(min(self.samples_per_round, len(sampled_parameters))):
                parameter = sampled_parameters[i]
                if self.verbose:
                    print(parameter)
                assert isinstance(parameter, dict)
                latency = config_descriptor.generate_code_and_profile(parameter)
                profiled_latency.append(latency)
            
            features = np.vstack([
                heuristics.parameter_to_feature(
                    parameter, problem_size) for parameter in sampled_parameters
            ])
            labels = profiled_latency
            if self.cold:
                self.model.fit(features, labels)
                self.cold = False
            else:
                self.model.fit(features, labels, xgb_model=self.model)
            array_labels = np.array(labels)
            top_idx = np.argsort(array_labels)[0]
            if self.verbose:
                print("round: ", round_idx, ", best parameter:", sampled_parameters[top_idx], ", measured latency: ", array_labels[top_idx], ", mean sampled latency: ", np.mean(labels))
            if array_labels[top_idx] < best_latency:
                best_parameter = sampled_parameters[top_idx]
                best_latency = array_labels[top_idx]

            sampled_latency["mean"].append(np.mean(labels))
            sampled_latency["std"].append(np.std(labels))
        
        if self.verbose:
            print("best parameter:", best_parameter, ", best latency: ", best_latency)

        if self.save_model:
            self.model.save_model("./xgboost.model")

        config_descriptor.cache_insert(list(best_parameter.values()))
        
        return best_parameter


# for test only
if __name__ == "__main__":
    pycutlass.get_memory_pool(init_pool_size=2**10, max_pool_size=2**32)
    pycutlass.compiler.nvcc()
    input_shape = [1024, 1024, 16384]
    autotuner = Autotuner(verbose=True)
    gemm_descriptor = GemmConfigDescriptor(
        input_shape, 
        cutlass.float16, cutlass.RowMajor, 
        cutlass.float16, cutlass.ColumnMajor, 
        cutlass.float16, cutlass.RowMajor, 
        cutlass.float32,
        8, 8, 8, True
    )
    best_parameter = autotuner(gemm_descriptor)
    print(best_parameter)
