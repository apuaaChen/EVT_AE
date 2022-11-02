import numpy as np
from autotuner.code_generator import generate_code
from autotuner.expert_knowledge_metafile import a100_metafile, Heuristics
import xgboost as xgb


from pycutlass.test.profiler import GpuTimer
from pycutlass import *
import cutlass
import torch
import sqlite3

# Number of Features
NUM_FEATURES = 15


def sample_implementation_parameters(heuristics):
    parameters = heuristics.propose_parameters()
    while not heuristics.is_valid(parameters):
        parameters = heuristics.propose_parameters()
    return parameters

def sample_parameters_without_ML(heuristics, num_samples, num_features, problem_size, checked_configs):
    # metafile_parameters: hardware constraints of the kernel
    metafile_parameters = heuristics.get_feature_from_metafile()
    
    sampled_parameter = np.zeros((0,num_features))
    while sampled_parameter.shape[0] < num_samples:        
        implementation_parameters = sample_implementation_parameters(heuristics)
        implementation_parameters = heuristics.get_feature_from_parameters(implementation_parameters)
        # [M, N, K, Mw, Nw, Kw, Stage, Swizzle, MaxSHmem, MaxNumReg/thread, global mem size, register file size per SM]
        parameter = implementation_parameters + metafile_parameters + problem_size
        parameter_str = str(parameter)
        # print("parameter_str: ", parameter_str)
        if parameter_str in checked_configs:
            continue
        else:
            parameter = np.array(parameter)
            checked_configs.add(parameter_str)
            sampled_parameter = np.vstack([sampled_parameter, parameter])
    return sampled_parameter

def sample_parameters_with_ML(heuristics, num_samples, num_features, model, problem_size, checked_configs):
    metafile_parameters = heuristics.get_feature_from_metafile()
    
    CONFIG_THIS_ROUND = set({})

    sampled_parameter = np.zeros((0,num_features))
    # sample 10 times more samples
    while sampled_parameter.shape[0] < num_samples*10:
        implementation_parameters = sample_implementation_parameters(heuristics)
        implementation_parameters = heuristics.get_feature_from_parameters(implementation_parameters)
        parameter = implementation_parameters + metafile_parameters + problem_size
        parameter_str = str(parameter)
        # print("parameter_str: ", parameter_str)
        if parameter_str in checked_configs or parameter_str in CONFIG_THIS_ROUND:
            continue
        else:
            CONFIG_THIS_ROUND.add(parameter_str)
            parameter = np.array(parameter)
            sampled_parameter = np.vstack([sampled_parameter, parameter])
    # predict performance
    predicted_performance = []
    for i in range(10*num_samples):
        cur_parameter = sampled_parameter[i]
        cur_parameter = cur_parameter.reshape((1, num_features))
        # perf = predict(model, cur_parameter)[0]
        perf = model.predict(cur_parameter)[0]
        predicted_performance.append(perf)
    
    predicted_performance = np.array(predicted_performance)
    # select the top $num_samples candidates
    top_idx = np.argsort(predicted_performance)[0:num_samples]
    selected_parameters = sampled_parameter[top_idx]

    for i in range(selected_parameters.shape[0]):
        parameter_str = str(selected_parameters[i])
        checked_configs.add(parameter_str)
    return selected_parameters


def generate_code_and_profile(
    parameters, input_shape, element_a, layout_a, element_b, layout_b, 
    element_c, layout_c, element_accumulator):
    #
    operation = generate_code(
        parameter=parameters, element_a=element_a, layout_a=layout_a,
        element_b=element_b, layout_b=layout_b, element_c=element_c,
        layout_c=layout_c, element_accumulator=element_accumulator)
    M, N, K = input_shape

    tensor_A = torch.empty(size=(M * K,), dtype=torch.float16, device="cuda")
    tensor_B = torch.empty(size=(N * K,), dtype=torch.float16, device="cuda")
    tensor_C = torch.empty(size=(M * N,), dtype=torch.float16, device="cuda")

    arguments = GemmArguments(
        operation=operation, problem_size=cutlass.gemm.GemmCoord(M, N, K),
        A=tensor_A, B=tensor_B, C=tensor_C, D=tensor_C, 
        output_op = operation.epilogue_type(1.0, 0.0),
        gemm_mode=cutlass.gemm.Mode.Gemm, split_k_slices=1
    )
    
    # warmup iterations
    for _ in range(200):
        operation.run(arguments)
    
    timer = GpuTimer()

    # profiling iterations
    timer.start()
    for _ in range(200):
        operation.run(arguments)
    timer.stop_and_wait()

    return timer.duration(200)


class ConfigCache:
    def __init__(self) -> None:
        self.arg_key = {
            cutlass.float16: "f16",
            cutlass.bfloat16: "bf16",
            cutlass.float32: "f32",
            cutlass.RowMajor: "row",
            cutlass.ColumnMajor: "col"
        }

        try:
            connection = sqlite3.connect("./compiled_cache.db")
            cursor = connection.cursor()
            sqlite_create_table_query = """CREATE TABLE best_config(
                op_key TEXT NOT NULL UNIQUE, block_x INTEGER, block_y INTEGER, 
                block_z INTEGER, warp_x INTEGER, warp_y INTEGER, warp_z INTEGER, 
                stage INTEGER, swizzle INTEGER)"""
            cursor.execute(sqlite_create_table_query)
            connection.commit()
            cursor.close()
        except:
            pass
        # pass
    
    def get_key(self, *args):
        key = ""
        for arg in list(args):
            try:
                key += self.arg_key[arg]
            except:
                for size in arg:
                    key += "_%d" % size
                key += "_"
        return key
                

    
    def insert(self, best_config, *args):
        key = self.get_key(*args)
        connection = sqlite3.connect("./compiled_cache.db")
        cursor = connection.cursor()
        sqlite_insert_blob_query = """ INSERT OR IGNORE INTO 
            best_config (op_key, block_x, block_y, block_z, warp_x, 
                warp_y, warp_z, stage, swizzle) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"""
        cursor.execute(sqlite_insert_blob_query, tuple([key,] + best_config))
        connection.commit()
        cursor.close()
    
    def load(self, *args):
        key = self.get_key(*args)

        connection = sqlite3.connect("./compiled_cache.db")
        cursor = connection.cursor()
        sqlite_fetch_blob_query = """SELECT * from best_config where op_key = ?"""

        cursor.execute(sqlite_fetch_blob_query, (key, ))
        record = cursor.fetchall()
        if len(record) == 0:
            return None
        else:
            return list(record[0])[1:]


class Autotuner:
    def __init__(self, 
        autotuning_rounds=10, samples_per_round=10, save_model=True, 
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
        
        self.cache = ConfigCache()
    
    def __call__(self, problem_size, element_a, layout_a, element_b, layout_b, 
        element_c, layout_c, element_accumulator):

        assert layout_c == cutlass.RowMajor, "Currently only support RowMajor output"

        # try load from cache
        best_parameter = self.cache.load(
            problem_size, element_a, layout_a, element_b, layout_b,
            element_c, layout_c, element_accumulator
        )
        if best_parameter is not None:
            return best_parameter

        # search for best config
        heuristics = Heuristics(
            a100_metafile, element_a=element_a, element_b=element_b,
            element_c=element_c, element_accumulator=element_accumulator,
            layout_a=layout_a, layout_b=layout_b
        )

        features = np.zeros((0, NUM_FEATURES))
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
                    num_features=NUM_FEATURES,
                    problem_size=problem_size,
                    checked_configs=checked_configs
                )
            else:
                sampled_parameters = sample_parameters_with_ML(
                    heuristics,
                    num_samples=self.samples_per_round,
                    num_features=NUM_FEATURES,
                    model=self.model,
                    problem_size=problem_size,
                    checked_configs=checked_configs
                )
            profiled_latency = []
            for i in range(self.samples_per_round):
                parameter = tuple(sampled_parameters[i][:8])
                if self.verbose:
                    print(parameter)
                latency = generate_code_and_profile(
                    parameter, problem_size, element_a=element_a, layout_a=layout_a,
                    element_b=element_b, layout_b=layout_b, element_c=element_c,
                    layout_c=layout_c, element_accumulator=element_accumulator
                )
                profiled_latency.append(latency)
            
            features = sampled_parameters
            labels = profiled_latency
            if self.cold:
                self.model.fit(features, labels)
                self.cold = False
            else:
                self.model.fit(features, labels, xgb_model=self.model)
            array_labels = np.array(labels)
            top_idx = np.argsort(array_labels)[0]
            if self.verbose:
                print("round: ", round_idx, ", best parameter:", features[top_idx][:8], ", measured latency: ", array_labels[top_idx], ", mean sampled latency: ", np.mean(labels))
            if array_labels[top_idx] < best_latency:
                best_parameter = features[top_idx][:8]
                best_latency = array_labels[top_idx]

            sampled_latency["mean"].append(np.mean(labels))
            sampled_latency["std"].append(np.std(labels))
                
        # with open("./log/sampled_latency.json", "w") as outfile:
        #     json.dump(sampled_latency, outfile)

        best_parameter = [int(p) for p in best_parameter]
        
        if self.verbose:
            print("best parameter:", best_parameter, ", best latency: ", best_latency)

        if self.save_model:
            self.model.save_model("./xgboost.model")
        
        self.cache.insert(best_parameter, problem_size, element_a, layout_a, 
            element_b, layout_b, element_c, layout_c, element_accumulator)
        return best_parameter


# for test only
if __name__ == "__main__":
    pycutlass.get_memory_pool(init_pool_size=2**10, max_pool_size=2**32)
    pycutlass.compiler.nvcc()
    input_shape = [3584, 32320, 1024]
    autotuner = Autotuner(verbose=True)
    autotuner(
        input_shape, cutlass.float16, cutlass.RowMajor, cutlass.float16, 
        cutlass.ColumnMajor, cutlass.float16, cutlass.RowMajor, cutlass.float32)
