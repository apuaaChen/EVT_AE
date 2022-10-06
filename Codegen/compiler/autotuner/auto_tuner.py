import numpy as np
from code_generator import generate_code
from expert_knowledge_metafile import a100_metafile, Heuristics
from cost_model import predict, train_model, collect_dataset


from pycutlass.test.profiler import GpuTimer
from pycutlass import *
import cutlass
import torch

# claim memory pool
pycutlass.get_memory_pool(init_pool_size=2**10, max_pool_size=2**32)
pycutlass.compiler.nvcc()

# Number of autotuning rounds
NUM_AUTOTUNING_ROUNDS = 5
# Number of samples per round
NUM_SAMPLES = 10
# Number of Features
NUM_FEATURES = 12

# number of skipped configuration caused by error
SKIPPED_CONFIG = 0

CHECKED_CONFIGs = set({})

def sample_implementation_parameters(heuristics):
    parameters = heuristics.propose_parameters()
    while not heuristics.is_valid(parameters):
        parameters = heuristics.propose_parameters()
    return parameters

def sample_parameters_without_ML(heuristics, num_samples, num_features):
    global CHECKED_CONFIGs
    # metafile_parameters: hardware constraints of the kernel
    metafile_parameters = heuristics.get_feature_from_metafile()
    
    sampled_parameter = np.zeros((0,num_features))
    while sampled_parameter.shape[0] < num_samples:        
        implementation_parameters = sample_implementation_parameters(heuristics)
        implementation_parameters = heuristics.get_feature_from_parameters(implementation_parameters)
        # [M, N, K, Mw, Nw, Kw, Stage, Swizzle, MaxSHmem, MaxNumReg/thread, global mem size, register file size per SM]
        parameter = implementation_parameters + metafile_parameters
        parameter_str = str(parameter)
        # print("parameter_str: ", parameter_str)
        if parameter_str in CHECKED_CONFIGs:
            continue
        else:
            parameter = np.array(parameter)
            CHECKED_CONFIGs.add(parameter_str)
            sampled_parameter = np.vstack([sampled_parameter, parameter])
    return sampled_parameter

def sample_parameters_with_ML(heuristics, num_samples, num_features, model):
    metafile_parameters = heuristics.get_feature_from_metafile()
    
    CONFIG_THIS_ROUND = set({})

    sampled_parameter = np.zeros((0,num_features))
    # sample 10 times more samples
    while sampled_parameter.shape[0] < num_samples*10:
        implementation_parameters = sample_implementation_parameters(heuristics)
        implementation_parameters = heuristics.get_feature_from_parameters(implementation_parameters)
        parameter = implementation_parameters + metafile_parameters
        parameter_str = str(parameter)
        # print("parameter_str: ", parameter_str)
        if parameter_str in CHECKED_CONFIGs or parameter_str in CONFIG_THIS_ROUND:
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
        perf = predict(model, cur_parameter)[0]
        predicted_performance.append(perf)
    
    predicted_performance = np.array(predicted_performance)
    # select the top $num_samples candidates
    top_idx = np.argsort(predicted_performance)[0:num_samples]
    selected_parameters = sampled_parameter[top_idx]

    for i in range(selected_parameters.shape[0]):
        parameter_str = str(selected_parameters[i])
        CHECKED_CONFIGs.add(parameter_str)
    return selected_parameters


def generate_code_and_profile(parameters, input_shape):
    operation = generate_code(parameter=parameters)
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
    for _ in range(100):
        operation.run(arguments)
    
    timer = GpuTimer()

    # profiling iterations
    timer.start()
    for _ in range(100):
        operation.run(arguments)
    timer.stop_and_wait()

    return timer.duration(100)


def autotuner(input_shape):
    global SKIPPED_CONFIG
    heuristics = Heuristics(
        a100_metafile, element_a=cutlass.float16, element_b=cutlass.float16, 
        element_c=cutlass.float16,
        element_accumulator=cutlass.float32, layout_a=cutlass.RowMajor, 
        layout_b=cutlass.RowMajor)
    features = np.zeros((0, NUM_FEATURES))
    labels = np.array([])

    # Autotuning for NUM_AUTOTUNING_ROUNDS
    for round_idx in range(NUM_AUTOTUNING_ROUNDS):
        print("round: ", round_idx)
        if round_idx == 0:
            # sampled_parameters
            # stack of 
            # [M, N, K, Mw, Nw, Kw, Stage, Swizzle, MaxSHmem, MaxNumReg/thread, 
            #  global mem size, register file size per SM]
            sampled_parameters = sample_parameters_without_ML(
                heuristics, 
                num_samples=NUM_SAMPLES,
                num_features=NUM_FEATURES,
                )
        else:
            sampled_parameters = sample_parameters_with_ML(
                heuristics,
                num_samples=NUM_SAMPLES,
                num_features=NUM_FEATURES,
                model=model,
            )
        profiled_latency = []
        for i in range(NUM_SAMPLES):
            parameter = sampled_parameters[i][:8]
            parameter = tuple(parameter)
            # try:
            print(parameter)
            latency = generate_code_and_profile(parameter, input_shape)
            profiled_latency.append(latency)
            # except:
            #     SKIPPED_CONFIG += 1
            #     print("SKIPPED_CONFIG: ", SKIPPED_CONFIG)
                
            #     exit()
        
        # Append newly sampled (features, labels) to previous (features, labels)
        (features, labels) = collect_dataset(
            features=sampled_parameters,
            labels=profiled_latency,
            existing_feature=features,
            existing_label=labels,
            num_feature=NUM_FEATURES,
        )

        model = train_model(features, labels)
        array_labels = np.array(labels)
        top_idx = np.argsort(array_labels)[0]
        print("round: ", round_idx, ", best parameter:", features[top_idx], ", measured latency: ", array_labels[top_idx], ", mean sampled latency: ", np.mean(labels))

        # hand tuned kernel
        latency = generate_code_and_profile(parameters=[128, 128, 32, 64, 64, 32, 5, 3], input_shape=input_shape)
        print(latency)


# [M, N, K]
input_shape = (3584, 32320, 1024)
autotuner(input_shape)