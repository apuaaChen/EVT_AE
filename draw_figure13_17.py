import matplotlib.pyplot as plt
import re
import pandas as pd
import os
from io import StringIO


# Results of each subgraph
results = {
    "bert": ["torch", "triton", "tvm", "evt"],
    "mlp": ["torch", "triton", "tvm", "evt"],
    "resnet": ["torch", "triton", "tvm", "evt"],
    "xmlcnn": ["torch", "triton", "tvm", "evt"],
    "gcn": ["torch", "triton", "tvm", "evt"],
}

offsets = {
    "bert": 0,
    "mlp": 0.155,
    "resnet": 0.058,
    "xmlcnn": 0.305,
    "gcn": 22.270
}
colors = {
    "torch": "grey",
    "triton": "orange",
    "tvm": "green",
    "evt": "blue"
}

pattern_seconds = r"CUDA time total: (\d+\.\d+)s"
pattern_milliseconds = r"CUDA time total: (\d+\.\d+)ms"

fig, axs = plt.subplots(1, 5, figsize=(25, 5))


def preprocess_tvm_results(key):
    # Step 1: grab autotvm & ansor result
    tvm_results = []
    for backend in ["autotvm", "ansor"]:
        path = f"./benchmark/subgraphs/{key}/{backend}_results.txt"
        if not os.path.exists(path):
            continue
        with open(f"./benchmark/subgraphs/{key}/{backend}_results.txt", 'r') as file:
            log = file.read()
            # Extract the table
            lines = log.split('\n')
            table_str = ""
            marker_cnt = 0
            for line in lines:
                if "------------------" in line:
                    marker_cnt += 1
                    if marker_cnt == 1:
                        continue
                if marker_cnt > 0 and marker_cnt < 3:
                    table_str += line + "\n"
            # Get number of 
            df = pd.read_csv(StringIO(table_str), delimiter=r"\s{2,}", engine='python')
            result_df = df[["Name", "CUDA total"]].to_dict(orient='records')
            
            results = {}
            # Clean the dict, combine it with the other record in ansor
            for op in result_df:
                if '-----------' in op['Name'] or op["CUDA total"] is None:
                    continue
                # unify the op time to ms
                cuda_time = op["CUDA total"]
                if "us" in cuda_time:
                    cuda_time = float(cuda_time[:-2]) / 1000.
                elif "ms" in cuda_time:
                    cuda_time = float(cuda_time[:-2])
                else:
                    raise NotImplementedError()

                results[op['Name']] = cuda_time
            
            tvm_results.append(results)
    
    # Step 2: unify the results from both sides
    tvm_cuda_time_total = 0

    if len(tvm_results) == 1:
        tvm_result = tvm_results[0]
        for op in tvm_result.keys():
            if key == "xmlcnn":
                if op in [
                    "tvmgen_default_fused_transpose_kernel",
                    "tvmgen_default_fused_transpose_1_kernel",
                    "Memcpy DtoD (Device -> Device)"
                ]:
                    continue
            
            tvm_cuda_time_total += tvm_result[op]
    else:
        autotvm_result = tvm_results[0]
        ansor_result = tvm_results[1]
        
        for op in autotvm_result.keys():
            # Filter for MLP-TVM
            if key == "mlp":
                if op in [
                    "tvmgen_default_fused_transpose_4_kernel",
                    "tvmgen_default_fused_transpose_3_kernel",
                    "tvmgen_default_fused_transpose_2_kernel",
                    "tvmgen_default_fused_transpose_1_kernel",
                    "tvmgen_default_fused_transpose_kernel"]:
                    continue
            elif key == "resnet":
                # TVM cannot generate efficient conv in this case
                # Fallback to cudnn
                if op == "tvmgen_default_fused_nn_conv2d_kernel":
                    tvm_cuda_time_total += 3.097
                    continue
            elif key == "xmlcnn":
                if op in [
                    "tvmgen_default_fused_transpose_kernel",
                    "tvmgen_default_fused_transpose_1_kernel"
                ]:
                    continue
            if op not in ansor_result:
                continue
            op_dur = min(autotvm_result[op], ansor_result[op])
            tvm_cuda_time_total += op_dur
    return tvm_cuda_time_total


def draw_subplot(idx, key):
    axs[idx].set_title(f"Figure {idx + 13}")
    result = {}
    for mt in results[key]:
        if mt == "tvm":
            result[mt] = preprocess_tvm_results(key)
            continue
        with open(f"./benchmark/subgraphs/{key}/{mt}_results.txt", 'r') as file:
            log = file.read()
            match_seconds = re.search(pattern_seconds, log)
            match_milliseconds = re.search(pattern_milliseconds, log)
            if match_seconds and not match_milliseconds:
                result[mt] = float(match_seconds.group(1)) * 1000
            elif match_milliseconds and not match_seconds:
                result[mt] = float(match_milliseconds.group(1))
            else:
                raise RuntimeError()
    
    methods = []
    duration = []
    speedup = []
    color = []
    torch_result = None
    offset = offsets[key]

    for k in result.keys():
        if k == "torch":
            torch_result = result[k] - offset
        methods.append(k)
        duration.append((result[k] - offset)  / torch_result)
        speedup.append(torch_result / (result[k] - offset))
        color.append(colors[k])
    
    axs[idx].bar(methods, duration, width=0.5, color=color)
    for i in range(len(methods)):
        axs[idx].text(i, 1./speedup[i] + 0.02, f'{speedup[i]:.2f}x', ha='center')

draw_subplot(0, "bert")
draw_subplot(1, "mlp")
draw_subplot(2, "resnet")
draw_subplot(3, "xmlcnn")
draw_subplot(4, "gcn")

plt.tight_layout()
plt.savefig("./figure13_17.png")