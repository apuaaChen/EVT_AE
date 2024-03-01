import matplotlib.pyplot as plt
import os
import re

# Results of each subgraph
results = {
    "bert": ["torch", "inductor", "nvfuser", "evt"],
    "vit": ["torch", "inductor", "nvfuser", "evt"],
    "resnet": ["torch", "torch_channel_last", "inductor", "nvfuser", "evt"],
    "xmlcnn": ["torch", "inductor", "nvfuser", "evt"],
    "gcn": ["torch", "nvfuser", "evt"]
}

colors = {
    "torch_channel_last": "grey",
    "inductor": "orange",
    "nvfuser": "green",
    "evt": "blue"
}

pattern_seconds = r"CUDA time total: (\d+\.\d+)s"
pattern_milliseconds = r"CUDA time total: (\d+\.\d+)ms"

fig, axs = plt.subplots(1, 5, figsize=(25, 5))

def draw_subplot(idx, key):
    axs[idx].set_title(key)
    result = {}
    for mt in results[key]:
        with open(f"./benchmark/{key}/{mt}_results.txt", 'r') as file:
            log = file.read()
            match_seconds = re.search(pattern_seconds, log)
            match_milliseconds = re.search(pattern_milliseconds, log)
            if match_seconds and not match_milliseconds:
                result[mt] = float(match_seconds.group(1))
            elif match_milliseconds and not match_seconds:
                result[mt] = float(match_milliseconds.group(1))
            else:
                raise RuntimeError()
        
    methods = []
    speedups = []
    color = []
    torch_result = None

    for k in result.keys():
        if k == "torch":
            torch_result = result[k]
        else:
            methods.append(k)
            speedups.append(torch_result / result[k])
            color.append(colors[k])

    axs[idx].bar(methods, speedups, width=0.5, color=color)
    if key in ["gcn"]:
        axs[idx].set_ylim(0.5)
    elif key in ["vit"]:
        axs[idx].set_ylim(0.9)
    else:
        axs[idx].set_ylim(1)
    axs[idx].text(len(speedups)-1, speedups[-1] + 0.02, f'{speedups[-1]:.2f}x', ha='center')
    axs[idx].grid(True)


draw_subplot(0, "bert")
draw_subplot(1, "vit")
draw_subplot(2, "resnet")
draw_subplot(3, "xmlcnn")
draw_subplot(4, "gcn")

plt.tight_layout()
plt.savefig("./figure12.png")