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
# Unit test for end-to-end bert
from gtl.helper import compiler_fn, partition_func, BaseTestCase, apex_autocast
from model_zoo.gcn import prepare_model_and_optimizer as gcn_model_optimizer
from model_zoo.gcn import example_inputs as gcn_input
from functools import partial
from gtl.compiler.passes import (
    GTLFrontend, pass_loss_elimination,
    pass_decomposition, pass_cse,
    pass_constant_propagation,
    pass_fusion,
    pass_clean_up,
    pass_print_graph)
from torch.profiler import profile, ProfilerActivity, record_function
import argparse

batch_size = 32


def joint_optimization(joint_module):
    frontend = GTLFrontend()
    joint_module, _ = frontend(joint_module)

    pass_loss_elimination(joint_module, joint_module.graph)
    pass_decomposition(joint_module, joint_module.graph)
    pass_cse(joint_module, joint_module.graph)
    pass_constant_propagation(joint_module, joint_module.graph)
    pass_fusion(joint_module, joint_module.graph)
    pass_clean_up(joint_module, joint_module.graph)
    joint_module.recompile()
    
    return joint_module

class BertProfile(BaseTestCase):
    
    def __init__(self, method, methodName: str = "runTest") -> None:
        super().__init__(methodName)        
        self.method = method

    def profile_bert_large(self):
        # Create the sample inputs
        features, labels, csr, csc, in_feats, n_classes = gcn_input("ogbn-mag")
        sample_inputs = [features, labels]

        if self.method == "gtl":
            f32_loss = False
        else:
            f32_loss = True

        if self.method in ["hand_tuned", "nvprims_nvfuser", "aot_ts_nvfuser"]:
            apex_loss = True
        else:
            apex_loss = False
        model, optimizer = gcn_model_optimizer(in_feats, n_classes, csr, csc, f32_loss=f32_loss, apex_loss=apex_loss)
        model, optimizer = apex_autocast(
            model, optimizer, True
        )

        if self.method == "gtl":
            model.aot_optimize(
                compiler_fn, compiler_fn,
                partial(
                    partition_func,
                    joint_compiler=joint_optimization
                )
            )
        elif self.method in ["inductor"]:
            model.torch_compile(backend=self.method, mode="max-autotune")
        elif self.method in ["aot_ts_nvfuser", "nvprims_nvfuser"]:
            model.torch_compile(backend=self.method)

        if self.method in ["torch", "gtl", "aot_ts_nvfuser", "nvprims_nvfuser", "hand_tuned"]:
            model.capture_graph(
                features, labels, optimizer=optimizer
            )
            model.set_features(features, labels)

            for _ in range(10):
                self.run_target_model(model, optimizer, [])
            
            with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
                with record_function(self.method):
                    for _ in range(10):
                        self.run_target_model(model, optimizer, [])
        else:
            # Max-autotune naturally has cuda graph
            for _ in range(10):
                self.run_reference_model(model, optimizer, sample_inputs, 4096.)
            with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
                with record_function(self.method):
                    for _ in range(10):
                        self.run_reference_model(model, optimizer, sample_inputs, 4096.)

        print(prof.key_averages().table(sort_by="cuda_time_total"))
    

if __name__ == '__main__':
    ################################################################################
    # parse args
    parser = argparse.ArgumentParser(description="Bert End-to-End Training with CUDA Graph")
    # method
    parser.add_argument('--method', '-mt', type=str, default="torch", choices=["torch", "gtl", "inductor", "aot_ts_nvfuser", "nvprims_nvfuser", "hand_tuned"])
    args = parser.parse_args()

    ################################################################################

    profiler = BertProfile(args.method)
    profiler.profile_bert_large()