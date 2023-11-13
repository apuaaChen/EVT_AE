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
from model_zoo.bert import prepare_model_and_optimizer as bert_model_optimizer
from model_zoo.bert import example_inputs as bert_inputs
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
        sample_inputs = bert_inputs(batch_size=batch_size, seq_len=512)

        model, optimizer = bert_model_optimizer(config_file="./large.json")
        model, optimizer = apex_autocast(
            model, optimizer, False
        )

        if self.method == "gtl":
            model.aot_optimize(
                compiler_fn, compiler_fn,
                partial(
                    partition_func,
                    joint_compiler=joint_optimization
                )
            )

        model.capture_graph(
            batch=batch_size, sequence_length=512, optimizer=optimizer
        )

        for _ in range(10):
            self.run_target_model(model, optimizer, sample_inputs)
        
        with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
            with record_function(self.method):
                for _ in range(10):
                    self.run_target_model(model, optimizer, sample_inputs)

        print(prof.key_averages().table(sort_by="cuda_time_total"))
    

if __name__ == '__main__':
    ################################################################################
    # parse args
    parser = argparse.ArgumentParser(description="Bert End-to-End Training with CUDA Graph")
    parser.add_argument('--iter', '-it', type=int, default=50, help="Profiling Iterations")
    # Hyper-parameter that defines the model size
    parser.add_argument('--batch_size', '-b', type=int, default=32, help="Training batch size per GPU")
    parser.add_argument('--seq_len', '-l', type=int, default=512, help="Sequence length")
    # method
    parser.add_argument('--method', '-mt', type=str, default="torch", choices=["torch", "gtl"])
    args = parser.parse_args()

    ################################################################################

    profiler = BertProfile(args.method)
    profiler.profile_bert_large()
