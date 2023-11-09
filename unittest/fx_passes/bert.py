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
import torch
import logging
import unittest


def joint_optimization(joint_module):
    frontend = GTLFrontend()
    joint_module, _ = frontend(joint_module)

    pass_loss_elimination(joint_module, joint_module.graph)
    pass_decomposition(joint_module, joint_module.graph)
    pass_cse(joint_module, joint_module.graph)
    pass_constant_propagation(joint_module, joint_module.graph)
    pass_fusion(joint_module, joint_module.graph)
    # pass_print_graph(joint_module, "./bert_optimized.svg")
    pass_clean_up(joint_module, joint_module.graph)
    joint_module.recompile()
    
    return joint_module


class BertTest(BaseTestCase):
    def test_bert_large(self):
        # Create the model
        model, optimizer = bert_model_optimizer(config_file="./large.json")
        model_ref, optimizer_ref = bert_model_optimizer(
            config_file="./large.json", reference=model)
        # Create the sample inputs
        sample_inputs = bert_inputs(batch_size=4, seq_len=512)
        # Cast to fp16
        model, optimizer = apex_autocast(
            model, optimizer, False
        )
        mode_ref, optimizer_ref = apex_autocast(
            model_ref, optimizer_ref, False
        )

        self.run_reference_model(model_ref, optimizer_ref, sample_inputs, 4096.)
    
        model.aot_optimize(
            compiler_fn, compiler_fn,
            partial(
                partition_func,
                joint_compiler=joint_optimization
            )
        )

        model.capture_graph(
            batch=4, sequence_length=512, optimizer=optimizer
        )

        self.run_target_model(model, optimizer, sample_inputs)
        
        self.verify(mode_ref, model, verbose=1)
    def is_close(self, grad1, grad2):
        return (
            torch.sum(
                torch.isclose(grad1, grad2, rtol=2e-1)
            ) / grad1.numel() > 0.9 
            or torch.allclose(grad1, grad2, atol=1e-3)
        )
    

if __name__ == '__main__':
    logging.basicConfig(level=getattr(logging, "DEBUG"))
    logging.basicConfig(format='%(message)s')
    unittest.main()
