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
# Unit test for end-to-end xml-cnn
from gtl.helper import compiler_fn, partition_func, BaseTestCase, apex_autocast
from model_zoo.xmlcnn import prepare_model_and_optimizer as xmlcnn_model_optimizer
from model_zoo.xmlcnn import example_inputs as xmlcnn_input
from model_zoo.xmlcnn import Params
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
from torch.profiler import profile, ProfilerActivity, record_function

################################################################################
# Model Configuration
params = Params(
    embedding_dim=304,  # alignment of 8
    filter_sizes=[2, 4, 8],
    sequence_length=512,
    batch_size=1024,
    num_filters=32,
    y_dim=670208,
    hidden_dims=512,
    pooling_units=32
)

def joint_optimization(joint_module):
    frontend = GTLFrontend()
    joint_module, _ = frontend(joint_module)

    pass_loss_elimination(joint_module, joint_module.graph)
    pass_decomposition(joint_module, joint_module.graph)
    pass_cse(joint_module, joint_module.graph)
    pass_constant_propagation(joint_module, joint_module.graph)
    pass_fusion(joint_module, joint_module.graph)
    pass_print_graph(joint_module, "./xmlcnn_optimized.svg")
    # pass_clean_up(joint_module, joint_module.graph)
    # joint_module.recompile()
    
    return joint_module


class XMLCNNTest(BaseTestCase):
    def test_xmlcnn(self):
        # Create the sample input
        sample_inputs = xmlcnn_input(params)
        # Create the model
        model_ref, optimizer_ref = xmlcnn_model_optimizer(params)

        # Cast to fp16
        model_ref, optimizer_ref = apex_autocast(
            model_ref, optimizer_ref, False
        )

        self.run_reference_model(model_ref, optimizer_ref, sample_inputs, 1)

        model, optimizer = xmlcnn_model_optimizer(params, reference=model_ref)
        model, optimizer = apex_autocast(
            model, optimizer, False
        )

        model.aot_optimize(
            compiler_fn, compiler_fn,
            partial(
                partition_func,
                joint_compiler=joint_optimization
            )
        )

        model.capture_graph(
            params.batch_size, params.sequence_length, params.embedding_dim,
            params.y_dim, optimizer=optimizer
        )

        self.run_target_model(model, optimizer, sample_inputs)
        self.verify(model_ref, model, verbose=1)

        with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
            with record_function("evt"):
                self.run_target_model(model, optimizer, sample_inputs)

        print(prof.key_averages().table(sort_by="cuda_time_total"))
        
        with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
            with record_function("torch"):
                self.run_reference_model(model_ref, optimizer_ref, sample_inputs, 4096.)
        print(prof.key_averages().table(sort_by="cuda_time_total"))

    def is_close(self, grad1, grad2):
        return torch.sum(
                torch.isclose(grad1, grad2, rtol=1e-1)) / grad1.numel() > 0.9

if __name__ == '__main__':
    logging.basicConfig(level=getattr(logging, "DEBUG"))
    logging.basicConfig(format='%(message)s')
    unittest.main()
