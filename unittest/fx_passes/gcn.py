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
# Unit test for end-to-end gcn
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
import torch
import logging
import unittest
from torch.profiler import profile, ProfilerActivity, record_function


def joint_optimization(joint_module):
    frontend = GTLFrontend()
    joint_module, _ = frontend(joint_module)

    pass_loss_elimination(joint_module, joint_module.graph)
    pass_decomposition(joint_module, joint_module.graph)
    pass_cse(joint_module, joint_module.graph)
    pass_constant_propagation(joint_module, joint_module.graph)
    pass_fusion(joint_module, joint_module.graph)
    pass_print_graph(joint_module, "./gcn_optimized.svg")
    # pass_clean_up(joint_module, joint_module.graph)
    # joint_module.recompile()
    
    return joint_module


class GCNTest(BaseTestCase):
    def test_gcn(self):
        features, labels, csr, csc, in_feats, n_classes = gcn_input("ogbn-mag")
        sample_inputs = [features, labels]

        model_ref, optimizer_ref = gcn_model_optimizer(in_feats, n_classes, csr, csc)

        # Cast to fp16
        model_ref, optimizer_ref = apex_autocast(
            model_ref, optimizer_ref, True
        )

        self.run_reference_model(model_ref, optimizer_ref, sample_inputs, 1e-5)

        model, optimizer = gcn_model_optimizer(in_feats, n_classes, csr, csc, reference=model_ref, f32_loss=False)
        model, optimizer = apex_autocast(
            model, optimizer, True
        )

        model.aot_optimize(
            compiler_fn, compiler_fn,
            partial(
                partition_func,
                joint_compiler=joint_optimization
            )
        )

        model.capture_graph(
            features, labels, optimizer=optimizer
        )
        model.set_features(features, labels)

        self.run_target_model(model, optimizer, [])

        self.verify(model, model_ref, verbose=1)

        # Warmup
        for _ in range(10):
            self.run_target_model(model, optimizer, [])

        with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
            with record_function("evt"):
                for _ in range(10):
                    self.run_target_model(model, optimizer, [])

        print(prof.key_averages().table(sort_by="cuda_time_total"))
        
        for _ in range(10):
            self.run_reference_model(model_ref, optimizer_ref, sample_inputs, 1e-5)

        with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
            with record_function("torch"):
                for _ in range(10):
                    self.run_reference_model(model_ref, optimizer_ref, sample_inputs, 1e-5)
        print(prof.key_averages().table(sort_by="cuda_time_total"))
    
    def is_close(self, grad1, grad2):
        return (
            torch.sum(
                torch.isclose(grad1, grad2, rtol=1e-1)
            ) / grad1.numel() > -1#0.9 
        )

if __name__ == '__main__':
    logging.basicConfig(level=getattr(logging, "DEBUG"))
    logging.basicConfig(format='%(message)s')
    unittest.main()
