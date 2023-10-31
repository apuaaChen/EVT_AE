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

import unittest
import torch
import logging
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
from gtl.compiler.passes import pass_print_graph
import re
import nvtx


class BaseTestCase(unittest.TestCase):
    @staticmethod
    def run_reference_model(model, optimizer, sample_inputs, loss_scale):
        scaler = torch.cuda.amp.GradScaler(init_scale=loss_scale)
        model.train()
        optimizer.zero_grad()
        loss = model(*sample_inputs)
        scaler.scale(loss).backward()
    
    @staticmethod
    def run_target_model(model, optimizer, sample_inputs):
        model.train()
        optimizer.zero_grad()
        model.train_with_graph(*sample_inputs)
    
    def grad_preprocess(self, grad):
        return grad
    
    def verify(self, reference, target, verbose=0, rtol=1e-1):
        for (param_ref, param_target) in zip(
            list(reference.named_parameters()),
            list(target.named_parameters())
        ):
            grad_ref = param_ref[1].grad.to(torch.float16)
            grad_target = self.grad_preprocess(param_target[1].grad).to(torch.float16)
            if verbose == 1:
                close_ratio = torch.sum(
                        torch.isclose(
                            grad_ref, grad_target, rtol=rtol
                        )
                    ) / grad_ref.numel()
                print(f"{param_ref[0]}: close ratio: {close_ratio.item()}")
            self.assertTrue(
                self.is_close(grad_ref, grad_target))

    def is_close(self, *args, **kwargs):
        raise NotImplementedError(
            "[Unit Test] is_close function is not overwritten!")


class UnitTestBase(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        # Change to True to visualize the graph before and after decomposition
        self.visualize = True
        self.warmup_iters = 20
        self.profiling_iters = 20
    
    # Helper function for launching test
    def util_test(self, cls, inputs, passes, criteria={"rtol": 5e-2}, profile=False):
        ## model instances
        model = cls()
        model_reference = cls()

        symbolic_traced: torch.fx.GraphModule = symbolic_trace(model)
        ShapeProp(symbolic_traced).propagate(*inputs)

        # Get snake case name
        name = cls.__name__
        words = re.findall(r'[A-Z][a-z0-9]*', name)
        snake_case_name = '_'.join(words).lower()

        if self.visualize:
            pass_print_graph(symbolic_traced, f"./{snake_case_name}.svg")
        
        for p in passes:
            p(symbolic_traced, symbolic_traced.graph)
        symbolic_traced.recompile()

        if self.visualize:
            pass_print_graph(symbolic_traced, f"./{snake_case_name}_decomposed.svg")
        
        out = symbolic_traced(*inputs)
        ref = model_reference(*inputs)
        torch.cuda.synchronize()
        if isinstance(out, tuple):
            for o, r in zip(out, ref):
                self.assertTrue(torch.allclose(o, r, **criteria))  
        else:
            self.assertTrue(torch.allclose(out, ref, **criteria))
        
        if not profile:
            return
        
        # Profiling
        for _ in range(self.warmup_iters):
            out = symbolic_traced(*inputs)
        
        with nvtx.annotate("gtl"):
            for _ in range(self.profiling_iters):
                out = symbolic_traced(*inputs)
        
        for _ in range(self.warmup_iters):
            ref = model_reference(*inputs)

        with nvtx.annotate("torch"):
            for _ in range(self.profiling_iters):
                ref = model_reference(*inputs)

