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

