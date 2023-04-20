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

import torch
from pycutlass.test.profiler import GpuTimer
import nvtx
import logging

class GTLProfiler:
    """
    Helper class that profiles the target model
    """
    def __init__(
            self, model, label="", use_cuda_graph=True, 
            warmup_iter=10, profile_iter=50) -> None:
        #
        self.model = model
        self.timer = GpuTimer()
        self.use_cuda_graph = use_cuda_graph
        self.warmup_iter = warmup_iter
        self.profile_iter = profile_iter
        self.label = label
        self.scaler = torch.cuda.amp.GradScaler()

    def profile(self, sample_inputs, batch_size=None):
        """
        Profile model with sample inputs
        """
        s = torch.cuda.Stream(priority=-1)
        s.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(s):
            for _ in range(self.warmup_iter):
                self.run(sample_inputs)
            
            s.synchronize()
            self.timer.start(stream=s.cuda_stream)
            with nvtx.annotate(
                f"{self.label}: {self.profile_iter}: Iterations"):
                #
                for _ in range(self.profile_iter):
                    with nvtx.annotate("Iter"):
                        self.run(sample_inputs)
            self.timer.stop_and_wait(stream=s.cuda_stream)
        
        iter_time = self.timer.duration(self.profile_iter)
        if batch_size is None:
            logging.info(f"Iter time: {iter_time} ms")
            return iter_time
        else:
            throughput = batch_size[1] / iter_time * 1000
            logging.info(
                f"Throughput: {throughput} {batch_size[0]}/sec, " \
                f"Iter time: {iter_time} ms")
    
    def run(self, sample_inputs):
        """
        Run a single iteration
        """
        if self.use_cuda_graph:
            self.model.train_with_graph(*sample_inputs)
        else:
            loss = self.model(*sample_inputs)
            self.scaler.scale(loss).backward()
            # loss.backward()
