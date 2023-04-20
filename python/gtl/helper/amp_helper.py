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
import warnings
from apex import amp, DeprecatedFeatureWarning
import torch


def apex_autocast(model, optimizer, bn_fp32=False):
    # while apex.amp is deprecated, it support more opt levels (O0~4)
    # whereas pytorch only supports O1
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecatedFeatureWarning)
        return amp.initialize(
            model, optimizer, 
            cast_model_outputs=torch.float16, 
            opt_level="O2", keep_batchnorm_fp32=bn_fp32, 
            loss_scale="dynamic", verbosity=0
        )
