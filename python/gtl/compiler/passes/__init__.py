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
# from gtl.compiler.autotuner.auto_tuner import Autotuner
# autotuner = Autotuner(verbose=True, load_model=False, autotuning_rounds=15)
# from gtl.compiler.passes.suffix_elimination import pass_suffix_elimination
from gtl.compiler.passes.pass_print_graph import pass_print_graph
from gtl.compiler.passes.pass_fake_shape_infer import pass_fake_shape_infer
from gtl.compiler.passes.pass_eliminate_loss import pass_loss_elimination
from gtl.compiler.passes.pass_decomposition import pass_decomposition
from gtl.compiler.passes.pass_constant_propagation import pass_constant_propagation
from gtl.compiler.passes.pass_cse import pass_cse
# from gtl.compiler.passes.strength_reduction import pass_stength_reduction
# from gtl.compiler.passes.stream_manager import pass_assign_stream
# from gtl.compiler.passes.instruction_reorder import pass_instruction_reorder
# from gtl.compiler.passes.weight_grad_tuner import pass_weight_gradient_tuner
# from gtl.compiler.passes.conv_fusion import pass_conv_fusion
# from gtl.compiler.passes.batch_norm_preprocessing import pass_batchnorm_preprocessing
# from gtl.compiler.passes.layout_transform import pass_nchw_to_nhwc
# from gtl.compiler.passes.tvm_preprocess import pass_tvm_preprocessing
from gtl.compiler.passes.pass_fusion import pass_fusion
from gtl.compiler.passes.pass_clean_up import pass_clean_up

################################################################################
# Reformulated passes
################################################################################
from gtl.compiler.passes.pass_frontend import GTLFrontend