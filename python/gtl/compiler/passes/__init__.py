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
from gtl.compiler.autotuner.auto_tuner import Autotuner
autotuner = Autotuner(verbose=True, load_model=False, autotuning_rounds=15)
from gtl.compiler.passes.suffix_elimination import pass_suffix_elimination
from gtl.compiler.passes.print_graph import pass_print_graph
# from gtl.compiler.passes.eliminate_transparent import pass_eliminate_transparent_node
from gtl.compiler.passes.eliminate_loss import pass_loss_elimination
from gtl.compiler.passes.composed_op_breakdown import pass_composed_op_breakdown
# from gtl.compiler.passes.substitution import pass_graph_substitution
from gtl.compiler.passes.shape_prop import pass_shape_prop
from gtl.compiler.passes.remove_duplicated import pass_remove_duplicated_node
from gtl.compiler.passes.extract_common_factor import pass_merge_common_factor
from gtl.compiler.passes.update_attr import pass_update_attributes
from gtl.compiler.passes.constant_folding import pass_constant_folding
from gtl.compiler.passes.strength_reduction import pass_stength_reduction
from gtl.compiler.passes.layer_norm_preprocessing import pass_layernorm_preprocessing
from gtl.compiler.passes.gemm_fusion import pass_gemm_fusion
from gtl.compiler.passes.softmax_fusion import pass_softmax_fusion
from gtl.compiler.passes.stream_manager import pass_assign_stream
from gtl.compiler.passes.trans_2_permute import pass_trans_2_permute
from gtl.compiler.passes.mark_epilogue_permute import pass_mark_epilogue_permutations
from gtl.compiler.passes.instruction_reorder import pass_instruction_reorder
from gtl.compiler.passes.weight_grad_tuner import pass_weight_gradient_tuner
from gtl.compiler.passes.conv_fusion import pass_conv_fusion
from gtl.compiler.passes.batch_norm_preprocessing import pass_batchnorm_preprocessing
from gtl.compiler.passes.layout_transform import pass_nchw_to_nhwc
from gtl.compiler.passes.fix_permute_issue import pass_permute_view_fix
from gtl.compiler.passes.tvm_preprocess import pass_tvm_preprocessing

################################################################################
# Reformulated passes
################################################################################
from gtl.compiler.passes.frontend import GTLFrontend