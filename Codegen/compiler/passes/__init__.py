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
from autotuner.auto_tuner import Autotuner
autotuner = Autotuner(verbose=True, load_model=False, autotuning_rounds=15)
from passes.suffix_elimination import pass_suffix_elimination
from passes.print_graph import pass_print_graph
from passes.eliminate_transparent import pass_eliminate_transparent_node
from passes.eliminate_loss import pass_loss_elimination
from passes.composed_op_breakdown import pass_composed_op_breakdown
from passes.substitution import pass_graph_substitution
from passes.shape_prop import pass_shape_prop
from passes.remove_duplicated import pass_remove_duplicated_node
from passes.extract_common_factor import pass_merge_common_factor
from passes.update_attr import pass_update_attributes
from passes.constant_folding import pass_constant_folding
from passes.strength_reduction import pass_stength_reduction
from passes.gemm_fusion import pass_gemm_fusion
from passes.softmax_fusion import pass_softmax_fusion
from passes.stream_manager import pass_assign_stream
from passes.trans_2_permute import pass_trans_2_permute
from passes.mark_epilogue_permute import pass_mark_epilogue_permutations