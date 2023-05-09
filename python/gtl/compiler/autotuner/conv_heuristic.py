import cutlass
import random
from gtl.compiler.autotuner.gemm_heuristic import DefaultMmaCore, DefaultEpilogueTensorOp, MmaBase, GemmHeuristics, a100_metafile, EpilogueWithVisitor, DefaultThreadMapTensorOp
# from gemm_heuristic import DefaultMmaCore, DefaultEpilogueTensorOp, MmaBase, GemmHeuristics, a100_metafile, EpilogueWithVisitor
import numpy as np
from pycutlass import *

class Conv2dHeuristics(GemmHeuristics):
    def __init__(
        self, element_a, element_b, element_c, element_accumulator,
        alignment_a, alignment_b, alignment_c, problem_size, split_k_mode="None",
        conv_kind = cutlass.conv.Operator.fprop
        ) -> None:
        #
        self.metafile = a100_metafile
        self.element_a = element_a
        self.element_b = element_b
        self.element_c = element_c
        self.alignment_a = alignment_a
        self.alignment_b = alignment_b
        self.alignment_c = alignment_c
        self.element_accumulator = element_accumulator
        self.split_k_mode = split_k_mode
        self.enable_split_k = self.split_k_mode != "None"
        self.problem_size = problem_size
        self.conv_kind = conv_kind
        # set stride support
        self.stride_support = StrideSupport.Strided
        if self.conv_kind == cutlass.conv.Operator.dgrad:
            if self.problem_size.stride_h == 1 or self.problem_size.stride_w == 1:
                self.stride_support == StrideSupport.Unity
        

        self.set_parameter_range()

        self.available_iterator_algorithms = [
            cutlass.conv.IteratorAlgorithm.analytic,
            cutlass.conv.IteratorAlgorithm.fixed_channels,
            cutlass.conv.IteratorAlgorithm.few_channels,
            cutlass.conv.IteratorAlgorithm.optimized
        ]
    
    def determine_iterator_algorithm(self):
        if self.conv_kind == cutlass.conv.Operator.fprop:
            # check whether the fixed channel is applicable
            if self.problem_size.C == self.alignment_a:
                return 1
            # not sure which is faster
            elif self.problem_size.C % self.alignment_a == 0:
                return random.randint(2, 3)
            else:
                return 0
        elif self.conv_kind == cutlass.conv.Operator.dgrad:
            if self.problem_size.K % self.alignment_a == 0 and self.problem_size.R <= 32 and self.problem_size.S <= 32 and self.problem_size.C % self.alignment_b == 0:
                return 3
            else:
                return 0
        elif self.conv_kind == cutlass.conv.Operator.wgrad:
            if self.problem_size.K % self.alignment_a == 0 and self.problem_size.C % self.alignment_b == 0:
                return 3
            else:
                return 0

    def set_parameter_range(self):
        if self.enable_split_k:
            max_split_k_slices = 16
        else:
            max_split_k_slices = 1
        self.parameter_range = {
            "max_tiling_size" : 128,
            "max_stages" : 5,
            "max_log_swizzling": 3,
            "max_split_k_slices": max_split_k_slices
        }
    
    def propose_split_k_slices(self, block_tiling_size, log_swizzle):
        # A more flexible way to limit the design space for split_k_slices
        # This is motivated by the observation that for some special case
        # like the last wgrad layer in resnet, the "optimal" split_k_slice 
        # could be 128, yet considering such a large amount of candidates
        # makes the searching more difficult

        if self.conv_kind == cutlass.conv.Operator.wgrad:
            max_wavefronts = 15 # obtained from observation
            # step 1: get the number of m & n tiles
            swizzling_functor = getattr(cutlass, "IdentitySwizzle%d"%int(pow(2, log_swizzle)))
            tile_size = cutlass.gemm.GemmCoord(
                block_tiling_size[0], block_tiling_size[1], block_tiling_size[2]
            )

            grid_size = swizzling_functor().get_tiled_shape(
                self.conv_kind, self.problem_size, tile_size, 1
            )

            num_tile_base = grid_size.m() * grid_size.n()
            # number of wavefronts before split k
            num_wavefront_base = (num_tile_base + 79) // 80

            if num_wavefront_base >= max_wavefronts:
                # no split k is required
                return 1
            else:
                num_wave_front = random.randint(num_wavefront_base, max_wavefronts)
                # compute the split k based on num_wave_front
                num_threadblocks = num_wave_front * 80
                split_k_slices = num_threadblocks // num_tile_base
                return split_k_slices

        else:
            # simply return the max split_k slices
            return random.randint(1, self.parameter_range["max_split_k_slices"])

    def propose_parameters(self):
        warp_tiling_size = []
        for i in range(3):
            v = self.propose_tiling_size(self.parameter_range["max_tiling_size"])
            warp_tiling_size.append(v)
        
        block_tiling_size = []
        for i in range(3):
            v = self.propose_tiling_size(self.parameter_range["max_tiling_size"])
            block_tiling_size.append(v)
        
        num_stages = random.randint(2, self.parameter_range["max_stages"])
        log_swizzle = random.randint(0, self.parameter_range["max_log_swizzling"])
        # split_k_slices = random.randint(1, self.parameter_range["max_split_k_slices"])
        split_k_slices = self.propose_split_k_slices(block_tiling_size, log_swizzle)

        # 0: analytic
        # 1: fixedchannels
        # 2: fewchannels
        # 3: optimized
        # iterator_alg = random.randint(0, 3)
        # need additional heuristic to quickly zoom into the optimal iterator_alg
        # otherwise it attracks too much attention, which stop us from finding 
        # the optimal config for others
        iterator_alg = self.determine_iterator_algorithm()

        parameters = {
            "block_x": block_tiling_size[0],
            "block_y": block_tiling_size[1],
            "block_z": block_tiling_size[2],
            "warp_x": warp_tiling_size[0],
            "warp_y": warp_tiling_size[1],
            "warp_z": warp_tiling_size[2],
            "stage": num_stages,
            "log_swizzle": log_swizzle,
            "split_k_slices": split_k_slices,
            "iterator_algorithm": iterator_alg
        }

        return parameters
    
    def parameter_to_feature(self, parameter, problem_size):
        feature = np.array(
            [
                parameter["block_x"], parameter["block_y"], parameter["block_z"],
                parameter["warp_x"], parameter["warp_y"], parameter["warp_z"],
                parameter["stage"], parameter["log_swizzle"], 
                parameter["split_k_slices"], parameter["iterator_algorithm"]
            ]
        )
        return feature.reshape((1, 10))
    
    def check_dgrad_swizzling(self, parameter):
        if self.conv_kind == cutlass.conv.Operator.dgrad:
            if self.problem_size.stride_h != 1 or self.problem_size.stride_w != 1:
                if parameter["log_swizzle"] not in [0,]:
                    return False
        return True
    
    def is_valid(self, parameters):
        block_tiling_size = [
            parameters["block_x"], parameters["block_y"], parameters["block_z"]]
        warp_tiling_size = [
            parameters["warp_x"], parameters["warp_y"], parameters["warp_z"]
        ]
        try:
            DefaultConv2d(
                self.element_a, cutlass.TensorNHWC, self.element_b, cutlass.TensorNHWC,
                self.element_c, cutlass.TensorNHWC, self.element_accumulator,
                cutlass.OpClass.TensorOp, block_tiling_size, warp_tiling_size,
                [16, 8, 16], parameters["stage"], self.available_iterator_algorithms[parameters["iterator_algorithm"]],
                self.alignment_a, self.alignment_b, self.alignment_c, self.problem_size,
                self.stride_support, self.conv_kind
            )
        except (AssertionError, ZeroDivisionError) as e:
            return False
        valid = self.check_block_tiling_size_valid(block_tiling_size, parameters["stage"]) \
            and self.check_warp_tiling_size_valid(warp_tiling_size) \
            and self.check_warp_count(warp_tiling_size, block_tiling_size) \
            and self.check_harmony_block_warp_tiling_size(warp_tiling_size, block_tiling_size) \
            and self.check_dgrad_swizzling(parameters)
        return valid

class AlignedArray:
    def __init__(self, type, elements) -> None:
        self.type = type
        self.elements = elements
        pass

class Conv2dFpropActivationTileAccessIteratorAnalytic:
    def __init__(self, shape, element, layout, threadmap, access_type) -> None:
        self.access_type = access_type
        assert threadmap.elements_per_access % access_type.elements == 0
        assert threadmap.iterations.contiguous() == 1
    
    def can_implement(self, problem_size):
        assert problem_size.C % self.access_type.elements == 0

class Conv2dFpropActivationTileAccessIteratorFixedChannels:
    def __init__(self, shape, element, layout, threadmap, access_type) -> None:
        self.access_type = access_type
        assert threadmap.elements_per_access % access_type.elements == 0
        assert threadmap.iterations.contiguous() == 1

        self.stride_h = 0
        self.stride_w = 0
        self.dilation_h = 0
        self.dilation_w = 0
    
    def can_implement(self, problem_size):
        assert problem_size.C == self.access_type.elements
        assert self.dilation_h == 0 or problem_size.dilation_h == self.dilation_h
        assert self.dilation_w == 0 or problem_size.dilation_w == self.dilation_w
        assert self.stride_h == 0 or problem_size.stride_h == self.stride_h
        assert self.stride_w == 0 or problem_size.stride_w == self.stride_w

class Conv2dFpropActivationTileAccessIteratorFewChannels:
    def __init__(self, shape, element, layout, threadmap, access_type) -> None:
        self.access_type = access_type
        assert threadmap.elements_per_access % access_type.elements == 0
        assert threadmap.iterations.contiguous() == 1

        self.stride_h = 0
        self.stride_w = 0
        self.dilation_h = 0
        self.dilation_w = 0
    
    def can_implement(self, problem_size):
        assert problem_size.C % self.access_type.elements == 0
        assert self.dilation_h == 0 or problem_size.dilation_h == self.dilation_h
        assert self.dilation_w == 0 or problem_size.dilation_w == self.dilation_w
        assert self.stride_h == 0 or problem_size.stride_h == self.stride_h
        assert self.stride_w == 0 or problem_size.stride_w == self.stride_w

class Conv2dFpropActivationTileAccessIteratorOptimized:
    def __init__(self, shape, element, layout, threadmap, access_type) -> None:
        self.access_type = access_type
        assert threadmap.elements_per_access % access_type.elements == 0
        assert threadmap.iterations.contiguous() == 1
    
    def can_implement(self, problem_size):
        assert problem_size.C % self.access_type.elements == 0
        assert problem_size.R <= 32
        assert problem_size.S <= 32

class Conv2dFpropFilterTileAccessIteratorAnalytic:
    def __init__(self, shape, element, layout, threadmap, access_type) -> None:
        self.access_type = access_type
        assert threadmap.elements_per_access % access_type.elements == 0
        assert threadmap.iterations.contiguous() == 1
    
    def can_implement(self, problem_size):
        assert problem_size.K % self.access_type.elements == 0

class Conv2dFpropFilterTileAccessIteratorFixedChannels:
    def __init__(self, shape, element, layout, threadmap, access_type) -> None:
        self.access_type = access_type
        assert threadmap.elements_per_access % access_type.elements == 0
        assert threadmap.iterations.contiguous() == 1
    
    def can_implement(self, problem_size):
        assert problem_size.C == self.access_type.elements

class Conv2dFpropFilterTileAccessIteratorFewChannels:
    def __init__(self, shape, element, layout, threadmap, access_type) -> None:
        self.access_type = access_type
        assert threadmap.elements_per_access % access_type.elements == 0
        assert threadmap.iterations.contiguous() == 1
    
    def can_implement(self, problem_size):
        assert problem_size.C % self.access_type.elements == 0

class Conv2dFpropFilterTileAccessIteratorOptimized:
    def __init__(self, shape, element, layout, threadmap, access_type) -> None:
        self.access_type = access_type
        assert threadmap.elements_per_access % access_type.elements == 0
        assert threadmap.iterations.contiguous() == 1
    
    def can_implement(self, problem_size):
        assert problem_size.C % self.access_type.elements == 0


class Conv2dDgradOutputGradientTileAccessIteratorAnalytic:
    def __init__(self, shape, element, threadmap, stride_support, access_type) -> None:
        self.access_type = access_type
        self.stride_support = stride_support
        assert threadmap.elements_per_access % access_type.elements == 0
        assert threadmap.iterations.contiguous() == 1
    
    def can_implement(self, problem_size):
        assert problem_size.K % self.access_type.elements == 0
        if self.stride_support == StrideSupport.Unity:
            assert problem_size.stride_h == 1
            assert problem_size.stride_w == 1


class Conv2dDgradOutputGradientTileAccessIteratorOptimized:
    def __init__(self, shape, element, threadmap, stride_support, access_type) -> None:
        self.access_type = access_type
        self.stride_support = stride_support
        assert threadmap.elements_per_access % access_type.elements == 0
        assert threadmap.iterations.contiguous() == 1
    
    def can_implement(self, problem_size):
        assert problem_size.K % self.access_type.elements == 0
        assert problem_size.R <= 32
        assert problem_size.S <= 32
    
        if self.stride_support == StrideSupport.Unity:
            assert problem_size.stride_h == 1
            assert problem_size.stride_w == 1


class Conv2dDgradFilterTileAccessIteratorAnalytic:
    def __init__(self, shape, element, threadmap, stride_support, access_type) -> None:
        self.access_type = access_type
        assert threadmap.elements_per_access % access_type.elements == 0
    
    def can_implement(self, problem_size):
        assert problem_size.C % self.access_type.elements == 0


class Conv2dDgradFilterTileAccessIteratorOptimized:
    def __init__(self, shape, element, threadmap, stride_support, access_type) -> None:
        self.access_type = access_type
        assert threadmap.elements_per_access % access_type.elements == 0
        assert threadmap.iterations.strided() * threadmap.iterations.contiguous() < 32
    
    def can_implement(self, problem_size):
        assert problem_size.C % self.access_type.elements == 0


class Conv2dWgradOutputGradientTileAccessIteratorAnalytic:
    def __init__(self, shape, element, threadmap, access_type) -> None:
        self.access_type = access_type
        assert threadmap.elements_per_access % access_type.elements == 0
    
    def can_implement(self, problem_size):
        assert problem_size.C % self.access_type.elements == 0


class Conv2dWgradOutputGradientTileAccessIteratorOptimized:
    def __init__(self, shape, element, threadmap, access_type) -> None:
        self.access_type = access_type
        assert threadmap.elements_per_access % access_type.elements == 0
    
    def can_implement(self, problem_size):
        # not consistent with cutlass, but seems more make sense
        assert problem_size.K % self.access_type.elements == 0


class Conv2dWgradActivationTileAccessIteratorAnalytic:
    def __init__(self, shape, element, threadmap, access_type) -> None:
        self.access_type = access_type
        assert threadmap.elements_per_access % access_type.elements == 0
    
    def can_implement(self, problem_size):
        assert problem_size.K % self.access_type.elements == 0


class Conv2dWgradActivationTileAccessIteratorOptimized:
    def __init__(self, shape, element, threadmap, access_type) -> None:
        self.access_type = access_type
        assert threadmap.elements_per_access % access_type.elements == 0
    
    def can_implement(self, problem_size):
        # not consistent with cutlass, but seems more make sense
        assert problem_size.C % self.access_type.elements == 0
        
class ConvTileIterator:
    def __init__(self, iterator) -> None:
        self.iterator = iterator
    
    def can_implement(self, problem_size):
        self.iterator.can_implement(problem_size)

class ImplicitGemmPipelined:
    def __init__(self, shape, iterator_a, smem_iterator_a, iterator_b,
        smem_iterator_b, element_c, layout_c, policy) -> None:
        self.base = MmaBase(shape, policy, 2)

class ImplicitGemmMultistage:
    def __init__(self, shape, iterator_a, smem_iterator_a, cache_op_a,
        iterator_b, smem_iterator_b, cache_op_b, policy, stages) -> None:
        self.base = MmaBase(shape, policy, stages)


class ImplicitGemmConvolution:
    def __init__(self, mma, epilogue, conv_operator) -> None:
        pass


class PredicatedTileIteratorStridedDgrad:
    def __init__(self, threadmap, element_C) -> None:
        assert threadmap.iterations.row > 0, "ThreadMap::Iterations::kRow must be > 0"
        assert threadmap.iterations.group > 0, "ThreadMap::Iterations::kGroup must be > 0"
        assert threadmap.iterations.cluster > 0, "ThreadMap::Iterations::kCluster must be > 0"
        assert threadmap.iterations.column > 0, "ThreadMap::Iterations::kColumn must be > 0"


class DefaultEpilogueTensorOpStridedDgrad:
    def __init__(self, threadblock_shape, warp_mma_tensor_op, partition_k, element_C, elements_per_access) -> None:
        self.output_tile_threadmp = DefaultThreadMapTensorOp(
            threadblock_shape, warp_mma_tensor_op.shape, partition_k, element_C, elements_per_access
        ).type

        self.output_tile_iterator = PredicatedTileIteratorStridedDgrad(
            self.output_tile_threadmp, element_C
        )

        # TODO: kFragmentsPerIteration

class DefaultConv2d:
    def __init__(
        self, element_a, layout_a, element_b, layout_b, element_c, layout_c,
        element_accumulator, operator_class, threadblock_shape,
        warp_shape, instruction_shape, stages, iterator_algorithm, 
        alignment_a, alignment_b, alignment_c, problem_size, 
        stride_support=StrideSupport.Strided,
        conv_kind=cutlass.conv.Operator.fprop) -> None:

        if operator_class == cutlass.OpClass.TensorOp:
            if conv_kind == cutlass.conv.Operator.fprop:
                self.mma_core = DefaultMmaCore(
                    threadblock_shape, warp_shape, instruction_shape, element_a,
                    cutlass.RowMajor, element_b, cutlass.ColumnMajor,
                    element_accumulator)
            elif conv_kind == cutlass.conv.Operator.dgrad:
                self.mma_core = DefaultMmaCore(
                    threadblock_shape, warp_shape, instruction_shape, element_a,
                    cutlass.RowMajor, element_b, cutlass.RowMajor,
                    element_accumulator)
            elif conv_kind == cutlass.conv.Operator.wgrad:
                self.mma_core = DefaultMmaCore(
                    threadblock_shape, warp_shape, instruction_shape, element_a,
                    cutlass.ColumnMajor, element_b, cutlass.RowMajor,
                    element_accumulator)
            
            self.threadmap_a = self.mma_core.iterator_threadmap_A
            # using AccessTypeA = cutlass::AlignedArray<ElementA, AlignmentA>;
            self.access_type_a = AlignedArray(element_a, alignment_a)
            if iterator_algorithm == cutlass.conv.IteratorAlgorithm.analytic:
                if conv_kind == cutlass.conv.Operator.fprop:
                    self.iterator_a = Conv2dFpropActivationTileAccessIteratorAnalytic(
                        [threadblock_shape[0], threadblock_shape[2]], element_a,
                        layout_a, self.threadmap_a, self.access_type_a
                    )
                elif conv_kind == cutlass.conv.Operator.dgrad:
                    self.iterator_a = Conv2dDgradOutputGradientTileAccessIteratorAnalytic(
                        [threadblock_shape[0], threadblock_shape[2]], element_a,
                        self.threadmap_a, stride_support, self.access_type_a
                    )
                elif conv_kind == cutlass.conv.Operator.wgrad:
                    self.iterator_a = Conv2dWgradOutputGradientTileAccessIteratorAnalytic(
                        [threadblock_shape[0], threadblock_shape[2]], element_a,
                        self.threadmap_a, self.access_type_a
                    )
            elif iterator_algorithm == cutlass.conv.IteratorAlgorithm.fixed_channels:
                if conv_kind == cutlass.conv.Operator.fprop:
                    self.iterator_a = Conv2dFpropActivationTileAccessIteratorFixedChannels(
                        [threadblock_shape[0], threadblock_shape[2]], element_a,
                        layout_a, self.threadmap_a, self.access_type_a
                    )
                elif conv_kind == cutlass.conv.Operator.dgrad:
                    assert False
                elif conv_kind == cutlass.conv.Operator.wgrad:
                    assert False
            elif iterator_algorithm == cutlass.conv.IteratorAlgorithm.few_channels:
                if conv_kind == cutlass.conv.Operator.fprop:
                    self.iterator_a = Conv2dFpropActivationTileAccessIteratorFewChannels(
                        [threadblock_shape[0], threadblock_shape[2]], element_a,
                        layout_a, self.threadmap_a, self.access_type_a
                    )
                elif conv_kind == cutlass.conv.Operator.dgrad:
                    assert False
                elif conv_kind == cutlass.conv.Operator.wgrad:
                    assert False
            elif iterator_algorithm == cutlass.conv.IteratorAlgorithm.optimized:
                if conv_kind == cutlass.conv.Operator.fprop:
                    self.iterator_a = Conv2dFpropActivationTileAccessIteratorOptimized(
                        [threadblock_shape[0], threadblock_shape[2]], element_a,
                        layout_a, self.threadmap_a, self.access_type_a
                    )
                elif conv_kind == cutlass.conv.Operator.dgrad:
                    self.iterator_a = Conv2dDgradOutputGradientTileAccessIteratorOptimized(
                        [threadblock_shape[0], threadblock_shape[2]], element_a,
                        self.threadmap_a, stride_support, self.access_type_a
                    )
                elif conv_kind == cutlass.conv.Operator.wgrad:
                    self.iterator_a = Conv2dWgradOutputGradientTileAccessIteratorOptimized(
                        [threadblock_shape[0], threadblock_shape[2]], element_a,
                        self.threadmap_a, self.access_type_a
                    )
            else:
                raise ValueError
            
            if stages == 2:
                self.iterator_a = ConvTileIterator(self.iterator_a)
            self.smem_iterator_a = self.mma_core.smem_iterator_A

            self.threadmap_b = self.mma_core.iterator_threadmap_B
            self.access_type_b = AlignedArray(element_b, alignment_b)
            if iterator_algorithm == cutlass.conv.IteratorAlgorithm.analytic:
                if conv_kind == cutlass.conv.Operator.fprop:
                    self.iterator_b = Conv2dFpropFilterTileAccessIteratorAnalytic(
                        [threadblock_shape[2], threadblock_shape[1]], element_b,
                        layout_b, self.threadmap_b, self.access_type_b
                    )
                elif conv_kind == cutlass.conv.Operator.dgrad:
                    self.iterator_b = Conv2dDgradFilterTileAccessIteratorAnalytic(
                        [threadblock_shape[2], threadblock_shape[1]],
                        element_b, self.threadmap_b,
                        stride_support, self.access_type_b
                    )
                elif conv_kind == cutlass.conv.Operator.wgrad:
                    self.iterator_b = Conv2dWgradActivationTileAccessIteratorAnalytic(
                        [threadblock_shape[2], threadblock_shape[1]],
                        element_b, self.threadmap_b, self.access_type_b
                    )
            elif iterator_algorithm == cutlass.conv.IteratorAlgorithm.fixed_channels:
                if conv_kind == cutlass.conv.Operator.fprop:
                    self.iterator_b = Conv2dFpropFilterTileAccessIteratorFixedChannels(
                        [threadblock_shape[2], threadblock_shape[1]], element_b,
                        layout_b, self.threadmap_b, self.access_type_b
                    )
                elif conv_kind == cutlass.conv.Operator.dgrad:
                    assert False
                elif conv_kind == cutlass.conv.Operator.wgrad:
                    assert False
                
            elif iterator_algorithm == cutlass.conv.IteratorAlgorithm.few_channels:
                if conv_kind == cutlass.conv.Operator.fprop:
                    self.iterator_b = Conv2dFpropFilterTileAccessIteratorFewChannels(
                        [threadblock_shape[2], threadblock_shape[1]], element_b,
                        layout_b, self.threadmap_b, self.access_type_b
                    )
                elif conv_kind == cutlass.conv.Operator.dgrad:
                    assert False
                elif conv_kind == cutlass.conv.Operator.wgrad:
                    assert False

            elif iterator_algorithm == cutlass.conv.IteratorAlgorithm.optimized:
                if conv_kind == cutlass.conv.Operator.fprop:
                    self.iterator_b = Conv2dFpropFilterTileAccessIteratorOptimized(
                        [threadblock_shape[2], threadblock_shape[1]], element_b,
                        layout_b, self.threadmap_b, self.access_type_b
                    )
                elif conv_kind == cutlass.conv.Operator.dgrad:
                    self.iterator_b = Conv2dDgradFilterTileAccessIteratorOptimized(
                        [threadblock_shape[2], threadblock_shape[1]],
                        element_b, self.threadmap_b,
                        stride_support, self.access_type_b
                    )
                elif conv_kind == cutlass.conv.Operator.wgrad:
                    self.iterator_b = Conv2dWgradActivationTileAccessIteratorOptimized(
                        [threadblock_shape[2], threadblock_shape[1]],
                        element_b, self.threadmap_b,
                        self.access_type_b
                    )
            else:
                raise ValueError
            if stages == 2:
                self.iterator_b = ConvTileIterator(self.iterator_b)

            self.smem_iterator_b = self.mma_core.smem_iterator_B

            self.warp_mma_tensor_op = self.mma_core.mma_tensor_op
            self.mma_policy = self.mma_core.mma_policy

            if element_b == cutlass.float16:
                element_b_size = 16
            else:
                raise NotImplementedError

            if element_b_size * alignment_b == 128:
                self.cache_op_b = "Global"
            else: 
                self.cache_op_b = "Always"
            
            if stages == 2:
                self.mma = ImplicitGemmPipelined(
                    threadblock_shape, self.iterator_a, self.smem_iterator_a,
                    self.iterator_b, self.smem_iterator_b, element_c, layout_c,
                    self.mma_policy
                )
            else:
                self.mma = ImplicitGemmMultistage(
                    threadblock_shape, self.iterator_a, self.smem_iterator_a,
                    "Always", self.iterator_b, 
                    self.smem_iterator_b, self.cache_op_b, self.mma_policy,
                    stages
                )

            self.partition_k = threadblock_shape[2] // warp_shape[2]

            if conv_kind == cutlass.conv.Operator.dgrad and stride_support == StrideSupport.Strided:
                self.epilogue = DefaultEpilogueTensorOpStridedDgrad(
                    threadblock_shape, self.warp_mma_tensor_op, self.partition_k,
                    element_c, alignment_c
                )
            else:
                self.epilogue = DefaultEpilogueTensorOp(
                    threadblock_shape, self.warp_mma_tensor_op, self.partition_k,
                    element_c, alignment_c
                )

            self.kernel = ImplicitGemmConvolution(
                self.mma, self.epilogue, cutlass.conv.Operator.fprop
            )

            self.Visitor = EpilogueWithVisitor(
                warp_shape, instruction_shape, self.epilogue, alignment_c)
        else:
            raise NotImplementedError
        
        self.iterator_a.can_implement(problem_size)
        self.iterator_b.can_implement(problem_size)


# TODO: Dgrad Strided requires special swizzling functor:
# threadblock::StridedDgradIdentityThreadblockSwizzle<1/4/8>

if __name__ == "__main__":

    problem_size = cutlass.conv.Conv2dProblemSize(
        cutlass.Tensor4DCoord(128, 224, 224, 4),
        cutlass.Tensor4DCoord(64, 7, 7, 4),
        cutlass.Tensor4DCoord(3, 3, 3, 3),
        cutlass.MatrixCoord(2, 2),
        cutlass.MatrixCoord(1, 1),
        cutlass.conv.Mode.cross_correlation,
        1, 1
    )

    rule = DefaultConv2d(
        cutlass.float16, cutlass.TensorNHWC,
        cutlass.float16, cutlass.TensorNHWC,
        cutlass.float16, cutlass.TensorNHWC,
        cutlass.float32, cutlass.OpClass.TensorOp, [64, 32, 64], [32, 32, 64],
        [16, 8, 16], 3, cutlass.conv.IteratorAlgorithm.fixed_channels,
        4, 4, 8, problem_size
    )
    print(rule)
    