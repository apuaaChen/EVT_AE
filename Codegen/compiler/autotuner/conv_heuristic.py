import cutlass
import random
from autotuner.gemm_heuristic import DefaultMmaCore, DefaultEpilogueTensorOp, MmaBase, GemmHeuristics, a100_metafile, EpilogueWithVisitor
# from gemm_heuristic import DefaultMmaCore, DefaultEpilogueTensorOp, MmaBase, GemmHeuristics, a100_metafile, EpilogueWithVisitor
import numpy as np

class Conv2dHeuristics(GemmHeuristics):
    def __init__(
        self, element_a, element_b, element_c, element_accumulator,
        alignment_a, alignment_b, alignment_c, problem_size, split_k_mode="None"
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
        self.set_parameter_range()

        self.available_iterator_algorithms = [
            cutlass.conv.IteratorAlgorithm.analytic,
            cutlass.conv.IteratorAlgorithm.fixed_channels,
            cutlass.conv.IteratorAlgorithm.few_channels,
            cutlass.conv.IteratorAlgorithm.optimized
        ]
    
    def determine_iterator_algorithm(self):
        # check whether the fixed channel is applicable
        if self.problem_size.C == self.alignment_a:
            return 1
        # not sure which is faster
        elif self.problem_size.C % self.alignment_a == 0:
            return random.randint(2, 3)
        else:
            return 0
    
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
        split_k_slices = random.randint(1, self.parameter_range["max_split_k_slices"])

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
    
    def is_valid(self, parameters):
        block_tiling_size = [
            parameters["block_x"], parameters["block_y"], parameters["block_z"]]
        warp_tiling_size = [
            parameters["warp_x"], parameters["warp_y"], parameters["warp_z"]
        ]
        try:
            DefaultConv2dFprop(
                self.element_a, cutlass.TensorNHWC, self.element_b, cutlass.TensorNHWC,
                self.element_c, cutlass.TensorNHWC, self.element_accumulator,
                cutlass.OpClass.TensorOp, block_tiling_size, warp_tiling_size,
                [16, 8, 16], parameters["stage"], self.available_iterator_algorithms[parameters["iterator_algorithm"]],
                self.alignment_a, self.alignment_b, self.alignment_c, self.problem_size
            )
        except (AssertionError, ZeroDivisionError):
            return False
        valid = self.check_block_tiling_size_valid(block_tiling_size, parameters["stage"]) \
            and self.check_warp_tiling_size_valid(warp_tiling_size) \
            and self.check_warp_count(warp_tiling_size, block_tiling_size) \
            and self.check_harmony_block_warp_tiling_size(warp_tiling_size, block_tiling_size)
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


class DefaultConv2dFprop:
    def __init__(
        self, element_a, layout_a, element_b, layout_b, element_c, layout_c,
        element_accumulator, operator_class, threadblock_shape,
        warp_shape, instruction_shape, stages, iterator_algorithm, 
        alignment_a, alignment_b, alignment_c, problem_size) -> None:

        if operator_class == cutlass.OpClass.TensorOp:
            self.mma_core = DefaultMmaCore(
                threadblock_shape, warp_shape, instruction_shape, element_a,
                cutlass.RowMajor, element_b, cutlass.ColumnMajor,
                element_accumulator)
            
            self.threadmap_a = self.mma_core.iterator_threadmap_A
            # using AccessTypeA = cutlass::AlignedArray<ElementA, AlignmentA>;
            self.access_type_a = AlignedArray(element_a, alignment_a)
            if iterator_algorithm == cutlass.conv.IteratorAlgorithm.analytic:
                self.iterator_a = Conv2dFpropActivationTileAccessIteratorAnalytic(
                    [threadblock_shape[0], threadblock_shape[2]], element_a,
                    layout_a, self.threadmap_a, self.access_type_a
                )
            elif iterator_algorithm == cutlass.conv.IteratorAlgorithm.fixed_channels:
                self.iterator_a = Conv2dFpropActivationTileAccessIteratorFixedChannels(
                    [threadblock_shape[0], threadblock_shape[2]], element_a,
                    layout_a, self.threadmap_a, self.access_type_a
                )
            elif iterator_algorithm == cutlass.conv.IteratorAlgorithm.few_channels:
                self.iterator_a = Conv2dFpropActivationTileAccessIteratorFewChannels(
                    [threadblock_shape[0], threadblock_shape[2]], element_a,
                    layout_a, self.threadmap_a, self.access_type_a
                )
            elif iterator_algorithm == cutlass.conv.IteratorAlgorithm.optimized:
                self.iterator_a = Conv2dFpropActivationTileAccessIteratorOptimized(
                    [threadblock_shape[0], threadblock_shape[2]], element_a,
                    layout_a, self.threadmap_a, self.access_type_a
                )
            else:
                raise ValueError
            
            if stages == 2:
                self.iterator_a = ConvTileIterator(self.iterator_a)
            self.smem_iterator_a = self.mma_core.smem_iterator_A

            self.threadmap_b = self.mma_core.iterator_threadmap_B
            self.access_type_b = AlignedArray(element_b, alignment_b)
            if iterator_algorithm == cutlass.conv.IteratorAlgorithm.analytic:
                self.iterator_b = Conv2dFpropFilterTileAccessIteratorAnalytic(
                    [threadblock_shape[2], threadblock_shape[1]], element_b,
                    layout_b, self.threadmap_b, self.access_type_b
                )
            elif iterator_algorithm == cutlass.conv.IteratorAlgorithm.fixed_channels:
                self.iterator_b = Conv2dFpropFilterTileAccessIteratorFixedChannels(
                    [threadblock_shape[2], threadblock_shape[1]], element_b,
                    layout_b, self.threadmap_b, self.access_type_b
                )
                
            elif iterator_algorithm == cutlass.conv.IteratorAlgorithm.few_channels:
                self.iterator_b = Conv2dFpropFilterTileAccessIteratorFewChannels(
                    [threadblock_shape[2], threadblock_shape[1]], element_b,
                    layout_b, self.threadmap_b, self.access_type_b
                )
            elif iterator_algorithm == cutlass.conv.IteratorAlgorithm.optimized:
                self.iterator_b = Conv2dFpropFilterTileAccessIteratorOptimized(
                    [threadblock_shape[2], threadblock_shape[1]], element_b,
                    layout_b, self.threadmap_b, self.access_type_b
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

    rule = DefaultConv2dFprop(
        cutlass.float16, cutlass.TensorNHWC,
        cutlass.float16, cutlass.TensorNHWC,
        cutlass.float16, cutlass.TensorNHWC,
        cutlass.float32, cutlass.OpClass.TensorOp, [128, 64, 32], [64, 32, 32],
        [16, 8, 16], 2, cutlass.conv.IteratorAlgorithm.fixed_channels,
        4, 4, 8, problem_size
    )
    print(rule)
    