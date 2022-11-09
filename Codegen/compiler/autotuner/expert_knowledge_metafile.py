import cutlass
from math import floor, log2
import random


v100_metafile = {
    # V100 has at most 96 KB shared memory per SM
    "MAX_SHMEM_SIZE": 96,
    # 1 thread has at most 256 registers. Otherwise there are register spilling
    "MAX_NUM_REGISTER_PER_THREAD": 256,
    "MAX_GLOBAL_MEM_SIZE" : 32, # Unit: GB
    "REGISTER_FILE_SIZE_PER_SM" : 256, # Unit: KB
}

a100_metafile = {
    # A100 has at most 164 KB shared memory per SM
    "MAX_SHMEM_SIZE": 164,
    # 1 thread has at most 256 registers. Otherwise there are register spilling
    "MAX_NUM_REGISTER_PER_THREAD": 256,
    "MAX_GLOBAL_MEM_SIZE" : 40, # Unit: GB
    "REGISTER_FILE_SIZE_PER_SM" : 256, # Unit: KB
}

element_size = {
    cutlass.float16: 2,
    cutlass.bfloat16: 2,
    cutlass.float32: 4,
    cutlass.float64: 8
}

# Heuristics helps to autotuning an operator implementation.
# It proposes parameters for autotuning and skip non-optimal parameters 
#   (e.g., parameters that may leads to register spilling).
# In Heuristics, we consider properties from both workload and hardware.
# While new workload requires manually writing new Heuristics, these Heuristics can be reused across
# a workload with different input sizes.
# We remind that, the same GEMM workload requires significantly different parameters under different 
#   input shapes.  
class Heuristics:
    # We only check a few heuristics such as tiling size should be power of 2.
    # We expect ML cost model to learn heuristics between parameters and GPU metafile.
    def __init__(
        self, metafile, element_a, element_b, element_c, element_accumulator, layout_a, layout_b, alignment_a, alignment_b, alignment_c) -> None:
        #
        self.metafile = metafile
        self.element_a = element_a
        self.element_b = element_b
        self.element_c = element_c
        self.layout_a = layout_a
        self.layout_b = layout_b
        self.alignment_a = alignment_a
        self.alignment_b = alignment_b
        self.alignment_c = alignment_c
        self.element_accumulator = element_accumulator
        self.set_parameter_range()

    def check_shared_memory_size(self, shared_memory_size):
        return shared_memory_size < self.metafile["MAX_SHMEM_SIZE"]*1024 # Unit: Bytes

    # Check that register usage is within the register file size per SM.
    # We make a conservative assumption that there is only 1 block allocated per SM.
    def check_register_size(self, register_size):
        return register_size <= self.metafile["MAX_NUM_REGISTER_PER_THREAD"]*4*32 # 4: Unit Bytes, 32: # threads per warp

    def get_feature_from_metafile(self):
        res = []
        for key in self.metafile:
            res.append(self.metafile[key])
        return res

    def get_feature_from_parameters(self, parameters):
        res = []
        res += parameters["block_tiling_size"]
        res += parameters["warp_tiling_size"]
        res.append(parameters["num_stages"])
        res.append(parameters["log_swizzle"])
        return res

    # A heuristic for proposing a tiling size within the max_size. Should be power of 2 for efficiency.
    def propose_tiling_size(self, max_size):
        max_log = floor(log2(max_size))
        exponent = random.randint(1, max_log+1)
        return 1<<exponent

    def set_parameter_range(self):
        self.parameter_range = {
            "max_tiling_size" : 128,
            "max_stages" : 5,
            "max_log_swizzling": 3
        }

    # propose the warp, threadblock tiling size, #stage, and swizzling stride
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
        parameters = {
            "warp_tiling_size" : warp_tiling_size,
            "block_tiling_size" : block_tiling_size,
            "num_stages" : num_stages,
            "log_swizzle": log_swizzle
        } 
        return parameters

    # A heuristic for gemm only. Block tiles need to fit in shared memory.
    # Assuming there are three tiling sizes [b_m, b_n, b_k]
    # Assuming each element is float and accounts for 4 bytes
    def check_block_tiling_size_valid(self, tiling_size_list, stages):
        assert len(tiling_size_list) == 3, "Warning: Too Little Block Tiling Sizes"
        bm = tiling_size_list[0]
        bn = tiling_size_list[1]
        bk = tiling_size_list[2]
        # Assuming each element is float and accounts for 4 bytes
        expected_shmem_size = element_size[self.element_a] * (bm * bk + bn * bk) * stages
        # print(expected_shmem_size)
        if not self.check_shared_memory_size(expected_shmem_size):
            return False
        return True

    # A heuristic for gemm only. Warp tiles need to fit in registers
    # Assuming there are three tiling sizes [w_m, w_n, w_k]
    def check_warp_tiling_size_valid(self, tiling_size_list):
        assert len(tiling_size_list) == 3, "Warning: Too Little Warp Tiling Sizes"
        wm = tiling_size_list[0]
        wn = tiling_size_list[1]
        wk = tiling_size_list[2]

        buffer_a_size = element_size[self.element_a] * wm * wk * 2 # double buffer
        buffer_b_size = element_size[self.element_b] * wn * wk * 2 # double buffer
        buffer_acc    = element_size[self.element_accumulator] * wm * wn
        expected_register_size = buffer_a_size + buffer_b_size + buffer_acc
        if not self.check_register_size(expected_register_size):
            return False
        # wm & wn % 16 = 0 to match the requirement of ldmatrix
        if not(wm % 16 == 0 and wn % 8 == 0 and wk % 16 == 0):
            return False
        return True

    def check_harmony_block_warp_tiling_size(self, warp_tiling_size, block_tiling_size):
        if (warp_tiling_size[2] != block_tiling_size[2]):
            return False
        # for i in range(3):
        #     if not (block_tiling_size[i] % warp_tiling_size[i] == 0):
        #         return False
        return True

    @staticmethod
    def check_too_many_predicate(num_iteration):
        if num_iteration <= 0:
            return False
        predicate_count = num_iteration
        predicate_byte_count = (predicate_count + 3) // 4
        predicate_word_count = (predicate_byte_count + 3) // 4
        return predicate_word_count <= 4

    
    def check_memory_load_iterations(self, warp_tiling_size, block_tiling_size):
        # Cutlass assume all the threads participate in loading the lhs and rhs operands
        # So at least one iteration is required for every thread
        bm = block_tiling_size[0]
        bn = block_tiling_size[1]
        bk = block_tiling_size[2]

        wm = warp_tiling_size[0]
        wn = warp_tiling_size[1]

        num_elements_per_access = (bm * bn / wm / wn) * 32 * (16 // self.element_a)
        lhs_elements_per_access = bk * bm
        rhs_elements_per_access = bk * bn

        lhs_iteration = lhs_elements_per_access // num_elements_per_access
        rhs_iteration = rhs_elements_per_access // num_elements_per_access

        if (lhs_elements_per_access % num_elements_per_access == 0 and rhs_elements_per_access % num_elements_per_access == 0 and Heuristics.check_too_many_predicate(lhs_iteration) and Heuristics.check_too_many_predicate(rhs_iteration)):
            return True
        else:
            return False

    def check_warp_count(self, warp_tiling_size, block_tiling_size):
        bm = block_tiling_size[0]
        bn = block_tiling_size[1]

        wm = warp_tiling_size[0]
        wn = warp_tiling_size[1]

        if (bm * bn / wm / wn >= 16):
            return False
        return True

    def check_limited_tiling_size_difference(self, tiling_size):
        m = tiling_size[0]
        n = tiling_size[1]

        return (m == n) or (m == 2*n)
    
    def check_threadmap_load_iterations(self, block_tiling_size):
        bm = block_tiling_size[0]
        bn = block_tiling_size[1]
        bk = block_tiling_size[2]

        if self.layout_a == cutlass.RowMajor:
            if bk % (4 * 16 // self.element_a) != 0:
                return False
            if bk >= 128:
                return False
        else:
            if bm % (8 * 16 // self.element_a) != 0:
                return False
        
        if self.layout_b == cutlass.RowMajor:
            if bn % (8 * 16 // self.element_b) != 0:
                return False
        else:
            if bk % (4 * 16 // self.element_b) != 0:
                return False
            if bk >= 128:
                return False
        
        return True


    # More heuristics can be added...
    def is_valid(self, parameters):
        # valid = self.check_warp_tiling_size_valid(parameters["warp_tiling_size"]) \
        #         and self.check_block_tiling_size_valid(parameters["block_tiling_size"], parameters["num_stages"]) \
        #          \
        #         and self.check_memory_load_iterations(parameters["warp_tiling_size"], parameters["block_tiling_size"]) \
        #          \
        #         and self.check_limited_tiling_size_difference(parameters["warp_tiling_size"]) \
        #         and self.check_limited_tiling_size_difference(parameters["block_tiling_size"]) \
        #         and self.check_threadmap_load_iterations(parameters["block_tiling_size"])
        try:
            GemmUniversalRule(
                self.element_a, self.layout_a, self.alignment_a,
                self.element_b, self.layout_b, self.alignment_b,
                self.element_c, self.alignment_c,
                self.element_accumulator, parameters["block_tiling_size"],
                parameters["warp_tiling_size"], [16, 8, 16], parameters["num_stages"])
        except:
            return False
        
        valid = self.check_block_tiling_size_valid(parameters["block_tiling_size"], parameters["num_stages"]) \
            and self.check_warp_tiling_size_valid(parameters["warp_tiling_size"]) \
            and self.check_warp_count(parameters["warp_tiling_size"], parameters["block_tiling_size"]) \
            and self.check_harmony_block_warp_tiling_size(parameters["warp_tiling_size"], parameters["block_tiling_size"])
        return valid

# if __name__ == "__main__":
#     heuristics = Heuristics(
#         a100_metafile, element_a=cutlass.float16, element_b=cutlass.float16, 
#         element_accumulator=cutlass.float32, layout_a=cutlass.RowMajor, 
#         layout_b=cutlass.RowMajor)
    
#     parameters = {
#         "warp_tiling_size" : [64, 64, 32],
#         "block_tiling_size" : [128, 128, 32],
#         "num_stages" : 5,
#         "log_swizzle": 3
#     }

#     print(heuristics.is_valid(parameters))


################################################################################
# Default Mma Core:
#
#   A: column-major
#   B: row-major
#   Operator: tensor op class
#
#  static_assert(
#       !(Shape::kM % WarpShape::kM) && !(Shape::kN % WarpShape::kN),
#       "Threadblock-scoped GEMM should be divisible by warp-scoped GEMM size.");


class PitchLinearShape:
    def __init__(self, contiguous, strided) -> None:
        self.contiguous_ = contiguous
        self.strided_ = strided
    
    def contiguous(self):
        return self.contiguous_
    def strided(self):
        return self.strided_
    def count(self):
        return self.contiguous_ * self.strided_

class MatrixShape:
    def __init__(self, row, column) -> None:
        self.row_ = row
        self.column_ = column
    
    def row(self):
        return self.row_
    
    def column(self):
        return self.column_

class Array:
    def __init__(self, element, aligment) -> None:
        self.elements = aligment


class GemmUniversalRule:
    def __init__(
        self, 
        element_A, layout_A, alignment_A, 
        element_B, layout_B, alignment_B,
        element_C, alignment_C,
        element_accumulator, threadblock_shape, warp_shape, instruction_shape, stages
        ) -> None:
        
        self.Mma = DefaultMma(
            element_A, layout_A, alignment_A, element_B, layout_B, alignment_B,
             element_accumulator, threadblock_shape, warp_shape, 
             instruction_shape, stages).threadblock_mma
        
        self.Epilogue = DefaultEpilogueTensorOp(
            threadblock_shape, self.Mma.operator, threadblock_shape[2] // warp_shape[2], element_C, alignment_C
        )

        self.Visitor = EpilogueWithVisitor(warp_shape, instruction_shape, self.Epilogue, alignment_C)

class DefaultMma:
    def __init__(
        self, 
        element_A, layout_A, alignment_A, 
        element_B, layout_B, alignment_B,
        element_accumulator, threadblock_shape, warp_shape, instruction_shape, stages
        ) -> None:
        
        self.mma_core = DefaultMmaCore(
            threadblock_shape, warp_shape, instruction_shape, element_A, layout_A, element_B, layout_B, element_accumulator
        )

        access_type_A = Array(element_A, alignment_A)

        self.iterator_A = PredicatedTileAccessIterator(
            MatrixShape(threadblock_shape[0], threadblock_shape[2]), 
            element_A, layout_A, 1, self.mma_core.iterator_threadmap_A, access_type_A
        )

        access_type_B = Array(element_A, alignment_B)
        
        self.iterator_B = PredicatedTileAccessIterator(
            MatrixShape(threadblock_shape[2], threadblock_shape[1]), 
            element_B, layout_B, 0, self.mma_core.iterator_threadmap_B, access_type_B
        )

        self.threadblock_mma = MmaMultistage(
            threadblock_shape, self.mma_core.mma_policy, stages
        )

class DefaultMmaCore:
    def __init__(
        self, threadblock_shape, warp_shape, instruction_shape,
        element_A, layout_A, element_B, layout_B, element_accumulator) -> None:

        M, N, K = threadblock_shape

        Mw, Nw, Kw = warp_shape

        self.threadblock_shape = threadblock_shape
        self.warp_shape = warp_shape

        # assertion
        assert M % Mw == 0 and N % Nw == 0, "Threadblock-scoped GEMM should be divisible by warp-scoped GEMM size"

        num_threads = (M // Mw) * (N // Nw) * (K // Kw) * 32
        access_size_in_bits = 128

        if layout_A == cutlass.RowMajor:
            self.smem_layout_A = RowMajorTensorOpMultiplicandCrosswise(
                8 * element_size[element_A], K
            )
            warp_thread_arrangement_contiguous_A = K // (access_size_in_bits // 8 // element_size[element_A])
            warp_thread_arrangement_strided_A = 32 // warp_thread_arrangement_contiguous_A
            self.iterator_threadmap_A = PitchLinearWarpRakedThreadMap(
                PitchLinearShape(K, M), num_threads, 
                PitchLinearShape(
                    warp_thread_arrangement_contiguous_A, warp_thread_arrangement_strided_A),
                access_size_in_bits // 8 // element_size[element_A]
            )
            self.smem_iterator_A = RegularTileAccessIterator(
                MatrixShape(M, K), element_A, self.smem_layout_A, 0, self.iterator_threadmap_A, crosswise=K
            )

        elif layout_A == cutlass.ColumnMajor:
            self.smem_layout_A = ColumnMajorTensorOpMultiplicandCongruous(
                8 * element_size[element_A], 128 // element_size[element_A]
            )
            self.iterator_threadmap_A = PitchLinearWarpRakedThreadMap(
                PitchLinearShape(M, K), num_threads, PitchLinearShape(8, 4),
                access_size_in_bits // 8 // element_size[element_A]
            )
            self.smem_iterator_A = RegularTileAccessIterator(
                MatrixShape(M, K), element_A, self.smem_layout_A, 1, self.iterator_threadmap_A
            )
        
        if layout_B == cutlass.RowMajor:
            self.smem_layout_B = RowMajorTensorOpMultiplicandCongruous(
                8 * element_size[element_B], 128 // element_size[element_B]
            )
            self.iterator_threadmap_B = PitchLinearWarpRakedThreadMap(
                PitchLinearShape(N, K), num_threads, PitchLinearShape(8, 4),
                access_size_in_bits // 8 // element_size[element_B]
            )
            self.smem_iterator_B = RegularTileAccessIterator(
                MatrixShape(K, N), element_B, self.smem_layout_B, 0, self.iterator_threadmap_B
            )
        elif layout_B == cutlass.ColumnMajor:
            self.smem_layout_B = ColumnMajorTensorOpMultiplicandCrosswise(
                8 * element_size[element_B], K
            )
            warp_thread_arrangement_contiguous_B = K // (access_size_in_bits // 8 // element_size[element_B])
            warp_thread_arrangement_strided_B = 32 // warp_thread_arrangement_contiguous_B
            self.iterator_threadmap_B = PitchLinearWarpRakedThreadMap(
                PitchLinearShape(K, N), num_threads, 
                PitchLinearShape(
                    warp_thread_arrangement_contiguous_B, warp_thread_arrangement_strided_B),
                access_size_in_bits // 8 // element_size[element_B]
            )
            self.smem_iterator_B = RegularTileAccessIterator(
                MatrixShape(K, N), element_B, self.smem_layout_B, 1, self.iterator_threadmap_B, crosswise=K
            )

        # using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
        #     WarpShape, InstructionShape, ElementA, SmemLayoutA, ElementB, SmemLayoutB,
        #     ElementC, LayoutC, Operator, WarpCount::kK>::Type;
        mma_tensor_op = DefaultMmaTensorOp(
            warp_shape, instruction_shape, element_A, self.smem_layout_A, element_B, self.smem_layout_B, element_accumulator, cutlass.RowMajor, K // Kw
        ).type
        
        # policy
        self.mma_policy = MmaPolicy(
            mma_tensor_op, MatrixShape(0, 0), MatrixShape(0, 0), K // Kw
        )

class MmaPolicy:
    def __init__(self, operator, smem_padding_A, smem_padding_B, partitions_K=1) -> None:
        self.operator = operator

class DefaultMmaTensorOp:
    def __init__(self, warp_shape, instruction_shape, element_A, layout_A, element_B, layout_B, element_C, Layout_C, partition_k) -> None:
        self.policy = MmaTensorOpPolicy(
            Mma(instruction_shape, 32, element_A, cutlass.RowMajor, element_B, cutlass.ColumnMajor, element_C, cutlass.RowMajor),
            MatrixShape(1, 1)
        )
        self.type = MmaTensorOp(
            warp_shape, element_A, layout_A, element_B, layout_B, element_C, Layout_C, self.policy, partition_k
        )

class MmaTensorOpPolicy:
    def __init__(self, operator, op_delta) -> None:
        self.operator = operator
        self.op_delta = op_delta
        self.mma_shape = operator.shape
        pass

class MmaTensorOp:
    def __init__(
        self, warp_shape, element_A, layout_A, element_B, layout_B, element_C, layout_C, policy, partition_k) -> None:
        self.shape = warp_shape
        self.policy = policy

        arch_mma_operator = policy.operator

        self.iterator_A = MmaTensorOpMultiplicandTileIterator(
            MatrixShape(warp_shape[0], warp_shape[2]), "A", element_A, layout_A,
            MatrixShape(arch_mma_operator.shape[0], arch_mma_operator.shape[2]),
            policy.op_delta.row(), partition_k
        )

        self.iterator_B = MmaTensorOpMultiplicandTileIterator(
            MatrixShape(warp_shape[2], warp_shape[1]), "B", element_B, layout_B,
            MatrixShape(arch_mma_operator.shape[2], arch_mma_operator.shape[1]),
            policy.op_delta.row(), partition_k
        )

        # self.iterator_C = MmaTensorOpAccumulatorTileIterator(
        #     MatrixShape(warp_shape[0], warp_shape[1]), element_C, layout_C,
        #     arch_mma_operator.shape, policy.opdelta
        # )


class MmaTensorOpMultiplicandTileIteratorTensorOpMultiplicandCongruous:
    def __init__(self, shape, operand, element, layout, instruction_shape, opdelta, partition_k) -> None:
        assert shape.contiguous() % instruction_shape.contiguous() == 0, "Shape of warp-level Mma must be divisible by operator shape."
        layout = TensorOpMultiplicandCongruous(8 * element_size[element], 64)
        ldsm_op_outer = layout.elements_per_access
        ldsm_op_inner = 8
        assert shape.contiguous() % ldsm_op_outer == 0, "Shape of warp-level mma must be divisible by LDSM's fundamental tile size."
        assert shape.strided() % ldsm_op_inner == 0, "Shape of warp-level mma must be divisible by LDSM's fundamental tile size."


class MmaTensorOpMultiplicandTileIteratorAColumnMajorTensorOpMultiplicandCongruous(MmaTensorOpMultiplicandTileIteratorTensorOpMultiplicandCongruous):
    def __init__(self, shape, operand, element, instruction_shape, opdelta, partition_k) -> None:
        super().__init__(
            PitchLinearShape(shape.row(), shape.column()), operand, element,
            TensorOpMultiplicandCongruous(8 * element_size[element], 128 // element_size[element]),
            PitchLinearShape(instruction_shape.row(), instruction_shape.column()),
            opdelta, partition_k)

class MmaTensorOpMiltiplicandTileIteratorBRowMajorTensorOpMultiplicandCongruous(MmaTensorOpMultiplicandTileIteratorTensorOpMultiplicandCongruous):
    def __init__(self, shape, operand, element, instruction_shape, opdelta, partition_k) -> None:
        super().__init__(
            PitchLinearShape(shape.column(), shape.row()), operand, element,
            TensorOpMultiplicandCongruous(8 * element_size[element], 128 // element_size[element]),
            PitchLinearShape(instruction_shape.column(), instruction_shape.row()),
            opdelta, partition_k)

class MmaTensorOpMultiplicandTileIteratorTensorOpMultiplicandCrosswise:
    def __init__(self, shape, operand, element, layout, instruction_shape, opdelta, partition_k) -> None:
        assert shape.contiguous() % instruction_shape.contiguous() == 0, "Shape of warp-level Mma must be divisible by operator shape."
        layout = TensorOpMultiplicandCongruous(8 * element_size[element], 64)
        ldsm_op_outer = layout.elements_per_access
        ldsm_op_inner = 8
        assert shape.contiguous() % ldsm_op_outer == 0, "Shape of warp-level mma must be divisible by LDSM's fundamental tile size."
        assert shape.strided() % ldsm_op_inner == 0, "Shape of warp-level mma must be divisible by LDSM's fundamental tile size."

class MmaTensorOpMultiplicandTileIteratorBColumnMajorTensorOpMultiplicandCrosswise(MmaTensorOpMultiplicandTileIteratorTensorOpMultiplicandCrosswise):
    def __init__(self, shape, operand, element, layout, instruction_shape, opdelta, partition_k) -> None:
        super().__init__(
            PitchLinearShape(shape.row(), shape.column()), operand, element,
            TensorOpMultiplicandCrosswise(8 * element_size[element], layout.crosswise),
            PitchLinearShape(instruction_shape.row(), instruction_shape.column()),
            opdelta, partition_k)

class MmaTensorOpMultiplicandTileIteratorARowMajorTensorOpMultiplicandCrosswise(MmaTensorOpMultiplicandTileIteratorTensorOpMultiplicandCrosswise):
    def __init__(self, shape, operand, element, layout, instruction_shape, opdelta, partition_k) -> None:
        super().__init__(
            PitchLinearShape(shape.column(), shape.row()), operand, element,
            TensorOpMultiplicandCrosswise(8 * element_size[element], layout.crosswise),
            PitchLinearShape(instruction_shape.column(), instruction_shape.row()),
            opdelta, partition_k)


class MmaTensorOpMultiplicandTileIterator:
    def __init__(self, shape, operand, element, layout, instruction_shape, opdelta, partition_k) -> None:
        if operand == "A" and isinstance(layout, ColumnMajorTensorOpMultiplicandCongruous):
            MmaTensorOpMultiplicandTileIteratorAColumnMajorTensorOpMultiplicandCongruous(
                shape, operand, element, instruction_shape, opdelta, partition_k
            )
        elif operand == "A" and isinstance(layout, RowMajorTensorOpMultiplicandCrosswise):
            MmaTensorOpMultiplicandTileIteratorARowMajorTensorOpMultiplicandCrosswise(
                shape, operand, element, layout, instruction_shape, opdelta, partition_k
            )
        elif operand == "B" and isinstance(layout, RowMajorTensorOpMultiplicandCongruous):
            MmaTensorOpMiltiplicandTileIteratorBRowMajorTensorOpMultiplicandCongruous(
                shape, operand, element, instruction_shape, opdelta, partition_k
            )
        elif operand == "B" and isinstance(layout, ColumnMajorTensorOpMultiplicandCrosswise):
            MmaTensorOpMultiplicandTileIteratorBColumnMajorTensorOpMultiplicandCrosswise(
                shape, operand, element, layout, instruction_shape, opdelta, partition_k
            )
        else:
            raise NotImplementedError()

class Mma:
    def __init__(
        self, instruction_shape, warp_size, element_A, layout_A, element_B, layout_B, element_C, layout_C) -> None:
        self.shape = instruction_shape
        pass


class TensorOpMultiplicand:
    def __init__(self, element_size, crosswise) -> None:
        access_size = 128  # this layout is optimized for 128b access
        elements_per_access = access_size // element_size
        # Contiguous dimension of the tile shape matches one shared memory cache
        # line - 128B.  For 128bit access size, it equals to 8 accesses.
        tile_shape_contiguous = 128 // (access_size // 8)
        factor = tile_shape_contiguous * elements_per_access // crosswise
        assert factor > 0, "kCrosswise should be no large than one shared memory cache line."
        self.crosswise = crosswise
        self.elements_per_access = elements_per_access
        

class TensorOpMultiplicandCrosswise(TensorOpMultiplicand):
    def __init__(self, element_size, crosswise) -> None:
        super().__init__(element_size, crosswise)

class TensorOpMultiplicandCongruous(TensorOpMultiplicand):
    def __init__(self, element_size, crosswise) -> None:
        super().__init__(element_size, crosswise)
        
class RowMajorTensorOpMultiplicandCrosswise(TensorOpMultiplicandCrosswise):
    def __init__(self, element_size, crosswise) -> None:
        super().__init__(element_size, crosswise)

class ColumnMajorTensorOpMultiplicandCongruous(TensorOpMultiplicandCongruous):
    def __init__(self, element_size, crosswise) -> None:
        super().__init__(element_size, crosswise)

class RowMajorTensorOpMultiplicandCongruous(TensorOpMultiplicandCongruous):
    def __init__(self, element_size, crosswise) -> None:
        super().__init__(element_size, crosswise)

class ColumnMajorTensorOpMultiplicandCrosswise(TensorOpMultiplicandCrosswise):
    def __init__(self, element_size, crosswise) -> None:
        super().__init__(element_size, crosswise)


class PitchLinearWarpRakedThreadMap:
    def __init__(self, threadblock_shape, num_threads, warp_thread_arrangement, elements_per_access) -> None:
        
        assert threadblock_shape.contiguous() % elements_per_access == 0, "Shape must be divisible by vector length."

        shape_in_accesses = PitchLinearShape(
            threadblock_shape.contiguous() // elements_per_access,
            threadblock_shape.strided()
        )

        self.elements_per_access = elements_per_access

        assert shape_in_accesses.contiguous() % warp_thread_arrangement.contiguous() == 0, "ShapeInAccesses must be divisible by WarpThreadArrangement."
        assert shape_in_accesses.strided() % warp_thread_arrangement.strided() == 0, "ShapeInAccesses must be divisible by WarpThreadArrangement."

        class Detail:
            def __init__(self) -> None:
                self.warp_thread_arrangement = warp_thread_arrangement
                warp_count = num_threads // 32
                self.warp_access_iterations = PitchLinearShape(
                    shape_in_accesses.contiguous() // warp_thread_arrangement.contiguous(),
                    shape_in_accesses.strided() // warp_thread_arrangement.strided()
                )
                if self.warp_access_iterations.strided() >= warp_count:
                    self.warps_strided = warp_count
                else:
                    self.warps_strided = self.warp_access_iterations.strided()
                
                if warp_count > self.warp_access_iterations.strided():
                    self.warps_contiguous = warp_count // self.warps_strided
                else:
                    self.warps_contiguous = 1

        
        detail = Detail()

        self.iterations = PitchLinearShape(
            detail.warp_access_iterations.contiguous() // detail.warps_contiguous,
            detail.warp_access_iterations.strided() // detail.warps_strided
        )
        assert self.iterations.count() != 0, "Number of iterations must be non-zero"

        self.delta = PitchLinearShape(
            detail.warp_thread_arrangement.contiguous() * elements_per_access,
            detail.warp_thread_arrangement.strided()
        )


class RegularTileAccessIteratorTensorOpMultiplicandCongruous:
    def __init__(self, threadblock_shape, element, advance_rank, threadmap, alignment) -> None:
        access_size_in_bits = 128
        assert 8 * element_size[element] * threadmap.elements_per_access == access_size_in_bits, "This iterator requires a policy whose access size is 128bs"


class RegularTileAccessIteratorColumnMajorTensorOpMultiplicandCongruous(RegularTileAccessIteratorTensorOpMultiplicandCongruous):
    def __init__(self, threadblock_shape, element, advance_rank, threadmap, alignment) -> None:
        super().__init__(
            PitchLinearShape(threadblock_shape.row(), threadblock_shape.column()),
            element, advance_rank, threadmap, alignment)


class RegularTileAccessIteratorRowMajorTensorOpMultiplicandCongruous(RegularTileAccessIteratorTensorOpMultiplicandCongruous):
    def __init__(self, threadblock_shape, element, advance_rank, threadmap, alignment) -> None:
        super().__init__(
            PitchLinearShape(threadblock_shape.column(), threadblock_shape.row()),
            element, 1 - advance_rank, threadmap, alignment)


class RegularTileAccessIteratorTensorOpMultiplicandCrosswise:
    def __init__(self, threadblock_shape, element, advance_rank, threadmap, alignment, crosswise) -> None:
        assert threadmap.delta.contiguous() % crosswise == 0, "kCrosswise is the smallest unit in the contiguous dimension for shared memory swizzling."
        access_size_in_bits = 128
        assert 8 * element_size[element] * threadmap.elements_per_access == access_size_in_bits, "This iterator requires a policy whose access size is 128bs"

class RegularTileAccessIteratorColumnMajorTensorOpMultiplicandCrosswise(RegularTileAccessIteratorTensorOpMultiplicandCrosswise):
    def __init__(self, threadblock_shape, element, advance_rank, threadmap, alignment, crosswise) -> None:
        super().__init__(
            PitchLinearShape(threadblock_shape.row(), threadblock_shape.column()),
            element, advance_rank, threadmap, alignment, crosswise),

class RegularTileAccessIteratorColumnMajorTensorOpMultiplicandCrosswise(RegularTileAccessIteratorTensorOpMultiplicandCrosswise):
    def __init__(self, threadblock_shape, element, advance_rank, threadmap, alignment, crosswise) -> None:
        super().__init__(
            PitchLinearShape(threadblock_shape.column(), threadblock_shape.row()),
            element, 1 - advance_rank, threadmap, alignment, crosswise),


class RegularTileAccessIterator:
    def __init__(self, threadblock_shape, element, layout, advance_rank, threadmap, alignment=None, crosswise=None) -> None:
        if alignment is None:
            alignment = element_size[element] * threadmap.elements_per_access
        if isinstance(layout, ColumnMajorTensorOpMultiplicandCongruous):
            RegularTileAccessIteratorColumnMajorTensorOpMultiplicandCongruous(
                threadblock_shape, element, advance_rank, threadmap, alignment
            )
        elif isinstance(layout, RowMajorTensorOpMultiplicandCongruous):
            RegularTileAccessIteratorRowMajorTensorOpMultiplicandCongruous(
                threadblock_shape, element, advance_rank, threadmap, alignment
            )
        elif isinstance(layout, ColumnMajorTensorOpMultiplicandCrosswise):
            RegularTileAccessIteratorColumnMajorTensorOpMultiplicandCrosswise(
                threadblock_shape, element, advance_rank, threadmap, alignment, crosswise
            )
        elif isinstance(layout, RowMajorTensorOpMultiplicandCrosswise):
            RegularTileAccessIteratorColumnMajorTensorOpMultiplicandCrosswise(
                threadblock_shape, element, advance_rank, threadmap, alignment, crosswise
            )


class PredicatedTileAccessIteratorPredicates:
    def __init__(self, threadblock_shape, element, advance_rank, threadmap, access_type) -> None:

        assert threadmap.elements_per_access % access_type.elements == 0, "Vectors implied by the thread map must be divisible by the access type."

        accesses_per_vector = threadmap.elements_per_access // access_type.elements

        predicates_per_byte = 4
        predicate_count = threadmap.iterations.count() * accesses_per_vector

        predicated_byte_count = (predicate_count + predicates_per_byte - 1) // predicates_per_byte
        predicated_word_count = (predicated_byte_count + 3) // 4

        assert predicated_word_count <= 4, "Too many prediacates"
        pass

class PredicatedTileAccessIteratorPitchLinear(PredicatedTileAccessIteratorPredicates):
    def __init__(self, threadblock_shape, element, advance_rank, threadmap, access_type) -> None:
        super().__init__(threadblock_shape, element, advance_rank, threadmap, access_type)

class PredicatedTileAccessIteratorRowMajor(PredicatedTileAccessIteratorPitchLinear):
    def __init__(self, threadblock_shape, element, advance_rank, threadmap, access_type) -> None:
        super().__init__(
            PitchLinearShape(threadblock_shape.column(), threadblock_shape.row()),
            element, 1 - advance_rank, threadmap, access_type)

class PredicatedTileAccessIteratorColumnMajor(PredicatedTileAccessIteratorPitchLinear):
    def __init__(self, threadblock_shape, element, advance_rank, threadmap, access_type) -> None:
        super().__init__(
            PitchLinearShape(threadblock_shape.row(), threadblock_shape.column()),
            element, advance_rank, threadmap, access_type)


class PredicatedTileAccessIterator:
    def __init__(self, threadblock_shape, element, layout, advance_rank, threadmap, access_type) -> None:
        if layout == cutlass.RowMajor:
            PredicatedTileAccessIteratorRowMajor(threadblock_shape, element, advance_rank, threadmap, access_type)
        elif layout == cutlass.ColumnMajor:
            PredicatedTileAccessIteratorColumnMajor(threadblock_shape, element, advance_rank, threadmap, access_type)


class MmaBase:
    def __init__(self, threadblock_shape, policy, stages) -> None:
        warp_gemm = policy.operator.shape
        operator = policy.operator
        warp_gemm_iterations =  warp_gemm[2] // operator.policy.mma_shape[2]
        assert warp_gemm_iterations > 1, "The pipelined structure requires at least two warp-level GEMM operations."
        assert warp_gemm_iterations % 2 == 0, "Inner loop iteration must be an even number."
        self.operator = operator


class MmaMultistage(MmaBase):
    def __init__(self, threadblock_shape, policy, stages) -> None:
        super().__init__(threadblock_shape, policy, stages)


class PredictedTileIteratorEpilogue:
    def __init__(self, threadmap, element_C) -> None:
        assert threadmap.iterations.row > 0, "ThreadMap::Iterations::kRow must be > 0"
        assert threadmap.iterations.group > 0, "ThreadMap::Iterations::kGroup must be > 0"
        assert threadmap.iterations.cluster > 0, "ThreadMap::Iterations::kCluster must be > 0"
        assert threadmap.iterations.column > 0, "ThreadMap::Iterations::kColumn must be > 0"

## Epilogue Rules
class DefaultEpilogueTensorOp:
    def __init__(self, threadblock_shape, warp_mma_tensor_op, partition_k, element_C, elements_per_access) -> None:
        # https://github.com/NVIDIA/cutlass/blob/12f4108ac2233022b92f0e3533c23bed399fcf45/include/cutlass/epilogue/threadblock/default_thread_map_tensor_op.h
        self.output_tile_threadmp = DefaultThreadMapTensorOp(
            threadblock_shape, warp_mma_tensor_op.shape, partition_k, element_C, elements_per_access
        ).type
        # No assertion

        self.output_tile_iterator = PredictedTileIteratorEpilogue(
            self.output_tile_threadmp, element_C
        )

class OutputTileShape:
    def __init__(
        self, column, row, group, cluster, tile) -> None:
        self.column = column
        self.row = row
        self.group = group
        self.cluster = cluster
        self.tile = tile
        self.count = column * row * group * cluster * tile


class RowArrangement:
    def __init__(self, shape, warps_remaining, elements_per_access, element_size, is_2d_tile) -> None:
        if is_2d_tile:
            warp_size = 32
            memory_access_size = 256
            class Detail:
                def __init__(self) -> None:
                    self.shape_width = shape.column // elements_per_access
                    self.shape_row = shape.row // warps_remaining
                    target_memory_access_width = memory_access_size // (elements_per_access * element_size // 8)
                    self.target_access_rows = warp_size // target_memory_access_width
            
            detail = Detail()
            if detail.target_access_rows > detail.shape_row:
                access_width = warp_size // detail.shape_row
            else:
                access_width = min(detail.shape_width, min(
                    warp_size, memory_access_size // (elements_per_access * element_size // 8)
                ))
            
            if detail.target_access_rows > detail.shape_row:
                access_rows = detail.shape_row
            else:
                access_rows = min(shape.row, warp_size // access_width)
            self.iterations_column = detail.shape_width // access_width
            self.iterations_row = detail.shape_row // access_rows

            assert access_width * elements_per_access <= shape.column, "Accessing too many elements per access"
            assert self.iterations_column > 0, "Iteration Count Column must be > 0"
            assert self.iterations_row > 0, "Iteration Count Row must be > 0"
        else:
            self.iterations_column = shape.column // elements_per_access // 32
            self.iterations_row = 1


class OutputTileOptimalThreadMap:
    def __init__(self, shape, count, threads, elements_per_access, element_size) -> None:
        self.count = count
        class Detail:
            def __init__(self) -> None:
                warp_count = threads // 32
                if shape.cluster > warp_count:
                    warps_remaining_for_groups = 1
                else:
                    warps_remaining_for_groups = warp_count // shape.cluster

                if shape.group > warps_remaining_for_groups:
                    warps_remaining_for_rows = 1
                else:
                    warps_remaining_for_rows = warps_remaining_for_groups // shape.group
                self.row_arrangement = RowArrangement(
                    shape, warps_remaining_for_rows,
                    elements_per_access,
                    element_size,
                    shape.row > warps_remaining_for_rows
                )
                if shape.group > warps_remaining_for_groups:
                    self.iterations_group = shape.group // warps_remaining_for_groups
                else:
                    self.iterations_group = 1
                
                if shape.cluster > warp_count:
                    self.iterations_cluster = shape.cluster // warp_count
                else:
                    self.iterations_cluster = 1
        
        detail = Detail()
        self.iterations = OutputTileShape(
            detail.row_arrangement.iterations_column,
            detail.row_arrangement.iterations_row,
            detail.iterations_group,
            detail.iterations_cluster, 
            1
        )
        pass

class DefaultThreadMapTensorOp:
    def __init__(self, threadblock_shape, warp_shape, partition_k, element_C, elements_per_access) -> None:

        class Detail:
            def __init__(self) -> None:
                self.tensor_op_rows = 8
                self.warp_count = [
                    threadblock_shape[0] // warp_shape[0],
                    threadblock_shape[1] // warp_shape[1],
                    threadblock_shape[2] // warp_shape[2]
                ]
                self.threads = self.warp_count[0] * self.warp_count[1] * self.warp_count[2] * 32
        
        detail = Detail()
        self.type = OutputTileOptimalThreadMap(
            OutputTileShape(
                threadblock_shape[1], detail.tensor_op_rows, detail.warp_count[0], 1, 1
            ),
            OutputTileShape(
                1, warp_shape[0] // detail.tensor_op_rows, 1, 1, warp_shape[0] // detail.tensor_op_rows
            ),
            detail.threads,
            elements_per_access, 8 * element_size[element_C]
        )
        pass


class EpilogueWithVisitor:
    def __init__(self, warp_shape, operator_shape, epilogue, elements_per_access) -> None:
        policy_operator_count_krow = (warp_shape[0] + operator_shape[0] - 1) // operator_shape[0]
        policy_operator_count_kcolumn = (warp_shape[1] + operator_shape[1] - 1) // operator_shape[1]
        operator_fragment_kelements = operator_shape[0] * operator_shape[1] // 32
        accumulator_tile_kelements = operator_fragment_kelements * policy_operator_count_krow * policy_operator_count_kcolumn
        accumulator_fragment_count = accumulator_tile_kelements // (epilogue.output_tile_threadmp.count.tile * elements_per_access)
        assert accumulator_fragment_count > 0



if __name__ == "__main__":

    rule = GemmUniversalRule(
        cutlass.float16, cutlass.RowMajor, 8,
        cutlass.float16, cutlass.ColumnMajor, 8,
        cutlass.float16, 8,
        cutlass.float32, [128, 16, 64], [64, 16, 64],
        [16, 8, 16], 2
    )
    