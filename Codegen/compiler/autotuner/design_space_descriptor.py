import cutlass
from autotuner.gemm_heuristic import GemmHeuristics, BatchedGemmHeuristics
from autotuner.conv_heuristic import Conv2dHeuristics
from pycutlass import *
import pycutlass
from pycutlass.test.profiler import GpuTimer

class ConfigDescriptor:
    def __init__(self, autotuning_rounds=15) -> None:
        self.autotuning_rounds = autotuning_rounds
        self.arg_key = {
            cutlass.float16: "f16",
            cutlass.bfloat16: "bf16",
            cutlass.float32: "f32",
            cutlass.RowMajor: "row",
            cutlass.ColumnMajor: "col"
        }
        pass
    
    def get_key(self):
        args = self.problem_description.values()
        key = self.OP_NAME
        for arg in list(args):
            try:
                if isinstance(arg, int):
                    key += "_%d" % arg
                elif isinstance(arg, bool):
                    key += "_%d" % int(arg)
                elif isinstance(arg, str):
                    key += "_" + arg
                else:
                    key += self.arg_key[arg]
            except:
                for size in arg:
                    key += "_%d" % size
                key += "_"
        return key


class GemmConfigDescriptor(ConfigDescriptor):
    CONFIG = [
        "block_x", "block_y", "block_z", "warp_x", "warp_y", 
        "warp_z", "stage", "swizzle", "split_k_slices"]
    OP_NAME = "GEMM"
    def __init__(self, problem_size, element_a, layout_a, element_b, layout_b,
        element_c, layout_c, element_accumulator, alignment_a, alignment_b,
        alignment_c, split_k_mode="None", autotuning_rounds=15) -> None:
        #
        super().__init__(autotuning_rounds)

        self.problem_description = {
            "element_a": element_a,
            "element_b": element_b,
            "element_c": element_c,
            "element_accumulator": element_accumulator,
            "layout_a": layout_a,
            "layout_b": layout_b,
            "layout_c": layout_c,
            "alignment_a": alignment_a,
            "alignment_b": alignment_b,
            "alignment_c": alignment_c,
            "split_k_mode": split_k_mode,
            "problem_size": problem_size
        }

        self.element_a = element_a
        self.element_b = element_b
       

        self.heuristic = GemmHeuristics(
            element_a, element_b, element_c, element_accumulator,
            layout_a, layout_b, alignment_a, alignment_b, alignment_c,
            split_k_mode
        )
    
    def cache_init(self):
        try:
            connection = sqlite3.connect("./compiled_cache.db")
            cursor = connection.cursor()
            sqlite_create_table_query = """CREATE TABLE gemm_best_config(
                op_key TEXT NOT NULL UNIQUE, block_x INTEGER, block_y INTEGER, 
                block_z INTEGER, warp_x INTEGER, warp_y INTEGER, warp_z INTEGER, 
                stage INTEGER, swizzle INTEGER, split_k_slices INTEGER)"""
            cursor.execute(sqlite_create_table_query)
            connection.commit()
            cursor.close()
        except:
            pass
    
    def cache_load(self, key):
        connection = sqlite3.connect("./compiled_cache.db")
        cursor = connection.cursor()
        sqlite_fetch_blob_query = """SELECT * from gemm_best_config where op_key = ?"""

        cursor.execute(sqlite_fetch_blob_query, (key, ))
        record = cursor.fetchall()
        
        if len(record) == 0:
            return None
        else:
            record = list(record[0])[1:]
            parameters = {
                "block_x": record[0],
                "block_y": record[1],
                "block_z": record[2],
                "warp_x": record[3],
                "warp_y": record[4],
                "warp_z": record[5],
                "stage": record[6],
                "log_swizzle": record[7],
                "split_k_slices": record[8]
            } 
            return parameters
    
    def cache_insert_with_key(self, key, best_config):
        connection = sqlite3.connect("./compiled_cache.db")
        cursor = connection.cursor()
        sqlite_insert_blob_query = """ INSERT OR IGNORE INTO 
            gemm_best_config (op_key, block_x, block_y, block_z, warp_x, 
                warp_y, warp_z, stage, swizzle, split_k_slices) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
        cursor.execute(sqlite_insert_blob_query, tuple([key,] + best_config))
        connection.commit()
        cursor.close()
    
    def cache_insert(self, best_config):
        key = self.get_key()
        connection = sqlite3.connect("./compiled_cache.db")
        cursor = connection.cursor()
        sqlite_insert_blob_query = """ INSERT OR IGNORE INTO 
            gemm_best_config (op_key, block_x, block_y, block_z, warp_x, 
                warp_y, warp_z, stage, swizzle, split_k_slices) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
        cursor.execute(sqlite_insert_blob_query, tuple([key,] + best_config))
        connection.commit()
        cursor.close()

    def generate_code(self, config):
        math_inst = MathInstruction(
            instruction_shape=[16, 8, 16],
            element_a=self.problem_description["element_a"], 
            element_b=self.problem_description["element_b"],
            element_accumulator=self.problem_description["element_accumulator"],
            opcode_class=cutlass.OpClass.TensorOp
        )

        threadblock_shape = [
            config["block_x"], config["block_y"], config["block_z"]
        ]

        warp_count = [
            config["block_x"] // config["warp_x"],
            config["block_y"] // config["warp_y"],
            config["block_z"] // config["warp_z"]
        ]

        stages = config["stage"]

        tile_description = TileDescription(
            threadblock_shape, stages, warp_count, math_inst
        )
        swizzling_functor = getattr(
            cutlass, "IdentitySwizzle%d"%int(pow(2, config["log_swizzle"])))
            

        A = TensorDescription(
            self.problem_description["element_a"], 
            self.problem_description["layout_a"], 
            self.problem_description["alignment_a"])
        B = TensorDescription(
            self.problem_description["element_b"], 
            self.problem_description["layout_b"], 
            self.problem_description["alignment_b"])
        C = TensorDescription(
            self.problem_description["element_c"], 
            self.problem_description["layout_c"], 
            self.problem_description["alignment_c"])

        epilogue_functor = LinearCombination(
            element_output=C.element, epilogue_vector_length=C.alignment,
            element_accumulator=math_inst.element_accumulator,
            element_epilogue=cutlass.float32
        )

        operation = GemmOperationUniversal(
            arch=80, tile_description=tile_description,
            A=A, B=B, C=C, epilogue_functor=epilogue_functor,
            swizzling_functor=swizzling_functor
        )

        pycutlass.compiler.add_module([operation])

        return operation
    
    def generate_code_and_profile(self, config):

        math_inst = MathInstruction(
            instruction_shape=[16, 8, 16],
            element_a=self.problem_description["element_a"], 
            element_b=self.problem_description["element_b"],
            element_accumulator=self.problem_description["element_accumulator"],
            opcode_class=cutlass.OpClass.TensorOp
        )

        threadblock_shape = [
            config["block_x"], config["block_y"], config["block_z"]
        ]

        warp_count = [
            config["block_x"] // config["warp_x"],
            config["block_y"] // config["warp_y"],
            config["block_z"] // config["warp_z"]
        ]

        stages = config["stage"]

        tile_description = TileDescription(
            threadblock_shape, stages, warp_count, math_inst
        )
        swizzling_functor = getattr(
            cutlass, "IdentitySwizzle%d"%int(pow(2, config["log_swizzle"])))
            

        A = TensorDescription(
            self.problem_description["element_a"], 
            self.problem_description["layout_a"], 
            self.problem_description["alignment_a"])
        B = TensorDescription(
            self.problem_description["element_b"], 
            self.problem_description["layout_b"], 
            self.problem_description["alignment_b"])
        C = TensorDescription(
            self.problem_description["element_c"], 
            self.problem_description["layout_c"], 
            self.problem_description["alignment_c"])

        epilogue_functor = LinearCombination(
            element_output=C.element, epilogue_vector_length=C.alignment,
            element_accumulator=math_inst.element_accumulator,
            element_epilogue=cutlass.float32
        )

        operation = GemmOperationUniversal(
            arch=80, tile_description=tile_description,
            A=A, B=B, C=C, epilogue_functor=epilogue_functor,
            swizzling_functor=swizzling_functor
        )

        pycutlass.compiler.add_module([operation])


        ## Profile
        M, N, K = self.problem_description["problem_size"]

        tensor_A = torch.empty(size=(M * K,), dtype=torch.float16, device="cuda")
        tensor_B = torch.empty(size=(N * K,), dtype=torch.float16, device="cuda")
        tensor_C = torch.empty(size=(M * N,), dtype=torch.float16, device="cuda")

        split_k_slices = config["split_k_slices"]

        if self.problem_description['split_k_mode'] in ["None", "Serial"]:
            arguments = GemmArguments(
                operation=operation, problem_size=cutlass.gemm.GemmCoord(M, N, K),
                A=tensor_A, B=tensor_B, C=tensor_C, D=tensor_C, 
                output_op = operation.epilogue_type(1.0, 0.0),
                gemm_mode=cutlass.gemm.Mode.Gemm, split_k_slices=split_k_slices
            )
        else:
            arguments = GemmArguments(
                operation=operation, problem_size=cutlass.gemm.GemmCoord(M, N, K),
                A=tensor_A, B=tensor_B, C=tensor_C, D=tensor_C,
                output_op = operation.epilogue_type(1.0, 0.0),
                gemm_mode=cutlass.gemm.Mode.GemmSplitKParallel,
                split_k_slices=split_k_slices
            )

        # warmup iterations
        for _ in range(200):
            operation.run(arguments)
        
        timer = GpuTimer()

        # profiling iterations
        timer.start()
        for _ in range(200):
            operation.run(arguments)
        timer.stop_and_wait()

        return timer.duration(200)
    

class BatchedGemmConfigDescriptor(ConfigDescriptor):
    CONFIG = [
        "block_x", "block_y", "block_z", "warp_x", "warp_y", 
        "warp_z", "stage"
    ]
    OP_NAME = "BGEMM"
    def __init__(self, problem_size, element_a, layout_a, element_b, layout_b,
        element_c, layout_c, element_accumulator, alignment_a, alignment_b,
        alignment_c, permute_a, permute_b, autotuning_rounds=15) -> None:
        #
        super().__init__(autotuning_rounds)

        self.problem_description = {
            "element_a": element_a,
            "element_b": element_b,
            "element_c": element_c,
            "element_accumulator": element_accumulator,
            "layout_a": layout_a,
            "layout_b": layout_b,
            "layout_c": layout_c,
            "alignment_a": alignment_a,
            "alignment_b": alignment_b,
            "alignment_c": alignment_c,
            "problem_size": problem_size,
            "permute_a": permute_a,
            "permute_b": permute_b
        }

        self.element_a = element_a
        self.element_b = element_b

        self.heuristic = BatchedGemmHeuristics(
            element_a, element_b, element_c, element_accumulator,
            layout_a, layout_b, alignment_a, alignment_b, alignment_c
        )

    def cache_init(self):
        try:
            connection = sqlite3.connect("./compiled_cache.db")
            cursor = connection.cursor()
            sqlite_create_table_query = """CREATE TABLE bmm_best_config(
                op_key TEXT NOT NULL UNIQUE, block_x INTEGER, block_y INTEGER, 
                block_z INTEGER, warp_x INTEGER, warp_y INTEGER, warp_z INTEGER, 
                stage INTEGER)"""
            cursor.execute(sqlite_create_table_query)
            connection.commit()
            cursor.close()
        except:
            pass
    
    def cache_load(self, key):
        connection = sqlite3.connect("./compiled_cache.db")
        cursor = connection.cursor()
        sqlite_fetch_blob_query = """SELECT * from bmm_best_config where op_key = ?"""

        cursor.execute(sqlite_fetch_blob_query, (key, ))
        record = cursor.fetchall()
        if len(record) == 0:
            return None
        else:
            record = list(record[0])[1:]
            parameters = {
                "block_x": record[0],
                "block_y": record[1],
                "block_z": record[2],
                "warp_x": record[3],
                "warp_y": record[4],
                "warp_z": record[5],
                "stage": record[6]
            } 
            return parameters
    
    def cache_insert(self, best_config):
        key = self.get_key()
        connection = sqlite3.connect("./compiled_cache.db")
        cursor = connection.cursor()
        sqlite_insert_blob_query = """ INSERT OR IGNORE INTO 
            bmm_best_config (op_key, block_x, block_y, block_z, warp_x, 
                warp_y, warp_z, stage) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)"""
        cursor.execute(sqlite_insert_blob_query, tuple([key,] + best_config[0:7]))
        connection.commit()
        cursor.close()
    
    def generate_code_and_profile(self, config):

        math_inst = MathInstruction(
            instruction_shape=[16, 8, 16],
            element_a=self.problem_description["element_a"], 
            element_b=self.problem_description["element_b"],
            element_accumulator=self.problem_description["element_accumulator"],
            opcode_class=cutlass.OpClass.TensorOp
        )

        threadblock_shape = [
            config["block_x"], config["block_y"], config["block_z"]
        ]

        warp_count = [
            config["block_x"] // config["warp_x"],
            config["block_y"] // config["warp_y"],
            config["block_z"] // config["warp_z"]
        ]

        stages = config["stage"]

        tile_description = TileDescription(
            threadblock_shape, stages, warp_count, math_inst
        )
        swizzling_functor = cutlass.BatchedIdentitySwizzle

        A = TensorDescription(
            self.problem_description["element_a"], 
            self.problem_description["layout_a"], 
            self.problem_description["alignment_a"])
        B = TensorDescription(
            self.problem_description["element_b"], 
            self.problem_description["layout_b"], 
            self.problem_description["alignment_b"])
        C = TensorDescription(
            self.problem_description["element_c"], 
            self.problem_description["layout_c"], 
            self.problem_description["alignment_c"])

        epilogue_functor = LinearCombination(
            element_output=C.element, epilogue_vector_length=C.alignment,
            element_accumulator=math_inst.element_accumulator,
            element_epilogue=cutlass.float32
        )

        operation = GemmOperationUniversal(
            arch=80, tile_description=tile_description,
            A=A, B=B, C=C, epilogue_functor=epilogue_functor,
            swizzling_functor=swizzling_functor
        )

        pycutlass.compiler.add_module([operation])


        ## Profile
        B, M, N, K = self.problem_description["problem_size"]

        tensor_A = torch.empty(size=(B * M * K,), dtype=torch.float16, device="cuda")
        tensor_B = torch.empty(size=(B * N * K,), dtype=torch.float16, device="cuda")
        tensor_C = torch.empty(size=(B * M * N,), dtype=torch.float16, device="cuda")

        arguments = BatchedGemmPermutedArguments(
            operation=operation, problem_size=cutlass.gemm.GemmCoord(M, N, K),
            A=tensor_A, B=tensor_B, C=tensor_C, D=tensor_C, 
            output_op = operation.epilogue_type(1.0, 0.0),
            gemm_mode=cutlass.gemm.Mode.Batched, batch=B,
            permute_A=self.problem_description["permute_a"],
            permute_B=self.problem_description["permute_b"]
        )

        # warmup iterations
        for _ in range(200):
            operation.run(arguments)
        
        timer = GpuTimer()

        # profiling iterations
        timer.start()
        for _ in range(200):
            operation.run(arguments)
        timer.stop_and_wait()

        return timer.duration(200)

class ConvConfigDescriptor(ConfigDescriptor):
    CONFIG = [
        "block_x", "block_y", "block_z", "warp_x", "warp_y",
        "warp_z", "stage", "swizzle", "split_k_slices", "iterator_algorithm"
    ]
    OP_NAME = "CONV"
    def __init__(self, problem_size, element_a, element_b, element_c,
        element_accumulator, alignment_a, alignment_b, alignment_c,
        split_k_mode="None", conv_kind = cutlass.conv.Operator.fprop, 
        autotuning_rounds=15) -> None:
        #
        super().__init__(autotuning_rounds)

        self.conv_kind = conv_kind
        self.problem_size = problem_size
        self.autotuning_rounds = autotuning_rounds

        if self.conv_kind == cutlass.conv.Operator.fprop:
            conv_kind_str = "fprop"
        elif self.conv_kind == cutlass.conv.Operator.dgrad:
            conv_kind_str = "dgrad_4"
        elif self.conv_kind == cutlass.conv.Operator.wgrad:
            conv_kind_str = "wgrad_t2"
        
        self.autotuning_rounds = autotuning_rounds
        

        self.problem_size_list = [
            problem_size.N, problem_size.H, problem_size.W, problem_size.C,
            problem_size.P, problem_size.Q, problem_size.K, problem_size.R,
            problem_size.S, problem_size.pad_h, problem_size.pad_w,
            problem_size.stride_h, problem_size.stride_w, problem_size.dilation_h,
            problem_size.dilation_w, problem_size.mode, problem_size.split_k_slices,
            problem_size.groups
        ]

        self.problem_description = {
            "element_a": element_a,
            "element_b": element_b,
            "element_c": element_c,
            "element_accumulator": element_accumulator,
            "alignment_a": alignment_a,
            "alignment_b": alignment_b,
            "alignment_c": alignment_c,
            "split_k_mode": split_k_mode,
            "problem_size": self.problem_size_list,
            "conv_kind": conv_kind_str
        }

        self.heuristic = Conv2dHeuristics(
            element_a, element_b, element_c, element_accumulator,
            alignment_a, alignment_b, alignment_c, problem_size,
            split_k_mode, conv_kind
        )
    
    def cache_init(self):
        try:
            connection = sqlite3.connect("./compiled_cache.db")
            cursor = connection.cursor()
            sqlite_create_table_query = """CREATE TABLE conv2d_best_config(
                op_key TEXT NOT NULL UNIQUE, block_x INTEGER, block_y INTEGER, 
                block_z INTEGER, warp_x INTEGER, warp_y INTEGER, warp_z INTEGER, 
                stage INTEGER, swizzle INTEGER, split_k_slices INTEGER, 
                iterator_algorithm INTEGER)"""
            cursor.execute(sqlite_create_table_query)
            connection.commit()
            cursor.close()
        except:
            pass
    
    def cache_load(self, key):
        connection = sqlite3.connect("./compiled_cache.db")
        cursor = connection.cursor()
        sqlite_fetch_blob_query = """SELECT * from conv2d_best_config where op_key = ?"""

        cursor.execute(sqlite_fetch_blob_query, (key, ))
        record = cursor.fetchall()
        
        if len(record) == 0:
            return None
        else:
            record = list(record[0])[1:]
            parameters = {
                "block_x": record[0],
                "block_y": record[1],
                "block_z": record[2],
                "warp_x": record[3],
                "warp_y": record[4],
                "warp_z": record[5],
                "stage": record[6],
                "log_swizzle": record[7],
                "split_k_slices": record[8],
                "iterator_algorithm": record[9]
            } 
            return parameters
    
    def cache_insert_with_key(self, key, best_config):
        connection = sqlite3.connect("./compiled_cache.db")
        cursor = connection.cursor()
        sqlite_insert_blob_query = """ INSERT OR IGNORE INTO 
            conv2d_best_config (op_key, block_x, block_y, block_z, warp_x, 
                warp_y, warp_z, stage, swizzle, split_k_slices, iterator_algorithm) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
        cursor.execute(sqlite_insert_blob_query, tuple([key,] + best_config))
        connection.commit()
        cursor.close()
    
    def cache_insert(self, best_config):
        key = self.get_key()
        connection = sqlite3.connect("./compiled_cache.db")
        cursor = connection.cursor()
        sqlite_insert_blob_query = """ INSERT OR IGNORE INTO 
            conv2d_best_config (op_key, block_x, block_y, block_z, warp_x, 
                warp_y, warp_z, stage, swizzle, split_k_slices, iterator_algorithm) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
        cursor.execute(sqlite_insert_blob_query, tuple([key,] + best_config))
        connection.commit()
        cursor.close()
    
    def generate_code(self, config):
        math_inst = MathInstruction(
            [16, 8, 16], self.problem_description["element_a"], 
            self.problem_description["element_b"],
            self.problem_description["element_accumulator"],
            cutlass.OpClass.TensorOp
        )

        threadblock_shape = [
            config["block_x"], config["block_y"], config["block_z"]
        ]

        warp_count = [
            config["block_x"] // config["warp_x"],
            config["block_y"] // config["warp_y"],
            config["block_z"] // config["warp_z"]
        ]

        stages = config["stage"]

        tile_description = TileDescription(
            threadblock_shape, stages, warp_count, math_inst
        )

        swizzling_functor = getattr(
            cutlass, "IdentitySwizzle%d"%int(pow(2, config["log_swizzle"])))

        stride_support = StrideSupport.Strided

        # case strided dgrad
        if self.conv_kind == cutlass.conv.Operator.dgrad:
            if self.problem_size.stride_h == 1 and self.problem_size.stride_w == 1:
                stride_support = StrideSupport.Unity
            else:
                swizzling_functor = getattr(
                    cutlass, "StridedDgradIdentitySwizzle%d"%int(pow(2, config["log_swizzle"]))
                )
        
        
        iterator_algorithm = self.heuristic.available_iterator_algorithms[
            config["iterator_algorithm"]
        ]

        A = TensorDescription(
            self.problem_description["element_a"], 
            cutlass.TensorNHWC,
            self.problem_description["alignment_a"])
        B = TensorDescription(
            self.problem_description["element_b"], 
            cutlass.TensorNHWC,
            self.problem_description["alignment_b"])
        C = TensorDescription(
            self.problem_description["element_c"], 
            cutlass.TensorNHWC, 
            self.problem_description["alignment_c"])

        epilogue_functor = LinearCombination(
            element_output=C.element, epilogue_vector_length=C.alignment,
            element_accumulator=math_inst.element_accumulator,
            element_epilogue=cutlass.float32
        )

        operation = Conv2dOperation(
            self.conv_kind, iterator_algorithm,
            80, tile_description, A, B, C, stride_support,
            epilogue_functor, swizzling_functor, visitor=False
        )

        pycutlass.compiler.add_module([operation])

        return operation
    
    def generate_code_and_profile(self, config):
        operation = self.generate_code(config)

        tensor_A_size = cutlass.conv.implicit_gemm_tensor_a_size(
            self.conv_kind, self.problem_size
        )

        tensor_B_size = cutlass.conv.implicit_gemm_tensor_b_size(
            self.conv_kind, self.problem_size
        )

        tensor_C_size = cutlass.conv.implicit_gemm_tensor_c_size(
            self.conv_kind, self.problem_size
        )

        tensor_A = torch.empty(size=(tensor_A_size,), dtype=torch.float16, device="cuda")
        tensor_B = torch.empty(size=(tensor_B_size,), dtype=torch.float16, device="cuda")
        tensor_C = torch.empty(size=(tensor_C_size,), dtype=torch.float16, device="cuda")

        self.problem_size.split_k_slices = config["split_k_slices"]

        arguments = Conv2dArguments(
            operation, self.problem_size,
            A=tensor_A, B=tensor_B, C=tensor_C, D=tensor_C,
            output_op=operation.epilogue_type(1.0, 0.0),
            split_k_mode=cutlass.conv.SplitKMode.Serial,
            split_k_slices=config["split_k_slices"]
        )

        # warmup iterations
        for _ in range(200):
            operation.run(arguments)
        
        timer = GpuTimer()

        # profiling iterations
        timer.start()
        for _ in range(200):
            operation.run(arguments)
        timer.stop_and_wait()

        return timer.duration(200)
