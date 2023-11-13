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
# Autotuner for EVT
import torch
from cutlass.backend.evt.ir.dag_ir import DAGIR
from cutlass.backend.evt.ir import (
    AuxLoadImpl, RowBroadcastImpl, ColumnBroadcastImpl, 
    AuxStoreImpl, RowReductionImpl, ColumnReductionImpl, RowStoreImpl
)
from torch.profiler import profile, ProfilerActivity
import sqlite3
from tqdm import tqdm


class AutoTunerBase:
    """
    We abstract each problem to be tuned as a ticket. Each type of operator (e.g.
    mm, bmm, softmax ...) has its unique derived ticket type to capture its diversed
    behaviors when constructing and launching kernel.
    """
    TUNER_CACHE_FILE = "tuner_cache.db"
    def __init__(self, epilogue_visitor, target) -> None:
        self.epilogue_visitor = epilogue_visitor
        self.target = target
        self.warmup_iterations=100
        self.profile_iterations=100

        self.inputs = [
            input for input in epilogue_visitor.inputs 
            if input.target != self.target]
        self.outputs = epilogue_visitor.outputs
        self.output2store = epilogue_visitor.output2store
        self.num_best_tds = 5

        table_entry = [f"{key} {value}" for key, value in self.config_columns.items()]
        table_entry_str = ", ".join(table_entry)
        connection = sqlite3.connect(self.TUNER_CACHE_FILE)
        cursor = connection.cursor()
        sqlite_create_best_config_table_query = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name}(key TEXT NOT NULL,
                                                  rank INTEGER NOT NULL,
                                                  {table_entry_str},
                                                  PRIMARY KEY (key, rank))
        """
        cursor.execute(sqlite_create_best_config_table_query)
        connection.commit()
        cursor.close()
    
    #
    # Functions to override
    #
    
    def construct_config_with_record(self, *args, **kwargs):
        """
        Convert config to sql record
        """
        raise NotImplementedError

    def construct_record_with_config(self, *args, **kwargs):
        """
        Convert the sql record to config
        """
        raise NotImplementedError
    
    @property
    def key_no_epilogue(self):
        """
        Return the key without epilogue
        """
        raise NotImplementedError
    
    def profile_with_config(self, config, with_epilogue, args, kwargs):
        """
        Profile under the current config and return duration
        """
        raise NotImplementedError
    
    def get_arguments_no_epilogue(self):
        """
        Create the arguments to be run without epilogue
        """
        raise NotImplementedError
    
    def profile_best_config_without_epilogue(self):
        raise NotImplementedError
    
    def profile_best_config_with_epilogue(self, configs):
        raise NotImplementedError
    
    #
    # Cache related helper functions
    #

    @property
    def key_with_epilogue(self):
        """
        Return the key with epilogue
        """
        return f"{self.key_no_epilogue}_{self.epilogue_key}"
    
    @property
    def epilogue_key(self):
        """
        Get the key of the epilogue based on statistics
        """
        dag_ir: DAGIR = self.epilogue_visitor.dag_ir
        self.stat = {
            "aux_load": 0,
            "row_broadcast": 0,
            "column_broadcast": 0,
            "aux_store": 0,
            "row_reduction": 0,
            "column_reduction": 0,
            "row_store": 0
        }
        self.get_epilogue_statistics(dag_ir, self.stat)
        values = [str(value) for value in self.stat.values()]
        return "_".join(values)
    
    def get_epilogue_statistics(self, dag_ir: DAGIR, statistics: dict):
        """
        Compute the epilogue statistics
        """
        for node_meta in dag_ir.nodes_meta:
            impl = node_meta.underlying_impl
            if isinstance(impl, AuxLoadImpl):
                statistics["aux_load"] += 1
            elif isinstance(impl, RowBroadcastImpl):
                statistics["row_broadcast"] += 1
            elif isinstance(impl, ColumnBroadcastImpl):
                statistics["column_broadcast"] += 1
            elif isinstance(impl, AuxStoreImpl):
                statistics["aux_store"] += 1
            elif isinstance(impl, RowReductionImpl):
                statistics["row_reduction"] += 1
            elif isinstance(impl, ColumnReductionImpl):
                statistics["column_reduction"] += 1
            elif isinstance(impl, RowStoreImpl):
                statistics["row_store"] +=1
    
    def fetch_record_with_key(self, key):
        connection = sqlite3.connect(self.TUNER_CACHE_FILE)
        cursor = connection.cursor()
        sqlite_fetch_config_query = f"""SELECT * from {self.table_name} where key = ?"""
        cursor.execute(sqlite_fetch_config_query, (key,))
        record = cursor.fetchall()
        if len(record) == 0:
            return None
        configs = []
        for row in record:
            configs.append(self.construct_config_with_record(row))
        return configs
    
    def insert_record(self, key, rank, config):
        connection = sqlite3.connect(self.TUNER_CACHE_FILE)
        cursor = connection.cursor()
        column_names = "key, rank, " + ", ".join(self.config_columns.keys())
        qms = ",".join(["?"] * len(self.config_columns.keys()))
        sqlite_insert_config = f""" INSERT OR IGNORE INTO {self.table_name} ({column_names}) VALUES (?,?,{qms})"""
        data_tuple = (key, rank) + self.construct_record_with_config(config)
        cursor.execute(sqlite_insert_config, data_tuple)
        connection.commit()
        cursor.close()
    
    #
    # Argument related helper functions
    #

    def get_arguments_with_epilogue(self):
        """
        Create the arguments to be run with epilogue
        """
        args, kwargs = self.get_arguments_no_epilogue()
        kwargs["visitor_args"] = self.get_visitor_args()
        return args, kwargs
    
    def get_output_name(self, output):
        if output.target == torch.ops.aten.clone:
            name = output.args[0].name
        elif output.target == self.target:
            name = "accum"
        else:
            name = output.name
        return name
    
    def get_visitor_args(self):
        visitor_args = {}
        for input in self.inputs:
            visitor_args[input.name] = torch.empty(
                size = input.meta["tensor_meta"].shape,
                dtype = input.meta["tensor_meta"].dtype,
                device = "cuda"
            )
        for output in self.outputs:
            output_name = self.get_output_name(output)
            if output_name not in self.output2store:
                self.output2store[output_name] = output_name
            if output_name in self.output2store and output_name != self.output2store[output_name]:
                continue
            visitor_args[output_name] = torch.empty(
                size=output.meta['tensor_meta'].shape,
                dtype=output.meta['tensor_meta'].dtype,
                device="cuda"
            )
        return visitor_args    
    
    def get_alignment(self, dim):
        if dim % 8 == 0: return 8
        elif dim % 4 == 0: return 4
        elif dim % 2 == 0: return 2
        else: return 1
    
    #
    # Profiling related helper functions
    #

    def profile_top_k_config(self, configs, k, with_epilogue: bool):
        # get the arguments
        if with_epilogue:
            args, kwargs = self.get_arguments_with_epilogue()
        else:
            args, kwargs = self.get_arguments_no_epilogue()
        durations = []
        measured_configs = []
        for config in tqdm(configs):
            duration = self.profile_with_config(config, with_epilogue, *args, **kwargs)
            durations.append(duration)
            measured_configs.append(config)
        # Sort
        combined = list(zip(durations, measured_configs))
        sorted_combined = sorted(combined, key=lambda x: x[0])
        _, sorted_configs = zip(*sorted_combined)
        return sorted_configs[:k]
    
    def torch_profiler(self, fn, *args, **kwargs):
        """
        Profile with torch profiler and return duration
        """
        with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
            fn(*args, **kwargs)
        duration = 0
        for item in prof.key_averages():
            duration += item.self_cuda_time_total
        return duration

    #
    # Autotuner Skeleton
    #

    def get_best_config(self):
        # Check if the same epilogue & problem is profiled
        config = self.fetch_cached_best_config_with_epilogue()
        if config is not None:
            return config
        # Need to profile the best configs without epilogue
        configs = self.get_best_config_no_epilogue()
        return self.profile_best_config_with_epilogue(configs)
    
    def get_best_config_no_epilogue(self):
        configs = self.fetch_cached_best_configs_without_epilogue()
        if configs is not None:
            return configs
        return self.profile_best_config_without_epilogue()
    
    def fetch_cached_best_configs_without_epilogue(self):
        key = self.key_no_epilogue
        return self.fetch_record_with_key(key)
    
    def fetch_cached_best_config_with_epilogue(self):
        key = self.key_with_epilogue
        configs = self.fetch_record_with_key(key)
        if configs is None:
            return None
        else:
            assert len(configs) == 1
            return configs[0]
    