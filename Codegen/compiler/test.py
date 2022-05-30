from functorch.compile import aot_function
import torch
import time

def benchmark(f):
  A_ = torch.randn(size=(32, 32), dtype=torch.float)
  B_ = torch.randn(size=(32, 32), dtype=torch.float)
  # Warmup
  for _ in range(10):
    _ = f(A_,B_)
  t = time.time()
  for _ in range(100):
    _ = f(A_,B_)
  return time.time() - t

# A = torch.randn(1024, requires_grad=True)
# B = torch.randn(1024, requires_grad=True)
A = torch.randn(size=(32, 32), dtype=torch.float)
B = torch.randn(size=(32, 32), dtype=torch.float)

@torch.jit.script
def foo_jit(a, b):
  c = torch.matmul(a, b)
  a = torch.matmul(c, b)
  return a

print("-- Default IR --\n", foo_jit.graph_for(A,B))
C_jit = foo_jit(A,B)
print(torch.jit.last_executed_optimized_graph())
print("Default version took {:.2f}ms".format(1000 * benchmark(foo_jit)))

import pointwise_compiler
print()

@torch.jit.script
def foo_compiled(a, b):
  c = torch.matmul(a, b)
  a = torch.matmul(c, b)
  return a

print("-- Transformed IR --\n", foo_compiled.graph_for(A,B))

C_compiled = foo_compiled(A,B)

print(torch.jit.last_executed_optimized_graph())
print("Compiled version took {:.2f}ms".format(1000 * benchmark(foo_compiled)))

print(C_jit)
print(C_compiled)

assert torch.allclose(C_jit, C_compiled)

# def foo_jit(a, b):
#   c = torch.matmul(a, b)
#   a = torch.matmul(c, b)
#   return a

# A = torch.randn(size=(32, 32), dtype=torch.float, requires_grad=True)
# B = torch.randn(size=(32, 32), dtype=torch.float, requires_grad=True)

# ref = foo_jit(A, B)

# def compiler_fn(fx_module: torch.fx.GraphModule, _):
#     print(fx_module.code)
#     return fx_module

# aot_print_fn = aot_function(foo_jit, fw_compiler=compiler_fn, bw_compiler=compiler_fn)

# res = aot_print_fn(A, B)
# assert torch.allclose(ref, res)

# from functorch.compile import clear_compile_cache
# clear_compile_cache()