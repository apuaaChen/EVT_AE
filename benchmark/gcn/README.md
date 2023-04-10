### DGL GCN implementation

Note: we modified the pytorch source code to make GCN traceable. In 
`/opt/conda/lib/python3.8/site-packages/torch/fx/experimental/proxy_tensor.py`
line 108 - 122, originally, it is
```python
r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
    cls,
    elem.shape, dtype=elem.dtype, layout=elem.layout, device=elem.device,
    requires_grad=requires_grad if requires_grad is not None else False, strides=elem.stride(),
    storage_offset=elem.storage_offset()
)
```
We update it to 
```python
try:
    r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
        cls,
        elem.shape, dtype=elem.dtype, layout=elem.layout, device=elem.device,
        requires_grad=requires_grad if requires_grad is not None else False, strides=elem.stride(),
        storage_offset=elem.storage_offset()
    )
except:
    r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
        cls,
        elem.shape, dtype=elem.dtype, layout=elem.layout, device=elem.device,
        requires_grad=requires_grad if requires_grad is not None else False, strides=(0, 0),
        storage_offset=elem.storage_offset()
    )
```

Also, in `/opt/conda/lib/python3.8/site-packages/torch/fx/passes/shape_prop.py` line 34, it is
```python
stride = result.stride()
```
We update it to
```python
try:
    stride = result.stride()
except:
    stride = (0, 0)
```
line 48
```python
if result.is_contiguous(memory_format=query_format):
    memory_format = query_format
    break
```
to
```python
try:
    if result.is_contiguous(memory_format=query_format):
        memory_format = query_format
        break
except:
    memory_format=torch.contiguous_format
```
otherwise the `elem.stride()` would raise `RuntimeError: Tensors of type SparseTensorImpl do not have strides`
