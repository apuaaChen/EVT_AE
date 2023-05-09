### DGL GCN implementation

Note: we modified the pytorch source code to make GCN traceable. 

In `/usr/local/lib/python3.8/dist-packages/torch/_subclasses/meta_utils.py` line 468 & 469 are commented, otherwise it raises FakeTensorError.
```python
if any(
    [
        # t.is_sparse_csr,  # line 468
        # t.layout in [torch.sparse_csc, torch.sparse_bsr, torch.sparse_bsc] # line 469
```
Also, in line 253, we add additional logic to avoid it from accessing the underlying storage
```python
# new logic
elif t.is_sparse_csr:
    is_leaf = safe_is_leaf(t)
    sizes, strides, storage_offset = sym_sizes_strides_storage_offset(t)
    r = callback(
        lambda: torch.empty_strided(
            sizes, strides, dtype=t.dtype, device="meta"
        )
    )
    assert safe_is_leaf(r), "the callback you passed in dosen't detach"
# end new logic
elif t.is_mkldnn:
    is_leaf = safe_is_leaf(t)
    sizes, strides, _storage_offset = sym_sizes_strides_storage_offset(
        t
    )
    r = callback(
        lambda: torch.empty_strided(
            sizes, strides, dtype=t.dtype, device="meta"
        )
    )
    assert safe_is_leaf(r), "the callback you passed in doesn't detach"
    if t.requires_grad:
        r.requires_grad = True
    if t.requires_grad and not is_leaf:
        with torch.enable_grad():
            r = r.clone()
```
Besides, at line 213, we add stride (0, 0) as sparse tensors have no stride
```python
def sym_sizes_strides_storage_offset(t):
    if make_symbolic:
        return shape_env.create_symbolic_sizes_strides_storage_offset(t, source)
    return (t.size(), t.stride(), t.storage_offset())
```
is changed to 
```python
def sym_sizes_strides_storage_offset(t):
    if make_symbolic:
        return shape_env.create_symbolic_sizes_strides_storage_offset(t, source)
    try:
        return (t.size(), t.stride(), t.storage_offset())
    except:
        return (t.size(), (0, 0), t.storage_offset())
```
Last, at line 112, we update 
```python
if t.is_sparse or t.is_mkldnn:
    weak_st = None
```
to
```python
if t.is_sparse or t.is_mkldnn or t.is_sparse_csr:
    weak_st = None
```

