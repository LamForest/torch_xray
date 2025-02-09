# torch_xray
算子dump &amp; 单测生成工具Demo

## 1. dump 规模
很简单，一行命令
```bash
export X_DEBUG=0x2    #dump输入的规模/dtype
export X_DEBUG=0x12   #dump输入的规模/dtype、checksum
export X_DEBUG=0x112  #dump输入的规模/dtype、checksum、数据
```
以`export X_DEBUG=0x2`为例：
```python
In [1]: import torch

In [2]: a = torch.randn(10,device="cuda")
[P:109388] 2025-02-10 00:47:00,584-xray-INFO-(python_dispatch.py:303) {"00000-aten.randn.default": {"arguments": {"size": [10], "device": "cuda", "pin_memory": false}}} 

In [3]: b = a + 3
[P:109388] 2025-02-10 00:47:08,114-xray-INFO-(python_dispatch.py:303) {"00001-aten.add.Tensor": {"arguments": {"self": {"dtype": "torch.float32", "shape": [10], "device": "cuda:0", "ptr": "0x4003000000"}, "other": 3}}}
```
其中`{"self": {"dtype": "torch.float32", "shape": [10], "device": "cuda:0"}`就代表了add第一个参数`self`的类型和规模，而`"other": 3`则代表了add第二个参数`other`的形状和规模，由于other是一个Scalar，所以直接记录other的值为3。

## 2. 生成单测
将上述stdout的json保存至文件a.json，
```json
{"00001-aten.add.Tensor": {"arguments": {"self": {"dtype": "torch.float32", "shape": [10], "device": "cuda:0", "ptr": "0x4003000000"}, "other": 3}}}
```
并执行`jprof a.json`，则会在`pytests/`下产生单测`xtest_00001_aten_add_Tensor.py`，单测代码截取如下：
```

```py
    op = torch.ops.aten.add.Tensor

    torch.manual_seed(seed)

    cuda_args = [
        torch.empty([10], dtype=torch.float32, device="cuda:0").uniform_(
            -0.1, 0.1
        ),  # self
        3,  # other
    ]
    cuda_kwargs = {}
    cuda_result = op(*cuda_args, **cuda_kwargs)

    # generate args
    if not cuda_only:
        cpu_args = tree_map(
            lambda x: to_cpu_device(x, is_clone=False, do_device=True), cuda_args
        )
        cpu_kwargs = tree_map(
            lambda x: to_cpu_device(x, is_clone=False, do_device=True), cuda_kwargs
        )
        cpu_result = op(*cpu_args, **cpu_kwargs)

        compare(cpu_result, cuda_result, atol=atol, rtol=rtol)
```

可用该单测进行问题复现、性能分析、精度对比。
