# PyTorch Version Compatibility

## Current Requirement

**PyTorch 2.4.x** is required for bert-tch and doclayout-yolo examples.

## Issue

tch-rs 0.17 is not compatible with PyTorch 2.8.0+ due to API changes in the `gelu` activation function.

### Error with PyTorch 2.8.0:
```
Undefined symbols for architecture arm64:
  "at::_ops::gelu::call(at::Tensor const&, c10::basic_string_view<char>)", referenced from:
      _atg_gelu in libtorch_sys-8bd317dfdb56a800.rlib
ld: symbol(s) not found for architecture arm64
```

## Solution

Install PyTorch 2.4.x:

```bash
pip install 'torch<2.5' --force-reinstall
```

## Future

When tch-rs releases a version compatible with PyTorch 2.8+, we can update to:
- tch 0.18+ or newer
- PyTorch 2.8+

## References

- tch-rs: https://github.com/LaurentMazare/tch-rs
- PyTorch C++ API changes: https://pytorch.org/cppdocs/
