#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试所有SS3D变体是否能正常导入和实例化
"""

import torch
import sys

def test_variant(method_name):
    """测试单个变体"""
    try:
        from architecture import model_generator
        print(f"Testing {method_name}...", end=" ")
        model = model_generator(method_name)
        # 测试前向传播
        x = torch.randn(1, 3, 64, 64).cuda()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 31, 64, 64), f"Output shape mismatch: {out.shape}"
        print("✓ OK")
        return True
    except Exception as e:
        print(f"✗ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """测试所有变体"""
    variants = [
        'mst_ss3d_plus_plus',      # 原始版本
        'mst_ss3d_6direction',     # 6方向版本
        'mst_ss3d_shared',         # 共享参数版本
        'mst_ss3d_grouped',        # 分组扫描版本
        'mst_ss3d_adaptive',       # 自适应权重版本
        'mst_ss3d_progressive',    # 渐进式扫描版本
        'mst_ss3d_selective',      # 选择性方向版本
        'mst_ss3d_hybrid',         # 集大成版本
    ]
    
    print("=" * 60)
    print("Testing all SS3D variants...")
    print("=" * 60)
    
    results = {}
    for variant in variants:
        results[variant] = test_variant(variant)
    
    print("=" * 60)
    print("Summary:")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for variant, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {variant}")
    
    print("=" * 60)
    print(f"Total: {passed}/{total} variants passed")
    print("=" * 60)
    
    if passed == total:
        print("All variants are working correctly! ✓")
        return 0
    else:
        print("Some variants failed. Please check the errors above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())








