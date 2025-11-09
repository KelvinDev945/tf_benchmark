#!/usr/bin/env python3
"""
ONNX 转换问题诊断脚本

此脚本帮助诊断 numpy/tf2onnx 兼容性问题
"""

import sys
from packaging import version


def check_numpy_compatibility():
    """检查 NumPy 版本和 API 可用性"""
    print("="*70)
    print("NumPy 兼容性检查")
    print("="*70)

    try:
        import numpy as np
        numpy_version = np.__version__
        print(f"✓ NumPy 已安装: {numpy_version}")

        # 检查版本
        v = version.parse(numpy_version)

        if v < version.parse("1.20.0"):
            print(f"  状态: ✅ 旧版本，np.bool 可用")
            status = "old"
        elif v < version.parse("1.24.0"):
            print(f"  状态: ⚠️  过渡版本，np.bool 已废弃但可用")
            status = "deprecated"
        else:
            print(f"  状态: ❌ 新版本，np.bool 已移除")
            status = "removed"

        # 测试 np.bool
        print("\n测试 np.bool API:")
        try:
            test = np.bool
            print(f"  ✅ np.bool 可访问: {test}")
            return True, status
        except AttributeError as e:
            print(f"  ❌ np.bool 不可用: {e}")
            print(f"  💡 替代方案: 使用 'bool' 或 'np.bool_'")
            return False, status

    except ImportError:
        print("✗ NumPy 未安装")
        return False, "not_installed"


def check_tf2onnx_compatibility():
    """检查 tf2onnx 版本和兼容性"""
    print("\n" + "="*70)
    print("tf2onnx 兼容性检查")
    print("="*70)

    try:
        import tf2onnx
        tf2onnx_version = tf2onnx.__version__
        print(f"✓ tf2onnx 已安装: {tf2onnx_version}")

        # 尝试导入问题模块
        print("\n测试 tf2onnx.utils 模块:")
        try:
            from tf2onnx.utils import ONNX_TO_NUMPY
            print(f"  ✅ ONNX_TO_NUMPY 导入成功")
            print(f"  ✅ tf2onnx 兼容当前 NumPy 版本")
            return True, tf2onnx_version
        except AttributeError as e:
            print(f"  ❌ ONNX_TO_NUMPY 导入失败")
            print(f"  错误: {e}")
            print(f"\n  💡 解决方案:")
            print(f"     1. 升级 tf2onnx: pip install tf2onnx --upgrade")
            print(f"     2. 或降级 NumPy: pip install numpy<1.24")
            return False, tf2onnx_version

    except ImportError:
        print("✗ tf2onnx 未安装")
        print("  安装: pip install tf2onnx")
        return False, "not_installed"


def check_tensorflow_compatibility():
    """检查 TensorFlow 版本"""
    print("\n" + "="*70)
    print("TensorFlow 兼容性检查")
    print("="*70)

    try:
        import tensorflow as tf
        tf_version = tf.__version__
        print(f"✓ TensorFlow 已安装: {tf_version}")

        v = version.parse(tf_version)

        # TensorFlow 2.20+ 需要 NumPy 1.26+
        if v >= version.parse("2.20.0"):
            print(f"  ⚠️  TensorFlow 2.20+ 需要 NumPy >= 1.26.0")
            import numpy as np
            if version.parse(np.__version__) >= version.parse("1.26.0"):
                print(f"  ✅ NumPy 版本满足要求")
            else:
                print(f"  ❌ NumPy 版本过低，可能导致问题")

        return True, tf_version

    except ImportError:
        print("✗ TensorFlow 未安装")
        return False, "not_installed"


def check_onnxruntime():
    """检查 ONNX Runtime"""
    print("\n" + "="*70)
    print("ONNX Runtime 检查")
    print("="*70)

    try:
        import onnxruntime as ort
        ort_version = ort.__version__
        print(f"✓ ONNX Runtime 已安装: {ort_version}")

        # 检查可用的执行提供程序
        providers = ort.get_available_providers()
        print(f"  可用的执行提供程序:")
        for p in providers:
            print(f"    - {p}")

        return True, ort_version

    except ImportError:
        print("✗ ONNX Runtime 未安装")
        print("  安装: pip install onnxruntime")
        return False, "not_installed"


def generate_compatibility_matrix():
    """生成兼容性矩阵"""
    print("\n" + "="*70)
    print("兼容性矩阵")
    print("="*70)

    print("""
┌─────────────────┬──────────────┬─────────────┬──────────────────┐
│ NumPy 版本      │ np.bool 状态 │ tf2onnx     │ TensorFlow 2.20  │
├─────────────────┼──────────────┼─────────────┼──────────────────┤
│ < 1.20          │ ✅ 可用       │ ✅ 兼容     │ ❌ 不兼容        │
│ 1.20 - 1.23     │ ⚠️  废弃      │ ⚠️  警告    │ ❌ 不兼容        │
│ 1.24 - 1.25     │ ❌ 已移除     │ ❌ 不兼容   │ ⚠️  部分兼容     │
│ 1.26+           │ ❌ 已移除     │ ✅ 1.17.0+  │ ✅ 需要此版本    │
└─────────────────┴──────────────┴─────────────┴──────────────────┘

说明:
  ✅ - 完全兼容
  ⚠️  - 有警告但可用
  ❌ - 不兼容
""")


def provide_recommendations(numpy_ok, tf2onnx_ok, tf_ok, ort_ok):
    """提供修复建议"""
    print("\n" + "="*70)
    print("修复建议")
    print("="*70)

    if numpy_ok and tf2onnx_ok and tf_ok and ort_ok:
        print("\n✅ 所有组件兼容，可以正常使用 ONNX 转换！")
        return

    print("\n检测到兼容性问题，以下是修复建议：\n")

    if not numpy_ok:
        print("【方案 1】升级 tf2onnx (推荐)")
        print("  pip install tf2onnx --upgrade")
        print("  # 确保 tf2onnx >= 1.17.0\n")

        print("【方案 2】使用 HuggingFace Optimum (推荐用于生产)")
        print("  pip install optimum[onnxruntime]")
        print("  # 完全绕过 tf2onnx，使用更现代的工具\n")

        print("【方案 3】降级 NumPy (不推荐)")
        print("  pip install 'numpy<1.24'")
        print("  # 可能与 TensorFlow 2.20 不兼容\n")

    if not tf2onnx_ok:
        print("【tf2onnx 问题】")
        print("  当前版本不兼容，需要升级")
        print("  pip install tf2onnx --upgrade\n")

    if not ort_ok:
        print("【ONNX Runtime 未安装】")
        print("  pip install onnxruntime\n")


def main():
    """主函数"""
    print("\n" + "="*70)
    print("ONNX 转换问题诊断工具")
    print("="*70)
    print()

    # 运行检查
    numpy_ok, numpy_status = check_numpy_compatibility()
    tf2onnx_ok, tf2onnx_ver = check_tf2onnx_compatibility()
    tf_ok, tf_ver = check_tensorflow_compatibility()
    ort_ok, ort_ver = check_onnxruntime()

    # 显示兼容性矩阵
    generate_compatibility_matrix()

    # 提供建议
    provide_recommendations(numpy_ok, tf2onnx_ok, tf_ok, ort_ok)

    # 总结
    print("\n" + "="*70)
    print("诊断总结")
    print("="*70)

    print(f"""
组件状态:
  - NumPy:        {'✅' if numpy_ok else '❌'} ({numpy_status})
  - tf2onnx:      {'✅' if tf2onnx_ok else '❌'} ({tf2onnx_ver})
  - TensorFlow:   {'✅' if tf_ok else '❌'} ({tf_ver})
  - ONNX Runtime: {'✅' if ort_ok else '❌'} ({ort_ver})

ONNX 转换: {'✅ 可用' if (numpy_ok and tf2onnx_ok) else '❌ 不可用'}
""")

    # 返回状态码
    return 0 if (numpy_ok and tf2onnx_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
