# CUDA 環境診斷工具
import sys
import os

def check_cuda_environment():
    print("🔍 CUDA 環境診斷")
    print("=" * 50)
    
    # 1. 檢查 CuPy 安裝
    print("\n1. 檢查 CuPy 安裝:")
    try:
        import cupy
        print(f"   ✅ CuPy 版本: {cupy.__version__}")
    except ImportError as e:
        print(f"   ❌ CuPy 未安裝: {e}")
        return
    
    # 2. 檢查 CUDA 版本
    print("\n2. 檢查 CUDA 版本:")
    try:
        cuda_version = cupy.cuda.runtime.runtimeGetVersion()
        print(f"   ✅ CUDA Runtime 版本: {cuda_version}")
    except Exception as e:
        print(f"   ❌ 無法獲取 CUDA 版本: {e}")
    
    # 3. 檢查 GPU 設備
    print("\n3. 檢查 GPU 設備:")
    try:
        device_count = cupy.cuda.runtime.getDeviceCount()
        print(f"   GPU 數量: {device_count}")
        for i in range(device_count):
            prop = cupy.cuda.runtime.getDeviceProperties(i)
            print(f"   GPU {i}: {prop['name'].decode()}")
    except Exception as e:
        print(f"   ❌ 無法獲取 GPU 信息: {e}")
    
    # 4. 測試基本 CuPy 操作
    print("\n4. 測試基本 CuPy 操作:")
    try:
        a = cupy.array([1, 2, 3])
        b = cupy.sum(a)
        result = b.get()
        print(f"   ✅ 基本操作成功: sum([1,2,3]) = {result}")
    except Exception as e:
        print(f"   ❌ 基本操作失敗: {e}")
        print(f"   詳細錯誤: {type(e).__name__}: {str(e)}")
        
        # 如果是 NVRTC 錯誤，提供詳細診斷
        if "nvrtc" in str(e).lower():
            print("\n   🔧 NVRTC 錯誤診斷:")
            print("   這是 CUDA 編譯器錯誤，可能原因:")
            print("   - CUDA Toolkit 未正確安裝")
            print("   - 系統 PATH 中缺少 CUDA 路徑")
            print("   - CuPy 版本與 CUDA 版本不匹配")
            
            # 檢查 PATH
            print("\n   檢查 CUDA 路徑:")
            path = os.environ.get('PATH', '')
            cuda_paths = [p for p in path.split(';') if 'cuda' in p.lower()]
            if cuda_paths:
                print("   找到的 CUDA 路徑:")
                for p in cuda_paths:
                    print(f"     {p}")
            else:
                print("   ❌ PATH 中未找到 CUDA 路徑")
    
    # 5. 檢查環境變數
    print("\n5. 檢查環境變數:")
    cuda_vars = ['CUDA_PATH', 'CUDA_HOME', 'CUDA_ROOT']
    for var in cuda_vars:
        value = os.environ.get(var)
        if value:
            print(f"   {var}: {value}")
        else:
            print(f"   {var}: 未設置")
    
    # 6. 建議解決方案
    print("\n6. 建議解決方案:")
    print("   如果 CuPy 基本操作失敗:")
    print("   1. 確認 NVIDIA GPU 驅動已安裝")
    print("   2. 下載並安裝 CUDA Toolkit 11.x")
    print("   3. 重新安裝匹配的 CuPy 版本:")
    print("      pip uninstall cupy-cuda11x")
    print("      pip install cupy-cuda11x")
    print("   4. 或者嘗試不同的 CUDA 版本:")
    print("      pip install cupy-cuda12x")

if __name__ == "__main__":
    check_cuda_environment()