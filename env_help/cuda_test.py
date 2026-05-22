# ========== Диагностика PyTorch GPU ==========
import torch
import subprocess
import sys

print("🔍 Диагностика GPU окружения PyTorch")
print("="*50)

# 1. Проверка NVIDIA драйвера
print("\n[1] Проверка NVIDIA драйвера...")
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, shell=True)
    if result.returncode == 0:
        for line in result.stdout.split('\n'):
            if 'Driver Version' in line:
                print(f"  ✅ Драйвер установлен: {line.strip()}")
                break
        print("  ✅ nvidia-smi работает")
    else:
        print("  ❌ nvidia-smi не работает. Возможно, драйвер не установлен.")
except FileNotFoundError:
    print("  ❌ nvidia-smi не найден. Установите драйвер NVIDIA.")

# 2. Проверка PyTorch
print("\n[2] Проверка PyTorch...")
print(f"  Версия PyTorch: {torch.__version__}")
print(f"  CUDA доступна: {torch.cuda.is_available()}")
print(f"  Путь установки: {torch.__file__}")

# Ключевая проверка: список поддерживаемых CUDA архитектур
cuda_arch_list = torch.cuda.get_arch_list()
print(f"  Поддерживаемые CUDA архитектуры: {cuda_arch_list}")
if cuda_arch_list:
    print(f"  ✅ Установлена CUDA-версия PyTorch")
else:
    print(f"  ❌ Установлена CPU-версия PyTorch (без поддержки CUDA)")

# 3. Если версия с CUDA, но не работает - получим детали ошибки
if not torch.cuda.is_available() and 'cu' in torch.__version__:
    print("\n[3] Детали ошибки CUDA...")
    try:
        torch.cuda.current_device()
    except Exception as e:
        print(f"  Ошибка CUDA: {e}")

print("\n" + "="*50)
print("Диагностика завершена. Смотрите рекомендации ниже.")