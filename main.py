from app.app_controller import AppController
import sys
import traceback
import multiprocessing  # ✅ necessário no Windows com subprocessos

if __name__ == '__main__':
    multiprocessing.freeze_support()  # ✅ necessário para multiprocessing em PyInstaller/Windows
    try:
        AppController().start()
    except Exception as e:
        print("❌ Erro crítico na execução do aplicativo:")
        print(str(e))
        print("Traceback:")
        traceback.print_exc()
        sys.exit(1)
