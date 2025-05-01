import sys
import time
import traceback
from PyQt6.QtWidgets import QApplication
from gui.main_window import MainWindow
from utils.logger import configurar_logger, limpar_logs_antigos, get_logger
from app.splash_screen import SplashScreen
from core.worker_manager import resource_manager

logger = get_logger(__name__)

class AppController:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.main_window = None
        self.splash = SplashScreen()

        # 🛡️ Captura de exceções globais para debug
        sys.excepthook = self._tratar_excecao

    def _tratar_excecao(self, tipo, valor, tb):
        mensagem = "".join(traceback.format_exception(tipo, valor, tb))
        logger.critical("🔥 Exceção global não capturada:\n" + mensagem)
        print("🔥 Exceção global não capturada:\n" + mensagem)

    def start(self):
        try:
            self._exibir_splash()
            self._inicializar_logs()
            self._inicializar_recursos()
            self._carregar_main_window()
            self._iniciar_loop_principal()
        except Exception:
            logger.exception("💥 Erro fatal ao iniciar a aplicação.")
            self._tratar_excecao(*sys.exc_info())
            sys.exit(1)

    def _exibir_splash(self):
        try:
            self.splash.show()
            self.splash.update_text("📁 Iniciando...")
            time.sleep(0.3)
        except Exception:
            logger.warning("⚠️ Splash screen não pôde ser exibida.")

    def _inicializar_logs(self):
        self.splash.update_text("📦 Configurando logs...")
        configurar_logger()
        limpar_logs_antigos()
        time.sleep(0.4)

    def _inicializar_recursos(self):
        """Inicializa o gerenciador de recursos (CPU/GPU)"""
        self.splash.update_text("🎮 Configurando recursos...")
        # O ResourceManager é um singleton, então apenas acessá-lo já o inicializa
        logger.info("✅ Gerenciador de recursos inicializado")
        time.sleep(0.4)

    def _carregar_main_window(self):
        self.splash.update_text("🧠 Carregando interface...")
        self.main_window = MainWindow()
        time.sleep(0.4)

    def _iniciar_loop_principal(self):
        self.splash.update_text("🚀 Pronto!")
        self.splash.finish(self.main_window)
        self.main_window.show()
        logger.info("🎨 Aplicação iniciada com sucesso.")
        sys.exit(self.app.exec())
