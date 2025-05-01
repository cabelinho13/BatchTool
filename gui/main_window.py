# main_window.py (Atualizado para novo fluxo de transcrição)
import os
import sys
import traceback
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QMessageBox,
    QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import Qt
from gui.layout_setup import setup_ui
from gui.actions import conectar_acoes, atualizar_status, atualizar_progresso
from utils.helpers import logar
from core.worker_manager import resource_manager
from utils.logger import get_logger

logger = get_logger(__name__)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        try:
            self.setWindowTitle("\U0001F3AC Video Batch Tool - GUI")
            self.setWindowIcon(QIcon("assets/icon.ico"))
            self.widgets = {}
            setup_ui(self, self.widgets)

            self.total_links = 0
            self.links_concluidos = 0
            self.cancelado = False
            self.pastas_criadas = []
            self.processamento_ativo = False

            # Configuração inicial dos botões
            self.widgets["btn_cancelar"].setEnabled(False)
            self.widgets["btn_limpar"].setEnabled(True)
            if "btn_reprocessar" in self.widgets:
                del self.widgets["btn_reprocessar"]  # Removendo botão reprocessar

            # Conecta ações dos botões
            self.widgets["btn_processar"].clicked.connect(self.iniciar_processamento)
            self.widgets["btn_cancelar"].clicked.connect(self.confirmar_parada)
            self.widgets["btn_limpar"].clicked.connect(self.limpar_interface)
            self.widgets["btn_abrir_saida"].clicked.connect(self.abrir_pasta_saida)
            
            # Configura a tabela
            self.configurar_tabela()

            # Status
            self.widgets["lbl_status"].setStyleSheet("color: gray;")
            self.statusBar().addWidget(self.widgets["lbl_status"])

        except Exception as e:
            logger.error(f"❌ Erro crítico na inicialização da MainWindow: {e}")
            logger.debug("Traceback completo:\n" + traceback.format_exc())
            self.exibir_popup_erro("A aplicação encontrou um erro crítico e será encerrada.")
            sys.exit(1)

    def exibir_popup_erro(self, mensagem: str):
        """Exibe uma mensagem de erro em um popup"""
        logger.error(f"❌ Erro exibido ao usuário: {mensagem}")
        QMessageBox.critical(self, "Erro", mensagem)
        if self.widgets.get("lbl_status"):
            self.widgets["lbl_status"].setText(f"❌ {mensagem}")
            self.widgets["lbl_status"].setStyleSheet("color: red; font-weight: bold;")

    def configurar_tabela(self):
        """Configura a tabela de links"""
        tabela = self.widgets["tabela_status"]
        tabela.setColumnCount(2)
        tabela.setHorizontalHeaderLabels(["Link", "Status"])
        header = tabela.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)

    def iniciar_processamento(self):
        """Inicia o processamento dos vídeos"""
        canal = self.widgets["input_canal"].text().strip()
        links = self.widgets["text_links"].toPlainText().strip().split('\n')
        links = [link.strip() for link in links if link.strip()]

        if not canal:
            self.exibir_popup_erro("Digite o nome do canal!")
            return
        if not links:
            self.exibir_popup_erro("Cole os links dos vídeos!")
            return

        self.processamento_ativo = True
        self.widgets["btn_processar"].setEnabled(False)
        self.widgets["btn_cancelar"].setEnabled(True)
        self.widgets["btn_limpar"].setEnabled(False)

        # Prepara a tabela
        tabela = self.widgets["tabela_status"]
        tabela.setRowCount(len(links))
        for i, link in enumerate(links):
            tabela.setItem(i, 0, QTableWidgetItem(link))
            tabela.setItem(i, 1, QTableWidgetItem("Aguardando..."))

        self.total_links = len(links)
        self.links_concluidos = 0
        
        # Processa cada link
        for i, link in enumerate(links):
            pasta_saida = os.path.join("output", f"{i+1} {canal}")
            os.makedirs(pasta_saida, exist_ok=True)
            self.pastas_criadas.append(pasta_saida)

            # Submete para processamento em CPU (download)
            worker = self.criar_download_worker(link, pasta_saida, i)
            resource_manager.submit_cpu_task(worker)

    def confirmar_parada(self):
        """Confirma se o usuário quer realmente parar o processamento"""
        if not self.processamento_ativo:
            return

        reply = QMessageBox.question(
            self, 'Confirmar Parada',
            'Deseja realmente parar o processamento?\nOs downloads em andamento serão cancelados.',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.parar_processamento()

    def parar_processamento(self):
        """Para o processamento atual"""
        self.processamento_ativo = False
        self.cancelado = True
        self.widgets["btn_cancelar"].setEnabled(False)
        self.widgets["btn_processar"].setEnabled(True)
        self.widgets["btn_limpar"].setEnabled(True)
        self.exibir_log("🛑 Processamento cancelado pelo usuário")

    def limpar_interface(self):
        """Limpa a interface (apenas quando não há processamento ativo)"""
        if self.processamento_ativo:
            return

        self.widgets["input_canal"].clear()
        self.widgets["text_links"].clear()
        self.widgets["tabela_status"].setRowCount(0)
        self.widgets["console"].clear()
        self.exibir_log("🧹 Interface limpa")

    def exibir_log(self, texto: str):
        """Exibe mensagem no console de log"""
        logar(self.widgets["console"], texto)

    def atualizar_status_linha(self, row: int, status: str):
        """Atualiza o status de uma linha na tabela"""
        if 0 <= row < self.widgets["tabela_status"].rowCount():
            self.widgets["tabela_status"].setItem(row, 1, QTableWidgetItem(status))

    def criar_download_worker(self, link, pasta_saida, row):
        """Cria um worker para download otimizado para CPU"""
        from workers.download_worker import DownloadWorker
        
        worker = DownloadWorker(
            link=link,
            caminho_saida=pasta_saida,
            row=row,
            logar_callback=self.exibir_log
        )

        # Conecta sinais
        worker.signals.status_linha.connect(self.atualizar_status_linha)
        worker.signals.finalizado.connect(lambda result: self.iniciar_transcricao(result))
        worker.signals.erro.connect(self.tratar_erro_download)

        return worker

    def iniciar_transcricao(self, download_result):
        """Inicia a transcrição após download bem-sucedido"""
        if not self.processamento_ativo:
            return

        from workers.transcriber_worker import TranscriberWorker
        
        video_path = download_result.get('video_path')
        row = download_result.get('row', -1)
        
        if not video_path or not os.path.exists(video_path):
            self.tratar_erro_download({"row": row, "erro": "Vídeo não encontrado para transcrição"})
            return

        # Cria worker de transcrição otimizado para GPU
        worker = TranscriberWorker(
            caminho_video=video_path,
            idioma=None,  # Auto-detecção
            device_id=0   # Primeira GPU disponível
        )

        # Conecta sinais
        worker.signals.status.connect(lambda msg: self.atualizar_status_linha(row, msg))
        worker.signals.resultado.connect(lambda result: self.finalizar_transcricao(row, result))
        worker.signals.erro.connect(lambda err: self.tratar_erro_transcricao(row, err))

        # Submete para processamento em GPU
        resource_manager.submit_gpu_task(worker)

    def finalizar_transcricao(self, row: int, resultado: dict):
        """Finaliza o processamento de um item"""
        if not self.processamento_ativo:
            return

        self.links_concluidos += 1
        self.atualizar_status_linha(row, "✅ Concluído")

        if self.links_concluidos >= self.total_links:
            self.processamento_ativo = False
            self.widgets["btn_cancelar"].setEnabled(False)
            self.widgets["btn_processar"].setEnabled(True)
            self.widgets["btn_limpar"].setEnabled(True)
            self.exibir_log("✅ Todos os vídeos foram processados!")

    def tratar_erro_download(self, erro: dict):
        """Trata erro no download"""
        row = erro.get("row", -1)
        msg = erro.get("erro", "Erro desconhecido")
        self.atualizar_status_linha(row, f"❌ Erro: {msg}")
        self.exibir_log(f"❌ Erro no download (linha {row+1}): {msg}")

    def tratar_erro_transcricao(self, row: int, erro: dict):
        """Trata erro na transcrição"""
        msg = erro.get("erro", "Erro desconhecido")
        self.atualizar_status_linha(row, f"❌ Erro na transcrição: {msg}")
        self.exibir_log(f"❌ Erro na transcrição (linha {row+1}): {msg}")

    def abrir_pasta_saida(self):
        """Abre a pasta de saída no explorador de arquivos"""
        try:
            pasta_output = "output"
            if not os.path.exists(pasta_output):
                os.makedirs(pasta_output)
                
            if os.path.exists(pasta_output):
                os.startfile(pasta_output)
            else:
                self.exibir_popup_erro("Pasta output não encontrada.")
        except Exception as e:
            logger.error(f"❌ Erro ao abrir pasta de saída: {e}")
            self.exibir_popup_erro(f"Erro ao abrir pasta: {e}")

    def closeEvent(self, event):
        """Manipula o evento de fechamento da janela"""
        if self.processamento_ativo:
            reply = QMessageBox.question(
                self, 'Confirmar Saída',
                'Há processamento em andamento. Deseja realmente sair?',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.parar_processamento()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()