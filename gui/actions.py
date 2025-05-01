# actions.py (atualizado para remover dependência direta do transcription_manager)
import os
import shutil
from functools import partial
from PyQt6.QtWidgets import QMessageBox, QProgressBar, QTableWidgetItem
from PyQt6.QtCore import Qt, QTimer
from utils.helpers import limpar_nome, logar, abrir_pasta

output_folder = os.path.join(os.getcwd(), "output")

def criar_barra_progresso(texto="🕒 Agendado") -> QProgressBar:
    barra = QProgressBar()
    barra.setRange(0, 100)
    barra.setValue(0)
    barra.setFormat(texto)
    barra.setAlignment(Qt.AlignmentFlag.AlignCenter)
    barra.setStyleSheet("background-color: #f0f0f0;")
    return barra

def conectar_acoes(main):
    w = main.widgets
    w["btn_processar"].clicked.connect(lambda: iniciar_processamento(main))
    w["btn_cancelar"].clicked.connect(lambda: cancelar_processamento(main))
    w["btn_limpar"].clicked.connect(lambda: limpar_formulario(main))
    w["btn_abrir_saida"].clicked.connect(lambda: abrir_pasta(output_folder))
    if "btn_reprocessar" in w:
        w["btn_reprocessar"].clicked.connect(lambda: reprocessar_linhas_com_erro(main))

def iniciar_processamento(main):
    w = main.widgets
    links = [l.strip() for l in w["text_links"].toPlainText().splitlines() if l.strip()]
    nome_canal = w["input_canal"].text().strip()

    if not links:
        QMessageBox.warning(main, "Aviso", "⚠️ Nenhum link fornecido.")
        return
    if not nome_canal:
        QMessageBox.warning(main, "Aviso", "⚠️ Informe o nome do canal.")
        return

    nome_canal_limpo = limpar_nome(nome_canal)
    main.resetar_contadores()
    main.cancelado = False
    main.pastas_criadas.clear()
    main.total_links = len(links)

    main.bloquear_botoes_processamento()
    w["console"].clear()
    w["tabela_status"].setRowCount(main.total_links)

    logar(w["console"], f"📋 Total de links: {main.total_links}")
    logar(w["console"], f"📂 Pasta de saída: {output_folder}")

    for i, link in enumerate(links):
        try:
            nome_pasta = f"{i+1} {nome_canal_limpo}"
            pasta_saida = os.path.join(output_folder, nome_pasta)
            os.makedirs(pasta_saida, exist_ok=True)
            main.pastas_criadas.append(pasta_saida)

            w["tabela_status"].setItem(i, 0, QTableWidgetItem(str(i + 1)))
            w["tabela_status"].setItem(i, 1, QTableWidgetItem(link))
            w["tabela_status"].setCellWidget(i, 2, criar_barra_progresso())

            log_cb = partial(atualizar_status, main, i)
            prog_cb = partial(atualizar_progresso, main, i)

            main.worker_manager.adicionar_tarefa(
                link=link,
                pasta=pasta_saida,
                index=i,
                logar_callback=log_cb,
                progresso_callback=prog_cb
            )

            logar(w["console"], f"🟡 Agendado: {link}")

        except Exception as e:
            logar(w["console"], f"❌ Erro ao agendar: {link}\n{str(e)}")

    logar(w["console"], f"🟢 Iniciado o processamento de {main.total_links} vídeo(s).")

def atualizar_status(main, row: int, mensagem: str):
    def _executar():
        w = main.widgets
        if 0 <= row < w["tabela_status"].rowCount():
            barra = w["tabela_status"].cellWidget(row, 2)
            if isinstance(barra, QProgressBar):
                barra.setFormat(mensagem)
                barra.setAlignment(Qt.AlignmentFlag.AlignCenter)
                if "✅" in mensagem:
                    barra.setStyleSheet("color: lightgreen;")
                    barra.setValue(100)
                elif "❌" in mensagem:
                    barra.setStyleSheet("color: red;")
                    barra.setValue(100)
                elif "🛑" in mensagem or "🚫" in mensagem:
                    barra.setStyleSheet("color: orange;")
                    barra.setValue(100)
                else:
                    barra.setStyleSheet("background-color: #f0f0f0;")

    QTimer.singleShot(0, _executar)

def atualizar_progresso(main, row: int, pct: int):
    def _executar():
        w = main.widgets
        if 0 <= row < w["tabela_status"].rowCount():
            barra = w["tabela_status"].cellWidget(row, 2)
            if isinstance(barra, QProgressBar):
                barra.setValue(pct)
    QTimer.singleShot(0, _executar)

def cancelar_processamento(main):
    w = main.widgets
    tem_downloads = main.worker_manager.tem_tarefas_ativas()

    if not tem_downloads:
        logar(w["console"], "ℹ️ Nenhuma tarefa em andamento.")
        return

    resposta = QMessageBox.question(
        main,
        "Cancelar processamento",
        "Tem certeza que deseja parar todos os vídeos em andamento?\nAs pastas temporárias serão excluídas.",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
    )

    if resposta != QMessageBox.StandardButton.Yes:
        return

    logar(w["console"], "🛑 Cancelamento solicitado pelo usuário.")
    main.worker_manager.cancelar_todas()

    for pasta in main.pastas_criadas:
        try:
            if os.path.exists(pasta):
                shutil.rmtree(pasta)
        except Exception as e:
            logar(w["console"], f"⚠️ Erro ao remover pasta {pasta}: {e}")

    for row in range(w["tabela_status"].rowCount()):
        barra = w["tabela_status"].cellWidget(row, 2)
        if isinstance(barra, QProgressBar) and barra.value() < 100:
            barra.setFormat("🚫 Cancelado")
            barra.setValue(100)

    main.links_concluidos = main.total_links
    main._habilitar_botoes_pos_execucao()
    logar(w["console"], "🧼 Processamento cancelado. Pastas limpas e status atualizados.")

def limpar_formulario(main):
    w = main.widgets
    if not main.worker_manager.tem_tarefas_ativas():
        w["text_links"].clear()
        w["input_canal"].clear()
        w["tabela_status"].setRowCount(0)
        w["console"].clear()
        main.pastas_criadas.clear()
        main.resetar_contadores()
        logar(w["console"], "🧹 Lista de links limpa.")
    else:
        logar(w["console"], "⚠️ Não é possível limpar enquanto vídeos estão em processamento.")

def reprocessar_linhas_com_erro(main):
    w = main.widgets
    total = w["tabela_status"].rowCount()
    if total == 0:
        logar(w["console"], "⚠️ Nenhum item para reprocessar.")
        return

    for row in range(total):
        barra = w["tabela_status"].cellWidget(row, 2)
        if isinstance(barra, QProgressBar) and "❌" in barra.format():
            link_item = w["tabela_status"].item(row, 1)
            if not link_item:
                continue
            link = link_item.text().strip()
            if not link:
                continue

            pasta_saida = main.pastas_criadas[row]
            log_cb = partial(atualizar_status, main, row)
            prog_cb = partial(atualizar_progresso, main, row)

            try:
                main.worker_manager.adicionar_tarefa(
                    link=link,
                    pasta=pasta_saida,
                    index=row,
                    logar_callback=log_cb,
                    progresso_callback=prog_cb
                )
                atualizar_status(main, row, "🌀 Reprocessando")
                logar(w["console"], f"🔁 Reprocessando linha {row + 1}")
            except Exception as e:
                logar(w["console"], f"❌ Falha ao reprocessar linha {row + 1}: {e}")
