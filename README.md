# 🎬 BatchTool

BatchTool é uma aplicação desktop com GUI para criadores de conteúdo — especialmente YouTube Shorts, TikTok e canais anônimos de terror, conspiração e espiritualidade.

🚀 Automatiza do início ao fim a criação de vídeos com download, transcrição, reencodificação e geração de assets.  

---

## 📦 Funcionalidades Principais

- 📥 **Download automático** de vídeos via `yt-dlp`
- 🎙️ **Transcrição de áudio** com OpenAI Whisper (torch, CUDA)
- 🧠 **Narração TTS** (integrável com gTTS, Bark, Tortoise, etc.)
- 🎬 **Reprocessamento de vídeo** com FFMPEG e moviepy
- 📄 **Geração de legendas** e áudio MP3 narrado
- 🖼️ **Thumbs e imagens** com IA (opcional com SD/MidJourney)
- ⚙️ **Pipeline estruturado** com `TaskManager` e `QueueManager`
- 📊 **Métricas e logs estruturados** com Loguru
- 🧪 **Modo Dry-Run CLI** (sem GUI, para automações)

---

## 📂 Estrutura de Pastas

```plaintext
BatchTool/
├── main.py                 # Script principal
├── app/                    # Controle da aplicação
├── assets/                 # Ícones e recursos
├── core/                   # Toda a lógica de negócio
│   ├── config/
│   ├── downloader/
│   ├── exceptions/
│   ├── logger/
│   ├── manager/
│   ├── metrics/
│   ├── processor/
│   ├── transcriber/
│   ├── utils/
│   └── validators/
├── gui/                    # Interface gráfica
├── workers/                # Processos paralelos
├── tests/                  # Testes automatizados
├── output/                 # Saídas e logs (gitignored)
├── logs/                   # Logs (gitignored)
├── requirements.txt        # Dependências
├── setup.py                # Configuração do pacote
├── README.md               # Documentação
└── .gitignore              # Ignorar arquivos sensíveis e temporários
🛠️ Tecnologias e Dependências
Python 3.10+

PyQt6

torch, torchaudio (CUDA support)

openai-whisper

FFMPEG

yt-dlp

moviepy

Loguru

pytest

✅ Instalação

Clone o repositório:

git clone https://github.com/seuusuario/BatchTool.git
cd BatchTool
Crie e ative um ambiente virtual:

python -m venv venv
source venv/bin/activate  # ou .\venv\Scripts\activate no Windows

Instale as dependências:

pip install -r requirements.txt

▶️ Como Rodar

🖥️ Modo GUI

python main.py

🧪 Modo Dry Run (CLI)

(Exemplo: processamento em lote sem interface)

python dry_run.py --input video.mp4 --output ./output/

🎛️ Personalização da GUI (em desenvolvimento)

Tema escuro

Seleção de vozes TTS

Pré-configuração de workflows (YouTube/TikTok)

Ajuste visual de timers e cortes

Editor de legendas integrado

📈 Roadmap
 Worker local para TTS com fallback

 Upload agendado para TikTok e YouTube

 SEO generator com IA local

 Dashboard com insights de engajamento

👨‍💻 Autor
Cauã Menezes
GitHub | menezescaua2505@gmail.com

📄 Licença
MIT License - veja LICENSE para mais informações.
