import os
import re
import time
import sys
import threading
import subprocess
from pathlib import Path
import numpy as np
import nibabel as nib
import cv2
from PIL import Image
import customtkinter as ctk
from tkinter import filedialog, messagebox
from tkinterdnd2 import TkinterDnD, DND_FILES
from ultralytics import YOLO
import matplotlib
matplotlib.use("TkAgg")  # Força o Matplotlib a renderizar no motor do Tkinter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Configuração global do formato estético da GUI
ctk.set_appearance_mode("light")

EXTENSOES_MEDICAS = [".nii.gz", ".npy"]
EXTENSOES_IMAGENS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]

# -------------------------------------------------------------------------
# METODOLOGIAS DE VALIDAÇÃO E FILTRAGEM GEOMÉTRICA (BACKEND ACOPLADO)
# -------------------------------------------------------------------------
def validar_nome(nome):
    if not nome:
        return False
    try:
        if len(nome) >= 3 and re.match(r'^[A-Za-zÀ-ÿ\s]+$', nome):
            return True
    except Exception:
        return False

def validar_path_img(imagem):
    if not imagem:
        return False
    try:
        caminho = Path(imagem)
        return caminho.exists() and caminho.is_file()
    except Exception:
        return False

def centralizar_volume(data, target_shape=(256, 256, 256), threshold=0):
    coords = np.where(data > threshold)
    x_min, x_max = coords[0].min(), coords[0].max()
    y_min, y_max = coords[1].min(), coords[1].max()
    z_min, z_max = coords[2].min(), coords[2].max()
    
    cropped = data[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]
    result = np.zeros(target_shape)
    start = [(target_shape[i] - cropped.shape[i]) // 2 for i in range(3)]
    
    result[
        start[0]:start[0]+cropped.shape[0],
        start[1]:start[1]+cropped.shape[1],
        start[2]:start[2]+cropped.shape[2]
    ] = cropped
    return result

def executar_hd_bet(app, nome, imagem):
    try:
        # 1. Configura as pastas de saída padrão
        pasta_saida = os.path.join("..", "Backend", "output", nome.upper())
        os.makedirs(pasta_saida, exist_ok=True)
        saida = os.path.join(pasta_saida, f"{nome}_brain.nii.gz")

        # Converte caminhos para o formato absoluto limpo
        imagem_abs = os.path.abspath(imagem)
        saida_abs = os.path.abspath(saida)

        # 2. Caminho absoluto direto para o executável gerado pelo pip no seu usuário
        home = str(Path.home())
        hd_bet_binario = os.path.join(
            home, "AppData", "Local", "Packages", 
            "PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0",
            "LocalCache", "local-packages", "Python313", "Scripts", "hd-bet.exe"
        )

        # Plano B: Caso o Windows use o diretório de roaming tradicional para scripts
        if not os.path.exists(hd_bet_binario):
            hd_bet_binario = os.path.join(home, "AppData", "Roaming", "Python", "Python313", "Scripts", "hd-bet.exe")

        # Plano C: Caso o Windows tenha colocado direto na pasta de scripts local padrão
        if not os.path.exists(hd_bet_binario):
            hd_bet_binario = os.path.join(home, "AppData", "Local", "Programs", "Python", "Python313", "Scripts", "hd-bet.exe")

        print(f"[DEBUG TCC] Forçando execução do binário isolado em: {hd_bet_binario}")

        # 3. Se mesmo assim o Windows esconder o .exe, usamos a string direta no shell para o PATH atual resolver
        if not os.path.exists(hd_bet_binario):
            print("[AVISO] Binário físico não mapeado no disco, tentando chamada via injeção de ambiente...")
            comando = f'hd-bet -i "{imagem_abs}" -o "{saida_abs}" -device cpu --disable_tta'
        else:
            comando = [
                hd_bet_binario,
                "-i", imagem_abs,
                "-o", saida_abs,
                "-device", "cpu",
                "--disable_tta"
            ]

        # 4. Dispara o subprocesso de forma totalmente isolada do interpretador python.exe do sistema
        processo = subprocess.Popen(
            comando,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True
        )

        inicio = time.time()
        tempo_estimado = 90

        while processo.poll() is None:
            tempo = time.time() - inicio
            progresso = min(tempo / tempo_estimado, 0.95)
            app.atualizar_loading(progresso, "Executando extração do crânio (HD-BET)...")
            time.sleep(0.5)

        stdout, stderr = processo.communicate()
        
        # Se o arquivo de saída foi gerado com sucesso, o pipeline funcionou!
        if os.path.exists(saida_abs):
            print("➔ [SUCESSO] O HD-BET gerou o arquivo de volume limpo!")
            return saida
            
        if processo.returncode != 0:
            print(f"Erro detalhado do terminal Windows:\n{stderr}\n{stdout}")
            raise subprocess.CalledProcessError(processo.returncode, comando, output=stdout, stderr=stderr)

        return saida
        
    except Exception as erro:
        print(f"Erro estrutural no HD-BET por binário isolado: {erro}")
        raise erro
    
def preprocessamento(app, nome, caminho):
    try:
        saida = os.path.join("..", "Backend", "output", nome.upper(), f"{nome}_brain.npy")
        app.atualizar_loading(0.3, "Carregando volume MRI...")
        img = nib.load(caminho)

        app.atualizar_loading(0.5, "Padronizando orientação canonical...")
        img_canonical = nib.as_closest_canonical(img)

        app.atualizar_loading(0.7, "Convertendo dados para matriz...")
        data = img_canonical.get_fdata()

        app.atualizar_loading(0.9, "Centralizando cérebro na matriz...")
        data_centralizada = centralizar_volume(data)

        np.save(saida, data_centralizada)
        return data_centralizada
    except Exception as erro:
        print(f"Erro no pré-processamento: {erro}")
        raise erro

def predict_cnn(app, fatias_rgb, indices_originais=None):
    app.atualizar_loading(0.95, "Classificando fatias com o YOLOv8...")
    
    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    raiz_projeto = os.path.dirname(diretorio_atual)
    
    caminho_modelo = os.path.join(raiz_projeto, "Backend", "runs", "classify", "train-3", "weights", "best.pt")
    if not os.path.exists(caminho_modelo):
        return "Erro: Pesos best.pt não localizados no Backend.", None

    model = YOLO(caminho_modelo)
    
    total_saudavel = 0
    total_tumor = 0
    
    maior_confianca_tumor = -1.0
    fatia_mais_critica_bgr = None
    slice_id_critico = None

    resultados = model.predict(source=fatias_rgb, device=0, verbose=False)

    for i, r in enumerate(resultados):
        classe_id = r.probs.top1
        classe_nome = r.names[classe_id]
        confianca = float(r.probs.top1conf)

        if classe_nome == "saudavel":
            total_saudavel += 1
        else:
            total_tumor += 1
            if confianca > maior_confianca_tumor:
                maior_confianca_tumor = confianca
                fatia_mais_critica_bgr = cv2.cvtColor(fatias_rgb[i], cv2.COLOR_RGB2BGR)
                if indices_originais:
                    slice_id_critico = indices_originais[i]

    total_fatias = len(fatias_rgb)
    
    if total_tumor > 0:
        porcentagem = (total_tumor / total_fatias) * 100
        laudo = f"PRESENÇA DE TUMOR DETECTADA\nConsistência: {porcentagem:.2f}% das fatias comprometidas"
    else:
        porcentagem = (total_saudavel / total_fatias) * 100
        laudo = f"EXAME CLASSIFICADO COMO SAUDÁVEL\nÍndice de integridade: {porcentagem:.2f}%"
        
        fatia_mais_critica_bgr = cv2.cvtColor(fatias_rgb[len(fatias_rgb)//2], cv2.COLOR_RGB2BGR)
        slice_id_critico = indices_originais[len(indices_originais)//2] if indices_originais else "2D"

    pasta_runs = os.path.join(raiz_projeto, "Backend", "runs")
    os.makedirs(pasta_runs, exist_ok=True)
    caminho_temp_fatia = os.path.join(pasta_runs, "temp_fatia_critica.jpg")
    
    if slice_id_critico is not None:
        cv2.putText(fatia_mais_critica_bgr, f"Slice: {slice_id_critico}", (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
    cv2.imwrite(caminho_temp_fatia, fatia_mais_critica_bgr)
    return laudo, caminho_temp_fatia

def pipeline(app, nome, imagem):
    try:
        extensao = os.path.splitext(imagem.lower())[1]
        if image_lower := imagem.lower().endswith(".nii.gz"):
            extensao = ".nii.gz"

        fatias_para_yolo = []
        indices_originais = []

        # --- FLUXO 1: ARQUIVOS DE IMAGEM TRADICIONAL (2D) ---
        if extensao in EXTENSOES_IMAGENS:
            app.atualizar_loading(0.4, "Processando imagem convencional...")
            img_bruta = cv2.imread(imagem)
            img_redim = cv2.resize(img_bruta, (256, 256))
            img_rgb = cv2.cvtColor(img_redim, cv2.COLOR_BGR2RGB)
            fatias_para_yolo.append(img_rgb)
            indices_originais.append("2D")

        # --- FLUXO 2: MATRIZES EXTRAÍDAS NUMPY (.npy 3D) ---
        elif extensao == ".npy":
            app.atualizar_loading(0.4, "Lendo e estruturando matriz .npy...")
            volume_3d = np.load(imagem)
            
            # CORREÇÃO CRÍTICA: Garante que o .npy seja salvo na pasta do paciente para o gráfico 3D ler
            pasta_saida = os.path.join("..", "Backend", "output", nome.upper())
            os.makedirs(pasta_saida, exist_ok=True)
            caminho_salvamento_npy = os.path.join(pasta_saida, f"{nome}_brain.npy")
            np.save(caminho_salvamento_npy, volume_3d) # Salva a cópia estruturada para o Matplotlib

            # Extrai as fatias para a classificação da YOLOv8
            for z in range(80, 180):
                fatia = volume_3d[:, :, z]
                if fatia.max() > 0:
                    fatia_norm = cv2.normalize(fatia, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    fatia_rgb = cv2.cvtColor(fatia_norm, cv2.COLOR_GRAY2RGB)
                    fatias_para_yolo.append(fatia_rgb)
                    indices_originais.append(z)

        # --- FLUXO 3: EXAMES BRUTOS NIFTI (.nii.gz) ---
        elif extensao == ".nii.gz":
            resultado_hdbet = executar_hd_bet(app, nome, imagem)
            volume_3d = preprocessamento(app, nome, resultado_hdbet)
            for z in range(80, 180):
                fatia = volume_3d[:, :, z]
                if fatia.max() > 0:
                    fatia_norm = cv2.normalize(fatia, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    fatia_rgb = cv2.cvtColor(fatia_norm, cv2.COLOR_GRAY2RGB)
                    fatias_para_yolo.append(fatia_rgb)
                    indices_originais.append(z)

        if not fatias_para_yolo:
            raise ValueError("Não foi possível coletar matrizes de pixels válidas.")

        laudo_final, caminho_slice = predict_cnn(app, fatias_para_yolo, indices_originais)
        
        app.atualizar_loading(1.0, "Análise concluída!")
        time.sleep(0.3)
        app.after(0, lambda: app.mostrar_pagina_resultado(nome, laudo_final, caminho_slice))

    except Exception as e:
        mensagem_erro = str(e)
        app.after(0, lambda: app.atualizar_loading(0, f"Erro: {mensagem_erro}"))
        app.after(0, lambda: messagebox.showerror("Erro no Pipeline", f"Falha no processamento:\n{mensagem_erro}"))
    finally:
        app.after(0, lambda: app.button_analise.configure(state="normal", text="Iniciar Análise"))
# -------------------------------------------------------------------------
# INTERFACE GRÁFICA MULTI-PÁGINAS
# -------------------------------------------------------------------------
class App(TkinterDnD.DnDWrapper, ctk.CTk):

    def __init__(self):
        ctk.CTk.__init__(self)
        self.TkdndVersion = TkinterDnD._require(self)
        
        self.font = "Consolas"
        self.title('Sistema de Detecção de Tumores Cerebrais com CNN')
        self.geometry('800x540')
        self.configure(fg_color="#F5F7FA")
        
        try:
            self.entry_mri_image = ctk.CTkImage(light_image=Image.open('images/upload.png'), size=(50, 50))
            self.iconbitmap('images/brain.ico')
        except Exception:
            self.entry_mri_image = None

        self.container = ctk.CTkFrame(self, fg_color="transparent")
        self.container.pack(fill="both", expand=True)

        self.criar_pagina_upload()
        self.criar_pagina_resultado()
        self.mostrar_pagina_upload()

    def criar_pagina_upload(self):
        self.frame_upload = ctk.CTkFrame(self.container, fg_color="transparent")
        self.frame_upload.pack(fill="both", expand=True)
        self.frame_upload.grid_columnconfigure((0, 2), weight=1, uniform="col")
        self.frame_upload.grid_rowconfigure(0, weight=1)

        # LADO ESQUERDO
        self.frame_esq = ctk.CTkFrame(self.frame_upload, fg_color="transparent")
        self.frame_esq.grid(row=0, column=0, sticky="nsew")

        self.label_paciente = ctk.CTkLabel(self.frame_esq, text="Nome do Paciente", text_color="#000000", font=(self.font, 24), anchor="w")
        self.label_paciente.pack(padx=20, pady=(20,0), fill="x")

        self.entry_paciente = ctk.CTkEntry(self.frame_esq, font=(self.font, 16), placeholder_text="Digite o nome do paciente", fg_color="#FFFFFF", text_color="#000000", corner_radius=100)
        self.entry_paciente.pack(padx=20, fill="x")

        self.label_mri = ctk.CTkLabel(self.frame_esq, text="Ressonância Magnética", text_color="#000000", font=(self.font, 24), anchor="w")
        self.label_mri.pack(padx=20, pady=(20,0), fill="x")

        self.frame_mri = ctk.CTkFrame(self.frame_esq, border_width=1, border_color="#848484", fg_color="#FFFFFF", corner_radius=10)
        self.frame_mri.pack(padx=20, fill="both", expand=True)
        self.frame_mri.pack_propagate(False)
        self.frame_mri.drop_target_register(DND_FILES)
        self.frame_mri.dnd_bind('<<Drop>>', self.ao_soltar_arquivo)

        self.label_image_entry = ctk.CTkLabel(self.frame_mri, image=self.entry_mri_image, text="" if self.entry_mri_image else "[Upload]")
        self.label_image_entry.pack(padx=20, pady=(20,5))

        self.button_entry = ctk.CTkButton(self.frame_mri, font=(self.font, 16), text_color="#3775E0", text="Escolha o arquivo", fg_color="transparent", hover=False, cursor="hand2", command=self.selecionar_arquivo)
        self.button_entry.pack(padx=20, pady=5)
        self.button_entry.bind("<Enter>", self.on_enter)
        self.button_entry.bind("<Leave>", self.on_leave)

        self.label_entry = ctk.CTkLabel(self.frame_mri, font=(self.font, 16), text_color="#848484", text="ou arraste ele aqui")
        self.label_entry.pack(padx=20, pady=5)

        self.label_entry_suport = ctk.CTkLabel(self.frame_mri, font=(self.font, 11), text_color="#848484", text="Formatos: .nii.gz, .npy, .png, .jpg")
        self.label_entry_suport.pack(padx=20, pady=(10,5))

        self.label_arquivo = ctk.CTkLabel(self.frame_mri, font=(self.font, 12), text_color="#FFFFFF", text="", corner_radius=10, fg_color="transparent", anchor="w")
        self.label_arquivo.pack(padx=20, pady=10, fill="x")

        self.button_analise = ctk.CTkButton(self.frame_esq, text="Iniciar Análise", text_color="#FFFFFF", font=(self.font, 22), corner_radius=100, fg_color="#3775E0", command=self.iniciar_deteccao)
        self.button_analise.pack(padx=20, pady=20)

        # DIVISOR CENTRAL
        self.linha = ctk.CTkFrame(self.frame_upload, width=2, fg_color="gray")
        self.linha.grid(row=0, column=1, sticky="ns", pady=20)

        # LADO DIREITO
        self.frame_dir = ctk.CTkFrame(self.frame_upload, fg_color="transparent")
        self.frame_dir.grid(row=0, column=2, sticky="nsew")

        self.label_resultado_painel = ctk.CTkLabel(self.frame_dir, text="Progresso do Pipeline", text_color="#000000", font=(self.font, 24))
        self.label_resultado_painel.pack(padx=20, pady=(20,0), fill="x")

        self.frame_resultados = ctk.CTkFrame(self.frame_dir, border_width=1, border_color="#848484", fg_color="#FFFFFF", corner_radius=10)
        self.frame_resultados.pack(padx=20, pady=(0,20), fill="both", expand=True)

        self.label_status = ctk.CTkLabel(self.frame_resultados, text="Aguardando envio de arquivo...", font=(self.font, 14), text_color="#848484")
        self.label_status.pack(pady=(40, 0))

        self.progressbar = ctk.CTkProgressBar(self.frame_resultados, width=300, height=20, corner_radius=100, fg_color="#E0E0E0", progress_color="#3775E0")
        self.progressbar.set(0)
        self.progressbar.pack_forget()

        self.label_porcentagem = ctk.CTkLabel(self.frame_resultados, text="0%", font=(self.font, 16), text_color="#848484")
        self.label_porcentagem.pack_forget()

    def criar_pagina_resultado(self):
        """ Inicializa e fixa os elementos estruturais da tela de Laudo e do Canvas 3D """
        self.frame_resultado_tela = ctk.CTkFrame(self.container, fg_color="transparent")
        
        self.label_titulo_res = ctk.CTkLabel(self.frame_resultado_tela, text="LAUDO MÉDICO COMPUTACIONAL", text_color="#000000", font=(self.font, 24, "bold"))
        self.label_titulo_res.pack(pady=(20, 5))

        self.box_laudo = ctk.CTkFrame(self.frame_resultado_tela, border_width=2, border_color="#3775E0", fg_color="#FFFFFF", corner_radius=15)
        self.box_laudo.pack(padx=40, pady=10, fill="both", expand=True)

        self.label_nome_res = ctk.CTkLabel(self.box_laudo, text="Paciente: ", text_color="#000000", font=(self.font, 18, "bold"), anchor="w")
        self.label_nome_res.pack(padx=20, pady=(15, 5), fill="x")

        self.label_diagnostico = ctk.CTkLabel(self.box_laudo, text="", font=(self.font, 20, "bold"), justify="center")
        self.label_diagnostico.pack(padx=20, pady=5, fill="x")

        # Label curinga para falhas ou exames estáticos 2D convencionais
        self.label_slice_mri = ctk.CTkLabel(self.box_laudo, text="")
        
        self.button_voltar = ctk.CTkButton(self.frame_resultado_tela, text="Analisar Novo Exame", text_color="#FFFFFF", font=(self.font, 16), corner_radius=100, fg_color="#848484", command=self.mostrar_pagina_upload)
        self.button_voltar.pack(pady=(5, 20))

    def mostrar_pagina_upload(self):
        self.frame_resultado_tela.pack_forget()
        self.frame_upload.pack(fill="both", expand=True)
        self.label_status.configure(text="Aguardando envio de arquivo...")
        self.progressbar.set(0)
        self.progressbar.pack_forget()
        self.label_porcentagem.pack_forget()

    def mostrar_pagina_resultado(self, nome_paciente, laudo_texto, caminho_slice):
        """ Executa o fatiamento em tempo real da matriz e renderiza o cérebro 3D interativo na tela """
        self.frame_upload.pack_forget()
        self.frame_resultado_tela.pack(fill="both", expand=True)
        
        self.label_nome_res.configure(text=f"Paciente: {nome_paciente.upper()}")
        self.label_diagnostico.configure(text=laudo_texto)
        
        if "SAUDÁVEL" in laudo_texto:
            self.label_diagnostico.configure(text_color="#2A9D8F")
        else:
            self.label_diagnostico.configure(text_color="#E63946")

        # Limpeza obrigatória de exibições e canvas anteriores para evitar vazamento de memória
        self.label_slice_mri.pack_forget() 
        if hasattr(self, 'canvas_3d'):
            self.canvas_3d.get_tk_widget().destroy()

        diretorio_atual = os.path.dirname(os.path.abspath(__file__))
        raiz_projeto = os.path.dirname(diretorio_atual)
        caminho_npy = os.path.join(raiz_projeto, "Backend", "output", nome_paciente.upper(), f"{nome_paciente}_brain.npy")

        # Se houver um volume 3D pré-processado válido, embuti a reconstrução tridimensional interativa
        if os.path.exists(caminho_npy):
            try:
                volume = np.load(caminho_npy)
                sub_amostragem = volume[::4, ::4, ::4]
                x, y, z = np.where(sub_amostragem > 10) 
                valores = sub_amostragem[x, y, z]

                fig = plt.figure(figsize=(4, 3.5), facecolor='white')
                ax = fig.add_subplot(111, projection='3d')
                ax.set_facecolor('white')
                
                ax.scatter(x, y, z, c=valores, cmap='bone', s=1, alpha=0.4)
                ax.axis('off') 
                ax.view_init(elev=20, azim=45) 

                self.canvas_3d = FigureCanvasTkAgg(fig, master=self.box_laudo)
                widget_3d = self.canvas_3d.get_tk_widget()
                widget_3d.pack(padx=20, pady=10, expand=True, fill="both")
                self.canvas_3d.draw()

                plt.close(fig)
            except Exception as e:
                self.label_slice_mri.configure(text=f"Erro ao gerar modelo 3D: {e}")
                self.label_slice_mri.pack(padx=20, pady=10, expand=True)
        else:
            # Caso seja uma imagem 2D comum convencional (.jpg/.png), renderiza a foto do slice estático
            self.label_slice_mri.pack(padx=20, pady=10, expand=True)
            if caminho_slice and os.path.exists(caminho_slice):
                try:
                    img_pil = Image.open(caminho_slice)
                    img_ctk = ctk.CTkImage(light_image=img_pil, size=(220, 220))
                    self.label_slice_mri.configure(image=img_ctk, text="")
                    self.label_slice_mri.image = img_ctk
                except Exception as e:
                    self.label_slice_mri.configure(text=f"Erro ao carregar imagem: {e}")
            else:
                self.label_slice_mri.configure(text="Slice de evidência indisponível.")

    def on_enter(self, event):
        self.button_entry.configure(font=(self.font, 16, "underline"))

    def on_leave(self, event):
        self.button_entry.configure(font=(self.font, 16))

    def arquivo_valido(self, caminho):
        caminho_lower = caminho.lower()
        validos = EXTENSOES_MEDICAS + EXTENSOES_IMAGENS
        return any(caminho_lower.endswith(ext) for ext in validos)
    
    def selecionar_arquivo(self):
         formatos_texto = " ".join([f"*{ext}" for ext in (EXTENSOES_MEDICAS + EXTENSOES_IMAGENS)])
         filename = filedialog.askopenfilename(
            title="Selecionar arquivo de exame", 
            filetypes=[("Todos os Exames Suportados", formatos_texto), ("Imagens Tradicionais", "*.jpg;*.png;*.jpeg;*.bmp"), ("Arquivos Médicos 3D", "*.nii.gz;*.npy")]
         )
         if self.arquivo_valido(filename):
            self.atualizar_label(filename)
         else:
            messagebox.showerror("Arquivo inválido", "Formato de arquivo não suportado pelo sistema.")

    def ao_soltar_arquivo(self, event):
        filename = event.data.strip('{ }')
        if self.arquivo_valido(filename):
            self.atualizar_label(filename)
        else:
            messagebox.showerror("Arquivo inválido", "Formato de arquivo não suportado pelo sistema.")

    def atualizar_label(self, texto):
        self.label_arquivo.configure(text=texto, fg_color="#848484")
        self.label_status.configure(text="Arquivo carregado pronto para análise.")

    def mostrar_loading(self):
        self.label_status.pack(pady=(40,0))
        self.progressbar.pack(padx=20, pady=(15,0), fill="x")
        self.label_porcentagem.pack(pady=5)

    def atualizar_loading(self, valor, texto):
        porcentagem = int(valor * 100)
        self.after(0, lambda: self.progressbar.set(valor))
        self.after(0, lambda: self.label_status.configure(text=texto))
        self.after(0, lambda: self.label_porcentagem.configure(text=f"{porcentagem}%"))

    def iniciar_deteccao(self):
        nome = self.entry_paciente.get().strip()
        imagem = self.label_arquivo.cget("text")

        if not validar_nome(nome):
            messagebox.showerror("Erro", "Insira um nome válido para o paciente (mínimo 3 letras).")
            return
        
        if not validar_path_img(imagem):
            messagebox.showerror("Erro", "Anexe um arquivo de exame válido.")
            return

        self.mostrar_loading()
        self.button_analise.configure(state="disabled", text="Processando...")

        thread = threading.Thread(target=pipeline, args=(self, nome, imagem), daemon=True)
        thread.start()

if __name__ == "__main__":
    app = App()
    app.mainloop()