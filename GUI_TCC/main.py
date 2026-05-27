import os
import re
import time
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

# Configuração global do CustomTkinter
ctk.set_appearance_mode("light")

# Listas de extensões suportadas pelo ecossistema do TCC
EXTENSOES_MEDICAS = [".nii.gz", ".npy"]
EXTENSOES_IMAGENS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]

# -------------------------------------------------------------------------
# FUNÇÕES DE VALIDAÇÃO E PROCESSAMENTO DE MATRIZES
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
        os.makedirs(f"output/{nome.upper()}", exist_ok=True)
        saida = f"output/{nome.upper()}/{nome}_brain.nii.gz"

        processo = subprocess.Popen(
            ["hd-bet", "-i", imagem, "-o", saida, "-device", "cpu", "--disable_tta"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        inicio = time.time()
        tempo_estimado = 90

        while processo.poll() is None:
            tempo = time.time() - inicio
            progresso = min(tempo / tempo_estimado, 0.95)
            app.atualizar_loading(progresso, "Executando extração do crânio (HD-BET)...")
            time.sleep(0.1)

        stdout, stderr = processo.communicate()
        if processo.returncode != 0:
            raise subprocess.CalledProcessError(processo.returncode, processo.args, output=stdout, stderr=stderr)

        return saida
    except Exception as erro:
        print(f"Erro no HD-BET: {erro}")
        raise erro

def preprocessamento(app, nome, caminho):
    try:
        saida = f"output/{nome.upper()}/{nome}_brain.npy"
        app.atualizar_loading(0.3, "Carregando volume extraído...")
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
    """ Roda a inferência do YOLOv8 e salva a imagem do corte com maior índice patológico """
    app.atualizar_loading(0.95, "Classificando imagens com o YOLOv8...")
    
    caminho_modelo = "runs/classify/train-6/weights/best.pt"
    if not os.path.exists(caminho_modelo):
        return "Erro: Modelo best.pt não encontrado.", None

    model = YOLO(caminho_modelo)
    
    total_saudavel = 0
    total_tumor = 0
    
    maior_confianca_tumor = -1.0
    fatia_mais_critica_bgr = None
    slice_id_critico = None

    # Executa a predição em lote tirando proveito da RTX 3060
    resultados = model.predict(source=fatias_rgb, device=0, verbose=False)

    for i, r in enumerate(resultados):
        classe_id = r.probs.top1
        classe_nome = r.names[classe_id]
        confianca = float(r.probs.top1conf)

        if classe_nome == "saudavel":
            total_saudavel += 1
        else:
            total_tumor += 1
            # Rastreia qual slice obteve maior certeza de tumor na rede
            if confianca > maior_confianca_tumor:
                maior_confianca_tumor = confianca
                fatia_mais_critica_bgr = cv2.cvtColor(fatias_rgb[i], cv2.COLOR_RGB2BGR)
                if indices_originais:
                    slice_id_critico = indices_originais[i]

    total_fatias = len(fatias_rgb)
    
    if total_tumor > total_saudavel:
        porcentagem = (total_tumor / total_fatias) * 100
        laudo = f"PRESENÇA DE TUMOR DETECTADA\nConsistência: {porcentagem:.2f}% dos inputs analisados"
    else:
        porcentagem = (total_saudavel / total_fatias) * 100
        laudo = f"EXAME SAUDÁVEL\nConsistência: {porcentagem:.2f}% dos inputs analisados"
        
        # Se for saudável, exibe a fatia do meio para ilustração em tons de cinza médicos
        fatia_mais_critica_bgr = cv2.cvtColor(fatias_rgb[len(fatias_rgb)//2], cv2.COLOR_RGB2BGR)
        slice_id_critico = indices_originais[len(indices_originais)//2] if indices_originais else "2D"

    # Salva a imagem do slice para exibição interna na interface
    caminho_temp_fatia = "output/temp_fatia_critica.jpg"
    os.makedirs("output", exist_ok=True)
    
    # Renderiza o número identificador do corte diretamente na matriz da imagem
    if slice_id_critico is not None:
        cv2.putText(fatia_mais_critica_bgr, f"Slice: {slice_id_critico}", (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
    cv2.imwrite(caminho_temp_fatia, fatia_mais_critica_bgr)
    return laudo, caminho_temp_fatia

def pipeline(app, nome, imagem):
    try:
        extensao = os.path.splitext(imagem.lower())[1]
        if imagem.lower().endswith(".nii.gz"):
            extensao = ".nii.gz"

        fatias_para_yolo = []
        indices_originais = []

        # --- FLUXO 1: IMAGENS CONVENCIONAIS ---
        if extensao in EXTENSOES_IMAGENS:
            app.atualizar_loading(0.4, "Convertendo imagem convencional...")
            img_bruta = cv2.imread(imagem)
            img_redim = cv2.resize(img_bruta, (256, 256))
            img_rgb = cv2.cvtColor(img_redim, cv2.COLOR_BGR2RGB)
            fatias_para_yolo.append(img_rgb)
            indices_originais.append("Único")

        # --- FLUXO 2: MATRIZES NUMPY ---
        elif extensao == ".npy":
            app.atualizar_loading(0.4, "Carregando matriz .npy...")
            volume_3d = np.load(imagem)
            for z in range(80, 180):
                fatia = volume_3d[:, :, z]
                if fatia.max() > 0:
                    img_norm = cv2.normalize(fatia, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    img_rgb = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2RGB)
                    fatias_para_yolo.append(img_rgb)
                    indices_originais.append(z)

        # --- FLUXO 3: ARQUIVOS NIFTI ---
        elif extensao == ".nii.gz":
            resultado_hdbet = executar_hd_bet(app, nome, imagem)
            volume_3d = preprocessamento(app, nome, resultado_hdbet)
            for z in range(80, 180):
                fatia = volume_3d[:, :, z]
                if fatia.max() > 0:
                    img_norm = cv2.normalize(fatia, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    img_rgb = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2RGB)
                    fatias_para_yolo.append(img_rgb)
                    indices_originais.append(z)

        if not fatias_para_yolo:
            raise ValueError("Nenhum dado de imagem válido pôde ser extraído.")

        laudo_final, caminho_slice = predict_cnn(app, fatias_para_yolo, indices_originais)
        
        app.atualizar_loading(1.0, "Análise concluída!")
        time.sleep(0.3)
        app.after(0, lambda: app.mostrar_pagina_resultado(nome, laudo_final, caminho_slice))

    except Exception as e:
        # CORREÇÃO AQUI: Converte o erro para texto ANTES de enviar para o after/lambda
        mensagem_erro = str(e)
        app.after(0, lambda: app.atualizar_loading(0, f"Erro: {mensagem_erro}"))
        app.after(0, lambda: messagebox.showerror("Erro no Pipeline", f"Falha no processamento:\n{mensagem_erro}"))
    finally:
        app.after(0, lambda: app.button_analise.configure(state="normal", text="Iniciar Análise"))

# -------------------------------------------------------------------------
# INTERFACE GRÁFICA INTERCONECTADA (CUSTOMTKINTER + TKINTERDND2)
# -------------------------------------------------------------------------
class App(ctk.CTk, TkinterDnD.DnDWrapper):

    def __init__(self):
        super().__init__()
        self.TkdndVersion = TkinterDnD._require(self)
        self.font = "Consolas"

        self.title('Sistema de Detecção de Tumores Cerebrais com CNN')
        self.geometry('800x520')
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

        # --- LADO ESQUERDO ---
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

        # --- DIVISOR CENTRAL ---
        self.linha = ctk.CTkFrame(self.frame_upload, width=2, fg_color="gray")
        self.linha.grid(row=0, column=1, sticky="ns", pady=20)

        # --- LADO DIREITO ---
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
        """ Segunda página estruturada para renderizar o laudo final junto com o slice crítico """
        self.frame_resultado_tela = ctk.CTkFrame(self.container, fg_color="transparent")
        
        self.label_titulo_res = ctk.CTkLabel(self.frame_resultado_tela, text="LAUDO MÉDICO COMPUTACIONAL", text_color="#000000", font=(self.font, 24, "bold"))
        self.label_titulo_res.pack(pady=(20, 5))

        self.box_laudo = ctk.CTkFrame(self.frame_resultado_tela, border_width=2, border_color="#3775E0", fg_color="#FFFFFF", corner_radius=15)
        self.box_laudo.pack(padx=40, pady=10, fill="both", expand=True)

        self.label_nome_res = ctk.CTkLabel(self.box_laudo, text="Paciente: ", text_color="#000000", font=(self.font, 18, "bold"), anchor="w")
        self.label_nome_res.pack(padx=20, pady=(15, 5), fill="x")

        self.label_diagnostico = ctk.CTkLabel(self.box_laudo, text="", font=(self.font, 20, "bold"), justify="center")
        self.label_diagnostico.pack(padx=20, pady=5, fill="x")

        # Container visual dedicado à renderização da fatia 2D enviada do pipeline
        self.label_slice_mri = ctk.CTkLabel(self.box_laudo, text="[Nenhum slice carregado]", text_color="#848484")
        self.label_slice_mri.pack(padx=20, pady=10, expand=True)

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
        self.frame_upload.pack_forget()
        self.frame_resultado_tela.pack(fill="both", expand=True)
        
        self.label_nome_res.configure(text=f"Paciente: {nome_paciente.upper()}")
        self.label_diagnostico.configure(text=laudo_texto)
        
        # Altera esteticamente a cor do laudo clínico dependendo da classe prevista
        if "SAUDÁVEL" in laudo_texto:
            self.label_diagnostico.configure(text_color="#2A9D8F")
        else:
            self.label_diagnostico.configure(text_color="#E63946")

        # Atualiza o widget de imagem com a foto JPG gerada dinamicamente pelo pipeline
        if caminho_slice and os.path.exists(caminho_slice):
            try:
                img_pil = Image.open(caminho_slice)
                img_ctk = ctk.CTkImage(light_image=img_pil, size=(220, 220))
                
                self.label_slice_mri.configure(image=img_ctk, text="")
                self.label_slice_mri.image = img_ctk  # Proteção de garbage collection do Tkinter
            except Exception as e:
                self.label_slice_mri.configure(text=f"Erro ao carregar imagem: {e}")
        else:
            self.label_slice_mri.configure(text="Imagem do slice não disponível.")

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

        # Dispara o pipeline em uma thread paralela isolando a inferência e mantendo a GUI responsiva
        thread = threading.Thread(target=pipeline, args=(self, nome, imagem), daemon=True)
        thread.start()

if __name__ == "__main__":
    app = App()
    app.mainloop()