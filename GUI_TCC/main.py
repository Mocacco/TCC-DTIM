import customtkinter as ctk
from tkinter import filedialog, messagebox
from tkinterdnd2 import TkinterDnD, DND_FILES
from PIL import Image
from pathlib import Path
import re
import os
import numpy as np
import nibabel as nib
import subprocess
import threading
import time

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

        if not caminho.exists():
            return False
        
        if not caminho.is_file():
            return False

        return True
    
    except Exception:
        return False

def centralizar_volume(data, target_shape=(256, 256, 256), threshold=0):
    # 1. Encontrar região do cérebro
    coords = np.where(data > threshold)
    
    x_min, x_max = coords[0].min(), coords[0].max()
    y_min, y_max = coords[1].min(), coords[1].max()
    z_min, z_max = coords[2].min(), coords[2].max()
    
    # 2. Crop (tirar fundo)
    cropped = data[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]
    
    # 3. Criar volume final
    result = np.zeros(target_shape)
    
    # 4. Centralizar
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
            [
                "hd-bet",
                "-i", imagem,
                "-o", saida,
                "-device", "cpu",
                "--disable_tta"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        inicio = time.time()

        tempo_estimado = 90

        while processo.poll() is None:

            tempo = time.time() - inicio

            progresso = min(
                tempo / tempo_estimado,
                0.95
            )

            app.atualizar_loading(
                progresso * 1.0,
                "Executando HD-BET..."
            )

            time.sleep(0.1)

        # pega saída do processo
        stdout, stderr = processo.communicate()

        # verifica erro
        if processo.returncode != 0:

            raise subprocess.CalledProcessError(
                processo.returncode,
                processo.args,
                output=stdout,
                stderr=stderr
            )

        app.atualizar_loading(
            1.0,
            "HD-BET concluído"
        )

        print("HD-BET executado com sucesso")

        return saida

    except subprocess.CalledProcessError as erro:

        app.atualizar_loading(
            0,
            "Erro ao executar HD-BET"
        )

        print("Erro ao executar HD-BET")
        print(erro.stderr)

    except FileNotFoundError:

        app.atualizar_loading(
            0,
            "HD-BET não encontrado"
        )

        print("HD-BET não está instalado ou não está no PATH")

    except Exception as erro:

        app.atualizar_loading(
            0,
            "Erro inesperado"
        )

        print("Erro inesperado:")
        print(erro)

def preprocessamento(app, nome, caminho):

    try:
        
        saida = f"output/{nome.upper()}/{nome}_brain.npy"

        # ---------------------------------
        # ETAPA 1 — LOAD
        # ---------------------------------

        app.atualizar_loading(
            0,
            "Carregando volume MRI..."
        )

        img = nib.load(caminho)

        # ---------------------------------
        # ETAPA 2 — CANONICAL
        # ---------------------------------

        app.atualizar_loading(
            0.3,
            "Padronizando orientação..."
        )

        img_canonical = nib.as_closest_canonical(img)

        # ---------------------------------
        # ETAPA 3 — GET DATA
        # ---------------------------------

        app.atualizar_loading(
            0.6,
            "Convertendo volume..."
        )

        data = img_canonical.get_fdata()

        # ---------------------------------
        # ETAPA 4 — CENTRALIZAÇÃO
        # ---------------------------------


        app.atualizar_loading(
            0.9,
            "Centralizando cérebro..."
        )

        data_centralizada = centralizar_volume(data)

        # ---------------------------------
        # FINAL
        # ---------------------------------

        app.atualizar_loading(
            1.0,
            "Pré-processamento concluído"
        )

        np.save(saida, data_centralizada)

        print("Pré-processamento executado com sucesso")

        

        return data_centralizada

    except Exception as erro:

        app.atualizar_loading(
            0,
            "Erro no preprocessamento"
        )

        print(erro)

def predict_cnn(app, imagem):
    #implementar
    return

def pipeline(app, nome, imagem):

    try:

        resultado = executar_hd_bet(app, nome, imagem)

        preprocessamento(app, nome, resultado)

        time.sleep(3) # placeholder = predict_cnn()

        app.atualizar_loading(
            1.0,
            "Análise concluída"
        )

    finally:

        app.after(
            0,
            lambda: app.button_analise.configure(
                state="normal",
                text="Iniciar Análise"
            )
        )

class App(ctk.CTk, TkinterDnD.DnDWrapper):

    def __init__(self):
        super().__init__()

        self.TkdndVersion = TkinterDnD._require(self)

        self.font = "Consolas"

        self.title('Sistema de Detecção de Tumores Cerebrais com CNN')
        self.iconbitmap('images/brain.ico')
        self.geometry('800x450')
        self.configure(fg_color="#F5F7FA")
        self.grid_columnconfigure((0,2), weight=1, uniform="col")
        self.grid_rowconfigure(0, weight=1)

        self.entry_mri_image = ctk.CTkImage(light_image=Image.open('images/upload.png'),size=(50, 50))
        
        # -----------------------------
        # LADO ESQUERDO
        # -----------------------------

        self.frame_esq = ctk.CTkFrame(
            self,
            fg_color="transparent"
            )
        self.frame_esq.grid(row=0, column=0, sticky="nsew")

        self.label_paciente = ctk.CTkLabel(
            self.frame_esq,
            text="Nome do Paciente",
            text_color="#000000",
            font=(self.font, 24),
            anchor="w"
        )
        self.label_paciente.pack(padx=20, pady=(20,0), fill="x")

        self.entry_paciente = ctk.CTkEntry(
            self.frame_esq,
            font=(self.font, 16),
            placeholder_text="Digite o nome do paciente",
            placeholder_text_color="#848484",
            border_width=1,
            border_color="#848484",
            fg_color="#FFFFFF",
            text_color="#000000",
            corner_radius=100,
        )
        self.entry_paciente.pack(padx=20, fill="x")

        self.label_mri = ctk.CTkLabel(
            self.frame_esq,
            text="Ressonância Magnética",
            text_color="#000000",
            font=(self.font, 24),
            anchor="w"
        )
        self.label_mri.pack(padx=20, pady=(20,0), fill="x")

        self.frame_mri = ctk.CTkFrame(
            self.frame_esq,
            border_width=1,
            border_color="#848484",
            fg_color="#FFFFFF",
            corner_radius=10,
        )
        self.frame_mri.pack(padx=20, fill="both", expand=True)
        self.frame_mri.pack_propagate(False)
        self.frame_mri.drop_target_register(DND_FILES)
        self.frame_mri.dnd_bind('<<Drop>>', self.ao_soltar_arquivo)

        self.label_image_entry = ctk.CTkLabel(
            self.frame_mri,
            image=self.entry_mri_image,
            text=""
        )
        self.label_image_entry.pack(padx=20, pady=(20,5))

        self.button_entry = ctk.CTkButton(
            self.frame_mri,
            font=(self.font, 16),
            text_color="#3775E0",
            text="Escolha o arquivo",
            fg_color="transparent",
            hover=False,
            cursor="hand2",
            command=self.selecionar_arquivo
        )
        self.button_entry.pack(padx=20, pady=5)
        self.button_entry.bind("<Enter>", self.on_enter)
        self.button_entry.bind("<Leave>", self.on_leave)

        self.label_entry = ctk.CTkLabel(
            self.frame_mri,
            font=(self.font, 16),
            text_color="#848484",
            text="ou arraste ele aqui",
            fg_color="transparent"
        )
        self.label_entry.pack(padx=20, pady=5)

        self.label_entry_suport = ctk.CTkLabel(
            self.frame_mri,
            font=(self.font, 12),
            text_color="#848484",
            text="Arquivo suportado: .nii.gz",
            fg_color="transparent"
        )
        self.label_entry_suport.pack(padx=20, pady=(20,5))

        self.label_arquivo = ctk.CTkLabel(
            self.frame_mri,
            font=(self.font, 16),
            text_color="#FFFFFF",
            text="",
            corner_radius=10,
            fg_color="transparent",
            anchor="w"
        )
        self.label_arquivo.pack(padx=20, pady=20, fill="x")

        self.button_analise = ctk.CTkButton(
            self.frame_esq,
            text="Iniciar Análise",
            text_color="#FFFFFF",
            font=(self.font, 22),
            corner_radius=100,
            fg_color="#3775E0",
            command=self.iniciar_deteccao
        )
        self.button_analise.pack(padx=20, pady=20)

        # -----------------------------
        # LINHA NO MEIO
        # -----------------------------

        self.linha = ctk.CTkFrame(self, width=2, fg_color="gray")
        self.linha.grid(row=0, column=1, sticky="ns", pady=20)

        # -----------------------------
        # LADO DIREITO
        # -----------------------------

        self.frame_dir = ctk.CTkFrame(
            self,
            fg_color="transparent"
        )
        self.frame_dir.grid(row=0, column=2, sticky="nsew")

        self.label_resultado = ctk.CTkLabel(
            self.frame_dir,
            text="Resultado",
            text_color="#000000",
            font=(self.font, 24),
        )
        self.label_resultado.pack(padx=20, pady=(20,0), fill="x")

        self.frame_resultados = ctk.CTkFrame(
            self.frame_dir,
            border_width=1,
            border_color="#848484",
            fg_color="#FFFFFF",
            corner_radius=10,
        )
        self.frame_resultados.pack(padx=20, pady=(0,20), fill="both", expand=True)

        self.label_status = ctk.CTkLabel(
            self.frame_resultados,
            text="Aguardando análise...",
            font=(self.font, 16),
            text_color="#848484",
        )
        self.label_status.pack_forget()

        self.progressbar = ctk.CTkProgressBar(
            self.frame_resultados,
            width=350,
            height=25,
            corner_radius=100,
            fg_color="#FFFFFF",
            progress_color="#3775E0",
            border_color="#848484",
            border_width=1
        )
        self.progressbar.set(0)
        self.progressbar.pack_forget()
        

        self.label_porcentagem = ctk.CTkLabel(
            self.frame_resultados,
            text="0%",
            font=(self.font, 16),
            text_color="#848484"
        )

        self.label_porcentagem.pack_forget()

    def on_enter(self, event):
        self.button_entry.configure(font=(self.font, 16, "underline"))

    def on_leave(self, event):
        self.button_entry.configure(font=(self.font, 16))

    def arquivo_valido(self, caminho):
        return caminho.lower().endswith(".nii.gz")
    
    def selecionar_arquivo(self):
        filename = filedialog.askopenfilename(
            title="Selecionar arquivo",
            filetypes=[("Arquivos NIFTI", "*.nii.gz")]
        )
        if self.arquivo_valido(filename):
            self.atualizar_label(filename)
        else:
             messagebox.showerror(
                "Arquivo inválido",
                "Selecione apenas arquivos .nii.gz"
            )

    def ao_soltar_arquivo(self, event):
        filename = event.data.strip('{ }')

        if self.arquivo_valido(filename):
            self.atualizar_label(filename)
        else:
             messagebox.showerror(
                "Arquivo inválido",
                "Selecione apenas arquivos .nii.gz"
            )

    def atualizar_label(self, texto):

        #nome_arquivo = os.path.basename(texto)
        self.label_arquivo.configure(
            text=texto,
            fg_color="#848484"
        )

    def mostrar_loading(self):
        self.label_status.pack(pady=(20,0))
        self.progressbar.pack(padx=20, pady=(20,0), fill="x")
        self.label_porcentagem.pack()

    def atualizar_loading(self, valor, texto):

        porcentagem = int(valor * 100)

        self.after(
            0,
            lambda: self.progressbar.set(valor)
        )

        self.after(
            0,
            lambda: self.label_status.configure(
                text=texto
            )
        )

        self.after(
            0,
            lambda: self.label_porcentagem.configure(
                text=f"{porcentagem}%"
            )
        )

    def iniciar_deteccao(self):

        nome = self.entry_paciente.get().strip()
        imagem = self.label_arquivo.cget("text")

        if not validar_nome(nome):
            messagebox.showerror(
                "Erro",
                "Insira o nome do paciente."
            )
            return
        
        if not validar_path_img(imagem):
            messagebox.showerror(
                "Erro",
                "Anexe um arquivo valido"
            )
            return

        self.mostrar_loading()

        self.button_analise.configure(
            state="disabled",
            text="Processando..."
        )

        thread = threading.Thread(
        target=pipeline,
        args=(self, nome, imagem),
        daemon=True
        )

        thread.start()

if __name__ == "__main__":
    app = App()
    app.mainloop()