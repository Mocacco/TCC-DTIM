import customtkinter as ctk

# configuração da janela
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.geometry("500x300")
app.title("Troca de Páginas")

# ----------------------
# funções de navegação
# ----------------------
def mostrar_pagina1():
    frame2.pack_forget()
    frame1.pack(fill="both", expand=True)

def mostrar_pagina2():
    frame1.pack_forget()
    frame2.pack(fill="both", expand=True)

# ----------------------
# página 1
# ----------------------
frame1 = ctk.CTkFrame(app)

label1 = ctk.CTkLabel(frame1, text="Página 1", font=("Arial", 24))
label1.pack(pady=30)

botao_ir_2 = ctk.CTkButton(
    frame1,
    text="Ir para Página 2",
    command=mostrar_pagina2
)
botao_ir_2.pack(pady=10)

# ----------------------
# página 2
# ----------------------
frame2 = ctk.CTkFrame(app)

label2 = ctk.CTkLabel(frame2, text="Página 2", font=("Arial", 24))
label2.pack(pady=30)

botao_ir_1 = ctk.CTkButton(
    frame2,
    text="Voltar para Página 1",
    command=mostrar_pagina1
)
botao_ir_1.pack(pady=10)

# começa mostrando a página 1
frame1.pack(fill="both", expand=True)

app.mainloop()