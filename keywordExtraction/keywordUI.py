import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
import PyPDF2

from KeywordExtr import KeywordExtr 

k = KeywordExtr

docs = []
threshold = 0.1
def open_pdfs():
    file_paths = filedialog.askopenfilenames(
        title="Select PDF files",
        filetypes=[("PDF Files", "*.pdf")]
    )

    if not file_paths:
        return

    
    docs.clear()
    for file_path in file_paths:
        try:
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                texts = ""
                for page in reader.pages:
                    texts += page.extract_text()
                    texts += "\n\n"
                docs.append(texts)
        except Exception as e:
            messagebox.showerror("Error", f"Could not read {file_path}\n{e}")
    
    global tf_idf
    tf_idf = k.get_tf_idf(docs)
    global ners
    ners = k.get_ner(docs)
    analyze()
       

def analyze():
    if(len(docs) == 0):
        return
    text_display.delete(1.0, tk.END)  # Clear existing text
    ner_display.delete(1.0, tk.END)  # Clear existing text
    for i in range(len(docs)):
        ind = i + 1
        text_display.insert(tk.END, f"\n keywords in Document {i+1}:")
        text_display.insert(tk.END, "\n")
        column = f'Doc{ind}'
        keywords = tf_idf.loc[tf_idf[column] >= threshold]
        keywords = keywords[[column, 'POS']].sort_values(by=column, ascending=False)
        #keywords = keywords[f'Doc{ind}'].sort_values(ascending=False)
        text_display.insert(tk.END, keywords.to_string())
        text_display.insert(tk.END, "\n\n")
        
        ner_display.insert(tk.END, f"\n NER in Document {ind}:")
        ner_display.insert(tk.END, "\n")
        ner = ners.loc[ners['Doc'] == str(ind)]
        ner = ner.drop_duplicates(subset='Text').set_index('Text')
        ner = ner[['Label']].sort_values(by='Text')
        ner_display.insert(tk.END, ner.to_string())
        ner_display.insert(tk.END, "\n\n")


def update_threshold(value):
    global threshold
    thrs = str(threshold) 
    if (thrs == value):
        return 0
    else:
        threshold = float(value)
        analyze()
def UI_Build():
    # Main window
    root = tk.Tk()
    root.title("Multi PDF Reader")
    root.geometry("1024x768")
    open_button = tk.Button(root, text="Open PDF Files", command=open_pdfs)
    open_button.pack(pady=10)
    slider_frame = tk.Frame(root)
    slider_frame.pack(fill=tk.X, pady=5)
   
    text_frame = tk.Frame(root)
    text_frame.pack(fill=tk.BOTH, expand=True)
    # Configure grid to distribute space equally
    text_frame.columnconfigure(0, weight=2)
    text_frame.columnconfigure(1, weight=1)
    text_frame.rowconfigure(0, weight=1)
    
    # Slider itself
    threshold_slider = tk.Scale(
        slider_frame,
        from_=0.0,
        to=1.0,
        resolution=0.01,
        orient=tk.HORIZONTAL,
        command=update_threshold
    )
    threshold_slider.set(threshold)
    threshold_slider.pack(fill=tk.X, padx=10, expand=True)

    global text_display
    text_display = scrolledtext.ScrolledText(text_frame, wrap=tk.WORD)
    #text_display.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    global ner_display
    ner_display = scrolledtext.ScrolledText(text_frame, wrap=tk.WORD)
    #ner_display.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    
    # Place them in grid
    text_display.grid(row=0, column=0, sticky="nsew")
    ner_display.grid(row=0, column=1, sticky="nsew")


    root.mainloop()



if __name__ == "__main__":
    #k.ensure_nltk_resources()
    UI_Build()