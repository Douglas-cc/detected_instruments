import requests
import numpy as np 
import gradio as gr
from loguru import logger 


def extract_mfcc(audio):  
    url =  "http://127.0.0.1:8000/files/"
    file = {'file': open(audio ,'rb')}
    logger.info(file)
    res = requests.post(url, files=file) 
    return res.text

    
demo = gr.Interface(
    fn=extract_mfcc,
    inputs = gr.Audio(type="filepath"),
    outputs='text'
)


# with gr.Blocks() as demo:
#     gr.Markdown("#Classificador de Instrumentos")
    
#     with gr.Tab("Gravar Audio"):
#         mic_input = gr.Audio(source="microphone")
#         record_button = gr.Button('predict') 
#         gr.Button('predict').click(fn=test, inputs=record_button, outputs='audio') 
        
#     with gr.Tab("Importar Audio"):
#         audio_input = gr.Audio(type="filepath")
        # gr.Button('predict').click(fn=test, inputs=audio_input, outputs='audio') 
        

if __name__ == "__main__":
    demo.launch()