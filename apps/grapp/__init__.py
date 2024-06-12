import gradio as gr
import torch
import torchaudio

from utils.audio.features import FeatureType, get_feature_transformer
from utils.plot import plot_spectrogram, plt2ndarray
import yaml
import json

with open("configs.yml", "r") as f:
    config = yaml.safe_load(f)
    FeatureParams = config["Features"]

def get_feature_params(feature_type):
    json_str = json.dumps(FeatureParams[feature_type], indent=4, ensure_ascii=False)
    return json_str

# 定义Gradio接口
def demo():
    with gr.Blocks() as demo:
        audio = gr.Audio(label="Audio", sources=['upload'])
        with gr.Row():
            sr_label = gr.Label(value=None, label="Sample Rate")
            shape_label = gr.Label(value=None, label="Shape")
            feature_type = gr.Dropdown(choices=[ft.name for ft in FeatureType], label="Feature Type", value=FeatureType.SPECTROGRAM.name)
        with gr.Row():
            with gr.Column():
                param_input = gr.TextArea(value=get_feature_params(FeatureType.SPECTROGRAM.name),label="Feature Parameters", interactive=True)
                log_text = gr.Text(label="Log")
            result_img = gr.Image(label="Result")

        audio.change(lambda x: x[0], inputs=audio, outputs=sr_label)
        audio.change(lambda x: str(x[1].shape), inputs=audio, outputs=shape_label)
        feature_type.change(get_feature_params, inputs=[feature_type], outputs=param_input)

        # 提交按钮
        submit_btn = gr.Button("Submit")

        # 处理提交
        def on_submit(audio, feature_type, param_input):
            try:
                params = json.loads(param_input)
                for k, v in params.items():
                    if v == 'None':
                        params[k] = None

                sr, signal = audio
                
                if len(signal.shape) == 2:
                    signal = torch.from_numpy(signal).T
                elif len(signal.shape) == 1:
                    signal = torch.from_numpy(signal).view(1, -1)
                max_ = torch.abs(signal).max()
                signal = signal / max_
                if 'sample_rate' in params.keys():
                    if sr != params['sample_rate']:
                        signal = torchaudio.transforms.Resample(sr, params['sample_rate'])(signal)
                        sr = params['sample_rate']

                signal = signal.mean(axis=0).flatten()
                feature_transformer = get_feature_transformer(feature_type, **params)

                feature = feature_transformer(signal)

                fig = plot_spectrogram(feature, title=feature_type)

                img = plt2ndarray(fig)

                return img, feature.shape
            except Exception as e:
                return None, str(e)

        submit_btn.click(on_submit, inputs=[audio, feature_type, param_input], outputs=[result_img, log_text])

    return demo



