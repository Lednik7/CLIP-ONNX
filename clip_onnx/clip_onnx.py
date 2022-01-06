from .clip_converter import clip_converter
import torch
import onnxruntime


class clip_onnx(clip_converter):
    def __init__(self, model=None,
                 visual_path: str = "clip_visual.onnx",
                 textual_path: str = "clip_textual.onnx"):
        if not isinstance(model, (type(None))):
            super().__init__(model, visual_path, textual_path)
        else:
            print("[CLIP ONNX] Load mode")

    def load_onnx(self, visual_path=None, textual_path=None, logit_scale=None):
        if visual_path and textual_path:
            if not logit_scale:
                raise Exception("For this mode logit_scale must be specified. Example: model.logit_scale.exp()")
            self.logit_scale = logit_scale
        if visual_path:
            self.visual_path = visual_path
            self.visual_flag = True
        if textual_path:
            self.textual_path = textual_path
            self.textual_flag = True

    def start_sessions(self, providers=['TensorrtExecutionProvider',
                                        'CUDAExecutionProvider',
                                        'CPUExecutionProvider']):
        if self.visual_flag:
            self.visual_session = onnxruntime.InferenceSession(self.visual_path,
                                                               providers=providers)
        if self.textual_flag:
            self.textual_session = onnxruntime.InferenceSession(self.textual_path,
                                                                providers=providers)

    def visual_run(self, onnx_image):
        onnx_input_image = {self.visual_session.get_inputs()[0].name: onnx_image}
        visual_output, = self.visual_session.run(None, onnx_input_image)
        return visual_output

    def textual_run(self, onnx_text):
        onnx_input_text = {self.textual_session.get_inputs()[0].name: onnx_text}
        textual_output, = self.textual_session.run(None, onnx_input_text)
        return textual_output

    def __call__(self, image, text, device: str = "cpu"):
        assert self.visual_flag and self.textual_flag
        image_features = torch.from_numpy(self.visual_run(image)).to(device)
        text_features = torch.from_numpy(self.textual_run(text)).to(device)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logits_per_image = self.logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text

    def encode_image(self, image):
        return self.visual_run(image)

    def encode_text(self, text):
        return self.textual_run(text)
