import torch
import onnx
from torch import nn
from onnxruntime.quantization import quantize_dynamic, QuantType
from .utils import Textual, DEFAULT_EXPORT


class clip_converter(nn.Module):
    def __init__(self, model, visual_path: str = "clip_visual.onnx",
                 textual_path: str = "clip_textual.onnx"):
        super().__init__()
        self.model = model
        self.visual_path = visual_path
        self.textual_path = textual_path
        self.visual_flag = False
        self.textual_flag = False
        self.logit_scale = self.model.logit_scale.exp()

        self.model.eval()
        for x in self.model.parameters():
            x.requires_grad = False

    def quantization(self, mode: str = "dynamic"):
        assert mode in ["dynamic"]
        if mode == "dynamic":
            model_quant_visual = f"{self.visual_path}.quant"
            quantize_dynamic(self.visual_path,
                             model_quant_visual,
                             weight_type=QuantType.QUInt8)
            self.visual_path = model_quant_visual

            model_quant_textual = f"{self.textual_path}.quant"
            quantize_dynamic(self.textual_path,
                             model_quant_textual,
                             weight_type=QuantType.QUInt8)
            self.textual_path = model_quant_textual

    def torch_export(self, model, dummy_input, path: str, export_params=DEFAULT_EXPORT):
        torch.onnx.export(model, dummy_input, path, **export_params)

    def onnx_checker(self, path: str):
        model = onnx.load(path)
        onnx.checker.check_model(model)
        del model

    def convert_visual(self, dummy_input, wrapper=lambda x: x,
                       export_params=DEFAULT_EXPORT):
        visual = wrapper(self.model.visual)
        self.torch_export(visual, dummy_input, self.visual_path,
                          export_params=export_params)
        self.onnx_checker(self.visual_path)

    def convert_textual(self, dummy_input, wrapper=Textual,
                        export_params=DEFAULT_EXPORT):
        textual = wrapper(self.model)
        self.torch_export(textual, dummy_input, self.textual_path,
                          export_params=export_params)
        self.onnx_checker(self.textual_path)

    def convert2onnx(self, visual_input=None, textual_input=None, verbose=True,
                     visual_wrapper=lambda x: x,
                     textual_wrapper=Textual,
                     visual_export_params=DEFAULT_EXPORT,
                     textual_export_params=DEFAULT_EXPORT):
        isinstance_visual_input = isinstance(visual_input, (torch.Tensor))
        isinstance_textual_input = isinstance(textual_input, (torch.Tensor))

        if (not isinstance_visual_input) and (not isinstance_textual_input):
            raise Exception("[CLIP ONNX] Please, choose a dummy input")
        elif not isinstance_visual_input:
            print("[CLIP ONNX] Convert only textual model")
        elif not isinstance_textual_input:
            print("[CLIP ONNX] Convert only visual model")

        if isinstance_visual_input:
            self.visual_flag = True
            if verbose:
                print("[CLIP ONNX] Start convert visual model")
            self.convert_visual(visual_input, visual_wrapper, visual_export_params)
            if verbose:
                print("[CLIP ONNX] Start check visual model")
            self.onnx_checker(self.visual_path)

        if isinstance_textual_input:
            self.textual_flag = True
            if verbose:
                print("[CLIP ONNX] Start convert textual model")
            self.convert_textual(textual_input, textual_wrapper, textual_export_params)
            if verbose:
                print("[CLIP ONNX] Start check textual model")
            self.onnx_checker(self.textual_path)

        if verbose:
            print("[CLIP ONNX] Models converts successfully")
