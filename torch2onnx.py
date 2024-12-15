from transformers import AutoTokenizer, AutoModel
import torch
import onnxruntime as ort
from thop import profile

import numpy as np

class roberta_model(torch.nn.Module):
    def __init__(self, N: int = 2048, batch_size: int = 8):
        super().__init__()

        self.batch_size = batch_size

        self.root = "/home/ubuntu/Max"

        self.backbone = AutoModel.from_pretrained('roberta-base', return_dict=True, output_hidden_states=True)
        self.fc = torch.nn.Linear(self.backbone.config.hidden_size, N)

        self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')

        dummy_string = "This is a dummy string used to act as an input for roberta model."

        encoding = self.tokenizer(dummy_string, return_tensors='pt')

        input_ids = torch.tensor(encoding["input_ids"], dtype=torch.int32, device="cpu")
        attention_mask = torch.tensor(encoding["attention_mask"], dtype=torch.int32, device="cpu")
        
        self.tokenized_input = input_ids
        self.attention_mask = attention_mask


    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        features = self.backbone(input_ids = input_ids, attention_mask = attention_mask)
        output = self.fc(features.last_hidden_state)

        return output

    def convert_to_onnx(self):
        self.eval()
        print(self.tokenized_input.shape, self.attention_mask.shape)
        torch.onnx.export(
            self,
            (self.tokenized_input, self.attention_mask),
            f"{self.root}/models/model.onnx",
            opset_version=19,
            input_names=['INPUT_IDS', 'ATTENTION_MASK'],
            output_names=['PROJECTION'],
            dynamic_axes={
                'INPUT_IDS': {0: 'BATCH_SIZE', 1: 'SEQUENCE_LENGTH'},
                'ATTENTION_MASK': {0: 'BATCH_SIZE', 1: 'SEQUENCE_LENGTH'},
                'PROJECTION': {0: 'BATCH_SIZE', 1: 'SEQUENCE_LENGTH'}
            },
        )
        # Sanity Check
        torch_output = self(self.tokenized_input, self.attention_mask).detach().numpy()

        ort_inputs = {
            "INPUT_IDS": self.tokenized_input.numpy(),
            "ATTENTION_MASK": self.attention_mask.numpy()
        }
        ort_session = ort.InferenceSession(f"{self.root}/models/model.onnx")
        ort_outputs = ort_session.run(None, ort_inputs)[0]

        assert np.allclose(torch_output, ort_outputs, atol=1e-5)

    
def calc_flops(model: roberta_model, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    macs, params, layer_info = profile(model.backbone, inputs=(input_ids, attention_mask), ret_layer_info=True)
    print(f"FLOPS count: {macs * 2}")
    print(f"Params count: {params}")
    print(f"Model's batch size is {input_ids.shape[0]}")

    for layer, (layer_macs, layer_params, _) in layer_info.items():
        memory_consumption = layer_params * 64
        print(f"Current layer: {layer}, FLOPs: {2 * layer_macs}, FLOPs / Memory usage: {2 * layer_macs / memory_consumption}")
        print(f"    is layer memory constrained? {(2 * layer_macs / memory_consumption) < 38}")


def main():
    model = roberta_model()
    print("Model loaded")
    model.convert_to_onnx()
    print("Model converted successfully")

    calc_flops(model, model.tokenized_input, model.attention_mask)

    model.tokenizer.save_pretrained('roberta-base')


if __name__ == "__main__":
    main()