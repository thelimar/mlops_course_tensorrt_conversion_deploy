import numpy as np
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput
from tritonclient.utils import np_to_triton_dtype

def call_triton(input_text: str):

    client = InferenceServerClient(url="localhost:8070")
    text = np.array([input_text], dtype=object)
    
    input_text_tensor = InferInput(
        name="TEXT", shape=text.shape, datatype=np_to_triton_dtype(text.dtype)
    )
    input_text_tensor.set_data_from_numpy(text)

    requested_tensors = ["PROJECTION_ONNX", "PROJECTION_BEST", "PROJECTION_FP16", "PROJECTION_FP32", "PROJECTION_INT8"]
    outputs = [InferRequestedOutput(tensor_name) for tensor_name in requested_tensors]
    
    response = client.infer(
        "ensemble",
        [input_text_tensor],
        outputs=outputs,
    )

    embeds = {tensor_name:response.as_numpy(tensor_name) for tensor_name in requested_tensors}
    
    return embeds

def check_quality(input_text: str):
    embeds = call_triton(input_text)

    deviations = {}

    for embed_key in embeds.keys():
        deviations[embed_key] = np.mean(np.abs(embeds[embed_key] - embeds["PROJECTION_ONNX"]))

    return deviations

def main():
    texts = [
        "Sample texts for checking",
        "They should be informative",
        "Or not",
        "It's 30 min before dedline so Im desperate",
        "Pls work",
        "P.s. It didnt work... Thank for not deleting tachka after deadline!"
    ]

    deviations = {'PROJECTION_BEST': [], 'PROJECTION_FP16': [], 'PROJECTION_FP32': [], 'PROJECTION_INT8': []}

    for text in texts:
        cur_deviations = check_quality(text)
        for deviation in cur_deviations.keys():
            deviations[deviation].append(cur_deviations[deviation])

    print("The differences for converted models:")
    print("###########################################################")

    for model_type in deviations.keys():
        print(f"Difference between {model_type.split("_")[-1]} and ONNX: {np.mean(deviations[model_type])}")

if __name__ == "__main__":
    main()

