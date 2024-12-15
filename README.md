# mlops_course_tensorrt_conversion_deploy
PyTorch model conversion in ONNX and TensorRT, performance measuring via Triton

# Results
## ONNX conversion
After converting Roberta-base model in onnx we will measure its performance: flops and flops / memory_consumption. If we will compare our number with peak flops / memory_bandwidth for our GPU (RTX 3080) we will know, if our layer is constrained by memory of bandwidth. Here are the results: 

![FLOPS](https://github.com/thelimar/mlops_course_tensorrt_conversion_deploy/blob/main/Images/flops.png)

## PLAN conversion
Then we convert .onnx to .plan via onnx2trt.sh in a TensorRT docker container.

![Offsets](https://github.com/thelimar/mlops_course_tensorrt_conversion_deploy/blob/main/Images/offsets.png)

## Triton server inference
Then we can inference our model via Triton Inference Server (also in a docker container) and measure the mae between tensorRT and onnx model outputs.
P.S. I tried using perf_analyzer from SDK, but it didn'd see the inference server fro some reason :(
