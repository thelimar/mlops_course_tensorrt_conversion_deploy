FROM nvcr.io/nvidia/tritonserver:23.12-py3-sdk

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY assets /assets
COPY model_repository /models

EXPOSE 7055
#EXPOSE 8070 8077 8079

#CMD ["tritonserver", "--http-port=8070", "--grpc-port=8077", "--metrics-port=8079", "--model-repository=/models"]
# docker run --gpus=all -it --rm -p8070:8070 -p8077:8077 -p8079:8079 max_triton_server