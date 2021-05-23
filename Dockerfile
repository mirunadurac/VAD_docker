FROM python:3.7
RUN pip install --upgrade pip
RUN pip install librosa==0.7.1
RUN pip install flask==1.1.2
RUN pip install werkzeug
RUN pip install numpy==1.17
RUN pip install torch==1.6.0

WORKDIR /opt/vad
COPY vad vad 
WORKDIR /workspace

CMD ["python"]