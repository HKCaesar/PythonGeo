
import os

if "INCLUDE" in os.environ.keys():
    os.environ["PATH"] = os.environ["PATH"] + r";M:\SDKs\cuda\cuda\bin;M:\SDKs\BLAS\OpenBLAS-v0.2.15-Win64-int32\bin"
    os.environ["INCLUDE"] = os.environ["INCLUDE"] + r";M:\SDKs\cuda\cuda\include"
    os.environ["LIB"] = os.environ["LIB"] + r";M:\SDKs\cuda\cuda\lib\x64;"
    os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32,dnn.enabled=True" #,lib.cnmem=0.4"
