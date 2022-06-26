import ctypes
import numpy as np
np.random.seed(0)
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    
    for idx, binding in enumerate(engine):
        size = trt.volume(engine.get_binding_shape(binding)) * 1
        print(size)
        host_mem = cuda.pagelocked_empty(size, dtype=trt.nptype(engine.get_binding_dtype(idx)))
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))

        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    
    return inputs, outputs, bindings, stream

def trt_inference():
    engine_file = "/target/decoder.plan"
    logger = trt.Logger(trt.Logger.WARNING)

    dummy_input = np.random.rand(16, 256, 256)
    np.save('input.npy', dummy_input)

    with trt.Runtime(logger) as trt_runtime:
        trt.init_libnvinfer_plugins(None, "")             
        with open(engine_file, 'rb') as f:
            engine_data = f.read()
        engine = trt_runtime.deserialize_cuda_engine(engine_data)
        inputs, outputs, bindings, stream = allocate_buffers(engine)

        with engine.create_execution_context() as context:
            if len(inputs) == 1:
                np.copyto(inputs[0].host, dummy_input.ravel())
            else:   
                for idx in range(len(inputs)):
                    np.copyto(inputs[idx].host, dummy_input[idx].ravel())        

            for inp in inputs:
                cuda.memcpy_htod_async(inp.device, inp.host, stream)
            context.execute_async(batch_size=1, bindings=bindings, stream_handle=stream.handle)
            for out in outputs:
                cuda.memcpy_dtoh_async(out.host, out.device, stream) 
                
            stream.synchronize()

            trt_output = [out.host for out in outputs]
            print(trt_output)
            np.save('ouput.npy', trt_output[0])
ctypes.cdll.LoadLibrary("/target/wenet-TensorRT/LayerNormPlugin.so")
trt_inference()
