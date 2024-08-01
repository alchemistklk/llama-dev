
import tensorrt_llm

class AtomTRTApi:
    def __init__(self, engine_dir, tokenizer_dir, max_input_length=4096):
        self.runtime_rank = tensorrt_llm.mpi_rank()
        self.model_name = read
