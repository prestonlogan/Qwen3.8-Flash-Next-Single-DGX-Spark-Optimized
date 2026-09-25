import os, time
from torch.utils.cpp_extension import load
d = "/qf"; os.makedirs(d + "/build", exist_ok=True); t = time.time()
load(name="qf_moe", sources=[d + "/qf_bind_moe.cpp", d + "/qf_moe.cu"], build_directory=d + "/build", verbose=False,
     extra_cflags=["-O3"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_121a,code=sm_121a", "--use_fast_math", "-lineinfo"])
print("built", round(time.time() - t, 1), "s")
