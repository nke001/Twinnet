import re
import os
import numpy
from time import sleep


def get_free_gpuid():
    lines = os.popen("nvidia-smi").read()
    if len(lines) == 0:
        return None
    gpu_mems = [line for line in lines.split('\n') if
                re.search('([0-9]+?)MiB / ([0-9]+?)MiB', line) is not None]
    gpu_free = []
    for gpu_id, gpu_line in enumerate(gpu_mems):
        fields = re.search('([0-9]+?)MiB / ([0-9]+?)MiB', gpu_line)
        assert fields is not None
        used = int(fields.group(1))
        total = int(fields.group(2))
        print(gpu_line)
        if used <= 20:
            gpu_free.append(gpu_id)
    if len(gpu_free) > 0:
        return gpu_free[0]
    return None


tasks = ["--twin 0.1 --deep_out --nlayers 3 --rnn_dim 512",
         "--twin 0.5 --deep_out --nlayers 3 --rnn_dim 512",
         "--twin 2.5 --deep_out --nlayers 3 --rnn_dim 512",
         "--twin 1.8 --deep_out --nlayers 3 --rnn_dim 512",
         "--twin 1.5 --deep_out --nlayers 3 --rnn_dim 512",
         "--twin 0.0 --deep_out --nlayers 3 --rnn_dim 512"]
base_dir = "cond_mnist_fancy_deep"
all_commands = []
status = open("test_seqmnist_twin.txt", "w")
for task in tasks:
    command = "python train_condmnist_twin.py"
    command += " %s" % task
    command += " --expname %s" % base_dir
    command += " > /dev/null 2>&1 &"
    all_commands.append(command)

print("Dispatching %s commands" % len(all_commands))
print(all_commands)
while len(all_commands) > 0:
    gpu_id = None
    while gpu_id is None:
        sleep(60.)
        gpu_id = get_free_gpuid()

    print("Got free gpu: %s" % gpu_id)
    assert len(all_commands) > 0

    command = all_commands.pop()
    prefix = "CUDA_VISIBLE_DEVICES=%s " % gpu_id
    command = prefix + command
    os.system(command)
    print("Dispatching: %s" % command)
    status.write("Dispatching: %s\n" % command)
    status.write("Commands left : %s\n" % len(all_commands))
    status.flush()

status.close()

