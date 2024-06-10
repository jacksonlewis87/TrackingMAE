import time
from tensorboard import program

from constants import ROOT_DIR


if __name__ == "__main__":
    experiment_name = "mae_v1"
    # experiment_name = "decoding_v1"

    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", f"{ROOT_DIR}\\data\\training\\{experiment_name}"])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")
    time.sleep(10000)
