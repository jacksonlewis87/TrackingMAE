import time
from tensorboard import program

from constants import ROOT_DIR


if __name__ == "__main__":
    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", f"{ROOT_DIR}\\data\\training\\mae_v0"])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")
    time.sleep(10000)
