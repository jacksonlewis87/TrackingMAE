from constants import ROOT_DIR
from visualization.tracking import do_work


def main():
    tensor_path = f"{ROOT_DIR}/data/tensors/0021500492_9.pt"
    do_work(path=tensor_path)


if __name__ == "__main__":
    main()
