"""Report the configured PyTorch and NVIDIA GPU runtime."""

import torch


def main() -> None:
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA runtime: {torch.version.cuda or 'not included'}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            print(f"GPU {index}: {torch.cuda.get_device_name(index)}")


if __name__ == "__main__":
    main()
