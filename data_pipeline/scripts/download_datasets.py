from datasets import load_dataset
import os


DATA_ROOT = "datasets/raw"


# ------------------------------------------------
# OpenWebText (small subset)
# ------------------------------------------------
def download_openwebtext():

    print("Downloading OpenWebText subset...")

    dataset = load_dataset(
        "openwebtext",
        split="train[:0.05%]"
    )

    path = os.path.join(DATA_ROOT, "openwebtext")

    dataset.save_to_disk(path)

    print("OpenWebText subset saved to", path)


# ------------------------------------------------
# GSM8K (small dataset already)
# ------------------------------------------------
def download_gsm8k():

    print("Downloading GSM8K...")

    dataset = load_dataset("gsm8k", "main")

    path = os.path.join(DATA_ROOT, "gsm8k")

    dataset.save_to_disk(path)

    print("GSM8K saved to", path)


# ------------------------------------------------
# Pile subset (very small sample)
# ------------------------------------------------
def download_pile_subset():

    print("Downloading Pile subset...")

    dataset = load_dataset(
        "EleutherAI/pile",
        split="train[:0.02%]"
    )

    path = os.path.join(DATA_ROOT, "pile")

    dataset.save_to_disk(path)

    print("Pile subset saved to", path)


# ------------------------------------------------
# Main
# ------------------------------------------------
def main():

    os.makedirs(DATA_ROOT, exist_ok=True)

    download_openwebtext()
    download_gsm8k()
    download_pile_subset()

    print("\nAll datasets downloaded successfully.")


if __name__ == "__main__":

    main()
