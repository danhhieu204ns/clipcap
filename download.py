from pathlib import Path
import shutil

import kagglehub


DATASET_HANDLE = "hsankesara/flickr-image-dataset"


def main() -> None:
	project_root = Path(__file__).resolve().parent

	# KaggleHub downloads into its cache; copy to project root for local use.
	cache_path = Path(kagglehub.dataset_download(DATASET_HANDLE)).resolve()
	dataset_name = DATASET_HANDLE.split("/", 1)[1]
	target_path = project_root / dataset_name

	if target_path.exists() and any(target_path.iterdir()):
		print(f"Dataset already exists in project root: {target_path}")
	else:
		shutil.copytree(cache_path, target_path, dirs_exist_ok=True)
		print(f"Copied dataset to project root: {target_path}")

	print(f"Kaggle cache path: {cache_path}")


if __name__ == "__main__":
	main()