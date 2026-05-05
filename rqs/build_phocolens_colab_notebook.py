from __future__ import annotations

import json
import textwrap
from pathlib import Path


def _src(text: str) -> list[str]:
    text = textwrap.dedent(text).strip("\n")
    if not text:
        return []
    return [line + "\n" for line in text.splitlines()]


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _src(text),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _src(text),
    }


cells: list[dict] = []

cells.append(
    md(
        """
        # PhoCoLens for Downsampled DiffuserCam (`3x67x120`)

        This notebook adapts the latest PhoCoLens codebase to the same setting used in this repository:

        - dataset: `bezzam/DiffuserCam-Lensless-Mirflickr-Dataset-NORM`
        - preprocessing: `downsample=4`, `flip_ud=False`, background subtraction from the PSF
        - image size after preprocessing: `3x67x120`

        The official PhoCoLens Stage 1 code assumes the original DiffuserCam resolution (`270x480`) and hard-codes several size-specific paths. This notebook patches those assumptions so PhoCoLens can be trained end-to-end on the padded size `3x68x120`, and then crops back to `3x67x120` for evaluation.

        The notebook is self-contained and designed for Google Colab with a single A100 GPU.
        """
    )
)

cells.append(
    code(
        """
        from pathlib import Path
        import json
        import os
        import random
        import shutil
        import subprocess
        import sys

        USE_DRIVE = False
        DRIVE_ROOT = Path("/content/drive/MyDrive")
        RUN_NAME = "phocolens_diffusercam_67x120"
        HF_DATASET_REPO = "bezzam/DiffuserCam-Lensless-Mirflickr-Dataset-NORM"
        HF_TOKEN = ""
        PHOCOLENS_COMMIT = "154fe32aea5c2b623f6dd1c07e90c2900d076486"

        DOWNSAMPLE = 4
        FINAL_H, FINAL_W = 67, 120
        PAD_H, PAD_W = 68, 120
        LETTERBOX_SIZE = 512

        STAGE1_BATCH_SIZE = 32
        STAGE1_EPOCHS = 80
        STAGE1_NUM_WORKERS = 4
        STAGE1_FFT_GAMMA = 100.0
        PROXY_DECODE_GAMMA = 1.0e-3

        STAGE2_BATCH_SIZE = 2
        STAGE2_NUM_WORKERS = 2
        STAGE2_MAX_STEPS = 50000
        STAGE2_VAL_INTERVAL = 0.25
        STAGE2_DDPM_STEPS = 200

        SEED = 123
        random.seed(SEED)
        os.environ["PYTHONHASHSEED"] = str(SEED)

        WORK_ROOT = DRIVE_ROOT / RUN_NAME if USE_DRIVE else Path("/content") / RUN_NAME
        PHOCOLENS_ROOT = WORK_ROOT / "PhoCoLens"
        STAGE1_DATA_ROOT = PHOCOLENS_ROOT / "SVDeconv" / "data" / "diffusercam67x120"
        STAGE2_DATA_ROOT = PHOCOLENS_ROOT / "NullSpaceDiff" / "data" / RUN_NAME
        ARTIFACT_ROOT = WORK_ROOT / "artifacts"
        ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

        print(json.dumps(
            {
                "work_root": str(WORK_ROOT),
                "phocolens_root": str(PHOCOLENS_ROOT),
                "stage1_data_root": str(STAGE1_DATA_ROOT),
                "stage2_data_root": str(STAGE2_DATA_ROOT),
                "artifact_root": str(ARTIFACT_ROOT),
                "phocolens_commit": PHOCOLENS_COMMIT,
            },
            indent=2,
        ))
        """
    )
)

cells.append(
    code(
        """
        if USE_DRIVE:
            from google.colab import drive
            drive.mount("/content/drive")

        def run(cmd, cwd=None):
            print("$", " ".join(str(x) for x in cmd))
            subprocess.check_call([str(x) for x in cmd], cwd=str(cwd) if cwd else None)

        def pip_install(*packages):
            run([sys.executable, "-m", "pip", "install", "-q", *packages])

        WORK_ROOT.mkdir(parents=True, exist_ok=True)
        os.chdir(WORK_ROOT)
        os.environ["WANDB_MODE"] = "offline"

        pip_install(
            "albumentations==1.3.0",
            "basicsr==1.4.2",
            "datasets>=2.18.0",
            "einops==0.7.0",
            "huggingface_hub>=0.23.0",
            "imageio",
            "imageio-ffmpeg",
            "invisible-watermark>=0.1.5",
            "kornia==0.6.0",
            "lpips",
            "matplotlib",
            "omegaconf==2.1.1",
            "open_clip_torch==2.0.2",
            "opencv-python-headless==4.8.1.78",
            "protobuf==3.20.3",
            "pytorch-lightning==1.4.2",
            "recordclass",
            "sacred",
            "scikit-image",
            "test-tube>=0.7.5",
            "torch-fidelity",
            "torchmetrics==0.6.0",
            "transformers==4.19.2",
            "wandb",
            "waveprop==0.0.10",
            "webdataset==0.2.5",
        )
        pip_install(
            "git+https://github.com/CompVis/taming-transformers.git@master",
            "git+https://github.com/openai/CLIP.git@main",
        )

        if PHOCOLENS_ROOT.exists():
            shutil.rmtree(PHOCOLENS_ROOT)

        run(["git", "clone", "https://github.com/OpenImagingLab/PhoCoLens.git", str(PHOCOLENS_ROOT)])
        run(["git", "checkout", PHOCOLENS_COMMIT], cwd=PHOCOLENS_ROOT)

        run([sys.executable, "-c", "import torch; import torchvision; print(torch.__version__); print(torchvision.__version__)"])
        """
    )
)

cells.append(
    code(
        """
        import subprocess
        import torch

        try:
            subprocess.run(["nvidia-smi"], check=False)
        except FileNotFoundError:
            print("nvidia-smi not found")

        print("torch.cuda.is_available() =", torch.cuda.is_available())
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable in this Colab session. Switch to a GPU runtime before training: Runtime -> Change runtime type -> GPU."
            )
        print("CUDA device:", torch.cuda.get_device_name(0))
        """
    )
)

cells.append(
    md(
        """
        ## 1. Export DiffuserCam data in PhoCoLens format

        This cell recreates the preprocessing used by the current project:

        - resize both lensless and lensed images by `downsample=4`
        - estimate a per-channel background from the downsampled PSF
        - subtract that background from the lensless image
        - pad the bottom by one row so Stage 1 can use `pixelshuffle_ratio=2`

        For the PhoCoLens Stage 1 supervision target, we build a **range-space proxy**
        by convolving the padded ground truth with the padded PSF and then applying a
        Wiener decode. This keeps the two-stage PhoCoLens structure while staying
        self-contained for the downsampled `67x120` setting.
        """
    )
)

cells.append(
    code(
        """
        import json
        import math
        from pathlib import Path

        import numpy as np
        import torch
        import torch.nn.functional as F
        from datasets import load_dataset
        from huggingface_hub import hf_hub_download, login
        from PIL import Image
        from torchvision.transforms.functional import resize
        from tqdm.auto import tqdm

        if HF_TOKEN:
            login(token=HF_TOKEN)

        dataset = load_dataset(HF_DATASET_REPO)
        print(dataset)

        psf_path = hf_hub_download(repo_id=HF_DATASET_REPO, filename="psf.tiff", repo_type="dataset")
        psf_full = np.asarray(Image.open(psf_path)).astype(np.float32) / 255.0
        if psf_full.ndim == 2:
            psf_full = np.repeat(psf_full[..., None], 3, axis=2)

        def resize_rgb(image, size_hw):
            tensor = torch.from_numpy(image).permute(2, 0, 1)
            tensor = resize(tensor, list(size_hw), antialias=True)
            return tensor.permute(1, 2, 0).numpy()

        def pad_bottom(image):
            assert image.shape[0] == FINAL_H, image.shape
            return np.pad(image, ((0, PAD_H - FINAL_H), (0, 0), (0, 0)), mode="edge")

        def to_uint8(image):
            return np.clip(np.round(image * 255.0), 0, 255).astype(np.uint8)

        psf_small = resize_rgb(psf_full, (FINAL_H, FINAL_W))
        bg = psf_small[:15, :15].mean(axis=(0, 1), keepdims=True)
        psf_small = np.clip(psf_small - bg, 0.0, None)
        psf_pad = pad_bottom(psf_small).astype(np.float32)
        psf_gray = psf_pad.mean(axis=2).astype(np.float32)

        def otf_from_psf(psf_hw):
            kernel = torch.from_numpy(psf_hw)
            return torch.fft.fft2(torch.fft.ifftshift(kernel))

        OTF = otf_from_psf(psf_gray)

        def fft_conv_rgb(image_hwc):
            tensor = torch.from_numpy(image_hwc.astype(np.float32)).permute(2, 0, 1)
            result = torch.fft.ifft2(torch.fft.fft2(tensor, dim=(-2, -1)) * OTF, dim=(-2, -1)).real
            return result.permute(1, 2, 0).numpy()

        def wiener_decode_rgb(image_hwc, gamma=PROXY_DECODE_GAMMA):
            tensor = torch.from_numpy(image_hwc.astype(np.float32)).permute(2, 0, 1)
            filt = torch.conj(OTF) / (OTF.abs().square() + gamma)
            result = torch.fft.ifft2(torch.fft.fft2(tensor, dim=(-2, -1)) * filt, dim=(-2, -1)).real
            result = result.permute(1, 2, 0).numpy()
            result = np.clip(result, 0.0, None)
            denom = result.max() - result.min()
            if denom < 1.0e-8:
                return np.zeros_like(result, dtype=np.float32)
            return ((result - result.min()) / denom).astype(np.float32)

        if STAGE1_DATA_ROOT.exists():
            shutil.rmtree(STAGE1_DATA_ROOT)

        for rel in [
            "diffuser_images",
            "ground_truth_lensed",
            "decode_sim_padding_png",
        ]:
            (STAGE1_DATA_ROOT / rel).mkdir(parents=True, exist_ok=True)

        Image.fromarray(to_uint8(psf_pad)).save(STAGE1_DATA_ROOT / "psf.tiff")

        split_to_csv = {"train": "dataset_train.csv", "test": "dataset_test.csv"}
        metadata = {
            "dataset_repo": HF_DATASET_REPO,
            "downsample": DOWNSAMPLE,
            "final_size": [FINAL_H, FINAL_W],
            "padded_size": [PAD_H, PAD_W],
            "proxy_decode_gamma": PROXY_DECODE_GAMMA,
            "background_rgb": bg.reshape(-1).tolist(),
            "splits": {},
        }

        for split_name, csv_name in split_to_csv.items():
            names = []
            for idx, example in enumerate(tqdm(dataset[split_name], desc=f"Export {split_name}")):
                base = f"im_{idx:05d}"
                manifest_name = f"{base}.jpg.tiff"

                lensless = np.asarray(example["lensless"]).astype(np.float32) / 255.0
                lensed = np.asarray(example["lensed"]).astype(np.float32) / 255.0

                lensless = resize_rgb(lensless, (FINAL_H, FINAL_W))
                lensed = resize_rgb(lensed, (FINAL_H, FINAL_W))

                lensless = np.clip(lensless - bg, 0.0, 1.0)
                lensless_pad = pad_bottom(lensless).astype(np.float32)
                lensed_pad = pad_bottom(lensed).astype(np.float32)

                decode_proxy = wiener_decode_rgb(fft_conv_rgb(lensed_pad))

                np.save(STAGE1_DATA_ROOT / "diffuser_images" / f"{base}.npy", lensless_pad)
                np.save(STAGE1_DATA_ROOT / "ground_truth_lensed" / f"{base}.npy", lensed_pad)
                Image.fromarray(to_uint8(decode_proxy)).save(
                    STAGE1_DATA_ROOT / "decode_sim_padding_png" / f"{base}.png"
                )
                names.append(manifest_name)

            (STAGE1_DATA_ROOT / csv_name).write_text("\\n".join(names) + "\\n", encoding="utf-8")
            metadata["splits"][split_name] = len(names)

        (STAGE1_DATA_ROOT / "export_metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(metadata, indent=2))
        """
    )
)

cells.append(
    md(
        """
        ## 2. Patch PhoCoLens Stage 1 for `68x120`

        The next cell rewrites the Stage 1 config, DiffuserCam dataset loader, FFT
        inversion modules, and validation script so they work with the padded
        `68x120` inputs produced above.
        """
    )
)

cells.append(
    code(
        """
        from pathlib import Path
        import textwrap

        def write_text(path, content):
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(textwrap.dedent(content).lstrip("\\n"), encoding="utf-8")
            print("wrote", path)

        svd_root = PHOCOLENS_ROOT / "SVDeconv"

        write_text(
            svd_root / "config.py",
            f'''
            from pathlib import Path
            import torch
            from types import SimpleNamespace

            HEIGHT = {PAD_H}
            WIDTH = {PAD_W}

            fft_args_dict = {{
                "psf_mat": Path("data/diffusercam67x120/psf.tiff"),
                "psf_height": HEIGHT,
                "psf_width": WIDTH,
                "psf_centre_x": HEIGHT // 2,
                "psf_centre_y": WIDTH // 2,
                "psf_crop_size_x": HEIGHT,
                "psf_crop_size_y": WIDTH,
                "meas_height": HEIGHT,
                "meas_width": WIDTH,
                "meas_centre_x": HEIGHT // 2,
                "meas_centre_y": WIDTH // 2,
                "meas_crop_size_x": HEIGHT,
                "meas_crop_size_y": WIDTH,
                "pad_meas_mode": "replicate",
                "image_height": HEIGHT,
                "image_width": WIDTH,
                "fft_gamma": {STAGE1_FFT_GAMMA},
                "fft_requires_grad": True,
                "fft_epochs": 0,
                "use_mask": False,
            }}

            def base_config():
                exp_name = "fft-svd-diffusercam67x120-proxy_decode_sim"
                multi = 9
                use_spatial_weight = True
                weight_update = True
                preprocess_with_unet = True
                decode_sim = True
                zero_conv = False
                load_raw = False
                locals().update(fft_args_dict)

                image_dir = Path("data/diffusercam67x120")
                output_dir = Path("output/diffusercam67x120") / exp_name
                ckpt_dir = Path("ckpts/diffusercam67x120") / exp_name
                run_dir = Path("runs/diffusercam67x120") / exp_name

                dataset_name = "diffusercam"
                shuffle = True
                train_gaussian_noise = 5e-3

                model = "UNet270480"
                batch_size = {STAGE1_BATCH_SIZE}
                num_threads = {STAGE1_NUM_WORKERS}

                num_epochs = {STAGE1_EPOCHS}
                fft_epochs = 0
                learning_rate = 1e-4
                fft_learning_rate = 3e-5
                beta_1 = 0.9
                beta_2 = 0.999
                lr_scheduler = "cosine"
                T_0 = 1
                T_mult = 2
                step_size = 2

                save_filename_G = "model.pth"
                save_filename_FFT = "FFT.pth"
                save_filename_latest_G = "model_latest.pth"
                save_filename_latest_FFT = "FFT_latest.pth"

                log_interval = 100
                save_ckpt_interval = 1000
                save_copy_every_epochs = 10

                pixelshuffle_ratio = 2
                grad_lambda = 0.0
                G_finetune_layers = []
                num_groups = 8

                lambda_adversarial = 0.0
                lambda_contextual = 0.0
                lambda_perception = 0.05
                lambda_image = 1.0
                lambda_l1 = 0.0

                resume = False
                finetune = False
                concat_input = False
                inference_mode = "latest"

                device = "cuda" if torch.cuda.is_available() else "cpu"
                distdataparallel = False
                val_train = False
                static_val_image = "im_00000"

            def infer_train():
                val_train = True

            named_config_ll = [infer_train]

            def initialise(ex):
                ex.config(base_config)
                for named_config in named_config_ll:
                    ex.named_config(named_config)
                return ex

            fft_args = SimpleNamespace(**fft_args_dict)
            '''
        )

        write_text(
            svd_root / "datasets" / "diffusercam.py",
            '''
            from pathlib import Path
            import numpy as np
            import torch
            from PIL import Image
            from torch.utils.data import Dataset

            def region_of_interest(x):
                return x

            def _to_tensor(image):
                image = np.asarray(image).astype(np.float32)
                if image.ndim == 2:
                    image = image[..., None]
                tensor = torch.from_numpy(image).permute(2, 0, 1).float()
                if tensor.max() > 1.0:
                    tensor = tensor / 255.0
                return tensor.mul(2.0).sub(1.0)

            def load_psf(path):
                image = np.asarray(Image.open(path)).astype(np.float32)
                if image.ndim == 2:
                    image = image[..., None]
                tensor = torch.from_numpy(image).permute(2, 0, 1).float()
                if tensor.max() > 1.0:
                    tensor = tensor / 255.0
                return tensor

            class LenslessLearning(Dataset):
                def __init__(self, diffuser_images, ground_truth_images):
                    self.xs = diffuser_images
                    self.ys = ground_truth_images

                def __len__(self):
                    return len(self.xs)

                def __getitem__(self, idx):
                    diffused = self.xs[idx]
                    ground_truth = self.ys[idx]
                    x = _to_tensor(np.load(diffused))
                    if ground_truth.suffix.lower() == ".png":
                        y = _to_tensor(np.asarray(Image.open(ground_truth)))
                    else:
                        y = _to_tensor(np.load(ground_truth))
                    return x, y, diffused.stem

            class LenslessLearningInTheWild(Dataset):
                def __init__(self, path):
                    self.xs = sorted(path.glob("*.npy"))

                def __len__(self):
                    return len(self.xs)

                def __getitem__(self, idx):
                    return _to_tensor(np.load(self.xs[idx]))

            def load_manifest(path, csv_filename, decode_sim=False):
                manifest = (path / csv_filename).read_text(encoding="utf-8").split()
                xs, ys = [], []
                for filename in manifest:
                    stem = filename.replace(".jpg.tiff", "")
                    xs.append(path / "diffuser_images" / f"{stem}.npy")
                    if decode_sim:
                        ys.append(path / "decode_sim_padding_png" / f"{stem}.png")
                    else:
                        ys.append(path / "ground_truth_lensed" / f"{stem}.npy")
                return xs, ys

            class LenslessLearningCollection:
                def __init__(self, args):
                    path = Path(args.image_dir)
                    self.psf = load_psf(path / "psf.tiff")
                    train_diffused, train_ground_truth = load_manifest(path, "dataset_train.csv", decode_sim=args.decode_sim)
                    val_diffused, val_ground_truth = load_manifest(path, "dataset_test.csv", decode_sim=args.decode_sim)
                    self.train_dataset = LenslessLearning(train_diffused, train_ground_truth)
                    self.val_dataset = LenslessLearning(val_diffused, val_ground_truth)
                    self.region_of_interest = region_of_interest
            '''
        )

        write_text(
            svd_root / "dataloader.py",
            '''
            from dataclasses import dataclass
            import importlib.util
            import logging
            from pathlib import Path

            import torch
            import torch.distributed as dist
            from sacred import Experiment
            from torch.utils.data import DataLoader

            from config import initialise

            ex = Experiment("data")
            ex = initialise(ex)

            _diffusercam_path = Path(__file__).resolve().parent / "datasets" / "diffusercam.py"
            _spec = importlib.util.spec_from_file_location("svdeconv_diffusercam_local", _diffusercam_path)
            _module = importlib.util.module_from_spec(_spec)
            assert _spec.loader is not None
            _spec.loader.exec_module(_module)
            LenslessLearningCollection = _module.LenslessLearningCollection

            @dataclass
            class Data:
                train_loader: DataLoader
                val_loader: DataLoader
                test_loader: DataLoader = None

            def get_dataloaders(args, is_local_rank_0: bool = True):
                if "diffusercam" not in args.dataset_name:
                    raise ValueError(f"Only diffusercam is supported by this notebook patch, got {args.dataset_name!r}")

                dataset = LenslessLearningCollection(args)
                train_dataset = dataset.train_dataset
                val_dataset = dataset.val_dataset
                test_dataset = dataset.val_dataset

                if is_local_rank_0:
                    logging.info(
                        f"Dataset: {args.dataset_name} Len Train: {len(train_dataset)} Val: {len(val_dataset)} Test: {len(test_dataset)}"
                    )

                def make_loader(ds, shuffle, sampler, pin_memory, drop_last=False):
                    return DataLoader(
                        ds,
                        batch_size=args.batch_size,
                        shuffle=shuffle,
                        num_workers=args.num_threads,
                        pin_memory=pin_memory,
                        drop_last=drop_last,
                        sampler=sampler,
                    )

                if args.distdataparallel:
                    train_sampler = torch.utils.data.distributed.DistributedSampler(
                        train_dataset, num_replicas=dist.get_world_size(), shuffle=True
                    )
                    val_sampler = torch.utils.data.distributed.DistributedSampler(
                        val_dataset, num_replicas=dist.get_world_size(), shuffle=False
                    )
                    test_sampler = torch.utils.data.distributed.DistributedSampler(
                        test_dataset, num_replicas=dist.get_world_size(), shuffle=False
                    )
                    train_shuffle = False
                    eval_shuffle = False
                else:
                    train_sampler = None
                    val_sampler = None
                    test_sampler = None
                    train_shuffle = True
                    eval_shuffle = False

                train_loader = make_loader(train_dataset, train_shuffle, train_sampler, pin_memory=True)
                val_loader = make_loader(val_dataset, eval_shuffle, val_sampler, pin_memory=True)
                test_loader = make_loader(test_dataset, eval_shuffle, test_sampler, pin_memory=False, drop_last=False)
                return Data(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)
            '''
        )
        """
    )
)

cells.append(
    md(
        """
        ## 3. Train Stage 1 (SVDeconv) and dump intermediate reconstructions

        This runs the strongest PhoCoLens Stage 1 branch available in the public
        repository: SVD-based multi-Wiener inversion with the U-Net preprocessor and
        spatial weighting.
        """
    )
)

cells.append(
    code(
        """
        STAGE1_EXP_NAME = "fft-svd-diffusercam67x120-proxy_decode_sim"

        run([sys.executable, "train.py", "-p"], cwd=PHOCOLENS_ROOT / "SVDeconv")
        run([sys.executable, "val.py", "-p"], cwd=PHOCOLENS_ROOT / "SVDeconv")
        run([sys.executable, "val.py", "with", "infer_train", "-p"], cwd=PHOCOLENS_ROOT / "SVDeconv")

        stage1_output_root = PHOCOLENS_ROOT / "SVDeconv" / "output" / "diffusercam67x120" / STAGE1_EXP_NAME
        print("Stage 1 outputs:", stage1_output_root)
        """
    )
)

cells.append(
    md(
        """
        ## 4. Build Stage 2 (`NullSpaceDiff`) training pairs

        The official Stage 2 expects `512x512` paired PNGs. Since your images are
        non-square, we preserve aspect ratio with letterboxing instead of center
        cropping. The same padding is removed again during final evaluation.
        """
    )
)

cells.append(
    code(
        """
        from pathlib import Path

        import numpy as np
        from PIL import Image
        from tqdm.auto import tqdm

        def letterbox_to_square(image_hwc, size=LETTERBOX_SIZE):
            image = Image.fromarray(np.clip(np.round(image_hwc * 255.0), 0, 255).astype(np.uint8))
            src_w, src_h = image.size
            scale = min(size / src_h, size / src_w)
            new_h = max(1, int(round(src_h * scale)))
            new_w = max(1, int(round(src_w * scale)))
            image = image.resize((new_w, new_h), resample=Image.Resampling.BICUBIC)
            canvas = Image.new("RGB", (size, size))
            top = (size - new_h) // 2
            left = (size - new_w) // 2
            canvas.paste(image, (left, top))
            return np.asarray(canvas).astype(np.uint8), {"top": top, "left": left, "height": new_h, "width": new_w}

        def load_stage1_png(path):
            return np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0

        if STAGE2_DATA_ROOT.exists():
            shutil.rmtree(STAGE2_DATA_ROOT)

        mapping = {}
        stage1_output_root = PHOCOLENS_ROOT / "SVDeconv" / "output" / "diffusercam67x120" / STAGE1_EXP_NAME
        for split in ["train", "val"]:
            input_dir = STAGE2_DATA_ROOT / split / "inputs_512"
            gt_dir = STAGE2_DATA_ROOT / split / "gts_512"
            input_dir.mkdir(parents=True, exist_ok=True)
            gt_dir.mkdir(parents=True, exist_ok=True)

            stage1_dir = stage1_output_root / split / "outputs"
            stems = sorted(p.stem for p in stage1_dir.glob("*.png"))
            print(split, "samples", len(stems))

            for stem in tqdm(stems, desc=f"Stage 2 pairs: {split}"):
                stage1_pred = load_stage1_png(stage1_dir / f"{stem}.png")
                gt = np.load(STAGE1_DATA_ROOT / "ground_truth_lensed" / f"{stem}.npy").astype(np.float32)

                stage1_sq, meta = letterbox_to_square(stage1_pred, LETTERBOX_SIZE)
                gt_sq, _ = letterbox_to_square(gt, LETTERBOX_SIZE)

                Image.fromarray(stage1_sq).save(input_dir / f"{stem}.png")
                Image.fromarray(gt_sq).save(gt_dir / f"{stem}.png")
                mapping[stem] = meta

        (ARTIFACT_ROOT / "letterbox_meta.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
        print("Stage 2 data root:", STAGE2_DATA_ROOT)
        """
    )
)

cells.append(
    md(
        """
        ## 5. Download StableSR weights and write the Stage 2 config
        """
    )
)

cells.append(
    code(
        """
        from huggingface_hub import hf_hub_download

        stable_sr_ckpt = Path(
            hf_hub_download(
                repo_id="Iceclear/StableSR",
                filename="stablesr_000117.ckpt",
                local_dir=PHOCOLENS_ROOT / "ckpts",
            )
        )
        stage2_config_path = PHOCOLENS_ROOT / "NullSpaceDiff" / "configs" / "NullSpaceDiff" / f"{RUN_NAME}.yaml"

        stage2_config_text = f'''
        sf: 4
        model:
          base_learning_rate: 5.0e-05
          target: ldm.models.diffusion.ddpm.LatentDiffusionSRTextWTFFHQ
          params:
            parameterization: "v"
            linear_start: 0.00085
            linear_end: 0.0120
            num_timesteps_cond: 1
            log_every_t: 200
            timesteps: 1000
            first_stage_key: image
            cond_stage_key: caption
            image_size: 512
            channels: 4
            cond_stage_trainable: False
            conditioning_key: crossattn
            monitor: val/loss_simple_ema
            scale_factor: 0.18215
            use_ema: False
            ckpt_path: {stable_sr_ckpt}
            unfrozen_diff: False
            random_size: False
            time_replace: 1000
            use_usm: False
            p2_gamma: ~
            p2_k: ~
            unet_config:
              target: ldm.modules.diffusionmodules.openaimodel.UNetModelDualcondV2
              params:
                image_size: 32
                in_channels: 4
                out_channels: 4
                model_channels: 320
                attention_resolutions: [4, 2, 1]
                num_res_blocks: 2
                channel_mult: [1, 2, 4, 4]
                num_head_channels: 64
                use_spatial_transformer: True
                use_linear_in_transformer: True
                transformer_depth: 1
                context_dim: 1024
                use_checkpoint: False
                legacy: False
                semb_channels: 256
            first_stage_config:
              target: ldm.models.autoencoder.AutoencoderKL
              params:
                ckpt_path: {stable_sr_ckpt}
                embed_dim: 4
                monitor: val/rec_loss
                ddconfig:
                  double_z: true
                  z_channels: 4
                  resolution: 512
                  in_channels: 3
                  out_ch: 3
                  ch: 128
                  ch_mult: [1, 2, 4, 4]
                  num_res_blocks: 2
                  attn_resolutions: []
                  dropout: 0.0
                lossconfig:
                  target: torch.nn.Identity
            cond_stage_config:
              target: ldm.modules.encoders.modules.FrozenOpenCLIPEmbedder
              params:
                freeze: True
                layer: "penultimate"
            structcond_stage_config:
              target: ldm.modules.diffusionmodules.openaimodel.EncoderUNetModelWT
              params:
                image_size: 96
                in_channels: 4
                model_channels: 256
                out_channels: 256
                num_res_blocks: 2
                attention_resolutions: [4, 2, 1]
                dropout: 0
                channel_mult: [1, 1, 2, 2]
                conv_resample: True
                dims: 2
                use_checkpoint: False
                use_fp16: False
                num_heads: 4
                num_head_channels: -1
                num_heads_upsample: -1
                use_scale_shift_norm: False
                resblock_updown: False
                use_new_attention_order: False
        data:
          target: main.DataModuleFromConfig
          params:
            batch_size: {STAGE2_BATCH_SIZE}
            num_workers: {STAGE2_NUM_WORKERS}
            wrap: false
            train:
              target: basicsr.data.paired_image_dataset.PairedImageDataset
              params:
                dataroot_gt: {STAGE2_DATA_ROOT / "train" / "gts_512"}
                dataroot_lq: {STAGE2_DATA_ROOT / "train" / "inputs_512"}
                io_backend:
                  type: disk
                phase: train
                gt_size: 512
                scale: 1
                use_rot: true
                use_hflip: true
            validation:
              target: basicsr.data.paired_image_dataset.PairedImageDataset
              params:
                dataroot_gt: {STAGE2_DATA_ROOT / "val" / "gts_512"}
                dataroot_lq: {STAGE2_DATA_ROOT / "val" / "inputs_512"}
                io_backend:
                  type: disk
                phase: val
                gt_size: 512
                scale: 1
                use_rot: false
                use_hflip: false
        lightning:
          logger:
            target: pytorch_lightning.loggers.TestTubeLogger
            params:
              name: testtube
              save_dir: ./logs
          modelcheckpoint:
            params:
              every_n_train_steps: 500
          callbacks:
            image_logger:
              target: main.ImageLogger
              params:
                batch_frequency: 1000
                max_images: 4
                log_on_batch_idx: True
                increase_log_steps: False
          trainer:
            benchmark: True
            max_steps: {STAGE2_MAX_STEPS}
            accumulate_grad_batches: 1
            val_check_interval: {STAGE2_VAL_INTERVAL}
        '''

        stage2_config_path.parent.mkdir(parents=True, exist_ok=True)
        stage2_config_path.write_text(textwrap.dedent(stage2_config_text).lstrip("\\n"), encoding="utf-8")
        print("StableSR ckpt:", stable_sr_ckpt)
        print("Stage 2 config:", stage2_config_path)
        """
    )
)

cells.append(
    md(
        """
        ## 6. Train and run Stage 2
        """
    )
)

cells.append(
    code(
        """
        run(
            [
                sys.executable,
                "main.py",
                "--train",
                "True",
                "--no-test",
                "True",
                "--base",
                str(stage2_config_path),
                "--gpus",
                "0,",
                "--name",
                RUN_NAME,
                "--scale_lr",
                "False",
                "--seed",
                str(SEED),
            ],
            cwd=PHOCOLENS_ROOT / "NullSpaceDiff",
        )

        log_root = PHOCOLENS_ROOT / "NullSpaceDiff" / "logs"
        stage2_runs = sorted(log_root.glob(f"*_{RUN_NAME}"), key=lambda p: p.stat().st_mtime)
        assert stage2_runs, f"No Stage 2 run found in {log_root}"
        stage2_logdir = stage2_runs[-1]
        stage2_ckpt = stage2_logdir / "checkpoints" / "last.ckpt"
        assert stage2_ckpt.exists(), stage2_ckpt

        stage2_outdir = ARTIFACT_ROOT / "phocolens_stage2_val_512"
        stage2_outdir.mkdir(parents=True, exist_ok=True)
        run(
            [
                sys.executable,
                "scripts/sr_val_ddpm_lensless.py",
                "--init-img",
                str(STAGE2_DATA_ROOT / "val" / "inputs_512"),
                "--outdir",
                str(stage2_outdir),
                "--config",
                str(stage2_config_path),
                "--ckpt",
                str(stage2_ckpt),
                "--ddpm_steps",
                str(STAGE2_DDPM_STEPS),
                "--n_samples",
                "1",
                "--input_size",
                str(LETTERBOX_SIZE),
                "--gpu_id",
                "0",
                "--gpu_num",
                "1",
                "--colorfix_type",
                "nofix",
            ],
            cwd=PHOCOLENS_ROOT / "NullSpaceDiff",
        )

        print("Stage 2 logdir:", stage2_logdir)
        print("Stage 2 ckpt:", stage2_ckpt)
        print("Stage 2 outputs:", stage2_outdir)
        """
    )
)

cells.append(
    md(
        """
        ## 7. Evaluate PhoCoLens on the validation split

        Metrics are computed after removing the `512x512` letterbox padding and
        cropping from `68x120` back to `67x120`.
        """
    )
)

cells.append(
    code(
        """
        import json
        import math

        import lpips
        import numpy as np
        import torch
        from PIL import Image
        from skimage.metrics import structural_similarity as ssim
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchvision.transforms.functional import resize
        from tqdm.auto import tqdm

        device = "cuda" if torch.cuda.is_available() else "cpu"
        lpips_fn = lpips.LPIPS(net="alex").to(device).eval()
        letterbox_meta = json.loads((ARTIFACT_ROOT / "letterbox_meta.json").read_text(encoding="utf-8"))

        def resize_rgb_float(image_hwc, size_hw):
            tensor = torch.from_numpy(image_hwc.astype(np.float32)).permute(2, 0, 1)
            tensor = resize(tensor, list(size_hw), antialias=True)
            return tensor.permute(1, 2, 0).numpy()

        def load_gt(stem):
            gt = np.load(STAGE1_DATA_ROOT / "ground_truth_lensed" / f"{stem}.npy").astype(np.float32)
            return np.clip(gt[:FINAL_H], 0.0, 1.0)

        def load_stage1_pred(stem):
            pred = np.asarray(Image.open(stage1_output_root / "val" / "outputs" / f"{stem}.png").convert("RGB")).astype(np.float32) / 255.0
            if pred.shape[:2] != (PAD_H, PAD_W):
                pred = resize_rgb_float(pred, (PAD_H, PAD_W))
            return np.clip(pred[:FINAL_H], 0.0, 1.0)

        def load_stage2_pred(stem):
            pred = np.asarray(Image.open(stage2_outdir / f"{stem}.png").convert("RGB")).astype(np.float32) / 255.0
            meta = letterbox_meta[stem]
            top = meta["top"]
            left = meta["left"]
            height = meta["height"]
            width = meta["width"]
            pred = pred[top:top + height, left:left + width]
            pred = resize_rgb_float(pred, (PAD_H, PAD_W))
            return np.clip(pred[:FINAL_H], 0.0, 1.0)

        def to_lpips_tensor(image_hwc):
            return torch.from_numpy(image_hwc).permute(2, 0, 1).unsqueeze(0).to(device).mul(2.0).sub(1.0)

        def to_fid_tensor(image_hwc):
            return torch.from_numpy(np.clip(np.round(image_hwc * 255.0), 0, 255).astype(np.uint8)).permute(2, 0, 1).unsqueeze(0).to(device)

        def evaluate_loader(loader, stems):
            fid = FrechetInceptionDistance(feature=2048).to(device)
            totals = {"PSNR": 0.0, "SSIM": 0.0, "LPIPS": 0.0, "MSE": 0.0}
            for stem in tqdm(stems, desc="Evaluate"):
                pred = loader(stem)
                gt = load_gt(stem)
                mse = float(np.mean((pred - gt) ** 2))
                psnr = 10.0 * math.log10(1.0 / max(mse, 1.0e-12))
                ssim_val = float(ssim(gt, pred, channel_axis=-1, data_range=1.0))
                lpips_val = float(lpips_fn(to_lpips_tensor(pred), to_lpips_tensor(gt)).mean().item())

                totals["PSNR"] += psnr
                totals["SSIM"] += ssim_val
                totals["LPIPS"] += lpips_val
                totals["MSE"] += mse

                fid.update(to_fid_tensor(gt), real=True)
                fid.update(to_fid_tensor(pred), real=False)

            n = max(len(stems), 1)
            return {
                "count": len(stems),
                "PSNR": totals["PSNR"] / n,
                "SSIM": totals["SSIM"] / n,
                "LPIPS": totals["LPIPS"] / n,
                "MSE": totals["MSE"] / n,
                "FID": float(fid.compute().item()),
            }

        val_stems = sorted(p.stem for p in (stage1_output_root / "val" / "outputs").glob("*.png"))
        summary = {
            "stage1": evaluate_loader(load_stage1_pred, val_stems),
            "stage2": evaluate_loader(load_stage2_pred, val_stems),
        }

        summary_path = ARTIFACT_ROOT / "phocolens_eval_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        print("Saved summary to", summary_path)
        """
    )
)

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.x",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = Path("notebooks") / "phocolens_diffusercam_colab.ipynb"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
print(out_path.resolve())

cells.append(
    code(
        """
        write_text(
            svd_root / "models" / "multi_fftlayer_diff.py",
            '''
            import copy
            import numpy as np
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            from PIL import Image

            from models.unet import UNet270480
            from utils.ops import roll_n, unpixel_shuffle

            def load_psf(path):
                image = np.asarray(Image.open(path)).astype(np.float32)
                if image.ndim == 2:
                    image = image[..., None]
                tensor = torch.from_numpy(image).permute(2, 0, 1).float()
                if tensor.max() > 1.0:
                    tensor = tensor / 255.0
                return tensor.mean(0, keepdim=True)

            def fft_conv2d(input, kernel):
                return torch.fft.ifft2(torch.fft.fft2(input) * torch.fft.fft2(kernel)).real

            def get_wiener_matrix(psf, gamma=100.0, centre_roll=True):
                if centre_roll:
                    for dim in range(1, 3):
                        psf = roll_n(psf, axis=dim, n=psf.shape[dim] // 2)
                H = torch.fft.fft2(psf)
                W = torch.conj(H) / (H.abs().square() + gamma)
                return torch.fft.ifft2(W).real

            def zero_module(module):
                for p in module.parameters():
                    p.detach().zero_()
                return module

            def generate_vertices(k, h, w):
                grid = int(round(k ** 0.5))
                ys = torch.linspace(h / (2 * grid), h - h / (2 * grid), steps=grid)
                xs = torch.linspace(w / (2 * grid), w - w / (2 * grid), steps=grid)
                pts = torch.tensor([(float(y), float(x)) for y in ys for x in xs])
                return pts[:k]

            class SpatialVaryWeight(nn.Module):
                def __init__(self, args):
                    super().__init__()
                    self.multi = args.multi
                    self.height = args.image_height
                    self.width = args.image_width
                    self.weight = nn.Parameter(torch.rand(self.multi, self.height, self.width), requires_grad=args.weight_update)
                    self.init_weight()

                def init_weight(self):
                    vertices = generate_vertices(self.multi, self.height, self.width)
                    y_grid, x_grid = torch.meshgrid(torch.arange(self.height), torch.arange(self.width), indexing="ij")
                    grid = torch.stack((y_grid, x_grid), dim=-1).float()
                    dists = torch.sqrt(((grid.unsqueeze(0) - vertices.unsqueeze(1).unsqueeze(1)) ** 2).sum(dim=-1))
                    weights = 1.0 / (dists + 1.0e-6)
                    self.weight.data = weights / weights.sum(dim=0, keepdim=True)

                def forward(self, img):
                    weight = F.softmax(self.weight, dim=0).unsqueeze(0).unsqueeze(2)
                    return (img * weight).sum(dim=1)

            class MultiFFTLayer_diff(nn.Module):
                def __init__(self, args):
                    super().__init__()
                    self.args = args
                    requires_grad = not (args.fft_epochs == args.num_epochs)
                    psf = load_psf(args.psf_mat)
                    self.multi = args.multi + 1
                    self.zero_conv = args.zero_conv
                    self.preprocess_with_unet = args.preprocess_with_unet

                    top = args.psf_centre_x - args.psf_crop_size_x // 2
                    bottom = args.psf_centre_x + args.psf_crop_size_x // 2
                    left = args.psf_centre_y - args.psf_crop_size_y // 2
                    right = args.psf_centre_y + args.psf_crop_size_y // 2
                    psf_crop = psf[:, top:bottom, left:right]

                    _, self.psf_height, self.psf_width = psf_crop.shape
                    if self.zero_conv:
                        self.psf_crop = nn.Parameter(psf_crop.unsqueeze(0), requires_grad=requires_grad)
                        self.zero_res_conv = zero_module(nn.Conv2d(1, self.multi, 1))
                    else:
                        self.psf_crop = nn.Parameter(psf_crop.repeat(self.multi, 1, 1, 1), requires_grad=requires_grad)

                    self.gamma = nn.Parameter(torch.tensor([args.fft_gamma] * self.multi, dtype=torch.float32), requires_grad=requires_grad)
                    self.normalizer = nn.Parameter(torch.tensor([1 / 0.0008]).reshape(1, 1, 1, 1).repeat(self.multi, 1, 1, 1), requires_grad=requires_grad)

                    unet_args = copy.deepcopy(args)
                    unet_args.pixelshuffle_ratio = 2
                    self.unet = UNet270480(unet_args, in_c=3)
                    self.spatial_weight = SpatialVaryWeight(args) if args.use_spatial_weight else None

                def _maybe_pad(self, tensor, pad_x, pad_y):
                    if pad_x == 0 and pad_y == 0:
                        return tensor
                    return F.pad(tensor, (pad_y // 2, pad_y - pad_y // 2, pad_x // 2, pad_x - pad_x // 2), mode="replicate")

                def forward(self, img):
                    pad_x = self.args.psf_height - self.args.psf_crop_size_x
                    pad_y = self.args.psf_width - self.args.psf_crop_size_y

                    h, w = img.shape[2], img.shape[3]
                    img = img[:, :, (h - self.psf_height) // 2:(h + self.psf_height) // 2, (w - self.psf_width) // 2:(w + self.psf_width) // 2]
                    img = self._maybe_pad(img, pad_x, pad_y)

                    if self.preprocess_with_unet:
                        img = unpixel_shuffle(img, 2)
                        img = self.unet(img)
                        img = F.pixel_shuffle(img, 2)

                    if self.zero_conv:
                        zero_res = self.zero_res_conv(self.psf_crop).view(self.multi, 1, self.psf_height, self.psf_width)
                        psf_crop = self.psf_crop + zero_res
                    else:
                        psf_crop = self.psf_crop

                    fft_layers = []
                    for i in range(self.multi):
                        fft_layer = get_wiener_matrix(psf_crop[i], gamma=self.gamma[i], centre_roll=False)
                        fft_layer = self._maybe_pad(fft_layer, pad_x, pad_y)
                        for dim in range(1, 3):
                            fft_layer = roll_n(fft_layer, axis=dim, n=fft_layer.size(dim) // 2)
                        fft_layers.append(fft_layer.unsqueeze(0))

                    _, _, fft_h, fft_w = fft_layers[0].shape
                    img_h = self.args.image_height
                    img_w = self.args.image_width
                    imgs = []
                    for i, fft_layer in enumerate(fft_layers):
                        out = fft_conv2d(img, fft_layer) * self.normalizer[i]
                        out = out[:, :, fft_h // 2 - img_h // 2: fft_h // 2 + img_h // 2, fft_w // 2 - img_w // 2: fft_w // 2 + img_w // 2]
                        imgs.append(out)

                    if self.spatial_weight is not None:
                        img_0 = imgs[0]
                        img_1 = self.spatial_weight(torch.stack(imgs[1:], dim=1))
                        return torch.cat([img_0, img_1], dim=1)
                    return torch.cat(imgs, dim=1)
            '''
        )

        write_text(
            svd_root / "models" / "fftlayer_diff.py",
            '''
            import numpy as np
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            from PIL import Image

            from utils.ops import roll_n

            def load_psf(path):
                image = np.asarray(Image.open(path)).astype(np.float32)
                if image.ndim == 2:
                    image = image[..., None]
                tensor = torch.from_numpy(image).permute(2, 0, 1).float()
                if tensor.max() > 1.0:
                    tensor = tensor / 255.0
                return tensor

            def fft_conv2d(input, kernel):
                return torch.fft.ifft2(torch.fft.fft2(input) * torch.fft.fft2(kernel)).real

            def get_wiener_matrix(psf, gamma=100.0, centre_roll=True):
                if centre_roll:
                    for dim in range(1, 3):
                        psf = roll_n(psf, axis=dim, n=psf.shape[dim] // 2)
                H = torch.fft.fft2(psf)
                W = torch.conj(H) / (H.abs().square() + gamma)
                return torch.fft.ifft2(W).real

            class FFTLayer_diff(nn.Module):
                def __init__(self, args):
                    super().__init__()
                    self.args = args
                    psf = load_psf(args.psf_mat)
                    top = args.psf_centre_x - args.psf_crop_size_x // 2
                    bottom = args.psf_centre_x + args.psf_crop_size_x // 2
                    left = args.psf_centre_y - args.psf_crop_size_y // 2
                    right = args.psf_centre_y + args.psf_crop_size_y // 2
                    psf_crop = psf[:, top:bottom, left:right]
                    _, self.psf_height, self.psf_width = psf_crop.shape
                    self.wiener_crop = nn.Parameter(get_wiener_matrix(psf_crop, gamma=args.fft_gamma, centre_roll=False), requires_grad=True)
                    self.normalizer = nn.Parameter(torch.tensor([1 / 0.0008]).reshape(1, 1, 1, 1), requires_grad=True)

                def _maybe_pad(self, tensor, pad_x, pad_y):
                    if pad_x == 0 and pad_y == 0:
                        return tensor
                    return F.pad(tensor, (pad_y // 2, pad_y - pad_y // 2, pad_x // 2, pad_x - pad_x // 2), mode="replicate")

                def forward(self, img):
                    pad_x = self.args.psf_height - self.args.psf_crop_size_x
                    pad_y = self.args.psf_width - self.args.psf_crop_size_y
                    fft_layer = self._maybe_pad(self.wiener_crop, pad_x, pad_y)
                    for dim in range(1, 3):
                        fft_layer = roll_n(fft_layer, axis=dim, n=fft_layer.size(dim) // 2)
                    fft_layer = fft_layer.unsqueeze(0)

                    _, _, fft_h, fft_w = fft_layer.shape
                    img_h = self.args.image_height
                    img_w = self.args.image_width

                    h, w = img.shape[2], img.shape[3]
                    img = img[:, :, (h - self.psf_height) // 2:(h + self.psf_height) // 2, (w - self.psf_width) // 2:(w + self.psf_width) // 2]
                    img = self._maybe_pad(img, pad_x, pad_y)
                    out = fft_conv2d(img, fft_layer) * self.normalizer
                    return out[:, :, fft_h // 2 - img_h // 2: fft_h // 2 + img_h // 2, fft_w // 2 - img_w // 2: fft_w // 2 + img_w // 2]
            '''
        )

        write_text(
            svd_root / "val.py",
            '''
            from collections import defaultdict
            from pathlib import Path
            import json
            import time

            import cv2
            import lpips
            import numpy as np
            import torch
            import torch.nn.functional as F
            from sacred import Experiment
            from skimage.metrics import structural_similarity as ssim
            from tqdm import tqdm

            from config import initialise
            from dataloader import get_dataloaders
            from metrics import PSNR
            from models import get_model
            from utils.ops import unpixel_shuffle
            from utils.train_helper import AvgLoss_with_dict, load_models
            from utils.tupperware import tupperware

            ex = Experiment("val")
            ex = initialise(ex)
            torch.multiprocessing.set_sharing_strategy("file_system")

            @ex.automain
            def main(_run):
                args = tupperware(_run.config)
                args.batch_size = 1
                args.resume = True
                device = args.device

                data = get_dataloaders(args)
                if args.val_train:
                    data.val_loader = data.train_loader

                G, FFT = get_model.model(args)
                G = G.to(device)
                FFT = FFT.to(device)
                (G, FFT), _, global_step, start_epoch, _ = load_models(G, FFT, None, None, args)

                lpips_criterion = lpips.LPIPS(net="alex").to(device)
                avg_metrics = AvgLoss_with_dict(
                    loss_dict={"PSNR": 0.0, "LPIPS": 0.0, "SSIM": 0.0, "Time": 0.0},
                    args=args,
                )
                avg_metrics.reset()

                split_name = "train" if args.val_train else "val"
                val_path = args.output_dir / split_name
                output_path = val_path / "outputs"
                fft_path = val_path / "fft"
                output_path.mkdir(parents=True, exist_ok=True)
                fft_path.mkdir(parents=True, exist_ok=True)

                pbar = tqdm(range(len(data.val_loader) * args.batch_size), dynamic_ncols=True)

                with torch.no_grad():
                    G.eval()
                    FFT.eval()
                    for _, batch in enumerate(data.val_loader):
                        metrics_dict = defaultdict(float)
                        source, target, filename = batch
                        source, target = source.to(device), target.to(device)

                        start_time = time.time()
                        fft_output = FFT(source)
                        output_unpixel_shuffled = G(unpixel_shuffle(fft_output, args.pixelshuffle_ratio))
                        output = F.pixel_shuffle(output_unpixel_shuffled, args.pixelshuffle_ratio)
                        metrics_dict["Time"] = time.time() - start_time
                        metrics_dict["PSNR"] = PSNR(output, target).item()
                        metrics_dict["LPIPS"] = lpips_criterion(output, target).mean().item()

                        output_np = output[0].mul(0.5).add(0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                        target_np = target[0].mul(0.5).add(0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                        fft_np = fft_output[0][:3].mul(0.5).add(0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                        metrics_dict["SSIM"] = ssim(target_np, output_np, channel_axis=-1, data_range=1.0)

                        stem = str(filename[0])
                        cv2.imwrite(str(output_path / f"{stem}.png"), (output_np[:, :, ::-1] * 255.0).astype(np.uint8))
                        cv2.imwrite(str(fft_path / f"{stem}.png"), (fft_np[:, :, ::-1] * 255.0).astype(np.uint8))

                        avg_metrics += metrics_dict
                        pbar.update(args.batch_size)
                        pbar.set_description(
                            f"{split_name}: PSNR={avg_metrics.loss_dict['PSNR']:.3f} SSIM={avg_metrics.loss_dict['SSIM']:.3f} LPIPS={avg_metrics.loss_dict['LPIPS']:.3f}"
                        )

                summary = {
                    "exp_name": args.exp_name,
                    "split": split_name,
                    "epoch": int(start_epoch),
                    "global_step": int(global_step),
                    "metrics": avg_metrics.loss_dict,
                }
                (val_path / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
                print(json.dumps(summary, indent=2))
            '''
        )
        """
    )
)
