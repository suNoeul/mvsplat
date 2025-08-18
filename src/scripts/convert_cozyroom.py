import argparse, json
from pathlib import Path
import numpy as np
import torch
from jaxtyping import UInt8
from torch import Tensor
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, help="Input cozyroom colmap directory")
parser.add_argument("--output_dir", type=str, help="Output directory for the processed dataset")
args = parser.parse_args()

INPUT_DIR = Path(args.input_dir)
OUTPUT_DIR = Path(args.output_dir)


def load_cozyroom_cameras(colmap_dir: Path) -> tuple[dict, dict]:
    poses_arr = np.load(colmap_dir / "poses_bounds.npy")
    poses = poses_arr[:, :-2].reshape([-1, 3, 5])
    intrinsics, extrinsics = {}, {}
    for i in range(poses.shape[0]):
        h, w, f = poses[i, :, -1]
        K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float32)
        pose = np.concatenate([poses[i, :, :4], np.array([[0, 0, 0, 1]])], axis=0)
        pose[0:3, 1:3] *= -1
        w2c = np.linalg.inv(pose)
        intrinsics[i], extrinsics[i] = K, w2c
    return intrinsics, extrinsics

def load_raw(path: Path) -> UInt8[Tensor, " length"]:
    return torch.tensor(np.memmap(path, dtype="uint8", mode="r"))

def load_metadata(intrinsics, extrinsics) -> dict:
    timestamps, cameras = [], []
    for vid, intr in intrinsics.items():
        timestamps.append(int(vid))
        fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]
        camera = [fx / (2*cx), fy / (2*cy), 0.5, 0.5, 0.0, 0.0]
        camera.extend(extrinsics[vid][:3].flatten().tolist())
        cameras.append(np.array(camera))
    return {
        "url": "",
        "timestamps": torch.tensor(timestamps, dtype=torch.int64),
        "cameras": torch.tensor(np.stack(cameras), dtype=torch.float32),
    }

if __name__ == "__main__":
    all_intrinsics, all_extrinsics = load_cozyroom_cameras(INPUT_DIR)
    num_total_images = len(all_intrinsics)

    # Deblurred 이미지(29장)의 ID 목록만 생성
    deblurred_ids = [i for i in range(num_total_images) if i % 8 != 0]
    print(f"Processing {len(deblurred_ids)} deblurred images for the test set.")

    # Deblurred 이미지에 해당하는 데이터만 필터링
    intrinsics = {i: all_intrinsics[i] for i in deblurred_ids}
    extrinsics = {i: all_extrinsics[i] for i in deblurred_ids}
    images_dict = {
        i: load_raw(INPUT_DIR / "images" / f"{i:05d}.png")
        for i in tqdm(deblurred_ids, desc="Loading deblurred images")
    }

    # MVSPlat 형식으로 데이터 패키징
    metadata = load_metadata(intrinsics, extrinsics)
    example = {
        "key": "cozyroom_scene",
        "url": "",
        "timestamps": metadata["timestamps"],
        "cameras": metadata["cameras"],
        "images": [images_dict[ts.item()] for ts in metadata["timestamps"]],
    }

    # 'test' 폴더만 생성하고 그 안에 .torch 파일 저장
    test_dir = OUTPUT_DIR / "test"
    test_dir.mkdir(exist_ok=True, parents=True)
    torch.save([example], test_dir / "000000.torch")
    print(f"Saved 29 deblurred images to {test_dir / '000000.torch'}")

    # index.json에는 평가 대상인 Deblurred 이미지(29개)의 키만 저장
    test_index = {f"cozyroom_view_{i:03d}": "000000.torch" for i in deblurred_ids}
    with (test_dir / "index.json").open("w") as f:
        json.dump(test_index, f)
    print(f"Saved test index with {len(test_index)} keys.")
    print("\nConversion for reconstruction test complete! ✨")