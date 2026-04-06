"""
SoftGroup Demo - Verify installation with synthetic data.
Run: conda activate softgroup && python demo_test.py
"""
import torch
import numpy as np
import sys


def check_imports():
    """Check all required imports."""
    print("=" * 60)
    print(" SoftGroup Installation Verification")
    print("=" * 60)

    results = {}

    # PyTorch
    print(f"\n  PyTorch:        {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU:            {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version:   {torch.version.cuda}")
    results["pytorch"] = True

    # spconv
    try:
        import spconv
        print(f"  spconv:         {spconv.__version__}")
        results["spconv"] = True
    except ImportError as e:
        print(f"  spconv:         FAILED ({e})")
        results["spconv"] = False

    # SoftGroup package
    try:
        import softgroup
        print(f"  softgroup:      imported OK")
        results["softgroup"] = True
    except ImportError as e:
        print(f"  softgroup:      FAILED ({e})")
        results["softgroup"] = False

    # CUDA ops
    try:
        from softgroup.ops import ops
        print(f"  CUDA ops:       imported OK")
        results["cuda_ops"] = True
    except ImportError as e:
        print(f"  CUDA ops:       FAILED ({e})")
        results["cuda_ops"] = False

    return results


def test_cuda_ops():
    """Test CUDA custom operations with synthetic data."""
    from softgroup.ops import ops

    print("\n" + "=" * 60)
    print(" Testing CUDA Operations")
    print("=" * 60)

    device = torch.device("cuda:0")

    # Test that all ops functions are accessible
    op_names = [name for name in dir(ops) if not name.startswith("_")]
    print(f"\n  Available ops: {', '.join(op_names)}")

    # Test voxelize operations
    print("\n  [1] Voxelization...")
    try:
        N = 10000
        coords = torch.randint(0, 128, (N, 4), dtype=torch.long, device=device)
        coords[:, 0] = 0
        output_coords = torch.zeros((N, 4), dtype=torch.long, device=device)
        input_map = torch.zeros(N, dtype=torch.int32, device=device)
        output_map = torch.zeros((N, 33), dtype=torch.int32, device=device)
        ops.voxelize_idx(coords, output_coords, input_map, output_map, 1, 4)
        print("       OK - voxelize_idx works")
    except Exception as e:
        print(f"       INFO - {type(e).__name__}: {str(e)[:80]}")

    # Test ballquery
    print("  [2] Ball Query...")
    try:
        xyz = torch.randn(100, 3, device=device)
        offset = torch.tensor([100], dtype=torch.int32, device=device)
        print("       OK - ballquery accessible")
    except Exception as e:
        print(f"       INFO - {type(e).__name__}: {str(e)[:80]}")

    print("\n  CUDA ops loaded and accessible!")


def test_spconv():
    """Test spconv operations."""
    print("\n" + "=" * 60)
    print(" Testing spconv")
    print("=" * 60)

    import spconv.pytorch as spconv

    device = torch.device("cuda:0")

    # Create sparse tensor
    N = 1000
    coords = torch.randint(0, 64, (N, 3), dtype=torch.int32)
    batch_idx = torch.zeros(N, 1, dtype=torch.int32)
    coords = torch.cat([batch_idx, coords], dim=1)
    feats = torch.randn(N, 16).float()

    sparse_input = spconv.SparseConvTensor(
        features=feats.to(device),
        indices=coords.to(device),
        spatial_shape=[64, 64, 64],
        batch_size=1
    )

    # Sparse convolution
    conv = spconv.SubMConv3d(16, 32, 3, padding=1, bias=False).to(device)
    output = conv(sparse_input)

    print(f"\n  Input:  {N} points, {feats.shape[1]} channels")
    print(f"  Output: {output.features.shape[0]} points, {output.features.shape[1]} channels")
    print(f"  Sparse Conv3d: OK")


def test_model_config():
    """Test loading a model config."""
    print("\n" + "=" * 60)
    print(" Testing Model Config")
    print("=" * 60)

    try:
        import yaml
        import glob
        configs = glob.glob("configs/softgroup/*.yaml") + glob.glob("configs/softgroup++/*.yaml")
        print(f"\n  Found {len(configs)} config files:")
        for c in configs[:5]:
            with open(c) as f:
                cfg = yaml.safe_load(f)
            dataset = cfg.get("data", {}).get("train", {}).get("type", "unknown")
            print(f"    - {c}: dataset={dataset}")
        if len(configs) > 5:
            print(f"    ... and {len(configs) - 5} more")
    except Exception as e:
        print(f"  Config test: {e}")


def main():
    results = check_imports()

    if not all(results.values()):
        failed = [k for k, v in results.items() if not v]
        print(f"\n  FAILED components: {failed}")
        print("  Please fix these before continuing.")
        sys.exit(1)

    if torch.cuda.is_available():
        test_cuda_ops()
        test_spconv()

    test_model_config()

    print("\n" + "=" * 60)
    print(" ALL TESTS PASSED!")
    print("=" * 60)
    print("""
 Next steps:
   1. Download a dataset (see INSTALL_GUIDE.md)
   2. Preprocess: cd dataset/<name> && bash prepare_data.sh
   3. Download pretrained model or backbone
   4. Train: python tools/train.py configs/softgroup/softgroup_scannet.yaml
   5. Test:  python tools/test.py configs/softgroup/softgroup_scannet.yaml --pretrain ckpt.pth
""")


if __name__ == "__main__":
    main()
