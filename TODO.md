# Install from here:
https://my.visualstudio.com/Downloads?q=visual%20studio%202022&wt.mc_id=o~msft~vscom~older-downloads
https://developer.nvidia.com/cuda-12-4-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local


# [Not Working] For Linux:
- Follow pytorch3d guide (https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)
- Install 'experiments/requirements.txt'
<!-- - conda install mkl=2024.0.0 -->
<!-- - conda install -c conda-forge ittapi -->
<!-- - pip uninstall pytorch3d -->
<!-- - conda install pytorch3d -c pytorch3d -->
- pip install torch==2.7.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128


# For Linux (Tested on GPU: RTX 2080Ti):
- Clone to this repo root: `git clone https://github.com/OfirGiladBGU/pytorch3d.git`
- Install `pytorch3d` with the guide: [INSTALL-NEW.md](https://github.com/OfirGiladBGU/pytorch3d/blob/python3.9-support/INSTALL-NEW.md) (Make sure to call the env: `pc2`)
- Run: `pip install -r experiments/requirements.txt`
- Run: `pip install sqlalchemy`
