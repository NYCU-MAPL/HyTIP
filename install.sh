conda install -y -c anaconda -c conda-forge -c comet_ml comet_ml
conda install -y numpy=1.26.4
conda install -y pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -y torchinfo
conda install -y scipy
conda install -y timm
conda install -y GPUtil
conda install -y einops
conda install -y seaborn
conda install -y loguru
conda install -y conda-forge::libstdcxx-ng=12.1.0
python -m pip install flowiz
python -m pip install ptflops
python -m pip install pytorch-msssim==0.2.0
python -m pip install -U pip && python -m pip install -e .
