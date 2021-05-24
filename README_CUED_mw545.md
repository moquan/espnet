https://espnet.github.io/espnet/installation.html#requirements
https://github.com/espnet/espnet/blob/master/egs2/TEMPLATE/tts1/README.md#multi-speaker-model-with-x-vector-training


. /data/vectra2/tts/mw545/TorchTTS/anaconda/etc/profile.d/conda.sh && conda deactivate && conda activate espnetpy36

/data/vectra2/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/vctk/

./setup_anaconda.sh /data/vectra2/tts/mw545/TorchTTS/anaconda espnetpy36torch15cuda101 3.6
./setup_anaconda.sh /data/vectra2/tts/mw545/TorchTTS/anaconda espnetpy36 3.6

Change Make file to 
  1. manually set TH_VERSION=1.5.1
  1. bypass nvcc; 
  2. manual input cuda version

  Change activate_python.sh for more paths
export CUDA_HOME=/usr/local/cuda-10.1
export PATH=${PATH}:${CUDA_HOME}/bin:/data/vectra2/tts/mw545/TorchTTS/anaconda/bin
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CUDA_HOME}/lib64:${CUDA_HOME}/extras/CUPTI/lib64
unset LD_PRELOAD

  Install Pytorch
conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.1 -c pytorch

make TH_VERSION=1.5.0 CUDA_VERSION=10.1

./setup_anaconda.sh /data/vectra2/tts/mw545/TorchTTS/anaconda espnet 3.8
Change Make file to bypass nvcc; manual input cuda version
make


Installation Require gcc 4.9+
  air208, 209, 210+ has gcc 4.9+
local/data_prep.sh: line 104: bc: command not found
  bc not installed on air210+
./db.sh: line 38: realpath: command not found
  realpath not installed on air206
Could not find sox in PATH
  sox not installed on air210+


Running xvector on air208
  /data/vectra2/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/vctk/tts_xvector


    MKL_ROOT="${MKL_ROOT:-/opt/intel/mkl}"
       # Check the well-known mkl.h file location.
    if ! [[ -f "${MKL_ROOT}/include/mkl.h" ]] &&


 ./configure --openblas-root=../tools/OpenBLAS/install --cudatk-dir=/usr/local/cuda-10.1

cd /data/vectra2/tts/mw545/TorchTTS/espnet_py36/espnet/tools/kaldi/tools
wget http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.8.0.tar.gz
wget https://sites.google.com/site/openfst/home/openfst-down/openfst-1.6.7.tar.gz
tar xozf openfst-1.6.7.tar.gz
mkdir openfst
./configure --prefix=/data/vectra2/tts/mw545/TorchTTS/espnet_py36/espnet/tools/kaldi/tools/openfst --exec-prefix=/data/vectra2/tts/mw545/TorchTTS/espnet_py36/espnet/tools/kaldi/tools/openfst





VCTK eval distribution
{'498': 2, '494': 2, '495': 2, '496': 2, '497': 2, '490': 2, '491': 1, '407': 5, '288': 1, '346': 3, '404': 9, '403': 9, '402': 7, '342': 1, '400': 4, '348': 3, '349': 3, '287': 1, '408': 3, '366': 3, '367': 2, '364': 5, '422': 1, '425': 1, '424': 1, '414': 12, '415': 34, '416': 36, '417': 34, '410': 3, '411': 2, '412': 2, '413': 3, '371': 2, '370': 1, '373': 2, '426': 1, '375': 2, '374': 2, '290': 1, '376': 1, '303': 1, '313': 1, '312': 1, '311': 2, '289': 1, '315': 1, '314': 1, '393': 7, '392': 6, '391': 6, '390': 1, '397': 7, '396': 6, '395': 9, '394': 9, '399': 3, '398': 6, '429': 2, '428': 2, '368': 2, '369': 2, '421': 1, '420': 2, '423': 1, '365': 5, '362': 4, '363': 3, '427': 2, '361': 2, '308': 1, '309': 1, '449': 1, '448': 1, '300': 1, '301': 1, '302': 1, '299': 1, '447': 1, '446': 1, '445': 1, '307': 1, '381': 1, '382': 1, '383': 1, '384': 2, '406': 7, '386': 3, '387': 3, '388': 1, '389': 2, '372': 2, '418': 34, '419': 27, '430': 1, '431': 1, '458': 2, '459': 2, '450': 1, '451': 2, '452': 2, '453': 3, '454': 4, '455': 3, '456': 3, '457': 3, '344': 4, '385': 2, '345': 4, '405': 9, '347': 3, '310': 1, '469': 3, '468': 3, '465': 4, '464': 3, '467': 3, '466': 3, '461': 2, '460': 2, '463': 3, '462': 3, '401': 5, '409': 3, '343': 2, '286': 1, '357': 1, '356': 2, '470': 2, '471': 2, '476': 1, '351': 4, '350': 2, '489': 2, '488': 2, '487': 2, '486': 1, '472': 2, '473': 2, '355': 4, '354': 5, '353': 4, '352': 5, '474': 1, '475': 1}
