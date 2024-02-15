## Machine Learning

This project uses two concept of machine learning, computer vision to do semantic segmentation and automatic speech recognition to do spoken word classification. To make life easier, I use pretrained MobileNetV3 Large, replace and fine-tune the segmentation head for three classes segmentation: `__background__`, `lips`, and `rice`. To make life harder (so balanced with the computer vision), I use GMM-HMM as the classifier model with MFCC as the feature extractor.

### Dataset

#### ASR

I'm using the combination of `[google/fleurs](https://huggingface.co/datasets/google/fleurs)`, `[fruit names](https://github.com/RRisto/single_word_asr_gmm_hmm)`, and my own volunteer recorded dataset.

Using the `google/fleurs`, I filtered it to be only Bahasa Indonesia spoken recording that contain word `makan` or `suap` (including it being in any affix form), cut the recording into `makan` and `suap` only, and save it. With this method, I got 49 worth of recordings saying `makan` and 0 recording saying `suap`. You can find the dataset [here](ASR/Dataset/makan_suap/).

Using the `fruit names` dataset, I use it as the example of `others` category. You can find the dataset used in this project [here](ASR/Dataset/others/) and on its original-complete form [here](https://github.com/RRisto/single_word_asr_gmm_hmm/raw/master/data/audio.tar.gz).

My own dataset consists of 80 recording of `makan` and 179 recording of `suap`. Due to privacy issue, I cannot publish this dataset to public. To get the recordings, hit me up on my email (available on the parent README file) saying your intention and purpose for the request. I may or may not approve the request, depending on the purpose of the request.

#### CV

I'm using the combination of `[kapitanov/easyportrait](https://www.kaggle.com/datasets/kapitanov/easyportrait)` and my own volunteer recorded dataset.

I used 100, randomly picked, faces from `kapitanov/easyportrait` and 100 rice images from my own dataset. The faces images are available [here](CV/Dataset/Lips) while the rice images are available per-request basis, hit me up on my email if you need one.

### Model

#### ASR

Flow: Load file -> volume normalization -> noise reduction -> MFCC (39 features) -> GMM-HMM

While training the GMM-HMM, I'm using parameter search through "brute-force" for `n_components`, `n_mix`, and `n_iter`. Apparently, from the test using the test dataset, model with the combinations of 2, 5, 5 and 4, 5, 10 gives the best result.

Find the training script [here](ASR/Train%20GMM-HMM.ipynb).

#### CV

Flow: Image normalization (resize, rescale to 0-1, normalization using [MobileNetV3 Large configuration](https://pytorch.org/vision/main/models/generated/torchvision.models.segmentation.lraspp_mobilenet_v3_large.html)) -> Training phase 1 (backbone frozen, 10 epochs) -> Training phase 2 (backbone also trained, 5 epochs)

CV training follows the usual two-phase training. I'm only replacing the segmentation head with a newly initialized LRASPPHead class with 3 final classes.

Find the training script [here](CV/Fine%20Tuning%20LRASPP%20MobileNetV3%20Large.ipynb).

#### Final Model

All final and best model from those training is available [here](Final%20Model).
