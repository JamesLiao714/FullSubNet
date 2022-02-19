# Audio Noise Reduction competetion
[AI cup contest link](https://aidea-web.tw/topic/8d381596-ee9d-45d5-b779-188909ccb0c8?lang=en)

目前人工智慧語音識別的應用已越來越成熟、廣泛，然而背景音的干擾容易造成語音識別上的困難，因此，人聲去噪的技術將是一個重要的研究議題。

本議題期望以 AI 的方式識別混音檔案中的人聲，並保留人聲的部分以進行去噪，期望獲得更純淨的人聲音訊，提高語音識別應用的服務品質。

You can do speech enhancement(SE) in the frequency domain using various method through this repository.
  
The results of the network can be evaluated through various objective metrics (PESQ, STOI, CSIG, CBAK, COVL).

## Requirements
> This repository is tested on Ubuntu 20.04, and
* Python 3.7
* Cuda 11.1
* CuDNN 8.0.5
* Pytorch 1.9.0

## Getting Started   
1. Install the necessary libraries 
2. Make a dataset for train and validation
   ```sh
   # The shape of the dataset
   [data_num, 2 (inputs and targets), sampling_frequency * data_length]   
   
   # For example, if you want to use 1,000 3-second data sets with a sampling frequency of 16k, the shape is,   
   [1000, 2, 48000]
   ```
4. Set [dataloader.py](https://github.com/seorim0/Speech_enhancement_for_you/blob/main/dataloader.py)
   ```sh
   self.input_path = "DATASET_FILE_PATH"
   ```
5. Set [config.py](https://github.com/seorim0/Speech_enhancement_for_you/blob/main/config.py)
   ```sh
   # If you need to adjust any settings, simply change this file.   
   # When you run this project for the first time, you need to set the path where the model and logs will be saved. 
   ```
6. Run [train_interface.py](https://github.com/seorim0/Speech_enhancement_for_you/blob/main/train_interface.py)


<!-- NETWORKS -->
## Networks   
> You can find a list that you can adjust in various ways at config.py, and they are:   
* Real network   
   - convolutional recurrent network (CRN)   
   it is a real version of DCCRN   
   - FullSubNet [[1]](https://arxiv.org/abs/2010.15508)  
* Complex network   
   - deep complex convolutional recurrent network (DCCRN) [[2]](https://arxiv.org/abs/2008.00264)  

<!-- LEARNING METHODS -->
## Learning Methods
* T-F masking
* Spectral mapping

<!-- LOSS FUNCTIONS -->
## Loss Functions   
* MSE   
* SDR   
* SI-SNR   
* SI-SDR   

> and you can join the loss functions with perceptual loss.   
* LMS
* PMSQE(-0.5 ~ 4.5)

## Test

Currently support FullSubnet only.

## Reference   
**FullSubNet: A Full-Band and Sub-Band Fusion Model for Real-Time Single-Channel Speech Enhancement**    
Xiang Hao, Xiangdong Su, Radu Horaud, Xiaofei Li   
[[arXiv]](https://arxiv.org/abs/2010.15508)  [[code]](https://github.com/haoxiangsnr/FullSubNet)   
**DCCRN: Deep Complex Convolution Recurrent Network for Phase-Aware Speech Enhancement**   
Yanxin Hu, Yun Liu, Shubo Lv, Mengtao Xing, Shimin Zhang, Yihui Fu, Jian Wu, Bihong Zhang, Lei Xie   
[[arXiv]](https://arxiv.org/abs/2008.00264)  [[code]](https://github.com/huyanxin/DeepComplexCRN)   
**Other tools**   
https://github.com/usimarit/semetrics     
https://ecs.utdallas.edu/loizou/speech/software.htm   

