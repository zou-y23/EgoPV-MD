# EgoPV-MD
## Introduction
This repository provides the PyTorch implementation of the paper "Mistake Detection in Egocentric Procedural Videos via CoT-based Action Anticipation".

## Requirments
- Python == 3.10
- Pytorch == 2.1.2
- Cuda == 11.8

## Framework
![](https://github.com/zou-y23/EgoPV-MD/blob/main/EgoPV-MD-fig2.png)

## Datasets
Prepare the datasets (HoloAssist [1], and Assembly101 [2]) according to the instructions.

### HoloAssist
Follow the instructions to download [videos](https://holoassist.github.io/) and [labels](https://holoassist.github.io/). 

### Assembly101
Using official download [script](https://github.com/assembly-101/assembly101-download-scripts).

## Prepare Offline Models
Download the pre-trained models ([EILEV](https://huggingface.co/kpyu/eilev-blip2-opt-2.7b) , and [DeepSeek-V2-Lite](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite)).

## References
[1] Wang, X., Kwon, T., Rad, M., Pan, B., Chakraborty, I., Andrist, S., Bohus, D., Feniello, A., Tekin, B., Frujeri, F.V. and Joshi, N., ”Holoassist: An egocentric human interaction dataset for interactive ai assistants in the real world”, in Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 20270-20281, 2023.
[2] Sener, F., Chatterjee, D., Shelepov, D., He, K., Singhania, D., Wang, R. and Yao, A., ”Assembly101: A large-scale multi-view video dataset for understanding procedural activities”, in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 21096-21106, 2022.

## Citation
