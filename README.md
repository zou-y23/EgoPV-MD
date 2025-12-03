# EgoPV-MD
## Introduction
This repository provides the PyTorch implementation of the paper "Mistake Detection in Egocentric Procedural Videos via CoT-based Action Anticipation".

## Requirments
- **Python** >= 3.10
- **PyTorch** >= 2.4.1
- **CUDA** >= 12.1

> **Note:** Please refer to the specific `requirements.txt` files within each module folder for detailed dependency lists.

## Framework
![](https://github.com/zou-y23/EgoPV-MD/blob/main/EgoPV-MD-fig2.png)

## Datasets
Prepare the datasets (HoloAssist [1], and Assembly101 [2]) according to the instructions.

### HoloAssist Dataset File Structure
Follow the instructions to download [videos](https://holoassist.github.io/) and [labels](https://holoassist.github.io/). 
```text
data/
└── HoloAssist/
    ├── annotations/
    └── videos/
        ├── R007-7July-DSLR/
        │   └── Export_py/
        │       ├── cam_info/
        │       │   ├── Pose_sync.txt
        │       │   ├── Instrinsincs.txt
        │       │   └── VideoMp4Timing.txt
        │       └── clips/
        │           └── Video_compress.mp4
        ├── R012-7July-Nespresso/
        ├── R013-7July-Nespresso/
        ├── R014-7July-DSLR/
        └── ...
```
## Prepare Offline Models
Please download the models from the links below and place them in their respective module directories:

| Model | Full Name | Directory | Model Link |
| :--- | :--- | :--- | :---: |
| **ARM** | Action Recognition Module | [Link](https://github.com/zou-y23/EgoPV-MD/tree/7a71c31687e5cbf0091a6de8462a916ce2ba8026/ActionRecognitionModule) | [Google Drive](https://drive.google.com/drive/folders/1jqsvvKB86gB8WtgOAWtN5Y_AR6SmzZys?usp=drive_link) |
| **ACM** | Action Captioning Module | [Link](https://github.com/zou-y23/EgoPV-MD/tree/7a71c31687e5cbf0091a6de8462a916ce2ba8026/ActionCaptioningModule) | [Google Drive](https://drive.google.com/drive/folders/1AqGsfZU8sIni5eLziTnjEhHnqScX9tl9?usp=drive_link) |
| **AAM** | Action Anticipation Module | [Link](https://github.com/zou-y23/EgoPV-MD/tree/7a71c31687e5cbf0091a6de8462a916ce2ba8026/ActionAnticipationModule) | [Google Drive](https://drive.google.com/drive/folders/1cqkRx5Sj_oD0yf9nMJm7rJXWPoVby8-_?usp=drive_link) |

## Usage

Please follow the instructions below to run inference and training. We recommend running the commands within each module's specific directory.

### 1. Inference

#### Overall Mistake Detection 
To reproduce the final results of the paper (Mistake Detection), utilize the **AAM** module. 

1.  **Generate Predictions:** Load the pre-trained model and generate predictions. 
    ```bash
    cd ActionAnticipationModule
    python generate_text.py
    ```
> **Note:** This process can be time consuming, we provide the generated results for evaluations directly. 

2.  **Evaluation Results:** Evaluation using the generated and comparison results. 
    ```bash
    cd ActionAnticipationModule
    python test_precision_recall.py
    ```

#### Modular Performance Evaluation
If you wish to evaluate the performance of individual components (ARM and ACM) separately:

**1. ARM (Action Recognition Module)**
Evaluate the action recognition performance.
    ```bash
    cd ActionRecognitionModule
    python run_net.py --cfg configs/fine_action_recognition.yaml
    ```

**2. ACM (Action Captioning Module)**
Generate Narrations.
    ```bash
    cd ActionCaptioningModule
    python python main.py 
    ```
> **Note:** This process can be time consuming, we provide the generated results for evaluations directly. 

Evaluation using the generated results.
    ```bash
    cd ActionCaptioningModule
    python test.py
    ```

### 2. Training

#### Train ARM
To train the Action Recognition Model (ARM) from scratch or fine-tune it:
1.  **Modify Configuration:** Open configs/fine_action_recognition.yaml and ensure the mode is set to Train (e.g., enable training flags).
2.  **Start Training:** Run the training script (supports multi-GPU, e.g., 3 GPUs).
    ```bash
    cd ActionRecognitionModule
    python run_net.py --cfg configs/fine_action_recognition.yaml NUM_GPUS 3
    ```

## References
[1] Wang, X., Kwon, T., Rad, M., Pan, B., Chakraborty, I., Andrist, S., Bohus, D., Feniello, A., Tekin, B., Frujeri, F.V. and Joshi, N., ”Holoassist: An egocentric human interaction dataset for interactive ai assistants in the real world”, in Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 20270-20281, 2023.
[2] Sener, F., Chatterjee, D., Shelepov, D., He, K., Singhania, D., Wang, R. and Yao, A., ”Assembly101: A large-scale multi-view video dataset for understanding procedural activities”, in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 21096-21106, 2022.

## Citation
