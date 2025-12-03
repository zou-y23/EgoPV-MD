from .seq2seq import Encoder, Decoder, Seq2Seq
import torch


def build_model(cfg, gpu_id=None):
    """
    Builds the video model.
    Args:
        cfg (configs): configs that contains the hyper-parameters to build the
        backbone. Details can be seen in slowfast/config/defaults.py.
        gpu_id (Optional[int]): specify the gpu index to build model.
    """

    tasks = cfg.TASKS
    INPUT_DIM = 2 * 3 * 26  # dim * joint_num
    OUTPUT_DIM = 2 * 3 * 26
    ENC_EMB_DIM = 2 * 3 * 26
    DEC_EMB_DIM = 2 * 3 * 26

    if "rgb" in tasks:
        ENC_EMB_DIM += 1000  # + resnet
    if "depth-aligned" in tasks:
        ENC_EMB_DIM += 1000  # + resnet
    if "eye" in tasks:
        INPUT_DIM += 7
        ENC_EMB_DIM += 7
    if "head-pose" in tasks:
        INPUT_DIM += 16
        ENC_EMB_DIM += 16
    HID_DIM = 512
    N_LAYERS = 3
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5
    device = cfg.DEVICE
    enc = Encoder(
        INPUT_DIM,
        ENC_EMB_DIM,
        HID_DIM,
        N_LAYERS,
        ENC_DROPOUT,
        tasks,
        device,
    )
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
    model = Seq2Seq(enc, dec, device).to(device)

    if cfg.NUM_GPUS:
        if gpu_id is None:
            # Determine the GPU used by the current process
            cur_device = torch.cuda.current_device()
        else:
            cur_device = gpu_id
    

    if cfg.NUM_GPUS > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            module=model,
            device_ids=[cur_device],
            output_device=cur_device,
            find_unused_parameters=True,
        )
    def init_weights(m):
        for name, param in m.named_parameters():
            torch.nn.init.uniform_(param.data, -0.08, 0.08)

    model.apply(init_weights)

    def count_parameters(model):
        return sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )

    return model