import sys

from egovideo.vision_encoder import *
model = PretrainVisionTransformer(
        img_size=224, 
        num_frames=16,
        tubelet_size=1,
        patch_size=14, 
        embed_dim=1408,
        clip_embed_dim=768,
        clip_teacher_embed_dim=3200,
        clip_teacher_final_dim=768,
        clip_norm_type='l2',
        clip_return_layer=6,
        clip_student_return_interval=1,
        use_checkpoint=False,
        checkpoint_num=40,
        use_flash_attn=True,
        use_fused_rmsnorm=True,
        use_fused_mlp=True,
        sep_image_video_pos_embed=False,

    )

print(model)