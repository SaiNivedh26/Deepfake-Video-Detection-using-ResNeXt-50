
import timm

model = timm.create_model("hf_hub:timm/resnext101_32x16d.fb_swsl_ig1b_ft_in1k", pretrained=True)