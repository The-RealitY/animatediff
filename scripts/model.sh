#!/bin/bash
# shellcheck disable=SC1068

CONFIG_ID=$CONFIG_ID

aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/AnimateDiff/resolve/main/mm_sd_v14.ckpt -d /content/animatediff/models/Motion_Module -o mm_sd_v14.ckpt
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/AnimateDiff/resolve/main/mm_sd_v15.ckpt -d /content/animatediff/models/Motion_Module -o mm_sd_v15.ckpt

if [ "$CONFIG_ID" = "ToonYou" ]; then
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/AnimateDiff/resolve/main/toonyou_beta3.safetensors -d /content/animatediff/models/DreamBooth_LoRA -o toonyou_beta3.safetensors
elif [ "$CONFIG_ID" = "Lyriel" ]; then
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/AnimateDiff/resolve/main/lyriel_v16.safetensors -d /content/animatediff/models/DreamBooth_LoRA -o lyriel_v16.safetensors
elif [ "$CONFIG_ID" = "RcnzCartoon" ]; then
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/AnimateDiff/resolve/main/rcnzCartoon3d_v10.safetensors -d /content/animatediff/models/DreamBooth_LoRA -o rcnzCartoon3d_v10.safetensors
elif [ "$CONFIG_ID" = "MajicMix" ]; then
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/AnimateDiff/resolve/main/majicmixRealistic_v5Preview.safetensors -d /content/animatediff/models/DreamBooth_LoRA -o majicmixRealistic_v5Preview.safetensors
elif [ "$CONFIG_ID" = "RealisticVision" ]; then
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/AnimateDiff/resolve/main/realisticVisionV20_v20.safetensors -d /content/animatediff/models/DreamBooth_LoRA -o realisticVisionV20_v20.safetensors
elif [ "$CONFIG_ID" = "Tusun" ]; then
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/AnimateDiff/resolve/main/TUSUN.safetensors -d /content/animatediff/models/DreamBooth_LoRA -o TUSUN.safetensors
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/AnimateDiff/resolve/main/moonfilm_reality20.safetensors -d /content/animatediff/models/DreamBooth_LoRA -o moonfilm_reality20.safetensors
elif [ "$CONFIG_ID" = "FilmVelvia" ]; then
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/AnimateDiff/resolve/main/majicmixRealistic_v5Preview.safetensors -d /content/animatediff/models/DreamBooth_LoRA -o majicmixRealistic_v4.safetensors
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/AnimateDiff/resolve/main/FilmVelvia2.safetensors -d /content/animatediff/models/DreamBooth_LoRA -o FilmVelvia2.safetensors
elif [ "$CONFIG_ID" = "GhibliBackground" ]; then
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/AnimateDiff/resolve/main/CounterfeitV30_v30.safetensors -d /content/animatediff/models/DreamBooth_LoRA -o CounterfeitV30_v30.safetensors
  aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/AnimateDiff/resolve/main/Pyramid%20lora_Ghibli_n3.safetensors -d /content/animatediff/models/DreamBooth_LoRA -o lora_Ghibli_n3.safetensors

else
    echo "Invalid or missing CONFIG-ID"
fi
