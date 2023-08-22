import os

from omegaconf import OmegaConf, DictConfig


def config_obj(prompt, config_id=None) -> DictConfig:
    ToonYou = {'ToonYou': {'base': '',
                           'path': os.path.join(os.getcwd(), 'models', 'DreamBooth_LoRA', 'toonyou_beta3.safetensors'),
                           'motion_module': [os.path.join(os.getcwd(), 'models', 'Motion_Module', 'mm_sd_v14.ckpt'),
                                             os.path.join(os.getcwd(), 'models', 'Motion_Module', 'mm_sd_v15.ckpt')],
                           'seed': [10788741199826055526, 6520604954829636163, 6519455744612555650,
                                    16372571278361863751],
                           'steps': 25,
                           'guidance_scale': 7.5,
                           'prompt': [prompt],
                           'n_prompt': [
                               'badhandv4,easynegative,ng_deepnegative_v1_75t,verybadimagenegative_v1.3, bad-artist, bad_prompt_version2-neg, teeth']
                           }
               }
    Lyriel = {'Lyriel': {'base': '',
                         'path': os.path.join(os.getcwd(), 'models', 'DreamBooth_LoRA', 'lyriel_v16.safetensors'),
                         'motion_module': [os.path.join(os.getcwd(), 'models', 'Motion_Module', 'mm_sd_v14.ckpt'),
                                           os.path.join(os.getcwd(), 'models', 'Motion_Module', 'mm_sd_v15.ckpt')],
                         'seed': [10917152860782582783, 6399018107401806238, 15875751942533906793, 6653196880059936551],
                         'steps': 25,
                         'guidance_scale': 7.5,
                         'prompt': [prompt],
                         'n_prompt': [
                             'holding an item, cowboy, hat, cartoon, 3d, disfigured, bad art, deformed,extra limbs,close up,b&w, wierd colors, blurry, duplicate, morbid, mutilated, [out of frame], extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, ugly, blurry, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, out of frame, ugly, extra limbs, bad anatomy, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, mutated hands, fused fingers, too many fingers, long neck, Photoshop, video game, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, 3d render']
                         }
              }
    RcnzCartoon = {'RcnzCartoon': {'base': '',
                                   'path': os.path.join(os.getcwd(), 'models', 'DreamBooth_LoRA',
                                                        'rcnzCartoon3d_v10.safetensors'),
                                   'motion_module': [
                                       os.path.join(os.getcwd(), 'models', 'Motion_Module', 'mm_sd_v14.ckpt'),
                                       os.path.join(os.getcwd(), 'models', 'Motion_Module', 'mm_sd_v15.ckpt')],
                                   'seed': [16931037867122267877, 2094308009433392066, 4292543217695451092,
                                            15572665120852309890],
                                   'steps': 25,
                                   'guidance_scale': 7.5,
                                   'prompt': [prompt],
                                   'n_prompt': [
                                       'easynegative, cartoon, anime, sketches, necklace, earrings worst quality, low quality, normal quality, bad anatomy, bad hands, shiny skin, error, missing fingers, extra digit, fewer digits, jpeg artifacts, signature, watermark, username, blurry, chubby, anorectic, bad eyes, old, wrinkled skin, red skin, photograph By bad artist -neg, big eyes, muscular face,']
                                   }
                   }
    MajicMix = {'MajicMix': {'base': '',
                             'path': os.path.join(os.getcwd(), 'models', 'DreamBooth_LoRA',
                                                  'majicmixRealistic_v5Preview.safetensors'),
                             'motion_module': [os.path.join(os.getcwd(), 'models', 'Motion_Module', 'mm_sd_v14.ckpt'),
                                               os.path.join(os.getcwd(), 'models', 'Motion_Module', 'mm_sd_v15.ckpt')],
                             'seed': [1572448948722921032, 1099474677988590681, 6488833139725635347,
                                      18339859844376517918],
                             'steps': 25,
                             'guidance_scale': 7.5,
                             'prompt': [prompt],
                             'n_prompt': [
                                 'nude, nsfw, ng_deepnegative_v1_75t, badhandv4, worst quality, low quality, normal quality, lowres, bad anatomy, bad hands, monochrome, grayscale watermark, moles, people']
                             }
                }
    RealisticVision = {'RealisticVision': {'base': '',

                                           'path': os.path.join(os.getcwd(), 'models', 'DreamBooth_LoRA',
                                                                'realisticVisionV20_v20.safetensors'),
                                           'motion_module': [
                                               os.path.join(os.getcwd(), 'models', 'Motion_Module', 'mm_sd_v14.ckpt'),
                                               os.path.join(os.getcwd(), 'models', 'Motion_Module', 'mm_sd_v15.ckpt')],
                                           'seed': [5658137986800322009, 12099779162349365895, 10499524853910852697,
                                                    16768009035333711932],
                                           'steps': 25,
                                           'guidance_scale': 7.5,
                                           'prompt': [prompt],
                                           'n_prompt': [
                                               'semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck', ]
                                           }
                       }
    Tusun = {'Tusun': {'base': os.path.join(os.getcwd(), 'models', 'DreamBooth_LoRA',
                                            'moonfilm_reality20.safetensors'),
                       'path': os.path.join(os.getcwd(), 'models', 'DreamBooth_LoRA',
                                            'TUSUN.safetensors'),
                       'motion_module': [os.path.join(os.getcwd(), 'models', 'Motion_Module', 'mm_sd_v14.ckpt'),
                                         os.path.join(os.getcwd(), 'models', 'Motion_Module', 'mm_sd_v15.ckpt')],
                       'seed': [10154078483724687116, 2664393535095473805, 4231566096207622938, 1713349740448094493],
                       'steps': 25,
                       'guidance_scale': 7.5,
                       'lora_alpha': 0.6,
                       'prompt': [prompt],
                       'n_prompt': [
                           'worst quality, low quality, deformed, distorted, disfigured, bad eyes, bad anatomy, disconnected limbs, wrong body proportions, low quality, worst quality, text, watermark, signatre, logo, illustration, painting, cartoons, ugly, easy_negative']
                       }
             }
    FilmVelvia = {'FilmVelvia': {'base': os.path.join(os.getcwd(), 'models', 'DreamBooth_LoRA',
                                                      'majicmixRealistic_v4.safetensors'),
                                 'path': os.path.join(os.getcwd(), 'models', 'DreamBooth_LoRA',
                                                      'FilmVelvia2.safetensors'),
                                 'motion_module': [
                                     os.path.join(os.getcwd(), 'models', 'Motion_Module', 'mm_sd_v14.ckpt'),
                                     os.path.join(os.getcwd(), 'models', 'Motion_Module', 'mm_sd_v15.ckpt')],
                                 'seed': [358675358833372813, 3519455280971923743, 11684545350557985081,
                                          8696855302100399877],
                                 'steps': 25,
                                 'guidance_scale': 7.5,
                                 'lora_alpha': 0.6,
                                 'prompt': [prompt],
                                 'n_prompt': [
                                     'wrong white balance, dark, cartoon, anime, sketches,worst quality, low quality, deformed, distorted, disfigured, bad eyes, wrong lips, weird mouth, bad teeth, mutated hands and fingers, bad anatomy, wrong anatomy, amputation, extra limb, missing limb, floating limbs, disconnected limbs, mutation, ugly, disgusting, bad_pictures, negative_hand-neg']
                                 }
                  }
    GhibliBackground = {'GhibliBackground': {'base': os.path.join(os.getcwd(), 'models', 'DreamBooth_LoRA',
                                                                  'CounterfeitV30_30.safetensors'),
                                             'path': os.path.join(os.getcwd(), 'models', 'DreamBooth_LoRA',
                                                                  'lora_Ghibli_n3.safetensors'),
                                             'motion_module': [
                                                 os.path.join(os.getcwd(), 'models', 'Motion_Module', 'mm_sd_v14.ckpt'),
                                                 os.path.join(os.getcwd(), 'models', 'Motion_Module',
                                                              'mm_sd_v15.ckpt')],
                                             'seed': [8775748474469046618, 5893874876080607656, 11911465742147695752,
                                                      12437784838692000640],
                                             'steps': 25,
                                             'guidance_scale': 7.5,
                                             'lora_alpha': 1.0,
                                             'prompt': [prompt],
                                             'n_prompt': [
                                                 'easynegative,bad_construction,bad_structure,bad_wail,bad_windows,blurry,cloned_window,cropped,deformed,disfigured,error,extra_windows,extra_chimney,extra_door,extra_structure,extra_frame,fewer_digits,fused_structure,gross_proportions,jpeg_artifacts,long_roof,low_quality,structure_limbs,missing_windows,missing_doors,missing_roofs,mutated_structure,mutation,normal_quality,out_of_frame,owres,poorly_drawn_structure,poorly_drawn_house,signature,text,too_many_windows,ugly,username,uta,watermark,worst_quality']
                                             }
                        }
    config = {"ToonYou": ToonYou,
              "Lyriel": Lyriel,
              "RcnzCartoon": RcnzCartoon,
              "MajicMix": MajicMix,
              "RealisticVision": RealisticVision,
              "Tusun": Tusun,
              "FilmVelvia": FilmVelvia,
              "GhibliBackground": GhibliBackground
              }
    config_dict = OmegaConf.create(config[config_id] if config_id in config else config['ToonYou'])
    return config_dict
