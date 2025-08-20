# HyTIP Training Procedure

* Organize your training datasets according to the following file structure:
    ```
    {dataset_root}/
        ├ vimeo_septuplet/
        │   └ sequences/
        │       ├ 00001/0001/
        │       │   ├ im1.png
        │       │   ├ ...
        │       │   └ im7.png
        │       └ ...
        └ BVI-DVC/
            ├ AAdvertisingMassagesBangkokVidevo_3840x2176_25fps_10bit_420/
            │   ├ Frame_0.png
            │   ├ ...
            │   └ Frame_63.png
            └ ...
    ```

* Download the following pre-trained models and put them in the corresponding folder in `./models`.
    * DCVC-DC intra codec: [cvpr2023_image_psnr.pth.tar](https://github.com/microsoft/DCVC/tree/main/DCVC-family/DCVC-DC)
    * DCVC-FM inter codec: [cvpr2024_video.pth.tar](https://github.com/microsoft/DCVC/tree/main/DCVC-family/DCVC-FM)
    * SPyNet as motion estimation network: [spy_net-sintel-final.pytorch](https://github.com/Yvonne-ul6u45p/Transformer-based-Codec/releases/download/Pre-trained_Weight/spy_net-sintel-final.pytorch)

* Modify [user_info/key.yml](../user_info/key.yml) with your [Comet](https://www.comet.com/site/) account information for recording training/evaluation logs.
    ```
    api_key: *************************
    workspace: ********
    ```

* Add the following argument(s) to the training command for debugging:
    * `--debug`: Disable uploading training/evaluation data to the model evaluation platform [Comet](https://www.comet.com/site/).
    * `--no_sanity`: Skip the initial validation step before starting training.

* In the following training procedure, the notations are defined as follows:
    * "CRC" denotes conditional residual coding and "MCR" denotes masked conditional residual coding.
    * "Explicit/Hybrid + Explicit/Hybrid" indicates the combination of buffering schemes for the motion codec and the inter codec.


## Step 1. Single Rate, CRC, Explicit + Explicit
The model weights are initialized from the pre-trained DCVC-FM model.

* Train (highest rate)

    `python HyTIP_SingleRate.py --cond_motion_coder_conf ./config/Motion_Explicit.yml --residual_coder_conf ./config/Inter_CRC_Explicit.yml --train_conf ./train_cfg/train_Explicit_Explicit_CRC.json -n 12 --quality_level 5 --iframe_quality 5 --gpus 1 --project_name HyTIP --experiment_name Explicit_Explicit_CRC --restore load_pretrained --restore_exp_key None --restore_exp_epoch -1 -data {dataset_root}`
* Test (highest rate)

    `python HyTIP_SingleRate.py --cond_motion_coder_conf ./config/Motion_Explicit.yml --residual_coder_conf ./config/Inter_CRC_Explicit.yml -n 0 --quality_level 5 --iframe_quality 5 --gpus 1 --project_name HyTIP --experiment_name Explicit_Explicit_CRC_Crop64 --restore resume --restore_exp_key {exp_key_of_highest_rate} --restore_exp_epoch {exp_epoch_of_highest_rate} --test --gop 32 --test_dataset {HEVC-B UVG} --color_transform {BT601,BT709} --remove_scene_cut --test_crop -data {dataset_root}`

## Step 2. Single Rate, MCR, Explicit + Explicit
The model weights are initialized from Step 1.

* Train (highest rate)

    `python HyTIP_SingleRate.py --cond_motion_coder_conf ./config/Motion_Explicit.yml --residual_coder_conf ./config/Inter_MCR_Explicit.yml --train_conf ./train_cfg/train_Explicit_Explicit_MCR.json -n 12 --quality_level 5 --iframe_quality 5 --gpus 1 --project_name HyTIP --experiment_name Explicit_Explicit_MCR --restore CRC_to_MCR --restore_exp_key {exp_key_of_highest_rate_in_step_1} --restore_exp_epoch {exp_epoch_of_highest_rate_in_step_1} -data {dataset_root}`
* Test (highest rate)

    `python HyTIP_SingleRate.py --cond_motion_coder_conf ./config/Motion_Explicit.yml --residual_coder_conf ./config/Inter_MCR_Explicit.yml -n 0 --quality_level 5 --iframe_quality 5 --gpus 1 --project_name HyTIP --experiment_name Explicit_Explicit_MCR_Crop64 --restore resume --restore_exp_key {exp_key_of_highest_rate} --restore_exp_epoch {exp_epoch_of_highest_rate} --test --gop 32 --test_dataset {HEVC-B UVG} --color_transform {BT601,BT709} --remove_scene_cut --test_crop -data {dataset_root}`

## Step 3. Single Rate, MCR, Hybrid (B=6) + Explicit
The model weights are initialized from Step 2.

* Train (highest rate)

    `python HyTIP_SingleRate.py --cond_motion_coder_conf ./config/Motion_Hybrid_B=6.yml --residual_coder_conf ./config/Inter_MCR_Explicit.yml --train_conf ./train_cfg/train_Hybrid_Explicit_MCR.json -n 12 --quality_level 5 --iframe_quality 5 --gpus 1 --project_name HyTIP --experiment_name Hybrid_B=6_Explicit_MCR --restore Motion_Hybrid --restore_exp_key {exp_key_of_highest_rate_in_step_2} --restore_exp_epoch {exp_epoch_of_highest_rate_in_step_2} --index_map_training 0 0 0 0 --index_map 0 0 0 0 --rate_gop_size 4 -data {dataset_root}`
* Test (highest rate)

    `python HyTIP_SingleRate.py --cond_motion_coder_conf ./config/Motion_Hybrid_B=6.yml --residual_coder_conf ./config/Inter_MCR_Explicit.yml -n 0 --quality_level 5 --iframe_quality 5 --gpus 1 --project_name HyTIP --experiment_name Hybrid_B=6_Explicit_MCR_Crop64 --restore resume --restore_exp_key {exp_key_of_highest_rate} --restore_exp_epoch {exp_epoch_of_highest_rate} --index_map_training 0 0 0 0 --index_map 0 0 0 0 --rate_gop_size 4 --test --gop 32 --test_dataset {HEVC-B UVG} --color_transform {BT601,BT709} --remove_scene_cut --test_crop -data {dataset_root}`

## Step 4. Single Rate, MCR, Hybrid (B=6) + Hybrid (B=51)
The model weights are initialized from Step 3.

* Train (highest rate)

    `python HyTIP_SingleRate.py --cond_motion_coder_conf ./config/Motion_Hybrid_B=6.yml --residual_coder_conf ./config/Inter_MCR_Hybrid_B=51.yml --train_conf ./train_cfg/train_Hybrid_Hybrid_MCR.json -n 12 --quality_level 5 --iframe_quality 5 --gpus 1 --project_name HyTIP --experiment_name Hybrid_B=6_Hybrid_B=51_MCR --restore Inter_Hybrid --restore_exp_key {exp_key_of_highest_rate_in_step_3} --restore_exp_epoch {exp_epoch_of_highest_rate_in_step_3} --feature_channel 48 --Pretrain_frame_feature --index_map_training 0 0 0 0 --index_map 0 0 0 0 --rate_gop_size 4 --FGOP_Inter_XtRNN -data {dataset_root}`
* Test (highest rate)

    `python HyTIP_SingleRate.py --cond_motion_coder_conf ./config/Motion_Hybrid_B=6.yml --residual_coder_conf ./config/Inter_MCR_Hybrid_B=51.yml -n 0 --quality_level 5 --iframe_quality 5 --gpus 1 --project_name HyTIP --experiment_name Hybrid_B=6_Hybrid_B=51_MCR_Crop64 --restore resume --restore_exp_key {exp_key_of_highest_rate} --restore_exp_epoch {exp_epoch_of_highest_rate} --feature_channel 48 --Pretrain_frame_feature --index_map_training 0 0 0 0 --index_map 0 0 0 0 --rate_gop_size 4 --FGOP_Inter_XtRNN --test --gop 32 --test_dataset {HEVC-B UVG} --color_transform {BT601,BT709} --remove_scene_cut --test_crop -data {dataset_root}`

## Step 5. Single Rate, MCR, Hybrid (B=2.125) + Hybrid (B=51)
The model weights are initialized from Step 4.

* Train (highest rate)

    `python HyTIP_SingleRate.py --cond_motion_coder_conf ./config/Motion_Hybrid_B=2.125.yml --residual_coder_conf ./config/Inter_MCR_Hybrid_B=51.yml --train_conf ./train_cfg/train_ReduceMotionBufferSize.json -n 12 --quality_level 5 --iframe_quality 5 --gpus 1 --project_name HyTIP --experiment_name Hybrid_B=2.125_Hybrid_B=51_MCR --restore ReduceMotionBufferSize --restore_exp_key {exp_key_of_highest_rate_in_step_4} --restore_exp_epoch {exp_epoch_of_highest_rate_in_step_4} --feature_channel 48 --index_map_training 0 0 0 0 --index_map 0 0 0 0 --rate_gop_size 4 -data {dataset_root}`
* Test (highest rate)

    `python HyTIP_SingleRate.py --cond_motion_coder_conf ./config/Motion_Hybrid_B=2.125.yml --residual_coder_conf ./config/Inter_MCR_Hybrid_B=51.yml -n 0 --quality_level 5 --iframe_quality 5 --gpus 1 --project_name HyTIP --experiment_name Hybrid_B=2.125_Hybrid_B=51_MCR_Crop64 --restore resume --restore_exp_key {exp_key_of_highest_rate} --restore_exp_epoch {exp_epoch_of_highest_rate} --feature_channel 48 --index_map_training 0 0 0 0 --index_map 0 0 0 0 --rate_gop_size 4 --test --gop 32 --test_dataset {HEVC-B UVG} --color_transform {BT601,BT709} --remove_scene_cut --test_crop -data {dataset_root}`

## Step 6. Single Rate, MCR, Hybrid (B=2.125) + Hybrid (B=5)
The model weights are initialized from Step 5.

* Train (highest rate)

    `python HyTIP_SingleRate.py --cond_motion_coder_conf ./config/Motion_Hybrid_B=2.125.yml --residual_coder_conf ./config/Inter_MCR_Hybrid_B=5.yml --train_conf ./train_cfg/train_ReduceInterBufferSize.json -n 12 --quality_level 5 --iframe_quality 5 --gpus 1 --project_name HyTIP --experiment_name Hybrid_B=2.125_Hybrid_B=5_MCR --restore ReduceInterBufferSize --restore_exp_key {exp_key_of_highest_rate_in_step_5} --restore_exp_epoch {exp_epoch_of_highest_rate_in_step_5} --feature_channel 2 --index_map_training 0 0 0 0 --index_map 0 0 0 0 --rate_gop_size 4 --FGOP_Inter_XtRNN -data {dataset_root}`
* Test (highest rate)

    `python HyTIP_SingleRate.py --cond_motion_coder_conf ./config/Motion_Hybrid_B=2.125.yml --residual_coder_conf ./config/Inter_MCR_Hybrid_B=5.yml -n 0 --quality_level 5 --iframe_quality 5 --gpus 1 --project_name HyTIP --experiment_name Hybrid_B=2.125_Hybrid_B=5_MCR_Crop64 --restore resume --restore_exp_key {exp_key_of_highest_rate} --restore_exp_epoch {exp_epoch_of_highest_rate} --feature_channel 2 --index_map_training 0 0 0 0 --index_map 0 0 0 0 --rate_gop_size 4 --FGOP_Inter_XtRNN --test --gop 32 --test_dataset {HEVC-B UVG} --color_transform {BT601,BT709} --remove_scene_cut --test_crop -data {dataset_root}`

## Step 7. Single Rate, MCR, Hybrid (B=2.125) + Hybrid (B=5), +FA
The model weights are initialized from Step 6.

* Train (highest rate)

    `python HyTIP_SingleRate.py --cond_motion_coder_conf ./config/Motion_Hybrid_B=2.125.yml --residual_coder_conf ./config/Inter_MCR_Hybrid_B=5.yml --train_conf ./train_cfg/train_FrameTypeAdaptation.json -n 12 --quality_level 5 --iframe_quality 5 --gpus 1 --project_name HyTIP --experiment_name Hybrid_B=2.125_Hybrid_B=5_MCR_FA --restore FA --restore_exp_key {exp_key_of_highest_rate_in_step_6} --restore_exp_epoch {exp_epoch_of_highest_rate_in_step_6} --index_map_training 0 1 0 2 1 --frame_weight 1 1.2 0.5 0.9 1.2 --FGOP_Inter_XtRNN -data {dataset_root}`
* Test (highest rate)

    `python HyTIP_SingleRate.py --cond_motion_coder_conf ./config/Motion_Hybrid_B=2.125.yml --residual_coder_conf ./config/Inter_MCR_Hybrid_B=5.yml -n 0 --quality_level 5 --iframe_quality 5 --gpus 1 --project_name HyTIP --experiment_name Hybrid_B=2.125_Hybrid_B=5_MCR_FA_Crop64 --restore resume --restore_exp_key {exp_key_of_highest_rate} --restore_exp_epoch {exp_epoch_of_highest_rate} --FGOP_Inter_XtRNN --test --gop 32 --test_dataset {HEVC-B UVG} --color_transform {BT601,BT709} --remove_scene_cut --test_crop -data {dataset_root}`

## Step 8. Single Rate, MCR, Hybrid (B=2.125) + Hybrid (B=5), +FA+CTM
The model weights are initialized from Step 7.

* Train (highest rate)

    `python HyTIP_SingleRate.py --cond_motion_coder_conf ./config/Motion_Hybrid_B=2.125.yml --residual_coder_conf ./config/Inter_MCR_Hybrid_B=5_CTM.yml --train_conf ./train_cfg/Add_CTM.json -n 12 --quality_level 5 --iframe_quality 5 --gpus 1 --project_name HyTIP --experiment_name Hybrid_B=2.125_Hybrid_B=5_MCR_FA_CTM --restore CTM --restore_exp_key {exp_key_of_highest_rate_in_step_7} --restore_exp_epoch {exp_epoch_of_highest_rate_in_step_7} --index_map_training 0 1 0 2 1 --frame_weight 1 1.2 0.5 0.9 1.2 -data {dataset_root}`

* Test (highest rate)

    `python HyTIP_SingleRate.py --cond_motion_coder_conf ./config/Motion_Hybrid_B=2.125.yml --residual_coder_conf ./config/Inter_MCR_Hybrid_B=5_CTM.yml -n 0 --quality_level 5 --iframe_quality 5 --gpus 1 --project_name HyTIP --experiment_name Hybrid_B=2.125_Hybrid_B=5_MCR_FA_CTM_Crop64 --restore resume --restore_exp_key {exp_key_of_highest_rate} --restore_exp_epoch {exp_epoch_of_highest_rate} --test --gop 32 --test_dataset {HEVC-B UVG} --color_transform {BT601,BT709} --remove_scene_cut --test_crop -data {dataset_root}`

## Step 9. Single Rate, MCR, Hybrid (B=2.125) + Hybrid (B=5), +FA+CTM+Context Model
The model weights are initialized from Step 8.

* Train (highest rate)

    `python HyTIP_SingleRate.py --cond_motion_coder_conf ./config/Motion_Hybrid_B=2.125.yml --residual_coder_conf ./config/Inter_MCR_Hybrid_B=5_CTM_Context.yml --train_conf ./train_cfg/Add_ContextModel.json -n 12 --quality_level 5 --iframe_quality 5 --gpus 1 --project_name HyTIP --experiment_name Hybrid_B=2.125_Hybrid_B=5_MCR_FA_CTM_Context --restore context --restore_exp_key {exp_key_of_highest_rate_in_step_8} --restore_exp_epoch {exp_epoch_of_highest_rate_in_step_8} --index_map_training 0 1 0 2 1 --frame_weight 1 1.2 0.5 0.9 1.2 -data {dataset_root}`

* Test (highest rate)

    `python HyTIP_SingleRate.py --cond_motion_coder_conf ./config/Motion_Hybrid_B=2.125.yml --residual_coder_conf ./config/Inter_MCR_Hybrid_B=5_CTM_Context.yml -n 0 --quality_level 5 --iframe_quality 5 --gpus 1 --project_name HyTIP --experiment_name Hybrid_B=2.125_Hybrid_B=5_MCR_FA_CTM_Context_Crop64 --restore resume --restore_exp_key {exp_key_of_highest_rate} --restore_exp_epoch {exp_epoch_of_highest_rate} --test --gop 32 --test_dataset {HEVC-B UVG} --color_transform {BT601,BT709} --remove_scene_cut --test_crop -data {dataset_root}`

## Step 10. Variable Rate, MCR, Hybrid (B=2.125) + Hybrid (B=5), +FA+CTM+Context Model
The model weights are initialized from Step 9.

* Train

    `python HyTIP.py --cond_motion_coder_conf ./config/Motion.yml --residual_coder_conf ./config/Inter.yml --train_conf ./train_cfg/train_VariableRate.json -n 12 --quality_level 5 --iframe_quality 63 --QP 63 --gpus 1 --project_name HyTIP --experiment_name Hybrid_B=2.125_Hybrid_B=5_MCR_FA_CTM_Context_Var_5F --restore Variable --restore_exp_key {exp_key_of_highest_rate_in_step_9} --restore_exp_epoch {exp_epoch_of_highest_rate_in_step_9} --index_map_training 0 1 0 2 1 --frame_weight 1 1.2 0.5 0.9 1.2 -data {dataset_root}`

* Test

    `python HyTIP.py --cond_motion_coder_conf ./config/Motion.yml --residual_coder_conf ./config/Inter.yml -n 0 --quality_level 5 --iframe_quality {0...63} --QP {0...63} --gpus 1 --project_name HyTIP --experiment_name Hybrid_B=2.125_Hybrid_B=5_MCR_FA_CTM_Context_Var_5F_Crop64 --restore resume --restore_exp_key {exp_key} --restore_exp_epoch {exp_epoch} --test --gop 32 --test_dataset {HEVC-B UVG} --color_transform {BT601,BT709} --remove_scene_cut --test_crop -data {dataset_root}`


## Step 11. Variable Rate, MCR, Hybrid (B=2.125) + Hybrid (B=5), +FA+CTM+Context Model, 7-frame
The model weights are initialized from Step 10.

* Train

    `python HyTIP.py --cond_motion_coder_conf ./config/Motion.yml --residual_coder_conf ./config/Inter.yml --train_conf ./train_cfg/train_7frame_LongSequence.json -n 12 --quality_level 5 --iframe_quality 63 --QP 63 --gpus 1 --project_name HyTIP --experiment_name Hybrid_B=2.125_Hybrid_B=5_MCR_FA_CTM_Context_Var_7F --restore LongSequence --restore_exp_key {exp_key_in_step_10} --restore_exp_epoch {exp_epoch_in_step_10} --index_map_training 0 1 0 2 0 2 1 --frame_weight 1 1.2 0.5 0.9 0.5 0.9 1.2 -data {dataset_root}`

* Test

    `python HyTIP.py --cond_motion_coder_conf ./config/Motion.yml --residual_coder_conf ./config/Inter.yml -n 0 --quality_level 5 --iframe_quality {0...63} --QP {0...63} --gpus 1 --project_name HyTIP --experiment_name Hybrid_B=2.125_Hybrid_B=5_MCR_FA_CTM_Context_Var_7F_Crop64 --restore resume --restore_exp_key {exp_key} --restore_exp_epoch {exp_epoch} --test --gop 32 --test_dataset {HEVC-B UVG} --color_transform {BT601,BT709} --remove_scene_cut --test_crop -data {dataset_root}`


## Step 12. Variable Rate, MCR, Hybrid (B=2.125) + Hybrid (B=5), +FA+CTM+Context Model, 10-frame
The model weights are initialized from Step 11.

* Train

    `python HyTIP.py --cond_motion_coder_conf ./config/Motion.yml --residual_coder_conf ./config/Inter.yml --train_conf ./train_cfg/train_10frame_LongSequence.json -n 12 --quality_level 5 --iframe_quality 63 --QP 63 --gpus 1 --project_name HyTIP --experiment_name Hybrid_B=2.125_Hybrid_B=5_MCR_FA_CTM_Context_Var_10F --restore LongSequence --restore_exp_key {exp_key_in_step_11} --restore_exp_epoch {exp_epoch_in_step_11} --index_map_training 0 1 0 2 0 2 0 2 0 1 --frame_weight 1 1.2 0.5 0.9 0.5 0.9 0.5 0.9 0.5 1.2 --train_dataset BVI-DVC -data {dataset_root}`

* Test

    `python HyTIP.py --cond_motion_coder_conf ./config/Motion.yml --residual_coder_conf ./config/Inter.yml -n 0 --quality_level 5 --iframe_quality {0...63} --QP {0...63} --gpus 1 --project_name HyTIP --experiment_name Hybrid_B=2.125_Hybrid_B=5_MCR_FA_CTM_Context_Var_10F_Crop64 --restore resume --restore_exp_key {exp_key} --restore_exp_epoch {exp_epoch} --test --gop 32 --test_dataset {HEVC-B UVG} --color_transform {BT601,BT709} --remove_scene_cut --test_crop -data {dataset_root}`


