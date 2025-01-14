
PYTHON=/home/wittej/miniconda3/envs/cvnn4sar/bin/python
PIPELINE_PATH=/home/wittej/cvnn4sar/cvnn/pipeline

DETECT_PREPROCESS=${PIPELINE_PATH}/1_preprocess.py
DETECT_TRAIN=${PIPELINE_PATH}/2_train.py
DETECT_TEST=${PIPELINE_PATH}/3_test.py

do_detect_pre=false
do_detect_train=true
do_detect_test=true

data_qualities=(SARFish_data)
num_augmentations_list=(1)
representations=(complex)
fold_numbers_train=(2)


for data_quality in ${data_qualities[@]} 
do
    for num_augmentations in ${num_augmentations_list[@]} 
    do
        # DETECTION CONFIG
        DATA_ROOT_DIR=/mnt/sasa028/cvnn4sar/SARFish/
        DATA_QUALITY=${data_quality}
        PRODUCT_TYPE=SLC
        FOLD=smallfold
        FOLD_FILE=${PIPELINE_PATH}/utils/fold_files/${FOLD}.csv
        FOLD_NUMBERS_PRE="1,2,3,4,5"
        TILE_SIZE=256
        TILE_OVERLAP=16
        NUM_AUGMENTATIONS=${num_augmentations}
        TILE_PATH=${DATA_ROOT_DIR}${DATA_QUALITY}/detection_dataset/${PRODUCT_TYPE}/${FOLD}/${TILE_SIZE}_${TILE_OVERLAP}/${NUM_AUGMENTATIONS}/
        
        # DETECTION PIPELINE
        if [ "${do_detect_pre}" = true ]; then

        $PYTHON $DETECT_PREPROCESS --data_root_dir $DATA_ROOT_DIR \
                            --data_quality $DATA_QUALITY \
                            --product_type $PRODUCT_TYPE \
                            --fold_file $FOLD_FILE\
                            --fold_numbers $FOLD_NUMBERS_PRE \
                            --tile_path $TILE_PATH\
                            --tile_size $TILE_SIZE\
                            --tile_overlap $TILE_OVERLAP\
                            --num_augmentations $NUM_AUGMENTATIONS
        fi

        for representation in ${representations[@]} 
        do
            for fold_number in ${fold_numbers_train[@]}
            do

                DATASET_FILE=${TILE_PATH}fold_${fold_number}.json
                REPRESENTATION=${representation}
                FOLD_NUMBER_TRAIN=${fold_number}
                DROP_LOW=1
                DROP_EMPTY=1
                BACKBONE_NAME=resnet18
                TRAINED_DMODEL_PATH=${PIPELINE_PATH}/utils/pretrained_models/detection/${DATA_QUALITY}/${FOLD}/fold_${FOLD_NUMBER_TRAIN}/${REPRESENTATION}/${BACKBONE_NAME}/${TILE_SIZE}_${TILE_OVERLAP}/${NUM_AUGMENTATIONS}
                TENSORBOARD_FOLDER=${PIPELINE_PATH}/utils/logs/detection/${DATA_QUALITY}/${FOLD}/fold_${FOLD_NUMBER_TRAIN}/${REPRESENTATION}/${BACKBONE_NAME}/${TILE_SIZE}_${TILE_OVERLAP}/${NUM_AUGMENTATIONS}
                FROZEN_BACKBONE=None 
                NUM_EPOCHS=10
                BATCH_SIZE=8
                MODEL_TO_USE=without_fb_checkpoint
                SCORE_THRESHOLDS=0,0.1,0.2,0.3,0.35,0.375,0.4,0.425,0.45,0.5,0.55,0.6,0.65,0.7,0.8,0.9
                IOU_THRESHOLD=0.3

                # DETECTION PIPELINE
                if [ "${do_detect_train}" = true ]; then

                $PYTHON $DETECT_TRAIN  --dataset_file $DATASET_FILE \
                                --representation $REPRESENTATION \
                                --drop_low $DROP_LOW \
                                --drop_empty $DROP_EMPTY \
                                --backbone_name $BACKBONE_NAME \
                                --trained_model_path $TRAINED_DMODEL_PATH \
                                --tensorboard_folder $TENSORBOARD_FOLDER\
                                --frozen_backbone $FROZEN_BACKBONE\
                                --num_epochs $NUM_EPOCHS \
                                --batch_size $BATCH_SIZE \
                                --save="false"
                fi

                if [ "${do_detect_test}" = true ]; then

                
                $PYTHON $DETECT_TEST   --dataset_file $DATASET_FILE \
                                --representation $REPRESENTATION\
                                --drop_low $DROP_LOW\
                                --drop_empty $DROP_EMPTY\
                                --backbone_name $BACKBONE_NAME\
                                --trained_model_path $TRAINED_DMODEL_PATH\
                                --tensorboard_folder $TENSORBOARD_FOLDER\
                                --model_to_use=$MODEL_TO_USE \
                                --score_thresholds $SCORE_THRESHOLDS\
                                --iou_threshold $IOU_THRESHOLD\
                                --save="false"
            fi
            done
        done
    done
done
