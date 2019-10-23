**train_sharp.py - training and validation script**

usage: 
``` {
      python train_sharp.py -h -g GPU -e EPOCHS -b TRAIN_BATCH_SIZE \
                      -B VAL_BATCH_SIZE
                      --batches-before-val BATCHES_BEFORE_VAL
                      --mini-val-batches-n-per-subset MINI_VAL_BATCHES_N_PER_SUBSET
                      --sheduler SHEDULER --model-spec MODEL_SPEC_FILENAME
                      --infer-from-spec --log-dir-prefix LOGS_DIR
                      -m INIT_MODEL_FILENAME -s SAVE_MODEL_FILENAME
                      --batches_before_save BATCHES_BEFORE_SAVE
                      --data-root DATA_ROOT --num-points NUM_POINTS
                      --lr LR --scheduler SCHEDULER
                      --loss-funct LOSS_FUNCT
                      --end-batch-train END_BATCH_TRAIN
                      --end-batch-val END_BATCH_VAL --verbose
                      -l LOGGING_FILENAME -tl TBOARD_JSON_LOGGING_FILE
                      -x TBOARD_DIR -w, engine=bash }
```
**important:** --data-root - path to directory which contains data/ with hdf5 files with ABC dataset (see data.py for details)
           --loss-funct - currently only cal_loss is implemented (default cal_loss)
           
**TODO:** 
- [ ] metrics 
- [ ] consider mini-validation split
           
