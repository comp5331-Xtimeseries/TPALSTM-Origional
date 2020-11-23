#!/bin/bash

ATTEN=16
HORIZON=3
HORIZON_ELEC=3
BASHSIZE=32
EPOCHS=40

#Traffic train

python3 main.py --mode train  --attention_len $ATTEN --highway $ATTEN --horizon $HORIZON --data_set traffic --batch_size $BASHSIZE     --mts 1     --dropout 0.2     --learning_rate 1e-3     --model_dir ./models/traffic     --num_epochs $EPOCHS     --num_layers 3     --num_units 338 2>&1 | tee trafficTrainLog.txt


#ExchangeRate train

python3 main.py --mode train --attention_len $ATTEN --highway $ATTEN --horizon $HORIZON --data_set exchange_rate --batch_size $BASHSIZE      --mts 1    --dropout 0.2     --learning_rate 3e-3     --model_dir ./models/exchange_rate     --num_epochs $EPOCHS     --num_layers 3     --num_units 338 2>&1 | tee exchangeRateTrainLog.txt


#Electricity train

python3 main.py --mode train --attention_len $ATTEN --highway $ATTEN --horizon $HORIZON_ELEC --data_set electricity --batch_size $BASHSIZE     --mts 1     --dropout 0.2     --learning_rate 1e-3     --model_dir ./models/electricity     --num_epochs $EPOCHS     --num_layers 3     --num_units 338 2>&1 | tee electricityTrainLog.txt


#Solar train

python3 main.py --mode train --attention_len $ATTEN --highway $ATTEN --horizon $HORIZON --data_set solar --batch_size $BASHSIZE   --mts 1       --dropout 0.2     --learning_rate 1e-3     --model_dir ./models/solar     --num_epochs $EPOCHS     --num_layers 3     --num_units 338 2>&1 | tee solarTrainLog.txt

