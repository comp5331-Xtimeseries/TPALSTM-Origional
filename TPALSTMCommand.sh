#!/bin/bash

#Traffic train

python3 main.py --mode train  --attention_len 168 --highway 168 --horizon 12 --save_name traffic --data_set traffic --batch_size 32     --mts 1     --dropout 0.2     --learning_rate 1e-3     --model_dir ./models/model     --num_epochs 40     --num_layers 3     --num_units 338 2>&1 | tee trafficTrainLog.txt

#Traffic test

python3 main.py --mode test --attention_len 168 --highway 168 --horizon 12 --save_name traffic --data_set traffic --batch_size 32      --mts 1    --dropout 0.2     --learning_rate 1e-3     --model_dir ./models/model     --num_epochs 40     --num_layers 3     --num_units 338 2>&1 | tee trafficTestLog.txt

#ExchangeRate train

python3 main.py --mode train --attention_len 168 --highway 168 --horizon 12 --save_name exchange_rate --data_set exchange_rate --batch_size 32      --mts 1    --dropout 0.2     --learning_rate 3e-3     --model_dir ./models/model     --num_epochs 40     --num_layers 3     --num_units 338 2>&1 | tee exchangeRateTrainLog.txt

#ExchangeRate test

python3 main.py --mode test --attention_len 168 --highway 168 --horizon 12 --save_name exchange_rate --data_set exchange_rate --batch_size 32      --mts 1    --dropout 0.2     --learning_rate 3e-3     --model_dir ./models/model     --num_epochs 40     --num_layers 3     --num_units 338 2>&1 | tee exchangeRateTestLog.txt

#Electricity train

python3 main.py --mode train --attention_len 168 --highway 168 --horizon 24 --save_name electricity --data_set electricity --batch_size 32     --mts 1     --dropout 0.2     --learning_rate 1e-3     --model_dir ./models/model     --num_epochs 40     --num_layers 3     --num_units 338 2>&1 | tee electricityTrainLog.txt

#Electricity test

python3 main.py --mode test --attention_len 168 --highway 168 --horizon 24 --save_name electricity --data_set electricity --batch_size 32     --mts 1     --dropout 0.2     --learning_rate 1e-3     --model_dir ./models/model     --num_epochs 40     --num_layers 3     --num_units 338 2>&1 | tee electricityTestLog.txt


#Solar train

python3 main.py --mode train --attention_len 168 --highway 168 --horizon 12 --save_name solar --data_set solar --batch_size 32   --mts 1       --dropout 0.2     --learning_rate 1e-3     --model_dir ./models/model     --num_epochs 40     --num_layers 3     --num_units 338 2>&1 | tee solarTrainLog.txt

#Solar test

python3 main.py --mode test --attention_len 168 --highway 168 --horizon 12--save_name solar --data_set solar --batch_size 32   --mts 1       --dropout 0.2     --learning_rate 1e-3     --model_dir ./models/model     --num_epochs 40     --num_layers 3     --num_units 338 2>&1 | tee solarTestLog.txt
