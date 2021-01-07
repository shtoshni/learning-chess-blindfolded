# Learning Chess Blindfolded
Chess as a testbed for evaluating language models at world state tracking. 

## Setup
Data can be downloaded from [rebel](http://rebel13.nl/dl.html?file=dl/MillionBase%202.5%20(PGN).7z). <br/>
The rest of the processing assumes that we have the PGN file extracted. 

Install the packages specified in requirements.txt
```
pip install -r requirements.txt
```

```
cd src/
export PYTHONPATH=${PWD}:${PYTHONPATH}
```

## Data Preparation


Parsing PGN to get data in SAN annotation
```
python data_processing/parse_pgn.py --source_file ../data/chess/millionbase-2.5.pgn --output_dir ../data/lm_chess/human/millionbase/ --max_games 2e6
```
Filter data to remove duplicate games, games with skewed lengths (too short or too long), and games missing move annotations (rare case).
```
 python data_processing/filter_data.py --source_file ../data/lm_chess/human/millionbase/millionbase-san.txt --output_file ../data/lm_chess/human/millionbase/millionbase-san-uniq.txt
```
Converting between different chess notations can be done via:
```
python conversion_scripts/convert.py --source_file ../data/lm_chess/human/millionbase/millionbase-san-uniq.txt --target_notation uci
```

From the UCI converted file, random splits train/dev/test can be created. Currently, we use the first 100K games for train, and the last 80K games for various evaluations.
```
data_file="../data/lm_chess/human/millionbase/millionbase-uci-uniq.txt"
output_dir="../data/lm_chess/human/uci"
head -n 100000 ${data_file} > ${output_dir}/train.txt
tail -n 30000 ${data_file} | head -n 15000 > ${output_dir}/dev.txt
tail -n 15000 ${data_file} > ${output_dir}/test.txt

# Cloze task data
tail -n 80000 ${data_file} | head -n 50000 > ${output_dir}/other_eval.txt
```


Vocabulary creation - Script will automatically detect notation type
```
python data_processing/create_vocab_models.py --vocab_dir ../models/vocab --source_file ../data/lm_chess/human/uci/train.txt
```
Generate Cloze tasks
```
python data_processing/generate_cloze_eval_tasks.py -base_dir ../data/lm_chess/human/
```
Randomly Annotated Piecetypes (RAP) data can be generated as follows:
```
python conversion_scripts/rap_conversion.py --base_dir ../data/lm_chess/human/ --random_prob 0.15
```
Get Board Representation
```
python data_processing/generate_fen_repr.py -base_dir ../data/lm_chess/
python data_processing/get_fen_other_eval.py --base_dir ../data/lm_chess/
```
Querying data stats (average length of games etc.):
```
python data_processing/data_stats.py --data_dir ../data/lm_chess/human/uci/
```

## Training 

Default settings:
- Train-L i.e. 100K games for training
- GPT2-small configurations i.e. n_layer=12, n_head=12, n_embd=768
- Notation UCI

Here are the commands to train the various models. <br/>

Baseline UCI model
```
python main.py
```
UCI + RAP
```
python main.py --rap_prob 0.15
```
Multi-view
```
python main.py --multiview_margin 0.6
```
Oracle
```
python main.py --oracle
```
Custom training size and other model configurations can be specified as follows:
```
python main.py --train_size 15_000 --n_layer 16
``` 

## Analysis
_Random Legal Move Baseline_: Baseline where a random legal move is chosen 
as the predicted move. Performance of this baseline gives a sense of 
complexity of the task even if the exact board state is available. 

```
python analysis/random_legalmove_baseline.py ../data/lm_chess/other_eval/uci/
```
