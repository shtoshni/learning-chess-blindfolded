# lm-chess
Language model for Chess


## Data Preparation
Parsing PGN to get data in SAN annotation
```
python data_processing/parse_pgn.py --source_file ../data/chess/millionbase-2.5.pgn --output_dir /tmp/ --max_games 2e6
```
Filter data to remove duplicate games, games with skewed lengths, and games missing move annotations (rare case).
```
 python data_processing/filter_data.py --source_file ../data/lm_chess/human/millionbase/millionbase-san.txt --output_file ../data/lm_chess/human/millionbase/millionbase-san-uniq.txt
```
Converting between different chess notations can be done via:
```
python conversion_scripts/convert.py --source_file ../data/lm_chess/synthetic/source/synthetic_lan.txt --target_notation lan_with_p
```
Randomly Annotated Piecetypes (RAP) data can be generated as follows
```
python conversion_scripts/rap_conversion.py --base_dir ../data/lm_chess/synthetic/ --random_prob 0.15
```
Querying data stats:
```
python data_processing/data_stats.py --data_dir ../data/lm_chess/human/uci/
```
Vocabulary creation - Script will automatically detect notation type
```
python data_processing/create_vocab_models.py --vocab_dir ../models/vocab --source_file ../data/lm_chess/human/uci/train_medium.txt
```
Cloze tasks
```
python data_processing/generate_cloze_eval_tasks.py -base_dir ../data/lm_chess/synthetic/
```
Generate FEN Representation
```
python data_processing/generate_fen_repr.py -base_dir ../data/lm_chess/human/
```