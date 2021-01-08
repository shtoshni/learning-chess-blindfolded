# Learning Chess Blindfolded
Chess as a testbed for evaluating language models on world state tracking.

## Setup
Step 1
```
git clone https://github.com/shtoshni92/learning-chess-blindfolded.git
cd learning-chess-blindfolded/
```

Step 2: Install packages. The following are the core pacakage which can be separately installed.
```
chess==1.3.0
pytorch-lightning==0.9.0
torch==1.7.0.dev20200719
transformers==3.1.0
prettytable==0.7.2
```

Or just do:
```
pip install -r requirements.txt
```

Finally
```
cd src/
export PYTHONPATH=${PWD}:${PYTHONPATH}
```

## Data Preparation

The processed data is available [here](https://drive.google.com/file/d/1G4TA4KVdn5mLvMXsRI2bGC7VIdzqkr86/view?usp=sharing).
UCI-based language models can be trained using just this data.
To train models which require piecetype/board state, extract this additional information via steps described [below](#additional-board-state).

Next we described the steps used processing the data.

- Data can be downloaded from [rebel](http://rebel13.nl/dl.html?file=dl/MillionBase%202.5%20(PGN).7z). <br/>

- Parse PGN to get data in UCI annotation (max_games to extract can be specified)
```
python data_processing/parse_pgn.py --source_file PGN_FILE --output_dir OUTPUT_DIR --max_games 1e7
```
- Filter data to remove duplicate games, games with skewed lengths (too short or too long), and games missing move annotations (rare case). 
If output_file is not specified, a suffix of "-uniq" is added to source file name.  
```
 python data_processing/filter_data.py --source_file INPUT_FILE --output_file OUTPUT_FILE
```


- Next we create partitions of the processed data
```
src_file=OUTPUT_FILE

cd ../data
output_dir="lm_chess/uci"
mkdir -p ${output_dir}

head -n 500000 ${src_file} > ${output_dir}/train.txt

tail -n 30000 ${src_file} | head -n 15000 > ${output_dir}/dev.txt
tail -n 15000 ${src_file} > ${output_dir}/test.txt
# Cloze task data
tail -n 130000 ${src_file} | head -n 50000 > ${output_dir}/other_eval.txt

cd ../src
DATA_DIR="../data/lm_chess"
```

- Create vocabulary
```
python data_processing/create_vocab_models.py --vocab_dir $DATA_DIR/vocab --source_file $DATA_DIR/uci/train.txt
```
- Querying data stats (average length of games etc.):
```
python data_processing/data_stats.py --data_dir $DATA_DIR --vocab_dir $DATA_DIR/vocab/
```
- Create Cloze tasks (Ending Square and Starting Square)
```
python data_processing/generate_cloze_eval_tasks.py --data_dir $DATA_DIR
```

### Additional Board State
Tne next two steps create additional information regarding the world state.
- This step extracts piecetype information as a numpy file for all the language modeling data splits.
```
python data_processing/get_piece_type_rap.py --data_dir $DATA_DIR
```
- This step extracts the board state from the [FEN notation](https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation)
  and stores it as a numpy file for different splits.
```
python data_processing/get_fen_repr.py --data_dir $DATA_DIR
python data_processing/get_fen_other_eval.py --data_dir $DATA_DIR
```

## Training 

Default settings:
- Train-L i.e. 200K games for training
- GPT2-small configurations i.e. n_layer=12, n_head=12, n_embd=768
- Notation UCI

Here are the commands to train the various models. <br/>

Baseline UCI model
```
python main.py --data_dir $DATA_DIR
```
UCI + RAP 
```
python main.py --rap_prob 0.15 --data_dir $DATA_DIR
# To use the loss from predicting piecetypes in the sequence
python main.py --rap_prob 0.15 --rap_grad --data_dir $DATA_DIR
```
Multi-view
```
python main.py --multiview --multiview_margin 0.6 --data_dir $DATA_DIR
```
Oracle
```
python main.py --oracle --data_dir $DATA_DIR
# To use the loss from predicting piecetypes in the sequence
python main.py --oracle --rap_grad --data_dir $DATA_DIR
```
Custom training size, number of layers, context size, and other model configurations can be specified as follows:
```
python main.py --train_size 15_000 --n_layer 16 --window_size 50 --data_dir $DATA_DIR
``` 

RNN models can be trained via:
```
python main.py --model_type rnn --n_layer 3 --rnn_dropout 0.2 --data_dir $DATA_DIR
```

## Analysis
_Random Legal Move Baseline_: Baseline where a random legal move is chosen 
as the predicted move. Performance of this baseline gives a sense of 
complexity of the task even if the exact board state is available. 

```
python analysis/random_legalmove_baseline.py --data_dir $DATA_DIR
```

_Error Analysis for Ending Squares_: Classifies the error made by the model among 
three categories, namely syntactic, pseudo legal, and path obstruction. 
```
python analysis/error_analysis_end.py --model_dir $MODEL_DIR
```