

import pandas as pd
import logging

from sympy import false, true

from seq2seq_model import Seq2SeqModel

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


data = pd.read_csv("./data/train_causal_timebank.csv", sep=',',encoding = 'gbk',usecols=[0,1]).values.tolist()
# data = pd.read_csv("./data/train_eventstoryline.csv", sep=',',encoding = 'UTF-8',usecols=[0,1]).values.tolist()
df = pd.DataFrame(data, columns=["input_text", "target_text"])
train_df=df[:int(df.shape[0] * 0.8)]
eval_df=df[~df.index.isin(train_df.index)]


model_args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "max_seq_length": 160,
    "train_batch_size": 36,
    "num_train_epochs": 10,
    "save_eval_checkpoints": False,
    "save_model_every_epoch": True,
    "evaluate_during_training": True,
    "evaluate_generated_text": True,
    "evaluate_during_training_verbose": True,
    "use_multiprocessing": False,
    "max_length": 32,
    "manual_seed": 3407,
    "save_steps": 11898,
    "gradient_accumulation_steps": 1,
    # "output_dir": "./exp/eventstoryline_relpos_regu",
    "output_dir": "./exp/causal_timebank_relpos_regu",
}

# Initialize model
model = Seq2SeqModel(
    encoder_decoder_type="bart",
    encoder_decoder_name="facebook/bart-base",
    args=model_args,
    cuda_device = 1 ,
    regularization = True,
    # use_cuda=False,
)



# Train the model
model.train_model(train_df, eval_data=eval_df)

# Evaluate the model
results = model.eval_model(eval_df)

# Use the model for prediction

print(model.predict(["During Lohan's stay at the famed Betty Ford Center in 2010, Lohan was accused of attacking a female staffer"]))


