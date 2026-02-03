# Mistral 7b architecure:
#
# MistralForCausalLM(
#   (model): MistralModel(
#     (embed_tokens): Embedding(32000, 4096)
#     (layers): ModuleList(
#       (0-31): MistralDecoderLayer(
#         (self_attn): MistralSdpaAttention(
#           (q_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)
#           (k_proj): Linear4bit(in_features=4096, out_features=1024, bias=False)
#           (v_proj): Linear4bit(in_features=4096, out_features=1024, bias=False)
#           (o_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)
#           (rotary_emb): MistralRotaryEmbedding()
#         )
#         (mlp): MistralMLP(
#           (gate_proj): Linear4bit(in_features=4096, out_features=14336, bias=False)
#           (up_proj): Linear4bit(in_features=4096, out_features=14336, bias=False)
#           (down_proj): Linear4bit(in_features=14336, out_features=4096, bias=False)
#           (act_fn): SiLU()
#         )
#         (input_layernorm): MistralRMSNorm((4096,), eps=1e-05)
#         (post_attention_layernorm): MistralRMSNorm((4096,), eps=1e-05)
#       )
#     )
#     (norm): MistralRMSNorm((4096,), eps=1e-05)
#   )
#   (lm_head): Linear(in_features=4096, out_features=32000, bias=False)
# )

import gc
import os
import sys
import torch
from datasets import load_dataset
from lightning import Trainer
from lightning.pytorch import LightningDataModule, LightningModule
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from huggingface_hub import login

from dotenv import load_dotenv
load_dotenv()


# NOTE: FINE TUNER MODEL
#
class FineTuner(LightningModule):
    def __init__(self, model_name):
        super().__init__()
        self.save_hyperparameters()

        # quantization config
        # store values in 4bit. convert to 16bit during math
        # nf4 for bell curve lora compression
        # double quant for scaling factor compression
        bits_bytes_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dytpe=torch.float16,
            bnb_4bit_double_quant=True,
        )

        # load model in 4bit
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bits_bytes_config,
            device_map='auto',
            trust_remote_code=True,
        )
        # freeze the model so its pre-existing weights are uneffected
        for param in self.model.parameters():
            param.requires_grad = False

        # lora config adapter
        # r = rank (adapter size)
        # lora_alpha = scaling factor
        # only train specific layers: target_modules
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "lm_head",
            ],
        )

        # attach the adapter
        self.model = get_peft_model(self.model, lora_config)

        # sanity check
        self.model.print_trainable_parameters()


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=5e-5, weight_decay=1e-4)

    def forward(self, input_ids, labels=None):
        return self.model(input_ids=input_ids, labels=labels)

    def shared_step(self, mode, batch, batch_index):
        input_ids, labels = batch
        pred = self(input_ids=input_ids, labels=labels)
        loss = pred.loss # auto calculates the loss
        # perplexity = torch.exp(loss)
        self.log(f"{mode}_step_loss", loss, prog_bar=True)
        # log and show perplexity
        # self.log(f"{mode}_perplexity", perplexity, prog_bar=True)
        return loss

    def training_step(self, batch, batch_index):
        return self.shared_step("train", batch, batch_index)

    def validation_step(self, batch, batch_index):
        return self.shared_step("val", batch, batch_index)

    def test_step(self, batch, batch_index):
        return self.shared_step("test", batch, batch_index)


# NOTE: DATA PREPARATION
#
class DataModule(LightningDataModule):
    def __init__(self, model_name, num_workers=0, batch_size=2, max_length=128):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model_name = model_name
        self.max_length = max_length

        # get the tokenizer from the model
        # and add EOS pad tokens to models lacking it
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = 'right'

    def setup(self, stage=None):
        # load data from hugging face stream
        # training:      0 -> 1000
        # validation:    1000 -> 1200
        # testing:       1200 -> 1400
        dataset = load_dataset(
            "Despina/project_gutenberg", 
            "fiction_books", 
            split="train", 
            streaming=True
        )
        dataset = dataset.shuffle(seed=38, buffer_size=1000)
        self.train_dataset = dataset.take(1000)
        self.val_dataset = dataset.skip(1000).take(200)
        self.test_dataset = dataset.skip(1200).take(200)

    def collate_fn(self, batch):
        # the preprocessing is a function thats given to the dataloader
        # the dataloader preprocesses on the fly between batches
        # get each items text and convert to tokens(numbers)
        # return (features, labels)
        texts = [item['text'] for item in batch]
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt" # return pytorch tensors
        )
        return encodings['input_ids'], encodings['input_ids']

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

# NOTE: MODEL USAGE
#
def generate_output_from_input(model, tokenizer, prompt, max_length=60):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)

    output = model.model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,

        # baseline = without improvement
        # controlled = with improvement
        # for improved generation:
        top_k=50,
        do_sample=True,
        temperature=0.7,
        repetition_penalty=1.2
    )

    decoded_text = tokenizer.decode(output[0], skip_special_tokens=True)
    model.train()
    return decoded_text


# NOTE: MODEL TRAINING / TESTING
#
if __name__ == "__main__":
    # example: 'python milestone2.py 2 2 "The brave warrior"'
    if len(sys.argv) == 4:
        epochs = int(sys.argv[1])
        batch_size = int(sys.argv[2])
        input_text = str(sys.argv[3])
    else:
        epochs = 1
        batch_size = 2
        input_text = "The brave warrior"

    # attempt to fetch HF token from dotenv
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
    else:
        print('NO HF_TOKEN FOUND: REDUCED RATES')

    torch.set_float32_matmul_precision("medium")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "mistralai/Mistral-7B-v0.1"
    data_module = DataModule(batch_size=batch_size, model_name=model_name)
    untrained_model = FineTuner(model_name=model_name)

    # untrained model test
    untrained_model.to(device)
    untrained_output_text = generate_output_from_input(untrained_model, data_module.tokenizer, input_text)
    trainer = Trainer(logger=False, accelerator="auto")
    print('\nUNTRAINED MODEL:')
    trainer.test(model=untrained_model, datamodule=data_module)

    # training
    logger = CSVLogger("logs", name="module")
    early_stop = EarlyStopping(
        monitor="val_step_loss", min_delta=0.00, patience=5, verbose=True, mode="min"
    )
    checkpoint = ModelCheckpoint(monitor="val_step_loss", mode="min", save_top_k=1)
    trainer = Trainer(
        logger=logger,
        max_epochs=epochs,
        log_every_n_steps=1,
        accelerator="auto",
        callbacks=[early_stop, checkpoint],
        gradient_clip_val=1.0,
    )
    trainer.fit(model=untrained_model, datamodule=data_module)

    # memory cleanup
    del untrained_model
    del trainer
    gc.collect() # force garbage collection
    torch.cuda.empty_cache()

    # load the best model from checkpoints
    best_model_path = checkpoint.best_model_path
    trained_model = FineTuner.load_from_checkpoint(best_model_path)

    # trained model test
    trained_model.to(device)
    trained_output_text = generate_output_from_input(trained_model, data_module.tokenizer, input_text)
    trainer = Trainer(logger=False, accelerator="auto")
    print('\nTRAINED MODEL:')
    trainer.test(model=trained_model, datamodule=data_module)

    # generation comparison
    print('\nUNTRAINED MODEL:')
    print(f'OUTPUT: {untrained_output_text}')
    print('\nTRAINED MODEL:')
    print(f'OUTPUT: {trained_output_text}')
