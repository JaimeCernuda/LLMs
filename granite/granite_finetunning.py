# def get_model(
#     model_path,
#     model_name: str = "ttm",
#     context_length: int = None,
#     prediction_length: int = None,
#     freq_prefix_tuning: bool = None,
#     **kwargs,
# ):
    
#     TTM Model card offers a suite of models with varying context_length and forecast_length combinations.
#     This wrapper automatically selects the right model based on the given input context_length and prediction_length abstracting away the internal
#     complexity.

#     Args:
#         model_path (str):
#             HF model card path or local model path (Ex. ibm-granite/granite-timeseries-ttm-r1)
#         model_name (*optional*, str)
#             model name to use. Allowed values: ttm
#         context_length (int):
#             Input Context length. For ibm-granite/granite-timeseries-ttm-r1, we allow 512 and 1024.
#             For ibm-granite/granite-timeseries-ttm-r2 and  ibm/ttm-research-r2, we allow 512, 1024 and 1536
#         prediction_length (int):
#             Forecast length to predict. For ibm-granite/granite-timeseries-ttm-r1, we can forecast upto 96.
#             For ibm-granite/granite-timeseries-ttm-r2 and  ibm/ttm-research-r2, we can forecast upto 720.
#             Model is trained for fixed forecast lengths (96,192,336,720) and this model add required `prediction_filter_length` to the model instance for required pruning.
#             For Ex. if we need to forecast 150 timepoints given last 512 timepoints using model_path = ibm-granite/granite-timeseries-ttm-r2, then get_model will select the
#             model from 512_192_r2 branch and applies prediction_filter_length = 150 to prune the forecasts from 192 to 150. prediction_filter_length also applies loss
#             only to the pruned forecasts during finetuning.
#         freq_prefix_tuning (*optional*, bool):
#             Future use. Currently do not use this parameter.
#         kwargs:
#             Pass all the extra fine-tuning model parameters intended to be passed in the from_pretrained call to update model configuration.

# Load Model from HF Model Hub mentioning the branch name in revision field


model = TinyTimeMixerForPrediction.from_pretrained(
                "https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2", revision="main"
            )

or

from tsfm_public.toolkit.get_model import get_model
model = get_model(
            model_path="https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2",
            context_length=512,
            prediction_length=96
        )



# Do zeroshot
zeroshot_trainer = Trainer(
        model=model,
        args=zeroshot_forecast_args,
        )
    )

zeroshot_output = zeroshot_trainer.evaluate(dset_test)


# Freeze backbone and enable few-shot or finetuning:

# freeze backbone
for param in model.backbone.parameters():
  param.requires_grad = False

finetune_model = get_model(
            model_path="https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2",
            context_length=512,
            prediction_length=96,
            # pass other finetune params of decoder or head
            head_dropout = 0.2
        )

finetune_forecast_trainer = Trainer(
        model=model,
        args=finetune_forecast_args,
        train_dataset=dset_train,
        eval_dataset=dset_val,
        callbacks=[early_stopping_callback, tracking_callback],
        optimizers=(optimizer, scheduler),
    )
finetune_forecast_trainer.train()
fewshot_output = finetune_forecast_trainer.evaluate(dset_test)
