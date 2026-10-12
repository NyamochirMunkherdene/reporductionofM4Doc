import os
from typing import Optional, Union

import torch
import torch.nn as nn
from transformers import GenerationConfig, PreTrainedModel
from transformers.modeling_outputs import Seq2SeqLMOutput
from types import MethodType

def fixed_prepare_inputs_for_generation(
    self,
    input_ids,
    past_key_values=None,
    attention_mask=None,
    **kwargs,
):
    """
    Preserve encoder conditioning during decoder generation.
    """

    if past_key_values is not None:
        input_ids = input_ids[:, -1:]

    if attention_mask is None:
        attention_mask = torch.ones(
            input_ids.shape,
            dtype=torch.long,
            device=input_ids.device,
        )

    prepared_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "use_cache": kwargs.get("use_cache", True),
    }

    encoder_hidden_states = kwargs.get(
        "encoder_hidden_states"
    )

    if encoder_hidden_states is not None:
        prepared_inputs["encoder_hidden_states"] = (
            encoder_hidden_states
        )

    encoder_attention_mask = kwargs.get(
        "encoder_attention_mask"
    )

    if encoder_attention_mask is not None:
        prepared_inputs["encoder_attention_mask"] = (
            encoder_attention_mask
        )

    return prepared_inputs



class BaseDIMTModelTrain(PreTrainedModel):
    def __init__(self, config, nougat_encoder, trans_decoder):
        super().__init__(config)
        self.nougat_encoder = nougat_encoder
        self.trans_decoder = trans_decoder

        nougat_hidden_size = nougat_encoder.config.hidden_size
        decoder_hidden_size = trans_decoder.config.hidden_size

        # This is random only when the model is first created.
        # During training, it learns how to map Nougat features
        # into the translation decoder's hidden space.
        self.nougat_down_proj = nn.Linear(
            nougat_hidden_size,
            decoder_hidden_size,
        )
        self.loss_function = nn.CrossEntropyLoss(ignore_index=-100)

    def gradient_checkpointing_enable(self):
        self.nougat_encoder.gradient_checkpointing_enable()
        self.trans_decoder.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.nougat_encoder.gradient_checkpointing_disable()
        self.trans_decoder.gradient_checkpointing_disable()

    def forward(
        self,
        nougat_pixel_values: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = True,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs,):


        nougat_outputs = self.nougat_encoder(
            pixel_values=nougat_pixel_values,
            return_dict=True,
        )
        encoder_hidden_states = self.nougat_down_proj(
            nougat_outputs.last_hidden_state
        )

        decoder_outputs = self.trans_decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

                    

        loss = None
        logits = decoder_outputs.logits
        if labels is not None:
            if logits.shape[:2] != labels.shape:
                raise ValueError(
                    f"Logits sequence shape "
                    f"{logits.shape[:2]} does not match "
                    f"labels shape {labels.shape}"
                )

            loss = self.loss_function(
                logits.reshape(
                    -1,
                    logits.size(-1),
                ),
                labels.reshape(-1),
            )

        return Seq2SeqLMOutput(
            loss=loss,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_hidden_states,
        )

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        state_dict: Optional[dict] = None,
        safe_serialization: bool = False,
        **kwargs,
    ):
        save_dir = os.path.join(save_directory, "state_dict")
        os.makedirs(save_dir, exist_ok=True)
        self.config.save_pretrained(save_directory)
        torch.save(
            self.nougat_encoder.state_dict(),
            os.path.join(save_dir, "nougat_encoder.bin"),
        )
        torch.save(
            self.trans_decoder.state_dict(),
            os.path.join(save_dir, "trans_decoder.bin"),
        )
        torch.save(
            self.nougat_down_proj.state_dict(),
            os.path.join(save_dir, "nougat_down_proj.bin"),
        )


class BaseDIMTModelInference(PreTrainedModel):
    def __init__(self, config, nougat_encoder, trans_decoder):
        super().__init__(config)
        self.nougat_encoder = nougat_encoder
        self.trans_decoder = trans_decoder

        nougat_hidden_size = nougat_encoder.config.hidden_size
        decoder_hidden_size = trans_decoder.config.hidden_size
        self.nougat_down_proj = nn.Linear(
            nougat_hidden_size,
            decoder_hidden_size,
        )
        # The decoder's original generation preparation drops
        # encoder_hidden_states. Bind the corrected implementation.
        self.trans_decoder.prepare_inputs_for_generation = (
            MethodType(
                fixed_prepare_inputs_for_generation,
                self.trans_decoder,
            )
        )


    def generate(
        self,
        nougat_pixel_values=None,
        generation_config=None,
        **kwargs,
    ):
        if nougat_pixel_values is None:
            raise ValueError(
                "nougat_pixel_values must be provided."
            )

        if generation_config is None:
            generation_config = (
                GenerationConfig.from_model_config(
                    self.config
                )
            )

        decoder_start_token_id = (
            generation_config.decoder_start_token_id
        )

        if decoder_start_token_id is None:
            decoder_start_token_id = (
                self.config.decoder_start_token_id
            )

        if decoder_start_token_id is None:
            raise ValueError(
                "decoder_start_token_id must be configured."
            )

        nougat_outputs = self.nougat_encoder(
            pixel_values=nougat_pixel_values,
            return_dict=True,
        )

        encoder_hidden_states = self.nougat_down_proj(
            nougat_outputs.last_hidden_state
        )

        batch_size = encoder_hidden_states.size(0)

        decoder_input_ids = torch.full(
            (batch_size, 1),
            fill_value=decoder_start_token_id,
            dtype=torch.long,
            device=encoder_hidden_states.device,
        )

        return self.trans_decoder.generate(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            **kwargs,
        )