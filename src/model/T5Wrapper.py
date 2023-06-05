import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.utils import round_nestedList
from src.model.generation import greedy_generation


class T5Wrapper(nn.Module):
    """ """

    def __init__(self, transformer, tokenizer):

        super().__init__()
        self.transformer = transformer
        self.tokenizer = tokenizer

    def forward(self, batch):
        transformer_outputs = self.transformer(
            input_ids=batch["input_ids"],
            attention_mask=batch["input_mask"],
            labels=batch["target_ids"],
        )

        # [batch_size, max_target_len, vocab_size]
        target_logits = transformer_outputs[1].float()
        vocab_size = target_logits.shape[-1]

        # Compute the log probability of the ids for all choices with respect to the logits
        # [batch_size x max_target_len]
        logProbs_ofTargetIds = F.cross_entropy(
            target_logits.reshape(-1, vocab_size),
            batch["target_ids"].reshape(-1),
            reduction="none",
        )
        # Zero out log_probs for target_ids with no loss
        target_mask = batch["target_mask"].reshape(-1)
        logProbs_ofTargetIds_zeroOutPadIds = logProbs_ofTargetIds * target_mask

        loss = torch.sum(logProbs_ofTargetIds_zeroOutPadIds) / torch.sum(target_mask)

        return loss, {"loss": loss.detach().cpu().item()}

    def _broadcast_tensors(self, input_masks, encoder_outputs, num_choices):
        """
        Broadcast the input masks and encoder outputs to account for multiple choices per input

        Args:
            input_masks: [batch_size, max_input_len]
            encoder_outputs: BaseModelOutput object from HuggingFace where the first element is
                            the hidden states of the encoder at the last layer
                            [batch_size, max_input_len, ff_dim]
            num_choices:

        Returns:
            input_masks: [batch_size x num_choices, max_input_len]
            encoder_outputs: BaseModelOutput object from HuggingFace where the first element is
                            the hidden states of the encoder at the last layer
                            [batch_size x num_choices, max_input_len, ff_dim]
        """
        input_masks = torch.repeat_interleave(input_masks, num_choices, dim=0)
        encoder_outputs = (
            torch.repeat_interleave(encoder_outputs[0], num_choices, dim=0),
        )
        return input_masks, encoder_outputs

    def compute_logProb(
        self,
        logProbs_ofAllChoices_ids,
        allChoices_masks,
        num_choices,
        maxChoice_len,
        length_normalization,
    ):
        """
        Args:
            logProbs_forAllChoices_ids: [batch_size x num_choices x max_choice_len]
            allChoices_masks: [batch_size, num_choices, max_choice_len]
            num_choices:
            maxChoice_len:
            length_normalization:

        Returns:
            logProbs_ofAllChoices: [batch_size, num_choices]
            logProbs_ofAllChoicesIds_zeroOutPadIds: [batch_size, num_choices, max_choice_len]
            len_allChoices: [batch_size, num_choices ]
        """
        # Compute the log probabilities of all the choices by summing the log probabilities of
        # each token
        # [batch_size, num_choices, max_choice_len]
        logProbs_ofAllChoices_ids = logProbs_ofAllChoices_ids.reshape(
            -1, num_choices, maxChoice_len
        )
        allChoices_masks = allChoices_masks.reshape(-1, num_choices, maxChoice_len)
        # Zero out padded out tokens so we their log probability is not included
        logProbs_ofAllChoicesIds_zeroOutPadIds = (
            logProbs_ofAllChoices_ids * allChoices_masks
        )

        logProbs_ofAllChoices = torch.sum(logProbs_ofAllChoicesIds_zeroOutPadIds, dim=2)

        # Store the length of each choice as additional metadata that can be used later to
        # compute length normalized score
        len_allChoices = torch.sum(allChoices_masks, dim=2)

        if length_normalization:
            logProbs_ofAllChoices = logProbs_ofAllChoices / len_allChoices

        return (
            logProbs_ofAllChoices,
            logProbs_ofAllChoicesIds_zeroOutPadIds,
            len_allChoices,
        )

    def compute_logProb_ofAllChoices(
        self,
        input_ids,
        input_masks,
        allChoices_ids,
        allChoices_masks,
        length_normalization,
    ):
        """
        Computes log probabilties for all the choices. This should be used when the number of
        choices is greater than 1. It computes the encoder hidden representations once, broadcasts
         it to match the number of choices, and then computes the log prob for each choice.

        Args:
            input_ids: [batch_size, max_input_len]
            input_masks: [batch_size, max_input_len]
            allChoices_ids: [batch_size x num_choices, max_choice_len]
            allChoices_masks: [batch_size x num_choices, max_choice_len]
            length_normalization:

        Returns:
            log_prob: [batch_size x num_choices, max_choice_len]
            logProbs_ofAllChoicesIds_zeroOutPadIds: [batch_size, num_choices, max_choice_len]
            len_allChoices: [batch_size, num_choices ]
        """
        encoder_outputs = self.transformer.get_encoder()(input_ids, input_masks)

        assert allChoices_ids.shape[0] % input_masks.shape[0] == 0, (
            f"The batch size {allChoices_ids.shape[0]} of allChoices_ids is not a multiple of "
            f"the batch size {input_masks.shape[0]} of input_masks"
        )

        num_choices = allChoices_ids.shape[0] // input_masks.shape[0]

        input_masks, encoder_outputs = self._broadcast_tensors(
            input_masks, encoder_outputs, num_choices
        )

        # WARNING: The loss at transformer_outputs[0] is not valid, since allChoices_ids uses a
        # pad token of 0 and so the loss will not be ignored for the pad tokens
        # The input mask is passed in for the cross encoder-decoder attention.
        transformer_outputs = self.transformer(
            attention_mask=input_masks,
            encoder_outputs=encoder_outputs,
            labels=allChoices_ids,
        )

        # We used the logits for all choices to compute the log probs per example since
        # the loss returned in transformer_outputs will average the negative log probs across
        # examples
        # [batch_size x num_choices, max_choice_len, vocab_size]
        logits_ofAllChoices = transformer_outputs[1].float()
        maxChoice_len = logits_ofAllChoices.shape[1]
        vocab_size = logits_ofAllChoices.shape[-1]

        # Compute the log probability of the ids for all choices with respect to the logits
        # [batch_size x num_choices x max_choice_len]
        logProbs_ofAllChoices_ids = -F.cross_entropy(
            logits_ofAllChoices.view(-1, vocab_size),
            allChoices_ids.view(-1),
            reduction="none",
        )

        return self.compute_logProb(
            logProbs_ofAllChoices_ids,
            allChoices_masks,
            num_choices,
            maxChoice_len,
            length_normalization,
        )

    def predict_mulChoice(self, batch, length_normalization):
        """

        Args:
            batch:
            length_normalization:

        Returns:
            pred_choice: [batch_size, ]
            score_ofChoices: [batch_size, num_choices]
            logProbs_ofAllChoicesIds: [batch_size, num_choices, max_choice_len]
            len_allChoices: [batch_size]
        """
        # Compute log p(y|x)
        (
            score_ofChoices,
            logProbs_ofAllChoicesIds,
            len_allChoices,
        ) = self.compute_logProb_ofAllChoices(
            batch["input_ids"],
            batch["input_mask"],
            batch["all_choices_ids"],
            batch["all_choices_mask"],
            length_normalization,
        )

        _, predicted_choice = torch.max(score_ofChoices, dim=1)

        return (
            predicted_choice.cpu().numpy().tolist(),
            round_nestedList(score_ofChoices.cpu().numpy().tolist(), 5),
            round_nestedList(logProbs_ofAllChoicesIds.cpu().numpy().tolist(), 4),
            len_allChoices.cpu().numpy().tolist(),
        )

    def generate(self, batch, max_generationLength):
        """

        Args:
            batch:
            max_generationLength:

        Returns:

        """
        # The bos_token_id is equal to the pad_token_id for T5
        generated_ids = greedy_generation(
            self.transformer,
            batch["input_ids"],
            batch["input_mask"],
            self.tokenizer.pad_token_id,
            self.tokenizer.eos_token_id,
            self.tokenizer.pad_token_id,
            max_generationLength,
        )

        generated_ids = generated_ids.cpu().numpy().tolist()
        generated_txt = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        return generated_ids, generated_txt
