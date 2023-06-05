import torch


def greedy_generation(
    transformer,
    input_ids,
    input_mask,
    bos_tokenId,
    eos_tokenId,
    pad_tokenId,
    max_generationLength,
):
    """
    Assumes model is encoder_decoder model and caches input first

    Args:
        model:
        input_ids:
        input_mask:
        bos_tokenId:
        eos_tokenId:
        pad_tokenId:
        max_generationLength:

    Returns:
        generated_ids: [batch_size, max_generationLength]
    """
    past_key_values = None
    batch_size = input_ids.shape[0]

    # Decode starting with bos_token_id
    # [batch_size, 1]
    current_decoderInputIds = torch.tensor([bos_tokenId] * batch_size)[:, None].to(
        input_ids.device
    )
    # Decoder mask is fixed to always be 1. We don't need to ignore any tokens in the decoder
    # since we just truncate any token after the eos token
    # [batch_size, 1]
    current_decoderMask = torch.ones((batch_size, 1)).to(input_ids.device)

    encoder_outputs = transformer.get_encoder()(input_ids, input_mask)

    generated_ids = current_decoderInputIds

    hasSequence_hitEOS = torch.zeros(size=(batch_size, 1), dtype=torch.int).to(
        input_ids.device
    )

    for i in range(max_generationLength):
        # attention_mask must be passed in for encoder_decoder models, even if we pass the
        # encoder_outputs, since the attention_mask is used to compute the cross_attention mask
        # for encoder decoder models
        output = transformer(
            attention_mask=input_mask,
            decoder_input_ids=current_decoderInputIds,
            decoder_attention_mask=current_decoderMask,
            encoder_outputs=encoder_outputs,
            use_cache=True,
            past_key_values=past_key_values,
        )

        # Update current key values
        past_key_values = output.past_key_values

        predicted_nextToken = torch.argmax(output.logits, -1)

        # If sequence has hit end, then every token afterwards should be a PAD token
        predicted_nextToken = (
            1 - hasSequence_hitEOS
        ) * predicted_nextToken + hasSequence_hitEOS * pad_tokenId

        generated_ids = torch.cat((generated_ids, predicted_nextToken), dim=1)

        # Update whether has sequence has hit end of sequence
        isToken_EOSToken = predicted_nextToken == eos_tokenId
        hasSequence_hitEOS = torch.bitwise_or(hasSequence_hitEOS, isToken_EOSToken)

        # Exit loop if every sequence has hit EOS
        if torch.sum(hasSequence_hitEOS) == batch_size:
            break

        current_decoderInputIds = predicted_nextToken

    return generated_ids
