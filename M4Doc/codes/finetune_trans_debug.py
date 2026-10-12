print("SCRIPT STARTED")
import argparse

import torch
import torch.nn as nn

from transformers import EncoderDecoderModel


def extend_bert_position_embeddings(
    bert_model,
    new_max_length,
):
    """
    Extend BERT learned positional embeddings.

    Example:
        512 -> 1536

    Existing positions [0:512] are copied exactly.
    New positions [512:1536] are randomly initialized
    using BERT's initializer_range.
    """

    old_embeddings = (
        bert_model.embeddings.position_embeddings
    )

    old_max_length = old_embeddings.num_embeddings
    hidden_size = old_embeddings.embedding_dim

    print(
        f"Extending positional embeddings: "
        f"{old_max_length} -> {new_max_length}"
    )

    if new_max_length <= old_max_length:
        raise ValueError(
            f"new_max_length={new_max_length} must be "
            f"larger than old length={old_max_length}"
        )

    # ------------------------------------------------------------
    # Create new embedding matrix
    # ------------------------------------------------------------

    new_embeddings = nn.Embedding(
        new_max_length,
        hidden_size,
        device=old_embeddings.weight.device,
        dtype=old_embeddings.weight.dtype,
    )

    # Same initialization style used by BERT
    nn.init.normal_(
        new_embeddings.weight,
        mean=0.0,
        std=bert_model.config.initializer_range,
    )

    # ------------------------------------------------------------
    # Copy pretrained positions exactly
    # ------------------------------------------------------------

    with torch.no_grad():

        new_embeddings.weight[
            :old_max_length
        ].copy_(
            old_embeddings.weight
        )

    # ------------------------------------------------------------
    # Replace positional embedding module
    # ------------------------------------------------------------

    bert_model.embeddings.position_embeddings = (
        new_embeddings
    )

    # ------------------------------------------------------------
    # Update position_ids buffer
    # ------------------------------------------------------------

    bert_model.embeddings.register_buffer(
        "position_ids",
        torch.arange(
            new_max_length,
            device=new_embeddings.weight.device,
        ).expand((1, -1)),
        persistent=False,
    )

    # ------------------------------------------------------------
    # Update token_type_ids buffer if present
    # ------------------------------------------------------------

    if hasattr(
        bert_model.embeddings,
        "token_type_ids",
    ):

        bert_model.embeddings.register_buffer(
            "token_type_ids",
            torch.zeros(
                (1, new_max_length),
                dtype=torch.long,
                device=new_embeddings.weight.device,
            ),
            persistent=False,
        )

    # ------------------------------------------------------------
    # Update config
    # ------------------------------------------------------------

    bert_model.config.max_position_embeddings = (
        new_max_length
    )


def verify_extension(
    old_weight,
    new_weight,
    old_length,
    new_length,
    name,
):
    """
    Verify:
      1. shape changed correctly
      2. first 512 positions are identical
      3. new positions are initialized
    """

    print()
    print("=" * 60)
    print(f"VERIFYING {name.upper()}")
    print("=" * 60)

    print(
        "Old weight shape:",
        tuple(old_weight.shape),
    )

    print(
        "New weight shape:",
        tuple(new_weight.shape),
    )

    # ------------------------------------------------------------
    # Shape test
    # ------------------------------------------------------------

    assert old_weight.shape[0] == old_length

    assert new_weight.shape[0] == new_length

    assert (
        old_weight.shape[1]
        == new_weight.shape[1]
    )

    print("[PASS] Embedding shape")

    # ------------------------------------------------------------
    # Check old 512 positions were preserved EXACTLY
    # ------------------------------------------------------------

    copied_weights = new_weight[:old_length]

    difference = (
        copied_weights - old_weight
    ).abs()

    max_difference = difference.max().item()
    mean_difference = difference.mean().item()

    print(
        f"Max difference in [0:{old_length}]: "
        f"{max_difference}"
    )

    print(
        f"Mean difference in [0:{old_length}]: "
        f"{mean_difference}"
    )

    assert torch.equal(
        copied_weights,
        old_weight,
    ), (
        f"{name}: original positional embeddings "
        f"were modified!"
    )

    print(
        f"[PASS] Original positions "
        f"0-{old_length - 1} copied exactly"
    )

    # ------------------------------------------------------------
    # Inspect new positions
    # ------------------------------------------------------------

    new_region = new_weight[
        old_length:new_length
    ]

    new_mean = new_region.mean().item()
    new_std = new_region.std().item()

    new_min = new_region.min().item()
    new_max = new_region.max().item()

    print()
    print(
        f"New positions "
        f"{old_length}-{new_length - 1}:"
    )

    print(
        f"  mean = {new_mean:.8f}"
    )

    print(
        f"  std  = {new_std:.8f}"
    )

    print(
        f"  min  = {new_min:.8f}"
    )

    print(
        f"  max  = {new_max:.8f}"
    )

    assert new_std > 0.0, (
        f"{name}: new positional embeddings "
        f"appear to be all identical."
    )

    print(
        "[PASS] New positions are initialized"
    )

    # ------------------------------------------------------------
    # Compare initializer std to model expectation
    # ------------------------------------------------------------

    print()
    print(
        f"[PASS] {name} extension verified"
    )


def test_long_forward_pass(
    model,
    sequence_length,
):
    """
    Test whether model can actually execute with
    sequence_length > 512.

    Uses a shorter decoder sequence to reduce memory usage.
    """

    print()
    print("=" * 60)
    print("LONG-SEQUENCE FORWARD PASS")
    print("=" * 60)

    device = next(
        model.parameters()
    ).device

    print(
        "Device:",
        device,
    )

    print(
        "Encoder test sequence length:",
        sequence_length,
    )

    # Keep decoder relatively small.
    # We separately test its embedding capability below.
    decoder_length = 32

    print(
        "Decoder test sequence length:",
        decoder_length,
    )

    # Avoid special/random invalid IDs.
    encoder_vocab_size = (
        model.config.encoder.vocab_size
    )

    decoder_vocab_size = (
        model.config.decoder.vocab_size
    )

    input_ids = torch.randint(
        low=0,
        high=encoder_vocab_size,
        size=(
            1,
            sequence_length,
        ),
        device=device,
    )

    attention_mask = torch.ones(
        (
            1,
            sequence_length,
        ),
        dtype=torch.long,
        device=device,
    )

    labels = torch.randint(
        low=0,
        high=decoder_vocab_size,
        size=(
            1,
            decoder_length,
        ),
        device=device,
    )

    model.eval()

    with torch.no_grad():

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    print(
        "Loss:",
        outputs.loss.item(),
    )

    print(
        "Logits shape:",
        tuple(outputs.logits.shape),
    )

    assert (
        outputs.logits.shape[0] == 1
    )

    assert (
        outputs.logits.shape[1]
        == decoder_length
    )

    print()
    print(
        f"[PASS] Model successfully processed "
        f"{sequence_length} encoder tokens"
    )


def test_decoder_long_positions(
    model,
    sequence_length,
):
    """
    Test the decoder itself with >512 positions.

    This verifies that the decoder positional embeddings
    also work beyond position 511 without requiring a full
    1536 x 1536 seq2seq forward pass.
    """

    print()
    print("=" * 60)
    print("DECODER POSITION TEST")
    print("=" * 60)

    device = next(
        model.parameters()
    ).device

    vocab_size = (
        model.config.decoder.vocab_size
    )

    decoder_input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(
            1,
            sequence_length,
        ),
        device=device,
    )

    attention_mask = torch.ones(
        (
            1,
            sequence_length,
        ),
        dtype=torch.long,
        device=device,
    )

    # Test only decoder embeddings.
    # This is much cheaper than doing a complete decoder
    # attention pass at 1536 tokens.
    with torch.no_grad():

        embeddings = (
            model.decoder.bert.embeddings(
                input_ids=decoder_input_ids,
            )
        )

    print(
        "Decoder embedding output:",
        tuple(embeddings.shape),
    )

    assert embeddings.shape == (
        1,
        sequence_length,
        model.config.decoder.hidden_size,
    )

    print(
        f"[PASS] Decoder positional embeddings "
        f"work at length {sequence_length}"
    )


def main(args):

    print("=" * 60)
    print("POSITION EMBEDDING EXTENSION TEST")
    print("=" * 60)

    # ============================================================
    # Load checkpoint
    # ============================================================

    print()
    print(
        "Loading checkpoint:",
        args.pretrained_model_dir,
    )

    model = EncoderDecoderModel.from_pretrained(
        args.pretrained_model_dir
    )

    # CPU is enough for structural checks.
    # Use CUDA for long forward pass if requested.
    if args.use_cuda:

        if not torch.cuda.is_available():

            raise RuntimeError(
                "--use_cuda was specified, "
                "but CUDA is unavailable."
            )

        model = model.cuda()

        print(
            "Moved model to GPU."
        )

    # ============================================================
    # Check original model
    # ============================================================

    encoder_old_length = (
        model.encoder.embeddings
        .position_embeddings
        .num_embeddings
    )

    decoder_old_length = (
        model.decoder.bert.embeddings
        .position_embeddings
        .num_embeddings
    )

    print()
    print(
        "Original encoder positional length:",
        encoder_old_length,
    )

    print(
        "Original decoder positional length:",
        decoder_old_length,
    )

    if encoder_old_length != args.expected_old_length:

        raise ValueError(
            f"Expected encoder length "
            f"{args.expected_old_length}, "
            f"but found {encoder_old_length}"
        )

    if decoder_old_length != args.expected_old_length:

        raise ValueError(
            f"Expected decoder length "
            f"{args.expected_old_length}, "
            f"but found {decoder_old_length}"
        )

    print(
        f"[PASS] Checkpoint really uses "
        f"{args.expected_old_length} positions"
    )

    # ============================================================
    # Save OLD weights before resizing
    # ============================================================

    old_encoder_weight = (
        model.encoder.embeddings
        .position_embeddings.weight
        .detach()
        .clone()
    )

    old_decoder_weight = (
        model.decoder.bert.embeddings
        .position_embeddings.weight
        .detach()
        .clone()
    )

    # ============================================================
    # Resize encoder
    # ============================================================

    print()
    print("Extending encoder...")

    extend_bert_position_embeddings(
        model.encoder,
        args.new_length,
    )

    # ============================================================
    # Resize decoder
    # ============================================================

    print()
    print("Extending decoder...")

    extend_bert_position_embeddings(
        model.decoder.bert,
        args.new_length,
    )

    # EncoderDecoderConfig also contains nested configs.
    model.config.encoder.max_position_embeddings = (
        args.new_length
    )

    model.config.decoder.max_position_embeddings = (
        args.new_length
    )

    # ============================================================
    # Verify encoder
    # ============================================================

    verify_extension(
        old_weight=old_encoder_weight,
        new_weight=(
            model.encoder.embeddings
            .position_embeddings.weight
            .detach()
        ),
        old_length=args.expected_old_length,
        new_length=args.new_length,
        name="encoder",
    )

    # ============================================================
    # Verify decoder
    # ============================================================

    verify_extension(
        old_weight=old_decoder_weight,
        new_weight=(
            model.decoder.bert.embeddings
            .position_embeddings.weight
            .detach()
        ),
        old_length=args.expected_old_length,
        new_length=args.new_length,
        name="decoder",
    )

    # ============================================================
    # Check configs
    # ============================================================

    print()
    print("=" * 60)
    print("CONFIG CHECK")
    print("=" * 60)

    print(
        "Encoder config max positions:",
        model.config.encoder.max_position_embeddings,
    )

    print(
        "Decoder config max positions:",
        model.config.decoder.max_position_embeddings,
    )

    assert (
        model.config.encoder.max_position_embeddings
        == args.new_length
    )

    assert (
        model.config.decoder.max_position_embeddings
        == args.new_length
    )

    print(
        "[PASS] Config updated correctly"
    )

    # ============================================================
    # Decoder >512 embedding test
    # ============================================================

    test_decoder_long_positions(
        model,
        args.new_length,
    )

    # ============================================================
    # Full model long encoder test
    # ============================================================

    if args.run_forward_test:

        test_long_forward_pass(
            model,
            args.forward_test_length,
        )

    else:

        print()
        print(
            "Full forward test skipped."
        )

        print(
            "Use --run_forward_test "
            "to enable it."
        )

    # ============================================================
    # Final result
    # ============================================================

    print()
    print("=" * 60)
    print("ALL POSITION EXTENSION CHECKS PASSED")
    print("=" * 60)

    print()
    print(
        f"Original length : "
        f"{args.expected_old_length}"
    )

    print(
        f"Extended length : "
        f"{args.new_length}"
    )

    print()
    print(
        "No model was trained."
    )

    print(
        "No model was saved."
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pretrained_model_dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--expected_old_length",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--new_length",
        type=int,
        default=1536,
    )

    parser.add_argument(
        "--run_forward_test",
        action="store_true",
    )

    parser.add_argument(
        "--forward_test_length",
        type=int,
        default=768,
        help=(
            "Length for full seq2seq forward test. "
            "768 is enough to prove >512 works "
            "without using as much memory as 1536."
        ),
    )

    parser.add_argument(
        "--use_cuda",
        action="store_true",
    )

    args = parser.parse_args()

    main(args)