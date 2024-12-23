import torch
from transformer_project.modelling.transformer import Transformer
from transformer_project.modelling.huggingface_bpe_tokenizer import CustomTokenizer
from transformer_project.preprocessing.clean_data import load_or_clean_data


def load_model(checkpoint_path: str, dim_feedforward: int, device: str, d_model: int):
    model = Transformer(
        d_model=d_model,
        vocab_size=50000,
        n_heads=8,
        num_decoder_layers=4,
        num_encoder_layers=4,
        dim_feed_forward=128,
        dropout=0.0,
        maxlen=32,
    ).to(device)

    model.load_state_dict(torch.load(checkpoint_path))
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    return model.to(device)


def translate_sentence(model, text: str, tokenizer, device="cuda", max_len=32):
    model.eval()
    bos_token_id = tokenizer.bos_token_id
    pad_token_id = tokenizer.pad_token_id

    decoder_input_token_ids = torch.tensor(
        [bos_token_id] + [pad_token_id] * (max_len - 1)
    ).to(device)

    print(f"Decoder input token ids: {decoder_input_token_ids}")
    input_ids = tokenizer.encode(
        text, padding="max_length", max_length=max_len, truncation=True
    )

    input_tensor = torch.tensor(input_ids).unsqueeze(0).to(device)

    # Encoder mask to indicate which tokens are padding tokens
    encoder_mask = (input_tensor != pad_token_id).int()
    print(f"Input tensor: {input_tensor}")
    print(f"Input tensor shape: {input_tensor.shape}")

    for i in range(max_len - 1):
        with torch.no_grad():
            output = model(input_tensor, decoder_input_token_ids, encoder_mask)
        print(f"output shape: {output.shape}")
        print(f"output: {output}")
        next_token = torch.argmax(output, dim=-1)
        print(f"Next token id: {next_token}")
        next_token_id = next_token[0, i].item()

        decoder_input_token_ids[i + 1] = next_token_id

        if next_token_id == tokenizer.eos_token_id:
            break

        print(f"Decoder input: {decoder_input_token_ids}")

    translation = tokenizer.decode(decoder_input_token_ids, skip_special_tokens=True)

    return translation


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_model = 64
    dim_feed_forward = 4 * d_model
    model = load_model("best_model.pth", dim_feed_forward, device, d_model)

    tokenizer = CustomTokenizer(
        vocab_size=50000, corpus_file=str("byte-level-bpe_wmt17.tokenizer.json")
    ).load_gpt2_tokenizer()

    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    cleaned_test_data = load_or_clean_data("test[:1%]")
    text = "Und damit endete die Reitkarriere dann auch schon wieder."
    translation = translate_sentence(model, text, tokenizer, device)
    print(f"Input: {text}")
    print(f"Translation: {translation}")
