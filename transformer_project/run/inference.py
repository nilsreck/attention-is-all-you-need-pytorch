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
        dim_feed_forward=dim_feedforward,
        dropout=0.0,
        maxlen=64,
    ).to(device)

    model.load_state_dict(torch.load(checkpoint_path))
    #     print("Model's state_dict:")
    #     for param_tensor in model.state_dict():
    #         print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    return model.to(device)


def translate_sentence(model, text: str, tokenizer, device="cuda", max_len=32):
    model.eval()
    bos_token_id = tokenizer.bos_token_id
    pad_token_id = tokenizer.pad_token_id

    with torch.no_grad():
        src_seq = tokenizer.encode(text, return_tensors="pt").to(device)
        encoder_mask = (src_seq != pad_token_id).int()
        encoder_output = model.encode(src_seq, encoder_mask)

        init_dec_input = torch.tensor([[bos_token_id]], device=device)

        for _ in range(2, max_len - 1):
            output = model.decode(init_dec_input, encoder_output, encoder_mask)
            print(f"Output.shape: {output.shape}")
            next_token = torch.argmax(output[:, -1, :], dim=-1)

            init_dec_input = torch.cat([init_dec_input, next_token.unsqueeze(1)], dim=1)
            print(f"Decoder input: {init_dec_input}")

            if next_token.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(init_dec_input[0], skip_special_tokens=True)


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
    text = "wer hat das huhn gegessen?"
    translation = translate_sentence(model, text, tokenizer, device)
    print(f"Translation: {translation}")
