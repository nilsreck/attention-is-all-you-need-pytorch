import torch
from transformer_project.modelling.transformer import Transformer
from transformer_project.modelling.huggingface_bpe_tokenizer import CustomTokenizer


def load_model(checkpoint_path: str, device: str = "cuda"):
    model = Transformer(
        vocab_size=50000,
        d_model=32,
        n_heads=8,
        num_decoder_layers=2,
        num_encoder_layers=2,
        dim_feedforward=128,
        dropout=0.0,
        maxlen=32,
    ).to(device)

    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    return model


def translate_sentence(model, text: str, tokenizer, device="cuda", max_len=32):
    with torch.no_grad():
        input_ids = tokenizer.encode(
            text, padding="max_length", max_length=max_len, truncation=True
        )
        print(f"Input token ids: {input_ids}")
        input_tensor = torch.tensor(input_ids).unsqueeze(0).to(device)
        print(f"Input tensor: {input_tensor}")
        decoder_input = torch.tensor([[tokenizer.bos_token_id]]).to(device)

        output_ids = []
        for _ in range(max_len):
            output = model(input_tensor, decoder_input)
            print(f"output shape: {output.shape}")
            next_token = output[0, -1].argmax()
            print(f"Next token id: {next_token}")
            next_token_id = next_token.item()

            output_ids.append(next_token_id)

            if next_token_id == tokenizer.eos_token_id:
                break

            decoder_input = torch.cat(
                [decoder_input, next_token.unsqueeze(0).unsqueeze(0)], dim=1
            )

        return translation


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model("best_model.pt", device)

    tokenizer = CustomTokenizer(
        vocab_size=50000,
    ).load_gpt2_tokenizer()

    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    text = "Er wollte nie an irgendeiner Art von Auseinandersetzung teilnehmen."
    translation = translate_sentence(model, text, tokenizer, device)
    print(f"Input: {text}")
    print(f"Translation: {translation}")
