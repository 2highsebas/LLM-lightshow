import struct
import sentencepiece as spm

# 1. Load the tokenizer
sp = spm.SentencePieceProcessor(model_file="trained_001.model")

# 2. Load the binary weights (sanity check)
with open("trained_001_corpus.bin", "rb") as f:
    blob = f.read()

print(f"✅ Loaded corpus ({len(blob):,} bytes)")

# 3. Simple prompt — for example, frequency bands
prompt = "F 42 L0.204,0.261,0.086,0.021,0.011,0.009,0.010,0.006 ->"

# 4. Encode it to tokens
tokens = sp.encode(prompt)
print(f"Prompt tokens: {tokens[:20]} ...")

# 5. Decode back for sanity check
decoded = sp.decode(tokens)
print("Decoded text:", decoded)

# 6. Fake inference loop (for now)
print("\n=== Simulated LED output ===")
print("T42 B0:ON B1:OFF B2:OFF B3:ON B4:OFF B5:ON B6:OFF B7:ON")