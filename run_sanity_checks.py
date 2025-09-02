import sys
from src.models.model_loader import load_gpt2_model
from src.sanity_checks.reproduce_papers import run_all_sanity_checks

print("Loading model...")
model, tokenizer = load_gpt2_model()

print("\nRunning sanity checks...")
success = run_all_sanity_checks(model, tokenizer)

if success:
    print("\n🎉 READY FOR EXPERIMENTS!")
else:
    print("\n❌ Fix issues before proceeding")
    sys.exit(1)
