from hint_engine import HintEngine
import os

print("=" * 50)
print("🎓 Math Hint Chatbot - Model Training")
print("=" * 50)

# Purana cache delete karo taaki fresh embeddings bane
cache_file = 'data/embeddings_cache.pkl'
if os.path.exists(cache_file):
    os.remove(cache_file)
    print("🗑️ Old cache deleted — fresh embeddings banenge")

engine = HintEngine(csv_path='data/maths_only.csv')

print("\n✅ Training complete!")
print(f"📊 Total questions indexed: {len(engine.df)}")
print("\nAb now you can run the python app.py")
print("    python app.py")