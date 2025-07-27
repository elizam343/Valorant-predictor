db_path = "valorant_matches.db"
trainer = GPUTrainer()
trainer.data_loader = DatabaseDataLoader(db_path=db_path)
import time
start_time = time.time()
results = trainer.train_all_models(limit_matches=None)
elapsed = time.time() - start_time
print(f"Total training time: {elapsed/60:.2f} minutes ({elapsed:.1f} seconds)") 