# Google Colab Setup for Valorant Predictor Training

This folder contains all the necessary Python scripts and modules to run your Valorant kill prediction model training in Google Colab, except for the database file.

## How to Use

1. **Upload your database**
   - Upload your `valorant_matches.db` file (from `Scraper/valorant_matches.db`) to the Colab environment or to your Google Drive and mount it in Colab.

2. **Upload these files**
   - Upload all files from this `google_colab` folder to your Colab environment (or copy-paste code as needed).

3. **Install dependencies**
   - In a Colab cell, run:
     ```python
     !pip install torch pandas scikit-learn joblib tqdm
     ```

4. **Adjust paths if needed**
   - If your database is in Google Drive, update the path in your code to `/content/drive/MyDrive/valorant_matches.db` or wherever you place it.

5. **Run training**
   - Run `gpu_trainer.py` as you would locally. Colab will use its GPU automatically if you select a GPU runtime.

## Included Files
- `gpu_trainer.py`
- `database_data_loader.py`
- `advanced_matchup_predictor.py`
- `predict_kills.py`
- `enhanced_data_loader.py`
- `README.md` (this file)
- Any other utility scripts needed for model training (except the database)

## Not Included
- The database file (`valorant_matches.db`). Upload this separately.
- Large model files (pkl) or training reports.

---

If you need help with Colab setup or have questions about adapting your workflow, let me know! 