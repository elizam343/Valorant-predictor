#!/usr/bin/env python3
"""
Simple setup script for clean Valorant data collection
Run this to get started with the new data collection system
"""

import os
import sys
import subprocess

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 7):
        print("❌ Python 3.7+ required. Current version:", sys.version)
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def install_requirements():
    """Install required packages"""
    requirements = [
        "requests",
        "tqdm", 
        "pandas",
        "numpy",
        "scikit-learn"
    ]
    
    print("📦 Installing required packages...")
    for package in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")
            return False
    
    return True

def backup_old_database():
    """Backup corrupted database"""
    old_db = "Scraper/valorant_matches.db"
    backup_db = "Scraper/valorant_matches_CORRUPTED_BACKUP.db"
    
    if os.path.exists(old_db):
        try:
            os.rename(old_db, backup_db)
            print(f"✅ Backed up corrupted database to: {backup_db}")
        except Exception as e:
            print(f"⚠️ Could not backup old database: {e}")
    else:
        print("ℹ️ No old database found to backup")

def test_new_system():
    """Test if the new system can be imported"""
    try:
        sys.path.insert(0, 'Scraper')
        from riot_api_client import RiotAPIClient
        from clean_database_schema import CleanValorantDatabase
        from collect_clean_data import CleanDataCollector
        print("✅ New data collection system ready!")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def get_api_key_instructions():
    """Show instructions for getting Riot API key"""
    print("\n🔑 RIOT API KEY SETUP:")
    print("=" * 40)
    print("1. Go to: https://developer.riotgames.com/")
    print("2. Sign in with your Riot account")
    print("3. Click 'PERSONAL API KEY'")
    print("4. Copy the API key (starts with 'RGAPI-')")
    print("5. Note: Personal API keys expire every 24 hours")
    print("\n⚠️ IMPORTANT:")
    print("   - Personal API keys are for development only")
    print("   - They have a 100 requests/2 minutes rate limit")
    print("   - For production, apply for a production key")

def main():
    """Main setup function"""
    print("🎮 VALORANT CLEAN DATA COLLECTION SETUP")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed during package installation")
        return
    
    # Backup old database
    backup_old_database()
    
    # Test new system
    if not test_new_system():
        print("❌ Setup failed - could not import new system")
        return
    
    print("\n✅ SETUP COMPLETE!")
    print("\n📋 NEXT STEPS:")
    print("1. Get your Riot API key (see instructions below)")
    print("2. Run: python Scraper/collect_clean_data.py")
    print("3. Choose option 1 for a quick test")
    print("4. If test works, collect more data with option 2")
    
    # Show API key instructions
    get_api_key_instructions()
    
    print(f"\n🚀 Ready to collect clean Valorant data!")
    print(f"   Run: cd Scraper && python collect_clean_data.py")

if __name__ == "__main__":
    main() 