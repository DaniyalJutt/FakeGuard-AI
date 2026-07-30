"""Check which packages from requirements.txt are installed"""
import importlib
import sys

packages_to_check = {
    'streamlit': 'streamlit',
    'pandas': 'pandas',
    'numpy': 'numpy',
    'scikit-learn': 'sklearn',
    'xgboost': 'xgboost',
    'transformers': 'transformers',
    'torch': 'torch',
    'scipy': 'scipy',
    'plotly': 'plotly',
    'wordcloud': 'wordcloud',
    'langdetect': 'langdetect',
    'emoji': 'emoji',
    'Pillow': 'PIL',
    'matplotlib': 'matplotlib',
    'seaborn': 'seaborn'
}

installed = []
missing = []

for package_name, import_name in packages_to_check.items():
    try:
        mod = importlib.import_module(import_name)
        version = getattr(mod, '__version__', 'unknown')
        installed.append(f"{package_name} ({version})")
    except ImportError:
        missing.append(package_name)

print("="*60)
print("INSTALLED PACKAGES:")
print("="*60)
for pkg in installed:
    print(f"✅ {pkg}")

print("\n" + "="*60)
print("MISSING PACKAGES:")
print("="*60)
if missing:
    for pkg in missing:
        print(f"❌ {pkg}")
else:
    print("✅ All packages are installed!")

