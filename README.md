# Automatic Modulation Classification with Deep Learning

CNN (PyTorch) that classifies radio modulation types from raw IQ samples on RadioML 2016.10A.

## Run
```bash
pip install -r requirements.txt
python -m src.train --data path/to/RML2016.10a_dict.pkl
python -m src.evaluate --data path/to/RML2016.10a_dict.pkl
```

## Results
_Add accuracy-vs-SNR plot and confusion matrix from `results/` here._

## To do
- [ ] Classical baseline (cumulants + SVM)
- [ ] ResNet variant
- [ ] Gradio demo on Hugging Face Spaces
