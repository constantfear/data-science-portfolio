# Gemma2-2b Quantization

[Исходная модель](IlyaGusev/gemma-2-2b-it-abliterated) 

[Ссылка](https://www.kaggle.com/code/ramiltiteev/gemma2b-quantization) на Kaggle Notebook

### Методы квантования:
- GPTQ: [HF](https://huggingface.co/feelconstantfear/gemma-2-2b-it-abliterated-GPTQ-4bit)
- GGUF (Q_8): [HF](https://huggingface.co/feelconstantfear/gemma-2-2b-it-abliterated-Q8_0-GGUF)

### Сравнение моделей:
Сравнение проводилось по метрике Perplexity на датасете [c4](allenai/c4)

- **PPL Source Model**: 15.8594
- **PPL GPTQ Model**: 16.8347