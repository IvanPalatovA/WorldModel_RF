# WorldModel_RF
Запуск локально:
`python3 main_demo.py`
Технический стек: `main_demo.py`, PyTorch, Gymnasium/MiniGrid, Transformers. Обучение запускал в Google Colab на NVIDIA T4 (CUDA): сбор 80k переходов (`expert_mix`), затем обучение RSSM как world model. Планирование — MPC random shooting (`K=32`, `H=10`, `gamma=0.99`) по imagined rollouts. VLM-оценка — CLIP `openai/clip-vit-base-patch32` по будущим кадрам. Сравнение: Random, WM reward, WM+VLM; выходы — метрики и GIF.