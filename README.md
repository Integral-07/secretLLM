  # 1回目: Phase 1 (ベース学習) → 重みとシークレットを保存         
  python3 -m examples.demo_training                             
                                                                   
  # 以降: Phase 2 だけ                                             
  python3 -m examples.demo_finetune --session sess-001

  # Phase 2 + セッション切り替え
  python3 -m examples.demo_finetune --session sess-001 --adapt
  sess-002

  # Phase 2 スキップして adapt だけ
  python3 -m examples.demo_finetune --session sess-001 --epochs 0
  --adapt sess-002

  # パラメータ調整
  python3 -m examples.demo_finetune --session sess-001 --epochs 5
  --lr 2e-4 --adapt-steps 100



python3 -m examples.demo_training
  python3 -m examples.demo_finetune --session sess-001 --steps 300
  python3 -m examples.demo_inference --session sess-001
  --temperature 0.1
