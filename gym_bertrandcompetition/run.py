register_env("your_multi_env", lambda c: YourMultiEnv(...))
trainer = PPOAgent(env="your_multi_env")
while True:
    print(trainer.train())  # distributed training step
