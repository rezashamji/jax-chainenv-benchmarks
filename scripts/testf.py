import pandas as pd, matplotlib.pyplot as plt
for diff in ["easy","medium","hard"]:
    df = pd.read_csv(f"runs/pqn/{diff}.csv", header=None, names=["steps","return"])
    plt.plot(df.steps, df["return"].rolling(10).mean(), label=diff)
plt.legend(); plt.xlabel("Environment steps"); plt.ylabel("Mean episodic return"); plt.title("PQN ChainEnv performance");
plt.show()
