import numpy as np
from dezero import test_mode
import dezero.functions as F

x = np.random.rand(10)
print(x)

# 学習時
y = F.dropout(x)
print(y)

# テスト時
with test_mode():
    y = F.dropout(x)
    print(y)